"""
Registry de modele de EXTRACTIE (v1..v18, fara v14) pentru pipeline-ul
extractie -> clasificare miscare (clasificatoarele lui Edward 14/15/18).

Maparea checkpoint -> arhitectura -> pipeline de inferenta:
  - .res = '400'  : pipeline nativ 128x400 (movement_dataset, infer_128x400),
                    fara decimare. Folosit de v1..v13 si v18.
  - .res = '128'  : pipeline decimat 128x128 (movement_dataset_v15, infer_128x128).
                    Folosit de v15/v16/v17.
Mastile (soft/gain) sunt aplicate IN forward() pentru modelele care le folosesc,
deci model(x) intoarce direct estimarea spectrograma -> istft.

v2/v3/v4: scripturile de training (archive/) au disparut; maparea checkpoint
e DEDUSA (dimensiune fisier + EXPERIMENTS.md) si NEVERIFICATA. Poarta de
validare e F1 R-peak vs GT din smoke_pipeline.py.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import scipy.signal
import librosa
from scipy.signal import resample_poly

from movement_dataset import (
    _stft_multichannel, _to_resized_complex_tensor,
    NFFT as NFFT1k, HOP_LENGTH as HOP1k, WIN_LENGTH as WIN1k,
    TARGET_SIZE_F, TARGET_SIZE_T, FS as FS1k,
)
from movement_dataset_v15 import (
    NFFT as NFFT5, HOP_LENGTH as HOP5, WIN_LENGTH as WIN5,
    WINDOW as WINDOW5,
)

from complex_network import ComplexUNet as CUNet_v1
from complex_network_v7 import ComplexUNetV7
from complex_network_v9 import ComplexUNetV9
from complex_network_v10 import ComplexUNetV10
from complex_network_v11 import ComplexUNetV11
from complex_network_v12 import ComplexUNetV12
from complex_network_v13 import ComplexUNetV13
from complex_network_v15 import ComplexUNet as CUNet_v15
from complex_network_v16 import ComplexAttentionUNet

from config import MODELS_DIR

WINDOW_400 = 4 * FS1k                 # 4000
DIM_400    = TARGET_SIZE_F * TARGET_SIZE_T
DIM_128    = 128 * 128
ORIG_F_400 = NFFT1k // 2 + 1
ORIG_T_400 = 1 + WINDOW_400 // HOP1k

_hann128 = torch.hann_window(WIN5)
_hann400 = torch.hann_window(WIN1k)

# vname -> (ArchClass, dim, in_channels, checkpoint_filename, res, verified)
REGISTRY = {
    'v1':  (CUNet_v1,            DIM_400, 6, 'movement_CUNet_128x400_composed.pth',     '400', True),
    'v2':  (CUNet_v15,           DIM_128, 6, 'movement_CUNet_128x128_paper.pth',        '128', False),
    'v3':  (CUNet_v1,            DIM_400, 6, 'movement_CUNet_128x400_peakw.pth',        '400', False),
    'v4':  (CUNet_v1,            DIM_400, 6, 'movement_CUNet_128x400_ampw.pth',         '400', False),
    'v5':  (CUNet_v1,            DIM_400, 6, 'movement_CUNet_128x400_ampw_scratch.pth', '400', True),
    'v6':  (CUNet_v1,            DIM_400, 6, 'movement_CUNet_128x400_v6_baseline.pth',  '400', True),
    'v7':  (ComplexUNetV7,       DIM_400, 6, 'movement_CUNet_v7_paper_direct.pth',      '400', True),
    'v8':  (ComplexUNetV7,       DIM_400, 6, 'movement_CUNet_v8_mse.pth',               '400', True),
    'v9':  (ComplexUNetV9,       DIM_400, 6, 'movement_CUNet_v9_mask.pth',              '400', True),
    'v10': (ComplexUNetV10,      DIM_400, 6, 'movement_CUNet_v10_fqrs.pth',             '400', True),
    'v11': (ComplexUNetV11,      DIM_400, 6, 'movement_CUNet_v11_direct_peak.pth',      '400', True),
    'v12': (ComplexUNetV12,      DIM_400, 6, 'movement_CUNet_v12_gainmask.pth',         '400', True),
    'v13': (ComplexUNetV13,      DIM_400, 6, 'movement_CUNet_v13_instanorm.pth',        '400', True),
    'v15': (CUNet_v15,           DIM_128, 6, 'movement_CUNet_v15_paper.pth',            '128', True),
    'v16': (ComplexAttentionUNet,DIM_128, 6, 'movement_CUNet_v16_attention.pth',        '128', True),
    'v17': (ComplexAttentionUNet,DIM_128, 6, 'movement_CUNet_v17_attsup.pth',           '128', True),
    'v18': (ComplexAttentionUNet,DIM_400, 6, 'movement_CUNet_v18_500hz.pth',             '400', True),
}
MODEL_ORDER = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v13',
               'v15','v16','v17','v18']


def load_extractor(vname, device='cpu'):
    arch, dim, cin, ckpt, res, _ = REGISTRY[vname]
    path = os.path.join(MODELS_DIR, ckpt)
    m = arch(dim, in_channels=cin)
    state = torch.load(path, map_location=device)
    m.load_state_dict(state)        # strict=True implicit -> prinde mismatch arhitectura
    return m.to(device).eval()


def _infer_128x400(model, mixture, device='cpu'):
    n_ch, total = mixture.shape
    n_win = total // WINDOW_400
    out = np.zeros((n_ch, n_win * WINDOW_400), dtype=np.float32)
    for i in range(n_win):
        s = i * WINDOW_400; e = s + WINDOW_400
        win  = mixture[:, s:e].copy()
        stds = np.where(win.std(1, keepdims=True) < 1e-8, 1.0, win.std(1, keepdims=True))
        norm = (win / stds).astype(np.float32)
        spec = _stft_multichannel(norm, NFFT1k, HOP1k, WIN1k)
        x = _to_resized_complex_tensor(spec, TARGET_SIZE_F, TARGET_SIZE_T).unsqueeze(0).to(device)
        with torch.no_grad():
            o = model(x).squeeze(0).cpu()
        C, H, W = o.shape
        real_r = F.interpolate(o.real.reshape(C,1,H,W), size=(ORIG_F_400, ORIG_T_400),
                               mode='bilinear', align_corners=False).squeeze(1)
        imag_r = F.interpolate(o.imag.reshape(C,1,H,W), size=(ORIG_F_400, ORIG_T_400),
                               mode='bilinear', align_corners=False).squeeze(1)
        o_full = torch.complex(real_r, imag_r)
        pred = np.stack([torch.istft(o_full[ch], n_fft=NFFT1k, hop_length=HOP1k,
                                     win_length=WIN1k, window=_hann400,
                                     length=WINDOW_400).numpy()
                         for ch in range(n_ch)])
        out[:, s:e] = pred * stds
    return out


def _infer_128x128(model, mixture, device='cpu'):
    mix = scipy.signal.decimate(mixture, 2, axis=1).astype(np.float32)
    n_ch, total = mix.shape
    n_win = total // WINDOW5
    out = np.zeros((n_ch, n_win * WINDOW5), dtype=np.float32)
    for i in range(n_win):
        s = i * WINDOW5; e = s + WINDOW5
        win  = mix[:, s:e].copy()
        stds = np.where(win.std(1, keepdims=True) < 1e-8, 1.0, win.std(1, keepdims=True))
        norm = (win / stds).astype(np.float32)
        specs = np.stack([librosa.stft(norm[ch], n_fft=NFFT5, hop_length=HOP5,
                                       win_length=WIN5, center=True)
                          for ch in range(n_ch)])
        spec = specs[:, :-1, :]
        x = (torch.from_numpy(spec.real.copy()) +
             1j * torch.from_numpy(spec.imag.copy())).unsqueeze(0).to(device)
        with torch.no_grad():
            o = model(x).squeeze(0).cpu()
        zeros  = torch.zeros(o.shape[0], 1, o.shape[2], dtype=o.dtype)
        o_full = torch.cat([o, zeros], dim=1)
        pred = np.stack([torch.istft(o_full[ch], n_fft=NFFT5, hop_length=HOP5,
                                     win_length=WIN5, window=_hann128,
                                     center=True, length=WINDOW5).numpy()
                         for ch in range(n_ch)])
        out[:, s:e] = pred * stds
    return resample_poly(out, 2, 1, axis=1).astype(np.float32)


def infer(vname, model, mixture, device='cpu'):
    res = REGISTRY[vname][4]
    return _infer_128x400(model, mixture, device) if res == '400' \
        else _infer_128x128(model, mixture, device)
