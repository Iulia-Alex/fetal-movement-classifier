"""
MovementECGDatasetV17 — paper-faithful (500 Hz, 128×128 natural) + attention target.

Compared to v15:
  - Loads fqrs annotations (the fetal R-peak positions at 1000 Hz)
  - Builds a binary temporal mask at the T=128 resolution (spectrogram):
      1.0 in the ±80ms window (±3 T-frames) around each R-peak
      0.0 elsewhere
  - Returns (mix_spec, fecg_spec, fecg_time, att_target_T128)

att_target_T128 : (128,) float32 — the attention mask for AG1 at full resolution.
  In training, an avg_pool 128→64 and broadcast to (1, 64, 64) is applied to
  match the shape of AG1's alpha.
"""

import os
import numpy as np
import torch
import scipy.io as sio

from movement_dataset_v15 import (
    MovementECGDatasetPaper,
    NFFT, HOP_LENGTH, WIN_LENGTH,
    FS, WINDOW, TARGET_SIZE_F, TARGET_SIZE_T,
    _stft_multichannel, _load_signals_500hz,
)

N_FRAMES = TARGET_SIZE_T   # 128
DIL_MS   = 80              # ±80ms ±40 samples la 500Hz ≈ ±2.67 T-frame-uri → ±3
DIL_FRAMES = int(np.ceil(DIL_MS / 1000.0 * FS / HOP_LENGTH))  # = ceil(40/15) = 3


def _fqrs_npz_path(mat_path):
    return os.path.splitext(mat_path)[0] + '_fqrs.npz'


def _load_fqrs_1000hz(mat_path):
    """Load fqrs at 1000 Hz from .npz cache or directly from .mat."""
    cache = _fqrs_npz_path(mat_path)
    if os.path.exists(cache):
        return np.load(cache)['fqrs']
    mat = sio.loadmat(mat_path)
    raw = mat['out']['fqrs'][0][0]
    fqrs = raw.flat[0].astype(np.int32).flatten()
    del mat
    try:
        np.savez_compressed(cache, fqrs=fqrs)
    except Exception as e:
        print(f'[WARN] Could not write fqrs cache {cache}: {e}')
    return fqrs


def _make_att_target(fqrs_1000, start_500, n_frames=N_FRAMES,
                     hop=HOP_LENGTH, dil=DIL_FRAMES):
    """
    Builds the attention mask of shape (n_frames,) at the spectrogram's T resolution.
    fqrs_1000 : the R-peak positions at 1000 Hz
    start_500 : the start index of the window at 500 Hz
    """
    mask = np.zeros(n_frames, dtype=np.float32)
    fqrs_500 = fqrs_1000 // 2   # conversion 1000 Hz → 500 Hz
    local = (fqrs_500[(fqrs_500 >= start_500) & (fqrs_500 < start_500 + WINDOW)]
             - start_500)
    for p in local:
        t = int(round(p / hop))   # nearest T-frame
        lo = max(0, t - dil)
        hi = min(n_frames, t + dil + 1)
        mask[lo:hi] = 1.0
    return mask


class MovementECGDatasetV17(MovementECGDatasetPaper):
    """
    Extends v15 with the attention mask for AG1 supervision.

    Returns: (mix_spec, fecg_spec, fecg_time, att_target_T128)
      mix_spec, fecg_spec  : (6, 128, 128) complex tensors
      fecg_time            : (6, 1915) float32
      att_target_T128      : (128,) float32  — the temporal mask for AG1
    """

    def __init__(self, data_dir):
        super().__init__(data_dir)
        self._fqrs_cache: dict = {}

    def _get_fqrs(self, file_idx):
        if file_idx not in self._fqrs_cache:
            try:
                fqrs = _load_fqrs_1000hz(self.files[file_idx])
                self._fqrs_cache[file_idx] = fqrs
            except Exception as e:
                print(f'[WARN] Could not load fqrs for '
                      f'{os.path.basename(self.files[file_idx])}: {e}')
                self._fqrs_cache[file_idx] = np.array([], dtype=np.int32)
        return self._fqrs_cache[file_idx]

    def __getitem__(self, idx):
        for attempt in range(20):
            try:
                candidate = (idx + attempt * self._wins_per_file) % len(self.windows)
                file_idx, start = self.windows[candidate]
                mixture, fecg = self._get_signals(file_idx)
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f'Could not load valid data after 20 attempts (idx={idx})')

        end = start + WINDOW
        mix_win  = mixture[:, start:end].copy()
        fecg_win = fecg[:,   start:end].copy()

        stds = mix_win.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        mix_win  = mix_win  / stds
        fecg_win = fecg_win / stds

        mix_spec  = _stft_multichannel(mix_win,  NFFT, HOP_LENGTH, WIN_LENGTH)
        fecg_spec = _stft_multichannel(fecg_win, NFFT, HOP_LENGTH, WIN_LENGTH)

        mix_spec  = mix_spec[:,  :-1, :]
        fecg_spec = fecg_spec[:, :-1, :]

        x = torch.complex(
            torch.from_numpy(np.real(mix_spec).copy()),
            torch.from_numpy(np.imag(mix_spec).copy()),
        )
        y = torch.complex(
            torch.from_numpy(np.real(fecg_spec).copy()),
            torch.from_numpy(np.imag(fecg_spec).copy()),
        )

        fecg_time = torch.from_numpy(fecg_win).float()

        fqrs      = self._get_fqrs(file_idx)
        att_mask  = _make_att_target(fqrs, start_500=start)
        att_target = torch.from_numpy(att_mask)   # (128,)

        return x, y, fecg_time, att_target


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data_dir = '../data/movement_ecg'
    dset = MovementECGDatasetV17(data_dir)
    print(f'Total windows: {len(dset)}')
    x, y, ft, att = dset[0]
    print(f'  x           : {x.shape}  dtype={x.dtype}')
    print(f'  y           : {y.shape}  dtype={y.dtype}')
    print(f'  fecg_time   : {ft.shape}  dtype={ft.dtype}')
    print(f'  att_target  : {att.shape}  n_active={att.sum():.0f}/{len(att)} '
          f'coverage={att.mean()*100:.1f}%')
    print(f'  dil_frames  : {DIL_FRAMES}  (±{DIL_MS}ms = ±{DIL_FRAMES} T-frames)')
    print('OK')
