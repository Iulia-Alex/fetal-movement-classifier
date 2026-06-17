"""
Calcul F1 score pentru detectia R-peaks QRS pe Final_Test_DB.

Metoda:
  1. Detectam R-peaks in fECG-ul prezis si in GT folosind find_peaks.
  2. Doua peaks se "potrivesc" daca sunt la distanta <= TOLERANCE_MS.
  3. F1 = 2*TP / (2*TP + FP + FN)

Parametri QRS detection:
  - Bandpass: 3-40 Hz (izolam QRS fetal)
  - Min distanta intre peaks: 250ms (max ~240 bpm)
  - Toleranta matching: 50ms (standard in literatura fECG)
  - Polaritate: detectam peaks pozitive si negative, folosim setul mai mare
"""

import sys, os, json, re, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa

import numpy as np
import torch
import torch.nn.functional as F
import scipy.signal
import librosa
from scipy.signal import find_peaks, butter, filtfilt
from scipy.signal import resample_poly

from movement_dataset import (
    _stft_multichannel, _to_resized_complex_tensor,
    NFFT as NFFT1k, HOP_LENGTH as HOP1k, WIN_LENGTH as WIN1k,
    TARGET_SIZE_F, TARGET_SIZE_T, FS as FS1k,
)
from movement_dataset_v15 import (
    NFFT as NFFT5, HOP_LENGTH as HOP5, WIN_LENGTH as WIN5,
    FS as FS5, WINDOW as WINDOW5,
)
from complex_network import ComplexUNet as CUNetV1
from complex_network_v16 import ComplexAttentionUNet

from config import NPY as NPY_BASE, MODELS_DIR, RESULTS_DIR as _RES
OUT_JSON   = _RES + '/final_testdb_f1.json'

WINDOW_1k = 4 * FS1k
DIM_1k    = TARGET_SIZE_F * TARGET_SIZE_T
DIM_500   = 128 * 128
ORIG_F_1k = NFFT1k // 2 + 1
ORIG_T_1k = 1 + WINDOW_1k // HOP1k

TOLERANCE_MS  = 50    # ms fereastra matching peaks
MIN_DIST_MS   = 250   # ms distanta minima intre peaks (max 240 bpm)
BANDPASS_LOW  = 3     # Hz
BANDPASS_HIGH = 40    # Hz

PATTERN = re.compile(r'SNRmn=([-\d]+)dB_SNRfm=([-\d]+)dB_SNRfn=([-\d]+)dB')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}', flush=True)

_hann5  = torch.hann_window(WIN5)
_hann1k = torch.hann_window(WIN1k)


def best_epoch(p):
    h = json.load(open(p))
    vl = h.get('val_loss', [])
    return vl.index(min(vl)) + 1 if vl else 0


# ── load models ──────────────────────────────────────────────────────────────
print('Loading models...', flush=True)

m_v1 = CUNetV1(DIM_1k, in_channels=6)
m_v1.load_state_dict(torch.load(f'{MODELS_DIR}/movement_CUNet_128x400_composed.pth', map_location=device))
m_v1 = m_v1.to(device).eval()
ep_v1 = best_epoch(f'{MODELS_DIR}/movement_CUNet_128x400_composed_history.json')

m_v16 = ComplexAttentionUNet(DIM_500, in_channels=6)
m_v16.load_state_dict(torch.load(f'{MODELS_DIR}/movement_CUNet_v16_attention.pth', map_location=device))
m_v16 = m_v16.to(device).eval()
ep_v16 = best_epoch(f'{MODELS_DIR}/movement_CUNet_v16_attention_history.json')

m_v17 = ComplexAttentionUNet(DIM_500, in_channels=6)
m_v17.load_state_dict(torch.load(f'{MODELS_DIR}/movement_CUNet_v17_attsup.pth', map_location=device))
m_v17 = m_v17.to(device).eval()
ep_v17 = best_epoch(f'{MODELS_DIR}/movement_CUNet_v17_attsup_history.json')

print(f'  v1 ep{ep_v1} | v16 ep{ep_v16} | v17 ep{ep_v17}', flush=True)


# ── inferenta ────────────────────────────────────────────────────────────────
def infer_full_1k(model, mixture_1k):
    n_ch, total = mixture_1k.shape
    n_win = total // WINDOW_1k
    out   = np.zeros((n_ch, n_win * WINDOW_1k), dtype=np.float32)
    for i in range(n_win):
        s = i * WINDOW_1k; e = s + WINDOW_1k
        win  = mixture_1k[:, s:e].copy()
        stds = np.where(win.std(axis=1, keepdims=True) < 1e-8, 1.0, win.std(axis=1, keepdims=True))
        norm = (win / stds).astype(np.float32)
        spec = _stft_multichannel(norm, NFFT1k, HOP1k, WIN1k)
        x    = _to_resized_complex_tensor(spec, TARGET_SIZE_F, TARGET_SIZE_T).unsqueeze(0).to(device)
        with torch.no_grad():
            o = model(x).squeeze(0).cpu()
        C, H, W = o.shape
        real_r = F.interpolate(o.real.reshape(C,1,H,W), size=(ORIG_F_1k, ORIG_T_1k),
                               mode='bilinear', align_corners=False).squeeze(1)
        imag_r = F.interpolate(o.imag.reshape(C,1,H,W), size=(ORIG_F_1k, ORIG_T_1k),
                               mode='bilinear', align_corners=False).squeeze(1)
        o_full = torch.complex(real_r, imag_r)
        pred = np.stack([torch.istft(o_full[ch], n_fft=NFFT1k, hop_length=HOP1k,
                                     win_length=WIN1k, window=_hann1k,
                                     length=WINDOW_1k).numpy()
                         for ch in range(n_ch)])
        out[:, s:e] = pred * stds
    return out


def infer_full_500(model, mixture_1k):
    mix_500 = scipy.signal.decimate(mixture_1k, 2, axis=1).astype(np.float32)
    n_ch, total = mix_500.shape
    n_win = total // WINDOW5
    out   = np.zeros((n_ch, n_win * WINDOW5), dtype=np.float32)
    for i in range(n_win):
        s = i * WINDOW5; e = s + WINDOW5
        win  = mix_500[:, s:e].copy()
        stds = np.where(win.std(axis=1, keepdims=True) < 1e-8, 1.0, win.std(axis=1, keepdims=True))
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
                                     win_length=WIN5, window=_hann5,
                                     center=True, length=WINDOW5).numpy()
                         for ch in range(n_ch)])
        out[:, s:e] = pred * stds
    return resample_poly(out, 2, 1, axis=1).astype(np.float32)


# ── QRS detection ────────────────────────────────────────────────────────────
_bp_cache = {}

def _bandpass(sig, fs):
    key = (fs, len(sig))
    if key not in _bp_cache:
        b, a = butter(4, [BANDPASS_LOW, BANDPASS_HIGH], btype='band', fs=fs)
        _bp_cache[key] = (b, a)
    b, a = _bp_cache[key]
    return filtfilt(b, a, sig)


def detect_peaks(sig, fs=FS1k):
    filtered = _bandpass(sig.astype(np.float64), fs)
    min_dist = int(MIN_DIST_MS * fs / 1000)
    thresh   = 0.3 * np.std(filtered)
    pos, _ = find_peaks( filtered, height=thresh, distance=min_dist)
    neg, _ = find_peaks(-filtered, height=thresh, distance=min_dist)
    return pos if len(pos) >= len(neg) else neg


def match_peaks(pred_peaks, gt_peaks, tol):
    """Return (tp, fp, fn) with greedy matching."""
    tp = 0
    used_gt = set()
    for pp in pred_peaks:
        for i, gp in enumerate(gt_peaks):
            if i not in used_gt and abs(int(pp) - int(gp)) <= tol:
                tp += 1
                used_gt.add(i)
                break
    fp = len(pred_peaks) - tp
    fn = len(gt_peaks)   - tp
    return tp, fp, fn


def f1_score(tp, fp, fn):
    if tp == 0:
        return 0.0, 0.0, 0.0
    prec   = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1     = 2 * prec * recall / (prec + recall)
    return f1, prec, recall


def compute_qrs_f1(pred, gt, fs=FS1k):
    tol = int(TOLERANCE_MS * fs / 1000)
    n   = min(pred.shape[1], gt.shape[1])
    results = []
    for ch in range(pred.shape[0]):
        p_peaks = detect_peaks(pred[ch, :n], fs)
        g_peaks = detect_peaks(gt[ch, :n],   fs)
        tp, fp, fn = match_peaks(p_peaks, g_peaks, tol)
        f1, prec, rec = f1_score(tp, fp, fn)
        results.append({
            'f1': f1, 'precision': prec, 'recall': rec,
            'tp': tp, 'fp': fp, 'fn': fn,
            'n_pred_peaks': len(p_peaks), 'n_gt_peaks': len(g_peaks),
        })
    return results


# ── signal list ──────────────────────────────────────────────────────────────
mix_base = os.path.join(NPY_BASE, 'mixture')
sig_base = os.path.join(NPY_BASE, 'signals')
signals  = sorted(os.listdir(mix_base))
print(f'Total signals: {len(signals)}', flush=True)


def load_signal(base, sig_name):
    d = os.path.join(base, sig_name)
    chs = []
    for ch in range(1, 7):
        f = [x for x in os.listdir(d) if f'_ch{ch}.npy' in x][0]
        chs.append(np.load(os.path.join(d, f)).astype(np.float32))
    return np.stack(chs)


def snr_category(sig_name):
    m = PATTERN.search(sig_name)
    if not m:
        return 'unknown', 0
    avg = (int(m.group(1)) + int(m.group(2)) + int(m.group(3))) / 3
    if avg < -5:   return 'difficult', avg
    elif avg <= 5: return 'medium',    avg
    else:          return 'easy',      avg


# ── main loop ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

if os.path.exists(OUT_JSON):
    all_results = json.load(open(OUT_JSON))
    done = set(all_results.keys())
    print(f'Resumed: {len(done)} signals already processed', flush=True)
else:
    all_results = {}
    done = set()

t0 = time.time()
for si, sig in enumerate(signals):
    if sig in done:
        continue

    cat, avg_snr = snr_category(sig)
    print(f'\n[{si+1}/{len(signals)}] {sig[:60]}  [{cat}, avg={avg_snr:.1f}dB]', flush=True)
    t_sig = time.time()

    try:
        mixture = load_signal(mix_base, sig)
        fecg_gt = load_signal(sig_base, sig)
    except Exception as e:
        print(f'  SKIP: {e}', flush=True)
        continue

    sig_result = {'category': cat, 'avg_snr': avg_snr, 'models': {}}

    for ver, model, infer_fn in [
        ('v1',  m_v1,  infer_full_1k),
        ('v16', m_v16, infer_full_500),
        ('v17', m_v17, infer_full_500),
    ]:
        try:
            pred    = infer_fn(model, mixture)
            metrics = compute_qrs_f1(pred, fecg_gt)
            sig_result['models'][ver] = {
                'epoch': ep_v1 if ver == 'v1' else (ep_v16 if ver == 'v16' else ep_v17),
                'channels': metrics,
                'mean': {k: float(np.mean([m[k] for m in metrics]))
                         for k in ['f1', 'precision', 'recall', 'tp', 'fp', 'fn',
                                   'n_pred_peaks', 'n_gt_peaks']},
            }
            m = sig_result['models'][ver]['mean']
            print(f'  {ver}: F1={m["f1"]:.4f}  Prec={m["precision"]:.4f}  Rec={m["recall"]:.4f}'
                  f'  peaks_pred={m["n_pred_peaks"]:.0f}  peaks_gt={m["n_gt_peaks"]:.0f}', flush=True)
        except Exception as e:
            print(f'  {ver} EROARE: {e}', flush=True)

    all_results[sig] = sig_result
    json.dump(all_results, open(OUT_JSON, 'w'), indent=2)

    elapsed   = time.time() - t_sig
    total_e   = time.time() - t0
    remaining = (len(signals) - si - 1) * (total_e / max(si + 1 - len(done), 1))
    print(f'  [{elapsed:.0f}s/signal | ETA ~{remaining/60:.0f}min]', flush=True)


# ── raport final ──────────────────────────────────────────────────────────────
print('\n' + '='*70, flush=True)
print('RAPORT FINAL — QRS F1 Score', flush=True)
print('='*70, flush=True)

by_cat = {'easy': {}, 'medium': {}, 'difficult': {}, 'all': {}}
for ver in ['v1', 'v16', 'v17']:
    for cat in by_cat:
        by_cat[cat][ver] = {'f1': [], 'precision': [], 'recall': []}

for sig, res in all_results.items():
    cat = res['category']
    for ver in ['v1', 'v16', 'v17']:
        if ver not in res['models']:
            continue
        m = res['models'][ver]['mean']
        for k in ['f1', 'precision', 'recall']:
            by_cat[cat][ver][k].append(m[k])
            by_cat['all'][ver][k].append(m[k])

for cat in ['all', 'easy', 'medium', 'difficult']:
    n = len(list(by_cat[cat].values())[0]['f1'])
    if n == 0:
        continue
    print(f'\n--- {cat.upper()} ({n} signals) ---')
    print(f"{'Model':<10} {'F1':>10} {'Precision':>12} {'Recall':>12}")
    for ver in ['v1', 'v16', 'v17']:
        d = by_cat[cat][ver]
        if not d['f1']:
            continue
        print(f"{ver:<10} "
              f"{np.mean(d['f1']):.4f}±{np.std(d['f1']):.4f}   "
              f"{np.mean(d['precision']):.4f}±{np.std(d['precision']):.4f}   "
              f"{np.mean(d['recall']):.4f}±{np.std(d['recall']):.4f}")

print(f'\nSalvat: {OUT_JSON}', flush=True)
