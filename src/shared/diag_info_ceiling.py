"""
DIAGNOSTIC "information ceiling": does the amplitude modulation (the movement
signature) survive the v1 extraction?

For a sample of signals: detect R-peaks on GT, read the QRS amplitude (max-abs in a
small window) at the SAME positions in GT and in the v1 extraction, and measure the
Pearson correlation between the two beat-amplitude sequences:
  - global (all beats)
  - per movement CLASS (linear/spline/helix), within the segments
    (= is the SHAPE that the classifier reads preserved?)
Decision: high r -> promising adapter; r ~ 0 -> information lost, the adapter won't help.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import numpy as np
import torch
torch.set_num_threads(1)
from scipy.signal import butter, filtfilt, find_peaks
import pipeline_registry as R

from config import NPY
FS = 500
CLASS_NAME = {0: 'no-move', 1: 'linear', 2: 'spline', 3: 'helix'}
_bp = butter(2, [5, 40], btype='band', fs=FS)


def load_sig(sub, name):
    d = os.path.join(NPY, sub, name)
    return np.stack([np.load(os.path.join(d, [x for x in os.listdir(d) if f'_ch{c}.npy' in x][0])).astype(np.float32)
                     for c in range(1, 7)])


def load_mask(sub, name):
    return np.load([os.path.join(NPY, sub, f) for f in os.listdir(os.path.join(NPY, sub)) if name in f][0])


def detect_peaks(sig):
    z = (sig - sig.mean()) / (sig.std() + 1e-8)
    env = np.abs(filtfilt(_bp[0], _bp[1], z))
    pk, _ = find_peaks(env, distance=150, height=np.percentile(env, 90) * 0.5)
    return pk


def amp_at(sig, peaks, w=12):
    if len(peaks) == 0:
        return np.zeros(0, np.float32)
    idx = np.clip(np.asarray(peaks)[:, None] + np.arange(-w, w), 0, len(sig) - 1)
    return np.max(np.abs(sig[idx]), axis=1).astype(np.float32)


def main():
    signals = sorted(os.listdir(os.path.join(NPY, 'mixture')))
    sample = signals[::7]   # ~18 signals, spanning SNR conditions
    print(f'Diagnostic on {len(sample)} signals (v1 extracted vs GT)\n', flush=True)
    ext_model = R.load_extractor('v1')

    glob_r = []                              # global r per (signal, channel)
    seg_r = {c: [] for c in (0, 1, 2, 3)}    # r per segment, per class
    seg_w = {c: [] for c in (0, 1, 2, 3)}    # lengths (beat count)

    for si, name in enumerate(sample):
        gt = load_sig('signals', name)
        ext = R.infer('v1', ext_model, load_sig('mixture', name))
        mc = load_mask('mc_masks', name).astype(int)
        N = min(gt.shape[1], ext.shape[1], len(mc))
        for ch in range(6):
            pk = detect_peaks(gt[ch, :N])
            if len(pk) < 20:
                continue
            ag = amp_at(gt[ch, :N], pk); ae = amp_at(ext[ch, :N], pk)
            if ag.std() > 1e-6 and ae.std() > 1e-6:
                glob_r.append(np.corrcoef(ag, ae)[0, 1])
            cls = mc[pk]
            # segments = runs of constant class at beat level
            b = 0
            while b < len(pk):
                e = b
                while e + 1 < len(pk) and cls[e + 1] == cls[b]:
                    e += 1
                run = slice(b, e + 1); L = e + 1 - b; c = int(cls[b])
                if L >= 6 and ag[run].std() > 1e-6 and ae[run].std() > 1e-6:
                    seg_r[c].append(float(np.corrcoef(ag[run], ae[run])[0, 1]))
                    seg_w[c].append(L)
                b = e + 1
        print(f'  [{si+1}/{len(sample)}] {name.split("_SNR")[0]}', flush=True)

    print('\n' + '=' * 60)
    print('RESULT — Pearson correlation beat amplitude GT vs v1-extracted')
    print('=' * 60)
    gr = np.array(glob_r)
    print(f'\nGLOBAL (all beats): r_mean = {np.nanmean(gr):.3f}  '
          f'(median {np.nanmedian(gr):.3f}, n={len(gr)} channels)')
    print('\nPER CLASS (correlation within segments — signature SHAPE):')
    print(f"  {'class':<10}{'r_mean':>9}{'r_weighted':>12}{'n_seg':>7}{'beats':>8}")
    for c in (0, 1, 2, 3):
        if not seg_r[c]:
            print(f'  {CLASS_NAME[c]:<10}{"--":>9}'); continue
        r = np.array(seg_r[c]); w = np.array(seg_w[c])
        wr = np.nansum(r * w) / np.nansum(w[~np.isnan(r)]) if np.any(~np.isnan(r)) else float('nan')
        print(f'  {CLASS_NAME[c]:<10}{np.nanmean(r):>9.3f}{wr:>12.3f}{len(r):>7}{int(w.sum()):>8}')
    print('\nInterpretation: r>~0.5 = info preserved, promising adapter | '
          'r~0.2-0.4 = weak | r~0 = lost (the adapter does not help).')


if __name__ == '__main__':
    main()
