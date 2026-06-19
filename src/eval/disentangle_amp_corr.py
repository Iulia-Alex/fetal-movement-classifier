"""
DISAMBIGUATION of Table 4.8 (examiner Q8): the R-peak amplitude correlation GT-vs-extracted
is read at peaks DETECTED ON THE EXTRACTED signal. A low correlation can be either (a) an
amplitude distortion (reconstruction) OR (b) badly localized peaks (detection). We separate them:

  - DETECTED : peaks found on the EXTRACTED signal (= the actual Table 4.8)
  - REFERENCE: peaks found on the GT signal (= true beat positions)

At reference peaks the localization is correct by construction, so the correlation isolates
the amplitude RECONSTRUCTION error. The difference (reference - detected) = how much of the
loss comes from peak MISLOCALIZATION (detection) rather than from reconstruction.
Same models/sample as diag_compare_models.py (Baseline=v1, Att-gated=v16, Att-sup=v17).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import numpy as np
import torch
torch.set_num_threads(1)
from diag_info_ceiling import detect_peaks, amp_at, load_sig, load_mask
import pipeline_registry as R
from config import NPY, FECG_ROOT as OUT_BASE, RESULTS_DIR as RES

MODELS = ['v1', 'v16', 'v17']
NAME = {'v1': 'Baseline', 'v16': 'Attention-gated', 'v17': 'Attention-supervised'}


def get_ext(v, name, model):
    saved = os.path.join(OUT_BASE, f'inferred_{v}', f'{name}.npy')
    return np.load(saved) if os.path.exists(saved) else R.infer(v, model, load_sig('mixture', name))


def per_class_corr(gt, ext, mc, pk):
    """Per-class shape correlation (linear/spline/helix) at peaks pk, length-weighted."""
    seg = {c: [] for c in (1, 2, 3)}; seg_w = {c: [] for c in (1, 2, 3)}
    N = gt.shape[1]
    for ch in range(6):
        p = pk[ch]
        if len(p) < 20:
            continue
        ag = amp_at(gt[ch, :N], p); ae = amp_at(ext[ch, :N], p)
        cls = mc[p]; b = 0
        while b < len(p):
            e = b
            while e + 1 < len(p) and cls[e + 1] == cls[b]: e += 1
            run = slice(b, e + 1); L = e + 1 - b; c = int(cls[b])
            if L >= 6 and c in (1, 2, 3) and ag[run].std() > 1e-6 and ae[run].std() > 1e-6:
                seg[c].append(float(np.corrcoef(ag[run], ae[run])[0, 1])); seg_w[c].append(L)
            b = e + 1
    out = {}
    for c in (1, 2, 3):
        if seg[c]:
            r = np.array(seg[c]); w = np.array(seg_w[c])
            out[c] = float(np.nansum(r * w) / np.nansum(w[~np.isnan(r)])) if np.any(~np.isnan(r)) else float('nan')
        else:
            out[c] = float('nan')
    return out


def diagnose(v):
    signals = sorted(os.listdir(os.path.join(NPY, 'mixture')))
    sample = signals[::7]
    model = R.load_extractor(v) if not os.path.exists(os.path.join(OUT_BASE, f'inferred_{v}')) else None
    det = {c: [] for c in (1, 2, 3)}; ref = {c: [] for c in (1, 2, 3)}
    detw = {c: [] for c in (1, 2, 3)}; refw = {c: [] for c in (1, 2, 3)}
    for name in sample:
        gt = load_sig('signals', name); ext = get_ext(v, name, model)
        mc = load_mask('mc_masks', name).astype(int)
        N = min(gt.shape[1], ext.shape[1], len(mc)); gt, ext, mc = gt[:, :N], ext[:, :N], mc[:N]
        pk_det = [detect_peaks(ext[ch]) for ch in range(6)]   # peaks on EXTRACTED (Table 4.8)
        pk_ref = [detect_peaks(gt[ch]) for ch in range(6)]    # peaks on GT (reference)
        for store, sw, pk in ((det, detw, pk_det), (ref, refw, pk_ref)):
            pc = per_class_corr(gt, ext, mc, pk)
            for c in (1, 2, 3):
                if not np.isnan(pc[c]):
                    store[c].append(pc[c]); sw[c].append(1)
    avg = lambda d: {c: float(np.nanmean(d[c])) if d[c] else float('nan') for c in (1, 2, 3)}
    return avg(det), avg(ref)


def main():
    res = {}
    print(f"{'Model':<22}{'type':<8}{'detected':>10}{'reference':>11}{'gap(ref-det)':>14}", flush=True)
    print('-' * 66, flush=True)
    for v in MODELS:
        det, ref = diagnose(v); res[v] = {'detected': det, 'reference': ref}
        for c, nm in ((1, 'linear'), (2, 'spline'), (3, 'helix')):
            print(f"{NAME[v]:<22}{nm:<8}{det[c]:>10.3f}{ref[c]:>11.3f}{ref[c]-det[c]:>14.3f}", flush=True)
    out = os.path.join(RES, 'amp_corr_disentangle.json')
    json.dump(res, open(out, 'w'), indent=2); print('\nsaved:', out, flush=True)


if __name__ == '__main__':
    main()
