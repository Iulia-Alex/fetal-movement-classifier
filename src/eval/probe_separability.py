"""
SEPARABILITY PROBE (cheap, decides the linear/spline strategy):
At the WINDOW level (16s, exactly what Stage 1 detection sees), how separable are
linear-vs-no-move and spline-vs-no-move from cross-channel features on the EXTRACTED signal?

  - high AUC (>0.8) but the pipeline misses -> the issue is the operating point/weighting
    => reweighting/rebalancing helps.
  - weak AUC (~0.6) -> information-limited; no threshold saves it (the ROC is the answer).

Also tests SPATIAL COHERENCE features of the drift (real movement = the 6 channels drift
COHERENTLY; the spurious extraction noise in no-move is INCOHERENT) to see whether they RAISE
the AUC (add information) vs the existing features alone.
"""
import os, sys, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import two_stage_dense as T
from two_stage_dense import feats as base_feats, FS, MIN_BEATS

WIN = 16 * FS
STRIDE = 4 * FS
CLS = {1: 'linear', 2: 'spline', 3: 'helix'}


def coherence_feats(amps, smooth=9):
    """Detection features: spatial coherence + magnitude of the slow drift vs jitter."""
    N, C = amps.shape
    A = amps.astype(np.float64)
    med = np.median(np.abs(A), 0, keepdims=True); med[med < 1e-9] = 1.0; A = A / med
    if N >= smooth * 2:
        k = np.ones(smooth) / smooth
        As = np.stack([np.convolve(A[:, c], k, 'valid') for c in range(C)], 1)
    else:
        As = A.copy()
    out = []
    # 1. slow drift (smooth range) vs beat-to-beat jitter: movement -> drift >> jitter
    smooth_range = np.mean([As[:, c].max() - As[:, c].min() for c in range(C)])
    raw_jit = np.mean([np.median(np.abs(np.diff(A[:, c]))) for c in range(C)]) + 1e-9
    out.append(float(smooth_range / raw_jit))
    # 2. fraction of slow variance (temporal coherence)
    var_s = np.mean([As[:, c].var() for c in range(C)])
    var_r = np.mean([A[:, c].var() for c in range(C)]) + 1e-12
    out.append(float(var_s / var_r))
    # 3. SPATIAL coherence: mean |corr| between the derivatives of the smoothed channels
    d = np.diff(As, axis=0)
    if d.shape[0] > 3:
        cc = np.corrcoef(d.T); iu = np.triu_indices(C, 1)
        out.append(float(np.nanmean(np.abs(cc[iu]))))
    else:
        out.append(0.0)
    # 4. PC1 fraction of the smoothed trajectory (coherent drift = low-rank)
    Ac = As - As.mean(0, keepdims=True)
    try:
        s = np.linalg.svd(Ac, compute_uv=False); ev = s ** 2
        out.append(float(ev[0] / (ev.sum() + 1e-12)))
    except Exception:
        out.append(0.0)
    return out


def build(data):
    Xb, Xc, TYPE, FID = [], [], [], []
    for fi, d in data.items():
        peaks, amps, mc, N = d['peaks'], d['amps'], d['mc'], d['N']
        for s in range(0, N - WIN + 1, STRIDE):
            e = s + WIN
            m = (peaks >= s) & (peaks < e)
            if int(m.sum()) < MIN_BEATS:
                continue
            a = amps[m]
            Xb.append(base_feats(a)); Xc.append(coherence_feats(a))
            TYPE.append(int(np.bincount(mc[s:e], minlength=4).argmax()))
            FID.append(fi)
    return (np.array(Xb, np.float32), np.array(Xc, np.float32),
            np.array(TYPE), np.array(FID))


def cv_auc(X, y, fid, n=5):
    files = np.unique(fid); rng = np.random.default_rng(0); rng.shuffle(files)
    folds = np.array_split(files, n)
    proba = np.zeros(len(y))
    for k in range(n):
        tf = set(folds[k].tolist())
        te = np.array([i for i in range(len(y)) if fid[i] in tf])
        tr = np.array([i for i in range(len(y)) if fid[i] not in tf])
        rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                    n_jobs=-1, random_state=0).fit(X[tr], y[tr])
        proba[te] = rf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, proba), average_precision_score(y, proba)


def main():
    names = sorted([f.replace('_mc_mask.npy', '') for f in os.listdir(os.path.join(T.NPY, 'mc_masks'))
                    if os.path.exists(os.path.join(T.INF, f.replace('_mc_mask.npy', '') + '.npy'))])
    print(f'{len(names)} files. Building windows (16s/4s)...', flush=True)
    data = T.precompute(names)
    Xb, Xc, TY, FID = build(data)
    Xbc = np.concatenate([Xb, Xc], axis=1)
    print(f'windows: {len(TY)}  type dist: {np.bincount(TY, minlength=4).tolist()}', flush=True)
    print(f'base feat dim={Xb.shape[1]}  +coherence={Xc.shape[1]}\n', flush=True)

    print('=== SEPARABILITY movement-class vs NO-MOVE (window-level, extracted) ===')
    print(f'{"task":18s} {"ROC-AUC base":>13s} {"AUC +coh":>10s} | {"PR-AUC base":>12s} {"PR +coh":>9s}  (prevalence)')
    for c in (1, 2, 3):
        mask = (TY == 0) | (TY == c)
        y = (TY[mask] == c).astype(int)
        prev = y.mean()
        a_b, p_b = cv_auc(Xb[mask], y, FID[mask])
        a_c, p_c = cv_auc(Xbc[mask], y, FID[mask])
        print(f'{CLS[c]+" vs no-move":18s} {a_b:13.3f} {a_c:10.3f} | {p_b:12.3f} {p_c:9.3f}  ({prev:.2f})')

    # bonus: linear vs spline (separability of the two, if detection were perfect)
    mask = (TY == 1) | (TY == 2)
    y = (TY[mask] == 2).astype(int)
    a_b, p_b = cv_auc(Xb[mask], y, FID[mask]); a_c, p_c = cv_auc(Xbc[mask], y, FID[mask])
    print(f'\nlinear vs spline (type discrimination, if detected): ROC base={a_b:.3f} +coh={a_c:.3f}')


if __name__ == '__main__':
    main()
