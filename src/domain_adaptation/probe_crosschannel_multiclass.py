"""
INFORMATION PROBE (diagnostic, NOT deliverable): does the CROSS-CHANNEL signature
of movement TYPE (linear/spline/helix) survive extraction?

Physical mechanism (from run_ecg_generator.m): fetal heart = dipole at position traj(t);
fECG on each electrode = dipole projection. Trajectory type determines HOW the
amplitude vector across 6 channels evolves:
  - linear  -> STRAIGHT trajectory in 6D space
  - spline  -> CURVED trajectory
  - helix   -> ROTATING (periodic) trajectory
Straight/curved/rotating is GEOMETRIC cross-channel, potentially more robust to
extraction amplitude distortion than per-channel amplitude (r~0.5).

METHODOLOGICAL NOTE: this probe uses ORACLE phase boundaries (mc_mask) and
ORACLE R-peak positions (detected on GT) — ONLY to measure INFORMATION CONTENT,
not as a final method. The deliverable (dense classification, no oracle) comes AFTER,
only if the information exists.

DECISION METRIC = per-class F1 on linear(1) and spline(2). Macro-F1 is misleading
(helix+no-move inflate the average; linear/spline can collapse to 0 — latent adapter trap).
"""
import os, sys, time
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from config import NPY, FECG_ROOT as ROOT
INF = ROOT + '/inferred_v1'
FS = 500
MIN_S, MIN_BEATS = 10, 8
CLS = {0: 'no-move', 1: 'linear', 2: 'spline', 3: 'helix'}
_bp = butter(2, [5, 40], btype='band', fs=FS)


def load_gt(name):
    d = os.path.join(NPY, 'signals', name)
    return np.stack([np.load(os.path.join(d, [x for x in os.listdir(d) if f'_ch{c}.npy' in x][0])).astype(np.float32)
                     for c in range(1, 7)])


def load_mc(name):
    return np.load(os.path.join(NPY, 'mc_masks', name + '_mc_mask.npy')).astype(int)


def detect_fqrs(gt6):
    """Robust fetal QRS timing: sum of band-pass envelopes over the 6 GT channels."""
    env = np.zeros(gt6.shape[1])
    for ch in range(6):
        z = (gt6[ch] - gt6[ch].mean()) / (gt6[ch].std() + 1e-8)
        env += np.abs(filtfilt(_bp[0], _bp[1], z))
    pk, _ = find_peaks(env, distance=150, height=np.percentile(env, 75))
    return pk


def amp_at(sig, peaks, w=12):
    if len(peaks) == 0:
        return np.zeros(0, np.float32)
    idx = np.clip(np.asarray(peaks)[:, None] + np.arange(-w, w), 0, len(sig) - 1)
    return np.max(np.abs(sig[idx]), axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
def spectral_feats(amps):
    """36 spectral features per channel (single-channel baseline, as in v9/v15)."""
    N, C = amps.shape
    t = np.arange(N, dtype=np.float64)
    feats = []
    for ch in range(C):
        a = amps[:, ch].astype(np.float64)
        a_dm = a - a.mean()
        feats.append(float(a_dm.std()))
        p = np.polyfit(t, a_dm, 1); slope = p[0]
        feats.append(float(slope))
        pred = slope * t + p[1]
        ss_res = ((a_dm - pred) ** 2).sum(); ss_tot = ((a_dm - a_dm.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        feats.append(float(np.clip(r2, -1, 1)))
        F = np.fft.rfft(a_dm); Fabs = np.abs(F); Fabs[0] = 0.0
        feats.append(float(Fabs.max()))
        feats.append(float(Fabs.argmax()) / max(N, 1))
        if N > 5:
            c = np.corrcoef(a_dm[:-5], a_dm[5:])[0, 1]
            feats.append(0.0 if np.isnan(c) else float(c))
        else:
            feats.append(0.0)
    return feats


def geom_feats(amps, smooth=5):
    """
    Geometric features of the 6D trajectory of the amplitude vector.
    Smooth per channel (moving average) to extract the SLOW movement trajectory
    (above beat-to-beat HRV noise), then characterise the path shape in 6D.
    """
    N, C = amps.shape
    feats = []
    A = amps.astype(np.float64).copy()
    # per-channel normalisation (median scale) -> robust to per-channel gain
    med = np.median(np.abs(A), axis=0, keepdims=True); med[med < 1e-9] = 1.0
    A = A / med
    # smooth along the beat axis
    if N >= smooth * 2:
        k = np.ones(smooth) / smooth
        A = np.stack([np.convolve(A[:, ch], k, mode='valid') for ch in range(C)], axis=1)
    A = A - A.mean(axis=0, keepdims=True)
    M = A.shape[0]

    # --- PCA on the 6D trajectory ---
    try:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        ev = (S ** 2)
        tot = ev.sum() + 1e-12
        r = ev / tot
    except Exception:
        r = np.zeros(C)
    for i in range(3):
        feats.append(float(r[i]) if i < len(r) else 0.0)
    # participation ratio (cate dimensiuni "reale" ocupa drumul): linear~1, helix~2-3
    pr = float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12)) if 'ev' in dir() and ev.sum() > 0 else 0.0
    feats.append(pr)

    # --- projection onto top-3 PCs: scores = dimensionality-reduced path ---
    if M >= 4 and len(S) >= 2:
        scores = U[:, :3] * S[:3]  # (M, <=3)
        if scores.shape[1] < 3:
            scores = np.pad(scores, ((0, 0), (0, 3 - scores.shape[1])))
    else:
        scores = np.zeros((max(M, 1), 3))

    p1, p2 = scores[:, 0], scores[:, 1]

    # tortuosity: path length / end-to-end distance (linear~1, spline>1, helix>>1)
    seg = np.sqrt(np.sum(np.diff(scores, axis=0) ** 2, axis=1))
    path_len = seg.sum()
    endto = np.sqrt(np.sum((scores[-1] - scores[0]) ** 2)) + 1e-9
    feats.append(float(np.clip(path_len / endto, 1.0, 50.0)))

    # total turning in the PC1-PC2 plane (helix rotates -> large cumulative angle)
    ang = np.arctan2(p2, p1)
    dang = np.diff(ang)
    dang = (dang + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi,pi]
    total_turn = float(np.abs(dang.sum()))           # net rotation
    abs_turn = float(np.abs(dang).sum())             # total absolute rotation
    feats.append(total_turn)
    feats.append(abs_turn)
    # number of complete rotations
    feats.append(float(abs_turn / (2 * np.pi)))

    # mean curvature: how much the path bends (deviation from straight line)
    if M >= 3:
        d1 = np.diff(scores, axis=0)
        n1 = np.linalg.norm(d1, axis=1, keepdims=True); n1[n1 < 1e-9] = 1.0
        u = d1 / n1
        cosang = np.clip(np.sum(u[:-1] * u[1:], axis=1), -1, 1)
        curv = float(np.mean(np.arccos(cosang)))
    else:
        curv = 0.0
    feats.append(curv)

    # spectral radius of the PC1-PC2 path: helix -> energy at one frequency (periodic)
    if M > 4:
        c = p1 + 1j * p2
        Fc = np.abs(np.fft.fft(c - c.mean()))
        peak = Fc.max(); meanf = Fc.mean() + 1e-12
        feats.append(float(np.clip(peak / meanf, 0, 50)))
    else:
        feats.append(0.0)

    return feats


GEOM_NAMES = ['pca_r1', 'pca_r2', 'pca_r3', 'partic_ratio', 'tortuosity',
              'net_turn', 'abs_turn', 'n_rot', 'curvature', 'pc_spec_peak']


# ---------------------------------------------------------------------------
def build():
    names = sorted([f.replace('_mc_mask.npy', '') for f in os.listdir(os.path.join(NPY, 'mc_masks'))])
    Xs_ext, Xs_gt, Y, FID = [], [], [], []
    t0 = time.time()
    for fi, name in enumerate(names):
        if not os.path.exists(os.path.join(INF, name + '.npy')):
            continue
        gt = load_gt(name)
        ext = np.load(os.path.join(INF, name + '.npy'))
        mc = load_mc(name)
        N = min(gt.shape[1], ext.shape[1], len(mc))
        gt, ext, mc = gt[:, :N], ext[:, :N], mc[:N]
        peaks = detect_fqrs(gt)
        tr = np.where(np.diff(mc) != 0)[0]
        starts = np.concatenate([[0], tr + 1]); ends = np.concatenate([tr + 1, [len(mc)]])
        for s, e in zip(starts, ends):
            if e - s < MIN_S * FS:
                continue
            lbl = int(mc[s])
            pk = peaks[(peaks >= s) & (peaks < e)]
            if len(pk) < MIN_BEATS:
                continue
            amps_e = np.stack([amp_at(ext[ch], pk) for ch in range(6)], axis=1)
            amps_g = np.stack([amp_at(gt[ch], pk) for ch in range(6)], axis=1)
            Xs_ext.append(spectral_feats(amps_e) + geom_feats(amps_e))
            Xs_gt.append(spectral_feats(amps_g) + geom_feats(amps_g))
            Y.append(lbl); FID.append(fi)
        if (fi + 1) % 20 == 0:
            print(f'  {fi+1}/{len(names)} files, {len(Y)} phases ({time.time()-t0:.0f}s)', flush=True)
    return (np.array(Xs_ext, np.float32), np.array(Xs_gt, np.float32),
            np.array(Y), np.array(FID))


def kfold_eval(X, y, fid, n_splits=5, tag=''):
    """File-level CV: each fold holds disjoint files in the test set."""
    files = np.unique(fid)
    rng = np.random.default_rng(0); rng.shuffle(files)
    folds = np.array_split(files, n_splits)
    preds = np.full(len(y), -1)
    for k in range(n_splits):
        test_f = set(folds[k].tolist())
        te = np.array([i for i in range(len(y)) if fid[i] in test_f])
        trn = np.array([i for i in range(len(y)) if fid[i] not in test_f])
        rf = RandomForestClassifier(n_estimators=400, class_weight='balanced',
                                    n_jobs=-1, random_state=0)
        rf.fit(X[trn], y[trn])
        preds[te] = rf.predict(X[te])
    f1pc = f1_score(y, preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
    macro = f1_score(y, preds, labels=[0, 1, 2, 3], average='macro', zero_division=0)
    acc = accuracy_score(y, preds)
    print(f'\n=== {tag} ===')
    print(f'  acc={acc:.3f}  macro-F1={macro:.3f}')
    print(f'  F1 per-class: ' + '  '.join(f'{CLS[c]}={f1pc[c]:.3f}' for c in range(4)))
    print(f'  CONFUSION:\n{confusion_matrix(y, preds, labels=[0,1,2,3])}')
    return f1pc, macro, acc


def main():
    print('Building features (oracle phases + GT peaks, probe only)...', flush=True)
    Xe, Xg, y, fid = build()
    n_spec = 36
    print(f'\nTotal phases: {len(y)}  classes: {np.bincount(y, minlength=4).tolist()}')
    print(f'Files: {len(np.unique(fid))}  feat dim: {Xe.shape[1]} (spectral {n_spec} + geom {len(GEOM_NAMES)})')

    # 1. CEILING: GT amplitudes, all features
    kfold_eval(Xg, y, fid, tag='GT (ceiling) — spectral+geom')
    # 2. EXTRACTED, single-channel spectral only (what M15 sees ~ per-channel)
    kfold_eval(Xe[:, :n_spec], y, fid, tag='EXTRACTED — spectral per-channel only')
    # 3. EXTRACTED, +geometric cross-channel (hypothesis)
    kfold_eval(Xe, y, fid, tag='EXTRACTED — spectral+geom cross-channel')
    # 4. EXTRACTED, geometric only (isolates geometry power)
    kfold_eval(np.column_stack([Xe[:, n_spec:]]), y, fid, tag='EXTRACTED — geometric cross-channel only')


if __name__ == '__main__':
    main()
