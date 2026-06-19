"""
IMPROVED two-stage, based on the separability probe:
  - features += SPATIAL COHERENCE of the drift (probe_separability.coherence_feats);
  - Stage 1 detection: the linear/spline windows are WEIGHTED up (sample_weight) so the
    model learns their subtle signature — the probe showed AUC 0.73/0.82 but a poor operating point;
  - the detection threshold is chosen on TRAIN to maximise the BALANCED BINARY ACCURACY
    (mean of recall_movement & recall_nomove) — favours recall without destroying no-move.
Reports ALL four per-type recalls + overall + confusion + the per-type DETECTION recall
(visibility of the cost on no-move), vs the baseline (0.989/0.121/0.193/0.729).
"""
import os, sys, json
import numpy as np
from scipy.ndimage import median_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import two_stage_dense as T
from two_stage_dense import spectral_feats, geom_feats, FS, CLS
from probe_separability import coherence_feats

WIN = int(os.environ.get('DETWIN', '16')) * FS
STRIDE = 2 * FS
MIN_BEATS = 8
MIN_RUN_S = 6
FILL_GAP_S = 3
LSW = float(os.environ.get('LSW', '4.0'))   # linear/spline weight at detection
HW = float(os.environ.get('HW', '1.5'))     # helix weight
RES = T.RES


def feats(amps):
    return np.array(spectral_feats(amps) + geom_feats(amps) + coherence_feats(amps), np.float32)


def build_stage1(data):
    X, Yb, TYPE, FID, CEN = [], [], [], [], []
    for fi, d in data.items():
        peaks, amps, mc, N = d['peaks'], d['amps'], d['mc'], d['N']
        for s in range(0, N - WIN + 1, STRIDE):
            e = s + WIN
            m = (peaks >= s) & (peaks < e)
            if int(m.sum()) < MIN_BEATS:
                continue
            X.append(feats(amps[m]))
            ty = int(np.bincount(mc[s:e], minlength=4).argmax())
            Yb.append(int(ty > 0)); TYPE.append(ty)
            FID.append(fi); CEN.append((s + e) // 2)
    return (np.array(X, np.float32), np.array(Yb), np.array(TYPE),
            np.array(FID), np.array(CEN))


def build_stage2(data, fids):
    X, Y = [], []
    for fi in fids:
        d = data[fi]; peaks, amps, mc = d['peaks'], d['amps'], d['mc']
        tr = np.where(np.diff(mc) != 0)[0]
        st = np.concatenate([[0], tr + 1]); en = np.concatenate([tr + 1, [len(mc)]])
        for s, e in zip(st, en):
            lbl = int(mc[s])
            if lbl == 0 or e - s < 10 * FS:
                continue
            m = (peaks >= s) & (peaks < e)
            if int(m.sum()) < MIN_BEATS:
                continue
            X.append(feats(amps[m])); Y.append(lbl)
    return np.array(X, np.float32), np.array(Y)


def sample_weights(types):
    w = np.ones(len(types))
    w[(types == 1) | (types == 2)] = LSW
    w[types == 3] = HW
    return w


def pmove_files(rf1, X, FID, CEN, data, fids):
    out = {}
    for fi in fids:
        N = data[fi]['N']; acc = np.zeros(N, np.float32); cov = np.zeros(N, np.float32)
        idx = np.where(FID == fi)[0]
        if len(idx):
            pm = rf1.predict_proba(X[idx])[:, 1]
            for j, wi in enumerate(idx):
                c = CEN[wi]; s = max(0, c - WIN // 2); e = min(N, c + WIN // 2)
                acc[s:e] += pm[j]; cov[s:e] += 1.0
        cov[cov == 0] = 1.0; out[fi] = acc / cov
    return out


def tune_thr_balanced(pm, data, fids):
    yt = np.concatenate([(data[fi]['mc'][:data[fi]['N']] > 0).astype(int) for fi in fids])
    p = np.concatenate([pm[fi] for fi in fids])
    best_t, best = 0.5, -1
    for t in np.arange(0.2, 0.7, 0.05):
        pred = (p >= t).astype(int)
        rec_m = (pred[yt == 1] == 1).mean() if (yt == 1).any() else 0
        rec_n = (pred[yt == 0] == 0).mean() if (yt == 0).any() else 0
        bal = 0.5 * (rec_m + rec_n)
        if bal > best:
            best, best_t = bal, t
    return float(best_t)


def runs(pm, thr, N):
    bm = (median_filter((pm >= thr).astype(int), size=FS) > 0).astype(int)
    ed = np.diff(np.concatenate([[0], bm, [0]]))
    ss = np.where(ed == 1)[0]; ee = np.where(ed == -1)[0]
    merged = []
    for a, b in zip(ss, ee):
        if merged and a - merged[-1][1] < FILL_GAP_S * FS:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= MIN_RUN_S * FS]


def main():
    names = sorted([f.replace('_mc_mask.npy', '') for f in os.listdir(os.path.join(T.NPY, 'mc_masks'))
                    if os.path.exists(os.path.join(T.INF, f.replace('_mc_mask.npy', '') + '.npy'))])
    print(f'{len(names)} files. LSW={LSW} HW={HW}', flush=True)
    data = T.precompute(names)
    X1, Yb1, TY1, FID1, CEN1 = build_stage1(data)
    print(f'stage1 windows: {len(Yb1)}', flush=True)

    files = np.array(sorted(data.keys())); rng = np.random.default_rng(0); rng.shuffle(files)
    folds = np.array_split(files, 5)
    cm = np.zeros((4, 4), np.int64); thrs = []
    for k in range(5):
        tf = set(folds[k].tolist()); trf = [f for f in files if f not in tf]
        tr1 = np.array([i for i in range(len(Yb1)) if FID1[i] not in tf])
        rf1 = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                     n_jobs=-1, random_state=0)
        rf1.fit(X1[tr1], Yb1[tr1], sample_weight=sample_weights(TY1[tr1]))
        pm_tr = pmove_files(rf1, X1, FID1, CEN1, data, trf)
        thr = tune_thr_balanced(pm_tr, data, trf); thrs.append(thr)
        Xp, Yp = build_stage2(data, trf)
        rf2 = RandomForestClassifier(n_estimators=400, class_weight='balanced',
                                     n_jobs=-1, random_state=0).fit(Xp, Yp)
        pm_te = pmove_files(rf1, X1, FID1, CEN1, data, folds[k])
        for fi in folds[k]:
            d = data[fi]; N = d['N']; peaks = d['peaks']; amps = d['amps']
            pred = np.zeros(N, np.int64)
            for a, b in runs(pm_te[fi], thr, N):
                m = (peaks >= a) & (peaks < b)
                if int(m.sum()) < MIN_BEATS:
                    pred[a:b] = 1; continue
                pred[a:b] = int(rf2.predict(feats(amps[m])[None])[0])
            cm += confusion_matrix(d['mc'][:N], pred, labels=[0, 1, 2, 3])
        print(f'  fold {k+1}/5 thr={thr:.2f}', flush=True)

    rec = {CLS[c]: float(cm[c, c] / max(cm[c].sum(), 1)) for c in range(4)}
    det = {CLS[c]: float(cm[c, 1:].sum() / max(cm[c].sum(), 1)) for c in range(1, 4)}  # detected as ANY movement
    overall = float(np.trace(cm) / cm.sum())
    print(f'\n=== IMPROVED (LSW={LSW} HW={HW} thr~{np.mean(thrs):.2f}) ===')
    print('  ACCURACY PER TYPE (recall):')
    for c in range(4):
        print(f'    {CLS[c]:9s}: {rec[CLS[c]]:.3f}')
    print(f'    OVERALL  : {overall:.3f}')
    print('  DETECTION RECALL (detected as movement, regardless of type):')
    for c in range(1, 4):
        print(f'    {CLS[c]:9s}: {det[CLS[c]]:.3f}')
    print(f'\n  baseline for comparison: no-move 0.989 linear 0.121 spline 0.193 helix 0.729 overall 0.848')
    print(f'  CONFUSION:\n{cm}')
    json.dump({'recall': rec, 'detection_recall': det, 'overall': overall,
               'confusion': cm.tolist(), 'LSW': LSW, 'HW': HW, 'thr': float(np.mean(thrs))},
              open(os.path.join(RES, 'two_stage_improved.json'), 'w'), indent=2)
    print('saved.')


if __name__ == '__main__':
    main()
