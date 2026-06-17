"""
DELIVERABLE v3 — two-stage dens cu Etapa 1 imbunatatita:
  - fereastra de DETECTIE mai scurta (WIN1, default 10s, pas 1s): 16s dilua miscarile
    scurte (o faza de 20s e majoritar no-move in ferestrele care o ating);
  - PRAG de miscare calibrat pe fold-urile de TRAIN (maximizeaza binary-F1 per-sample
    pe train), aplicat pe test — fara leakage.
  - Etapa 2 (tip pe regiunea detectata intreaga) neschimbata.

Restul (features cross-canal, RF pe extras, CV 5-fold file-level) identic cu
two_stage_dense.py.
"""
import os, sys, time, json
import numpy as np
from scipy.ndimage import median_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import two_stage_dense as T
from two_stage_dense import feats, load_mc, CLS, FS

WIN1 = int(os.environ.get('WIN1', '10')) * FS
STRIDE1 = int(os.environ.get('STRIDE1', '1')) * FS
MIN_RUN_S = 6
FILL_GAP_S = 3
RES = T.RES
SRC = T.SRC


def build_stage1(data):
    X, Yb, FID, CEN = [], [], [], []
    for fi, d in data.items():
        peaks, amps, mc, N = d['peaks'], d['amps'], d['mc'], d['N']
        for s in range(0, N - WIN1 + 1, STRIDE1):
            e = s + WIN1
            m = (peaks >= s) & (peaks < e)
            if int(m.sum()) < T.MIN_BEATS:
                continue
            X.append(feats(amps[m]))
            Yb.append(int((mc[s:e] > 0).mean() >= 0.5))
            FID.append(fi); CEN.append((s + e) // 2)
    return np.array(X, np.float32), np.array(Yb), np.array(FID), np.array(CEN)


def pmove_for_files(rf1, X1, FID1, CEN1, data, fids):
    """Movement probability per-sample (overlap-add) for each file in fids."""
    out = {}
    for fi in fids:
        N = data[fi]['N']
        acc = np.zeros(N, np.float32); cov = np.zeros(N, np.float32)
        idx = np.where(FID1 == fi)[0]
        if len(idx):
            pm = rf1.predict_proba(X1[idx])[:, 1]
            for j, wi in enumerate(idx):
                c = CEN1[wi]; s = max(0, c - WIN1 // 2); e = min(N, c + WIN1 // 2)
                acc[s:e] += pm[j]; cov[s:e] += 1.0
        cov[cov == 0] = 1.0
        out[fi] = acc / cov
    return out


def tune_threshold(pmove, data, fids):
    """Choose the threshold that maximises binary-F1 per-sample on the TRAIN files."""
    yt = np.concatenate([(data[fi]['mc'][:data[fi]['N']] > 0).astype(int) for fi in fids])
    pm = np.concatenate([pmove[fi] for fi in fids])
    best_t, best_f = 0.5, -1
    for t in np.arange(0.2, 0.75, 0.05):
        f = f1_score(yt, (pm >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), float(best_f)


def runs_from_pmove(pm, thr, N):
    bm = (median_filter((pm >= thr).astype(int), size=FS) > 0).astype(int)
    edges = np.diff(np.concatenate([[0], bm, [0]]))
    starts = np.where(edges == 1)[0]; ends = np.where(edges == -1)[0]
    merged = []
    for r in zip(starts, ends):
        if merged and r[0] - merged[-1][1] < FILL_GAP_S * FS:
            merged[-1][1] = r[1]
        else:
            merged.append([r[0], r[1]])
    return [(a, b) for a, b in merged if b - a >= MIN_RUN_S * FS]


def main():
    names = sorted([f.replace('_mc_mask.npy', '') for f in os.listdir(os.path.join(T.NPY, 'mc_masks'))
                    if os.path.exists(os.path.join(T.INF, f.replace('_mc_mask.npy', '') + '.npy'))])
    print(f'{len(names)} files. source={SRC} WIN1={WIN1/FS:.0f}s stride1={STRIDE1/FS:.0f}s', flush=True)
    data = T.precompute(names)
    X1, Yb1, FID1, CEN1 = build_stage1(data)
    print(f'stage1 windows: {len(Yb1)}  movement frac={Yb1.mean():.3f}', flush=True)

    files = np.array(sorted(data.keys()))
    rng = np.random.default_rng(0); rng.shuffle(files)
    folds = np.array_split(files, 5)
    all_pred, all_true, thrs = [], [], []
    for k in range(5):
        test_f = set(folds[k].tolist())
        train_fids = [f for f in files if f not in test_f]
        tr1 = np.array([i for i in range(len(Yb1)) if FID1[i] not in test_f])
        rf1 = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                     n_jobs=-1, random_state=0)
        rf1.fit(X1[tr1], Yb1[tr1])
        # threshold on train split
        pm_train = pmove_for_files(rf1, X1, FID1, CEN1, data, train_fids)
        thr, ftr = tune_threshold(pm_train, data, train_fids)
        thrs.append(thr)
        # stage2
        Xp, Yp = T.build_stage2_phases(data, train_fids)
        rf2 = RandomForestClassifier(n_estimators=400, class_weight='balanced',
                                     n_jobs=-1, random_state=0)
        rf2.fit(Xp, Yp)
        # test
        pm_test = pmove_for_files(rf1, X1, FID1, CEN1, data, folds[k])
        for fi in folds[k]:
            d = data[fi]; N = d['N']; peaks = d['peaks']; amps = d['amps']
            pred = np.zeros(N, np.int64)
            for (a, b) in runs_from_pmove(pm_test[fi], thr, N):
                m = (peaks >= a) & (peaks < b)
                if int(m.sum()) < T.MIN_BEATS:
                    pred[a:b] = 1; continue
                pred[a:b] = int(rf2.predict(feats(amps[m])[None])[0])
            all_pred.append(pred); all_true.append(d['mc'][:N])
        print(f'  fold {k+1}/5 thr={thr:.2f} (train binF1={ftr:.3f})', flush=True)

    yp = np.concatenate(all_pred); yt = np.concatenate(all_true)
    f1pc = f1_score(yt, yp, labels=[0,1,2,3], average=None, zero_division=0)
    macro = f1_score(yt, yp, labels=[0,1,2,3], average='macro', zero_division=0)
    acc = accuracy_score(yt, yp)
    binf1 = f1_score((yt>0).astype(int), (yp>0).astype(int), zero_division=0)
    print(f'\n=== TWO-STAGE v2 [{SRC}] WIN1={WIN1/FS:.0f}s thr~{np.mean(thrs):.2f} ===')
    print(f'  PER-SAMPLE: acc={acc:.3f}  macro-F1={macro:.3f}  binary-move-F1={binf1:.3f}')
    print(f'  F1 per-class: ' + '  '.join(f'{CLS[c]}={f1pc[c]:.3f}' for c in range(4)))
    print(f'  CONFUSION:\n{confusion_matrix(yt, yp, labels=[0,1,2,3])}')
    json.dump({'src': SRC, 'win1_s': WIN1/FS, 'thr_mean': float(np.mean(thrs)),
               'persample_acc': float(acc), 'persample_macroF1': float(macro),
               'binary_move_F1': float(binf1), 'persample_f1': f1pc.tolist()},
              open(os.path.join(RES, f'two_stage_v2_{SRC}_w{WIN1//FS}.json'), 'w'), indent=2)
    print('saved.')


if __name__ == '__main__':
    main()
