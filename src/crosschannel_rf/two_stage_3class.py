"""
3-CLASS variant: {0 no-move, 1 directed (linear+spline), 2 helix}.
Reason: linear vs spline are ~inseparable even on the clean signal (ROC-AUC 0.67) — grouping
them is the information-honest target, we lose nothing recoverable. Same two-stage; Stage 2
becomes binary {directed, helix}. Two configs: REWEIGHT=0 (baseline detection, threshold 0.5)
and REWEIGHT=1 (directed reweighting + coherence + balanced threshold tuned on train).
"""
import os, sys, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import two_stage_dense as T
from two_stage_improved import feats, WIN, STRIDE, MIN_BEATS, runs, pmove_files, tune_thr_balanced
from two_stage_dense import FS

REWEIGHT = os.environ.get('REWEIGHT', '1') == '1'
LSW = 4.0; HW = 1.5
CLS3 = {0: 'no-move', 1: 'directed(lin+spl)', 2: 'helix'}
RES = T.RES


def remap(mc):
    o = mc.copy(); o[mc == 2] = 1; o[mc == 3] = 2
    return o


def build_stage1(data):
    X, Yb, TYPE, FID, CEN = [], [], [], [], []
    for fi, d in data.items():
        peaks, amps, mc, N = d['peaks'], d['amps'], d['mc'], d['N']
        for s in range(0, N - WIN + 1, STRIDE):
            e = s + WIN; m = (peaks >= s) & (peaks < e)
            if int(m.sum()) < MIN_BEATS:
                continue
            X.append(feats(amps[m]))
            ty = int(np.bincount(mc[s:e], minlength=3).argmax())
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
            X.append(feats(amps[m])); Y.append(lbl)   # lbl in {1,2}
    return np.array(X, np.float32), np.array(Y)


def sw3(types):
    if not REWEIGHT:
        return None
    w = np.ones(len(types)); w[types == 1] = LSW; w[types == 2] = HW
    return w


def main():
    names = sorted([f.replace('_mc_mask.npy', '') for f in os.listdir(os.path.join(T.NPY, 'mc_masks'))
                    if os.path.exists(os.path.join(T.INF, f.replace('_mc_mask.npy', '') + '.npy'))])
    tag = 'reweighted' if REWEIGHT else 'baseline-detection'
    print(f'{len(names)} files. 3-CLASS, config={tag}', flush=True)
    data = T.precompute(names)
    for fi in data:
        data[fi]['mc'] = remap(data[fi]['mc'])
    X1, Yb1, TY1, FID1, CEN1 = build_stage1(data)

    files = np.array(sorted(data.keys())); rng = np.random.default_rng(0); rng.shuffle(files)
    folds = np.array_split(files, 5)
    cm = np.zeros((3, 3), np.int64)
    for k in range(5):
        tf = set(folds[k].tolist()); trf = [f for f in files if f not in tf]
        tr1 = np.array([i for i in range(len(Yb1)) if FID1[i] not in tf])
        rf1 = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                     n_jobs=-1, random_state=0)
        rf1.fit(X1[tr1], Yb1[tr1], sample_weight=sw3(TY1[tr1]))
        if REWEIGHT:
            pm_tr = pmove_files(rf1, X1, FID1, CEN1, data, trf)
            thr = tune_thr_balanced(pm_tr, data, trf)
        else:
            thr = 0.5
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
            cm += confusion_matrix(d['mc'][:N], pred, labels=[0, 1, 2])
        print(f'  fold {k+1}/5 thr={thr:.2f}', flush=True)

    rec = {CLS3[c]: float(cm[c, c] / max(cm[c].sum(), 1)) for c in range(3)}
    overall = float(np.trace(cm) / cm.sum())
    print(f'\n=== 3-CLASS [{tag}] ===')
    for c in range(3):
        print(f'  {CLS3[c]:18s}: {rec[CLS3[c]]:.3f}  (n={cm[c].sum()})')
    print(f'  OVERALL accuracy   : {overall:.3f}')
    print(f'  CONFUSION:\n{cm}')
    json.dump({'config': tag, 'recall': rec, 'overall': overall, 'confusion': cm.tolist()},
              open(os.path.join(RES, f'two_stage_3class_{tag}.json'), 'w'), indent=2)
    print('saved.')


if __name__ == '__main__':
    main()
