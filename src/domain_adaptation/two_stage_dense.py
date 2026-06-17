"""
DELIVERABLE v2 — dense per-sample classification in TWO STAGES, no oracle phase boundaries,
no re-training of the extractor. Solves the linear/spline collapse from the fixed-window
variant: movement type is only visible on the COMPLETE trajectory, not on a 16s window.

  Stage 1 (detection): 16s sliding windows -> binary RF (movement vs no-move) ->
          per-sample probability (overlap-add) -> threshold + morphological cleanup ->
          detected MOVEMENT REGIONS (data-driven, NOT oracle).
  Stage 2 (type): for each detected region, cross-channel features over ALL beats
          in the region -> 3-class RF (linear/spline/helix) -> label assigned
          to all samples in the region.

Representation = cross-channel features (probe_crosschannel_multiclass). Both RFs
trained on EXTRACTED signals. 5-fold file-level CV. Stage 2 trained on oracle phases
only from TRAIN files (labels allowed at training); at TEST no oracle is used.
"""
import os, sys, time, json
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import median_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
from probe_crosschannel_multiclass import spectral_feats, geom_feats

from config import NPY, FECG_ROOT as ROOT, RESULTS_DIR as RES
INF = ROOT + '/inferred_v1'
FS = 500
WIN = 16 * FS
STRIDE = 2 * FS
MIN_BEATS = 8
MIN_RUN_S = 8           # detected regions shorter than this -> dropped
FILL_GAP_S = 3          # gaps shorter than this between regions -> filled
CLS = {0: 'no-move', 1: 'linear', 2: 'spline', 3: 'helix'}
_bp = butter(2, [5, 40], btype='band', fs=FS)
SRC = os.environ.get('SRC', 'ext')   # 'ext' (deliverable) or 'gt' (ceiling)


def load_gt(name):
    d = os.path.join(NPY, 'signals', name)
    return np.stack([np.load(os.path.join(d, [x for x in os.listdir(d) if f'_ch{c}.npy' in x][0])).astype(np.float32)
                     for c in range(1, 7)])


def load_mc(name):
    return np.load(os.path.join(NPY, 'mc_masks', name + '_mc_mask.npy')).astype(int)


def detect_fqrs(sig6):
    env = np.zeros(sig6.shape[1])
    for ch in range(6):
        z = (sig6[ch] - sig6[ch].mean()) / (sig6[ch].std() + 1e-8)
        env += np.abs(filtfilt(_bp[0], _bp[1], z))
    pk, _ = find_peaks(env, distance=150, height=np.percentile(env, 75))
    return pk


def amp_at(sig, peaks, w=12):
    if len(peaks) == 0:
        return np.zeros(0, np.float32)
    idx = np.clip(np.asarray(peaks)[:, None] + np.arange(-w, w), 0, len(sig) - 1)
    return np.max(np.abs(sig[idx]), axis=1).astype(np.float32)


def feats(amps):
    return np.array(spectral_feats(amps) + geom_feats(amps), np.float32)


# ---------------------------------------------------------------------------
def precompute(names):
    """Per file: peaks (on extracted), amps_all (Nb,6) from source, mc, N."""
    data = {}
    t0 = time.time()
    for fi, name in enumerate(names):
        ext = np.load(os.path.join(INF, name + '.npy'))
        gt = load_gt(name) if SRC == 'gt' else None
        mc = load_mc(name)
        N = min(ext.shape[1], len(mc), (gt.shape[1] if gt is not None else 10 ** 12))
        ext = ext[:, :N]; mc = mc[:N]
        peaks = detect_fqrs(ext)
        src = ext if SRC == 'ext' else gt[:, :N]
        amps_all = np.stack([amp_at(src[ch], peaks) for ch in range(6)], axis=1)
        data[fi] = dict(name=name, N=N, peaks=peaks, amps=amps_all, mc=mc)
        if (fi + 1) % 30 == 0:
            print(f'  precompute {fi+1}/{len(names)} ({time.time()-t0:.0f}s)', flush=True)
    return data


def build_stage1(data):
    """Sliding windows -> features + binary label (majority movement)."""
    X, Yb, FID, CEN = [], [], [], []
    for fi, d in data.items():
        peaks, amps, mc, N = d['peaks'], d['amps'], d['mc'], d['N']
        for s in range(0, N - WIN + 1, STRIDE):
            e = s + WIN
            m = (peaks >= s) & (peaks < e)
            if int(m.sum()) < MIN_BEATS:
                continue
            X.append(feats(amps[m]))
            Yb.append(int((mc[s:e] > 0).mean() >= 0.5))
            FID.append(fi); CEN.append((s + e) // 2)
    return np.array(X, np.float32), np.array(Yb), np.array(FID), np.array(CEN)


def build_stage2_phases(data, fids):
    """Oracle movement phases (classes 1/2/3) from given files -> features over the full phase."""
    X, Y = [], []
    for fi in fids:
        d = data[fi]; peaks, amps, mc = d['peaks'], d['amps'], d['mc']
        tr = np.where(np.diff(mc) != 0)[0]
        starts = np.concatenate([[0], tr + 1]); ends = np.concatenate([tr + 1, [len(mc)]])
        for s, e in zip(starts, ends):
            lbl = int(mc[s])
            if lbl == 0 or e - s < 10 * FS:
                continue
            m = (peaks >= s) & (peaks < e)
            if int(m.sum()) < MIN_BEATS:
                continue
            X.append(feats(amps[m])); Y.append(lbl)
    return np.array(X, np.float32), np.array(Y)


def detect_runs(p_move, N):
    """Prob per-sample -> threshold -> cleanup -> list of (s,e) movement regions."""
    bm = (p_move >= 0.5).astype(int)
    bm = (median_filter(bm, size=FS) > 0).astype(int)   # 1s smoothing
    # fill short gaps
    s = 0; out = []
    # extract runs of 1s
    edges = np.diff(np.concatenate([[0], bm, [0]]))
    starts = np.where(edges == 1)[0]; ends = np.where(edges == -1)[0]
    runs = list(zip(starts, ends))
    # fill gaps
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < FILL_GAP_S * FS:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(list(r))
    # remove short
    return [(a, b) for a, b in merged if b - a >= MIN_RUN_S * FS]


def run_cv(data):
    files = np.array(sorted(data.keys()))
    rng = np.random.default_rng(0); rng.shuffle(files)
    folds = np.array_split(files, 5)
    X1, Yb1, FID1, CEN1 = build_stage1(data)
    print(f'  stage1 windows: {len(Yb1)}  movement frac={Yb1.mean():.3f}', flush=True)

    all_pred, all_true = [], []
    for k in range(5):
        test_f = set(folds[k].tolist())
        train_fids = [f for f in files if f not in test_f]
        tr1 = np.array([i for i in range(len(Yb1)) if FID1[i] not in test_f])
        rf1 = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                     n_jobs=-1, random_state=0)
        rf1.fit(X1[tr1], Yb1[tr1])
        Xp, Yp = build_stage2_phases(data, train_fids)
        rf2 = RandomForestClassifier(n_estimators=400, class_weight='balanced',
                                     n_jobs=-1, random_state=0)
        rf2.fit(Xp, Yp)

        for fi in folds[k]:
            d = data[fi]; N = d['N']; peaks = d['peaks']; amps = d['amps']
            # stage 1: movement probability per-sample
            idx = np.where(FID1 == fi)[0]
            acc = np.zeros(N, np.float32); cov = np.zeros(N, np.float32)
            if len(idx):
                pm = rf1.predict_proba(X1[idx])[:, 1]
                for j, wi in enumerate(idx):
                    c = CEN1[wi]; s = max(0, c - WIN // 2); e = min(N, c + WIN // 2)
                    acc[s:e] += pm[j]; cov[s:e] += 1.0
            cov[cov == 0] = 1.0
            p_move = acc / cov
            runs = detect_runs(p_move, N)
            # stage 2: type for each detected region
            pred = np.zeros(N, np.int64)
            for (a, b) in runs:
                m = (peaks >= a) & (peaks < b)
                if int(m.sum()) < MIN_BEATS:
                    pred[a:b] = 1  # too few beats → fallback to linear (weakest signal class)
                    continue
                lbl = int(rf2.predict(feats(amps[m])[None])[0])
                pred[a:b] = lbl
            all_pred.append(pred); all_true.append(d['mc'][:N])
        print(f'  fold {k+1}/5 done', flush=True)

    yp = np.concatenate(all_pred); yt = np.concatenate(all_true)
    f1pc = f1_score(yt, yp, labels=[0, 1, 2, 3], average=None, zero_division=0)
    macro = f1_score(yt, yp, labels=[0, 1, 2, 3], average='macro', zero_division=0)
    acc = accuracy_score(yt, yp)
    # binary (movement yes/no) as reference
    binf1 = f1_score((yt > 0).astype(int), (yp > 0).astype(int), zero_division=0)
    print(f'\n=== TWO-STAGE DENSE [{SRC}] ===')
    print(f'  PER-SAMPLE: acc={acc:.3f}  macro-F1={macro:.3f}  binary-move-F1={binf1:.3f}')
    print(f'  F1 per-class: ' + '  '.join(f'{CLS[c]}={f1pc[c]:.3f}' for c in range(4)))
    print(f'  CONFUSION:\n{confusion_matrix(yt, yp, labels=[0,1,2,3])}')
    return {'src': SRC, 'persample_acc': float(acc), 'persample_macroF1': float(macro),
            'binary_move_F1': float(binf1), 'persample_f1': f1pc.tolist()}


def main():
    names = sorted([f.replace('_mc_mask.npy', '') for f in os.listdir(os.path.join(NPY, 'mc_masks'))
                    if os.path.exists(os.path.join(INF, f.replace('_mc_mask.npy', '') + '.npy'))])
    print(f'{len(names)} files. source={SRC}', flush=True)
    data = precompute(names)
    res = run_cv(data)
    out = os.path.join(RES, f'two_stage_dense_{SRC}.json')
    json.dump(res, open(out, 'w'), indent=2)
    print('saved:', out)


if __name__ == '__main__':
    main()
