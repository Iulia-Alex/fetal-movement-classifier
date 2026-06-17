"""
DELIVERABLE: clasificare DENSA per-sample a tipului de miscare (0..3) pe fECG EXTRAS,
FARA granite de faza si FARA re-antrenarea extractiei.

Diferente fata de probe (probe_crosschannel_multiclass.py, care folosea granite oracle):
  - ferestre GLISANTE de lungime fixa (W=16s, pas 2s) — nicio informatie despre
    lungimea/granitele miscarilor (cum cere un scenariu real);
  - R-peaks detectati pe semnalul EXTRAS (nu pe GT) — pipeline realist;
  - predictie per-sample prin overlap-add al probabilitatilor + netezire temporala.

Reprezentare = features CROSS-CANAL (spectral per-canal + geometrie 6D a traiectoriei
de amplitudini), pe care probe-ul a aratat-o robusta la extractie (vs per-canal M15).
Clasificator RF re-antrenat pe EXTRAS (adaptare de domeniu, label-only).

Eval: 5-fold file-level CV. Raporteaza per-sample macro-F1 + F1 per-clasa + confusion,
fata de plafonul DENS pe GT (acelasi pipeline, dar amplitudini din GT).
"""
import os, sys, time, json
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import median_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
from probe_crosschannel_multiclass import spectral_feats, geom_feats, GEOM_NAMES

from config import NPY, FECG_ROOT as ROOT, RESULTS_DIR as RES
INF = ROOT + '/inferred_v1'
FS = 500
WIN = 16 * FS          # 8000 samples / 16s
STRIDE = 2 * FS        # 1000 samples / 2s
MIN_BEATS = 8
CLS = {0: 'no-move', 1: 'linear', 2: 'spline', 3: 'helix'}
_bp = butter(2, [5, 40], btype='band', fs=FS)


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


def feats_for_window(amps):
    return spectral_feats(amps) + geom_feats(amps)


def build_windows(names, source='ext'):
    """
    Pentru fiecare fereastra glisanta: features + label majoritar + (fid, centru).
    Peaks detectati pe EXTRAS (pipeline realist). 'source' = de unde citim amplitudinile
    (ext = deliverable; gt = plafon dens, acelasi peaks/ferestre).
    Returneaza si, per fisier, lungimea N si peaks (pt reconstructia densa).
    """
    X, Y, FID, CENTER = [], [], [], []
    meta = {}   # fid -> (name, N, peaks)
    t0 = time.time()
    for fi, name in enumerate(names):
        ext = np.load(os.path.join(INF, name + '.npy'))
        gt = load_gt(name)
        mc = load_mc(name)
        N = min(ext.shape[1], gt.shape[1], len(mc))
        ext, gt, mc = ext[:, :N], gt[:, :N], mc[:N]
        peaks = detect_fqrs(ext)
        src = ext if source == 'ext' else gt
        amps_all = np.stack([amp_at(src[ch], peaks) for ch in range(6)], axis=1)  # (Nb,6)
        meta[fi] = (name, N, peaks)
        for s in range(0, N - WIN + 1, STRIDE):
            e = s + WIN
            m = (peaks >= s) & (peaks < e)
            nb = int(m.sum())
            if nb < MIN_BEATS:
                continue
            amps = amps_all[m]
            X.append(feats_for_window(amps))
            # label = majority class over the window
            Y.append(int(np.bincount(mc[s:e], minlength=4).argmax()))
            FID.append(fi); CENTER.append((s + e) // 2)
        if (fi + 1) % 20 == 0:
            print(f'  windows {fi+1}/{len(names)}: {len(Y)} ({time.time()-t0:.0f}s)', flush=True)
    return (np.array(X, np.float32), np.array(Y), np.array(FID),
            np.array(CENTER), meta)


def dense_predict(rf, X, FID, CENTER, meta, test_fids):
    """
    Per-sample: acumuleaza probabilitatile ferestrelor (overlap-add pe intervalul
    fiecarei ferestre) -> argmax -> netezire mediana. Returneaza dict fid -> (pred, mc).
    """
    out = {}
    for fid in test_fids:
        name, N, _ = meta[fid]
        acc = np.zeros((4, N), np.float32)
        cov = np.zeros(N, np.float32)
        idx = np.where(FID == fid)[0]
        if len(idx):
            proba = rf.predict_proba(X[idx])     # (n_win,4)
            for j, wi in enumerate(idx):
                c = CENTER[wi]; s = max(0, c - WIN // 2); e = min(N, c + WIN // 2)
                acc[:, s:e] += proba[j][:, None]
                cov[s:e] += 1.0
        pred = acc.argmax(0)
        pred[cov == 0] = 0       # zone neacoperite (margini) -> no-move
        pred = median_filter(pred, size=FS).astype(int)  # netezire 1s
        out[fid] = (pred, load_mc(name)[:N])
    return out


def run(X, Y, FID, CENTER, meta, tag=''):
    files = np.unique(FID)
    rng = np.random.default_rng(0); rng.shuffle(files)
    folds = np.array_split(files, 5)
    all_pred, all_true = [], []
    win_pred, win_true = np.full(len(Y), -1), Y
    for k in range(5):
        test_f = set(folds[k].tolist())
        te = np.array([i for i in range(len(Y)) if FID[i] in test_f])
        trn = np.array([i for i in range(len(Y)) if FID[i] not in test_f])
        rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                    n_jobs=-1, random_state=0)
        rf.fit(X[trn], Y[trn])
        win_pred[te] = rf.predict(X[te])
        dense = dense_predict(rf, X, FID, CENTER, meta, folds[k])
        for fid in folds[k]:
            p, t = dense[fid]
            all_pred.append(p); all_true.append(t)
        print(f'  fold {k+1}/5 done', flush=True)
    yp = np.concatenate(all_pred); yt = np.concatenate(all_true)
    # window-level (pt referinta)
    wf1 = f1_score(win_true, win_pred, labels=[0,1,2,3], average=None, zero_division=0)
    wmac = f1_score(win_true, win_pred, labels=[0,1,2,3], average='macro', zero_division=0)
    # per-sample (DELIVERABLE)
    f1pc = f1_score(yt, yp, labels=[0,1,2,3], average=None, zero_division=0)
    macro = f1_score(yt, yp, labels=[0,1,2,3], average='macro', zero_division=0)
    acc = accuracy_score(yt, yp)
    print(f'\n=== {tag} ===')
    print(f'  WINDOW-level : macro-F1={wmac:.3f}  ' + ' '.join(f'{CLS[c]}={wf1[c]:.3f}' for c in range(4)))
    print(f'  PER-SAMPLE   : acc={acc:.3f}  macro-F1={macro:.3f}')
    print(f'  PER-SAMPLE F1: ' + '  '.join(f'{CLS[c]}={f1pc[c]:.3f}' for c in range(4)))
    print(f'  CONFUSION (per-sample):\n{confusion_matrix(yt, yp, labels=[0,1,2,3])}')
    return {'tag': tag, 'persample_acc': float(acc), 'persample_macroF1': float(macro),
            'persample_f1': f1pc.tolist(), 'window_macroF1': float(wmac),
            'window_f1': wf1.tolist()}


def main():
    names = sorted([f.replace('_mc_mask.npy', '') for f in os.listdir(os.path.join(NPY, 'mc_masks'))
                    if os.path.exists(os.path.join(INF, f.replace('_mc_mask.npy', '') + '.npy'))])
    print(f'{len(names)} files. W={WIN/FS:.0f}s stride={STRIDE/FS:.0f}s', flush=True)
    print('Building EXTRACTED windows...', flush=True)
    Xe, Y, FID, CEN, meta = build_windows(names, source='ext')
    print(f'Windows: {len(Y)}  class dist (window labels): {np.bincount(Y, minlength=4).tolist()}', flush=True)
    res = {}
    res['extras'] = run(Xe, Y, FID, CEN, meta, tag='EXTRAS dense (deliverable)')

    print('\nBuilding GT windows (dense ceiling)...', flush=True)
    Xg, Yg, FIDg, CENg, metag = build_windows(names, source='gt')
    res['gt'] = run(Xg, Yg, FIDg, CENg, metag, tag='GT dense (ceiling)')

    os.makedirs(RES, exist_ok=True)
    json.dump(res, open(os.path.join(RES, 'dense_multiclass_crosschannel.json'), 'w'), indent=2)
    print('\nsaved:', os.path.join(RES, 'dense_multiclass_crosschannel.json'))


if __name__ == '__main__':
    main()
