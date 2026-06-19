"""
DECISIVE control for Q6: separates "domain matching" from "reading the leads together".

A FAITHFUL per-channel version of the M15 classifier (each lead classified on its own,
then the probabilities AVERAGED over the 6 leads), but TRAINED on the EXTRACTED signal
(5-fold file-level). Uses ONLY the 6 spectral features of each lead.

Comparison:
  M15 per-channel, trained on CLEAN, applied to extracted -> ~0.30 (in the thesis)
  per-channel (this control), trained on EXTRACTED         -> ?      (pure DOMAIN effect)
  6-channel spectral together, extracted                   -> 0.83  (+ joint reading)
  spectral+geom cross-channel, extracted                   -> 0.84  (+ geometry)
If per-channel-extracted ~0.30 -> the jump comes from the joint reading (representation).
If per-channel-extracted ~0.8  -> the jump comes from domain matching.
"""
import os, sys, json, time
import numpy as np
from scipy.ndimage import median_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import dense_multiclass_crosschannel as D

FPER = 6  # per-lead spectral features (std, slope, r2, fft_amp, fft_freq, autocorr5)


def dense_predict_perlead(rfs, X, FID, CENTER, meta, test_fids):
    """Per-sample: AVERAGE over 6 leads of the window probabilities (overlap-add)."""
    out = {}
    for fid in test_fids:
        name, N, _ = meta[fid]
        acc = np.zeros((4, N), np.float32); cov = np.zeros(N, np.float32)
        idx = np.where(FID == fid)[0]
        if len(idx):
            # average over leads of the per-window proba
            proba = np.zeros((len(idx), 4), np.float32)
            for ch in range(6):
                proba += rfs[ch].predict_proba(X[idx][:, ch*FPER:(ch+1)*FPER])
            proba /= 6.0
            for j, wi in enumerate(idx):
                c = CENTER[wi]; s = max(0, c - D.WIN // 2); e = min(N, c + D.WIN // 2)
                acc[:, s:e] += proba[j][:, None]; cov[s:e] += 1.0
        pred = acc.argmax(0); pred[cov == 0] = 0
        pred = median_filter(pred, size=D.FS).astype(int)
        out[fid] = (pred, D.load_mc(name)[:N])
    return out


def run_perlead(X, Y, FID, CENTER, meta, tag=''):
    files = np.unique(FID); rng = np.random.default_rng(0); rng.shuffle(files)
    folds = np.array_split(files, 5)
    all_pred, all_true = [], []
    for k in range(5):
        test_f = set(folds[k].tolist())
        trn = np.array([i for i in range(len(Y)) if FID[i] not in test_f])
        rfs = []
        for ch in range(6):
            rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                        n_jobs=-1, random_state=0)
            rf.fit(X[trn][:, ch*FPER:(ch+1)*FPER], Y[trn]); rfs.append(rf)
        dense = dense_predict_perlead(rfs, X, FID, CENTER, meta, folds[k])
        for fid in folds[k]:
            p, t = dense[fid]; all_pred.append(p); all_true.append(t)
        print(f'  fold {k+1}/5 done', flush=True)
    yp = np.concatenate(all_pred); yt = np.concatenate(all_true)
    f1pc = f1_score(yt, yp, labels=[0,1,2,3], average=None, zero_division=0)
    macro = f1_score(yt, yp, labels=[0,1,2,3], average='macro', zero_division=0)
    acc = accuracy_score(yt, yp)
    print(f'\n=== {tag} ===')
    print(f'  PER-SAMPLE acc={acc:.3f} macro-F1={macro:.3f}')
    print(f'  PER-SAMPLE F1: ' + '  '.join(f'{D.CLS[c]}={f1pc[c]:.3f}' for c in range(4)))
    print(f'  CONFUSION:\n{confusion_matrix(yt, yp, labels=[0,1,2,3])}')
    return {'tag': tag, 'persample_acc': float(acc), 'persample_macroF1': float(macro),
            'persample_f1': f1pc.tolist()}


def main():
    names = sorted([f.replace('_mc_mask.npy', '') for f in os.listdir(os.path.join(D.NPY, 'mc_masks'))
                    if os.path.exists(os.path.join(D.INF, f.replace('_mc_mask.npy', '') + '.npy'))])
    print(f'{len(names)} files. Building EXTRAS windows...', flush=True)
    t0 = time.time()
    X, Y, FID, CEN, meta = D.build_windows(names, source='ext')
    print(f'Windows: {len(Y)}  built in {time.time()-t0:.0f}s', flush=True)
    res = run_perlead(X[:, :36], Y, FID, CEN, meta,
                      tag='EXTRAS per-LEAD averaged (faithful per-channel, domain-matched)')
    out = os.path.join(D.RES, 'control_perlead_4class.json')
    json.dump(res, open(out, 'w'), indent=2)
    print('\nsaved:', out, flush=True)


if __name__ == '__main__':
    main()
