"""
Eval FINAL two-stage (16s, varianta cea mai buna) pe EXTRAS, 5-fold file-level.
  - metrica raportata: ACCURACY PER TIP DE MISCARE (recall pe clasa) + overall.
  - salveaza predictiile per-sample (binar + multiclasa) pentru FIECARE fisier,
    ca sa pot alege un semnal bun si sa-l plotez (plot_pipeline_demo.py).
Fara oracle la inferenta: Etapa 1 detecteaza regiunile, Etapa 2 le clasifica.
"""
import os, sys, json, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import two_stage_dense as T
from two_stage_dense import feats, CLS, FS, WIN, build_stage1, build_stage2_phases, detect_runs

RES = T.RES
OUT_PREDS = os.path.join(RES, 'two_stage_preds_ext')
os.makedirs(OUT_PREDS, exist_ok=True)


def main():
    names = sorted([f.replace('_mc_mask.npy', '') for f in os.listdir(os.path.join(T.NPY, 'mc_masks'))
                    if os.path.exists(os.path.join(T.INF, f.replace('_mc_mask.npy', '') + '.npy'))])
    print(f'{len(names)} files, source=ext, WIN={WIN/FS:.0f}s', flush=True)
    data = T.precompute(names)
    X1, Yb1, FID1, CEN1 = build_stage1(data)
    print(f'stage1 windows: {len(Yb1)} movefrac={Yb1.mean():.3f}', flush=True)

    files = np.array(sorted(data.keys()))
    rng = np.random.default_rng(0); rng.shuffle(files)
    folds = np.array_split(files, 5)

    cm = np.zeros((4, 4), dtype=np.int64)
    per_file = []
    for k in range(5):
        test_f = set(folds[k].tolist())
        train_fids = [f for f in files if f not in test_f]
        tr1 = np.array([i for i in range(len(Yb1)) if FID1[i] not in test_f])
        rf1 = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                     n_jobs=-1, random_state=0).fit(X1[tr1], Yb1[tr1])
        Xp, Yp = build_stage2_phases(data, train_fids)
        rf2 = RandomForestClassifier(n_estimators=400, class_weight='balanced',
                                     n_jobs=-1, random_state=0).fit(Xp, Yp)
        for fi in folds[k]:
            d = data[fi]; N = d['N']; peaks = d['peaks']; amps = d['amps']
            idx = np.where(FID1 == fi)[0]
            acc = np.zeros(N, np.float32); cov = np.zeros(N, np.float32)
            if len(idx):
                pm = rf1.predict_proba(X1[idx])[:, 1]
                for j, wi in enumerate(idx):
                    c = CEN1[wi]; s = max(0, c - WIN // 2); e = min(N, c + WIN // 2)
                    acc[s:e] += pm[j]; cov[s:e] += 1.0
            cov_safe = cov.copy(); cov_safe[cov_safe == 0] = 1.0
            p_move = acc / cov_safe
            pred = np.zeros(N, np.int64)
            for (a, b) in detect_runs(p_move, N):
                m = (peaks >= a) & (peaks < b)
                if int(m.sum()) < T.MIN_BEATS:
                    pred[a:b] = 1; continue
                pred[a:b] = int(rf2.predict(feats(amps[m])[None])[0])
            mc = d['mc'][:N]
            cm += confusion_matrix(mc, pred, labels=[0, 1, 2, 3])
            facc = accuracy_score(mc, pred)
            per_file.append((d['name'], float(facc),
                             np.bincount(mc, minlength=4).tolist()))
            np.savez_compressed(os.path.join(OUT_PREDS, d['name'] + '.npz'),
                                pred=pred.astype(np.int8), mc=mc.astype(np.int8),
                                p_move=p_move.astype(np.float16), peaks=peaks.astype(np.int32))
        print(f'  fold {k+1}/5 done ({time.time():.0f})', flush=True)

    # ACCURACY PER TYPE (recall per class = diag / row-sum)
    per_class_acc = {CLS[c]: float(cm[c, c] / max(cm[c].sum(), 1)) for c in range(4)}
    overall = float(np.trace(cm) / cm.sum())
    print('\n=== ACCURACY PER MOVEMENT TYPE (two-stage, extracted, per-sample) ===')
    for c in range(4):
        print(f'  {CLS[c]:9s}: {per_class_acc[CLS[c]]:.3f}  (n={cm[c].sum()})')
    print(f'  OVERALL  : {overall:.3f}')
    print(f'\nCONFUSION:\n{cm}')

    json.dump({'per_class_accuracy': per_class_acc, 'overall_accuracy': overall,
               'confusion': cm.tolist()},
              open(os.path.join(RES, 'two_stage_per_class_accuracy.json'), 'w'), indent=2)

    # top files for plot: high acc + contain all movement types
    per_file.sort(key=lambda r: -r[1])
    print('\nTop files (acc, [n_nomove,n_lin,n_spl,n_hel]):')
    for nm, fa, dist in per_file[:12]:
        has_all = all(d > 0 for d in dist[1:])
        print(f'  {fa:.3f} all3={has_all}  {nm}  {dist}')
    json.dump(per_file, open(os.path.join(RES, 'two_stage_per_file_acc.json'), 'w'), indent=2)
    print('\nsaved preds ->', OUT_PREDS)


if __name__ == '__main__':
    main()
