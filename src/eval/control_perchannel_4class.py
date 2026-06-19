"""
Control for the 0.30 -> 0.85 decomposition (examiner question Q6).

Same DENSE pipeline on EXTRACTED fECG, trained on the extracted domain (RF, 5-fold
file-level), varying ONLY the representation:
  - per-channel spectral (36)      = PER-CHANNEL representation (no cross-channel geometry),
                                      TRAINED on extracted  -> isolates the DOMAIN effect
                                      (vs M15 per-channel trained on clean, ~0.30)
  - spectral + geom (cross-channel)= full representation -> ~0.84 (= two-stage 0.85)
  - cross-channel geom only        = the pure power of the geometry
The dense GT ceiling (spectral+geom) is already reported in dense_multiclass_crosschannel.json.

The classifier family (RF) and the training domain (extracted) are FIXED across configs,
so the spectral-only -> +geom difference isolates the contribution of the cross-channel
REPRESENTATION.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import dense_multiclass_crosschannel as D

N_SPEC = 36  # 6 channels x 6 per-channel spectral features


def main():
    names = sorted([f.replace('_mc_mask.npy', '') for f in os.listdir(os.path.join(D.NPY, 'mc_masks'))
                    if os.path.exists(os.path.join(D.INF, f.replace('_mc_mask.npy', '') + '.npy'))])
    print(f'{len(names)} files. Building EXTRAS windows once...', flush=True)
    t0 = time.time()
    X, Y, FID, CEN, meta = D.build_windows(names, source='ext')
    print(f'Windows: {len(Y)}  dim={X.shape[1]} (spectral {N_SPEC} + geom {X.shape[1]-N_SPEC})'
          f'  built in {time.time()-t0:.0f}s', flush=True)

    res = {}
    res['perchannel_spectral'] = D.run(X[:, :N_SPEC], Y, FID, CEN, meta,
                                       tag='EXTRAS per-channel (spectral only) — DOMAIN-MATCHED control')
    res['geom_only'] = D.run(X[:, N_SPEC:], Y, FID, CEN, meta,
                             tag='EXTRAS geom only (cross-channel)')
    res['full'] = D.run(X, Y, FID, CEN, meta,
                        tag='EXTRAS full (spectral+geom) — reproduce 0.84')

    out = os.path.join(D.RES, 'control_perchannel_4class.json')
    json.dump(res, open(out, 'w'), indent=2)
    print('\nsaved:', out, flush=True)

    print('\n=== DECOMPOSITION SUMMARY (per-sample) ===', flush=True)
    for k in ('perchannel_spectral', 'geom_only', 'full'):
        r = res[k]
        f1 = r['persample_f1']
        print(f"  {k:<22} acc={r['persample_acc']:.3f} macroF1={r['persample_macroF1']:.3f}"
              f"  | no-move={f1[0]:.2f} linear={f1[1]:.2f} spline={f1[2]:.2f} helix={f1[3]:.2f}", flush=True)


if __name__ == '__main__':
    main()
