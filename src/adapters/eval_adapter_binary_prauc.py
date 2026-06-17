"""
Threshold-independent comparison for the binary adapter: base vs lat (vs GT ceiling).
F1@0.5 conflates "PR curve shifted" with "only threshold shifted" — base sits at the
high-recall/low-precision corner, lat at the high-precision/low-recall corner. So we report:
  - AP (average precision = area under precision-recall curve, threshold-independent)
  - best-F1 across all thresholds (best operating point for each detector)
If lat's PR curve dominates base's (higher AP) => the adapter genuinely improved the
detector, and flat F1@0.5 is a threshold artefact.

Uses the already-saved checkpoint and cached extractions. Per channel -> mean over channels
-> mean over set (held-out 9 / Test_DB Sem1..11). Variant: v1 (default) or v17.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import numpy as np
import torch
os.environ.setdefault('LA_THREADS', '4')
torch.set_num_threads(int(os.environ['LA_THREADS']))
from sklearn.metrics import average_precision_score, precision_recall_curve

import pretrained_clf as E
import pipeline_registry as R
from latent_adapter import LatentAdapter
import latent_adapter_binary as B


def best_f1(y, prob):
    p, r, _ = precision_recall_curve(y, prob)
    f = 2 * p * r / (p + r + 1e-9)
    return float(np.nanmax(f))


def ap(y, prob):
    return float(average_precision_score(y, prob)) if y.max() > 0 else float('nan')


def metrics_signal(net14, A, fecg, ext, mb):
    """Per-channel AP + best-F1 for gt(clean)/base(extracted)/lat(extracted+A); mean over channels."""
    N = min(fecg.shape[1], ext.shape[1], len(mb)); mb = mb[:N]
    out = {k: {'ap': [], 'bf1': []} for k in ('gt', 'base', 'lat')}
    for ch in range(6):
        pg = E.run_model(14, fecg[ch, :N], net14, stride=3840)[1][:N]
        pb = E.run_model(14, ext[ch, :N], net14, stride=3840)[1][:N]
        pl = B.predict_dense_binary(net14, A, ext[ch, :N], stride=3840, return_prob=True)[:N]
        for k, pr in (('gt', pg), ('base', pb), ('lat', pl)):
            out[k]['ap'].append(ap(mb, pr)); out[k]['bf1'].append(best_f1(mb, pr))
    return {k: {'ap': float(np.nanmean(v['ap'])), 'bf1': float(np.nanmean(v['bf1']))}
            for k, v in out.items()}


def main():
    variant = B.VARIANT
    net14 = E.load_model(14)
    A = LatentAdapter(); A.load_state_dict(torch.load(B.CKPT, map_location='cpu')); A.eval()
    print(f'[{variant}] CKPT {os.path.basename(B.CKPT)} loaded.', flush=True)

    heldout = sorted(set(B.heldout_signals()))
    EXT = R.load_extractor(variant)
    rows = []
    print('=== held-out (9) ===', flush=True)
    for nm in heldout:
        fecg, ext, mb = B.heldout_fn(nm)
        m = metrics_signal(net14, A, fecg, ext, mb); m['set'] = 'HELD-OUT'; m['name'] = nm.split('_SNR')[0]
        rows.append(m)
        print(f"  {m['name']:<12} AP base={m['base']['ap']:.3f} lat={m['lat']['ap']:.3f} gt={m['gt']['ap']:.3f}"
              f" | bestF1 base={m['base']['bf1']:.3f} lat={m['lat']['bf1']:.3f} gt={m['gt']['bf1']:.3f}", flush=True)
    print('=== Test_DB (Sem1..Sem11) ===', flush=True)
    for stem in B.SEMS:
        fecg, ext, mb = B.sem_fn(EXT, stem)
        m = metrics_signal(net14, A, fecg, ext, mb); m['set'] = 'TEST_DB'; m['name'] = stem
        rows.append(m)
        print(f"  {m['name']:<12} AP base={m['base']['ap']:.3f} lat={m['lat']['ap']:.3f} gt={m['gt']['ap']:.3f}"
              f" | bestF1 base={m['base']['bf1']:.3f} lat={m['lat']['bf1']:.3f} gt={m['gt']['bf1']:.3f}", flush=True)

    out = os.path.join(B.RES, f'adapter_binary_prauc_{variant}.json')
    json.dump(rows, open(out, 'w'), indent=2)
    print('\n=== SUMMARY threshold-independent (mean per set) ===', flush=True)
    for label in ('HELD-OUT', 'TEST_DB'):
        sub = [r for r in rows if r['set'] == label]
        for k in ('gt', 'base', 'lat'):
            mAP = np.mean([r[k]['ap'] for r in sub]); mBF = np.mean([r[k]['bf1'] for r in sub])
            print(f'  {label:<9} {k:<5} AP={mAP:.3f}  bestF1={mBF:.3f}', flush=True)
    print(f'\njson: {out}', flush=True)


if __name__ == '__main__':
    main()
