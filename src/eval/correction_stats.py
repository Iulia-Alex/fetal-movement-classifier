"""
Statistics for the binary correction experiments (examiner Q9): a paired Wilcoxon
signed-rank test (one-sided, alternative='greater') plus a 95% bootstrap confidence
interval over recordings, for the Average Precision (AP) gain of each correction over the
baseline (uncorrected extracted signal), reported separately on HELD-OUT (n=9) and
TEST_DB (n=11).

Reproduces results_fECG_extraction/correction_stats.json from the per-recording AP values
that are already saved (nothing is re-trained):
  - adapter-v1   : adapter_binary_prauc_v1.json        (column 'lat' vs 'base', ceiling 'gt')
  - adapter-v17  : adapter_binary_prauc_v17.json        (column 'lat')
  - finetuned    : clf_finetuned_extracted_binary.json  (column 'ft')

The point estimates (base/corrected/ceiling/gain/gap_recovered) and the Wilcoxon test are
DETERMINISTIC; only the bootstrap intervals depend on the RNG (seed fixed below).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import numpy as np
from scipy.stats import wilcoxon
from config import RESULTS_DIR as RES

N_BOOT = 10000
SEED = 0
SETS = ['HELD-OUT', 'TEST_DB']
# (reported name, source file, key of the corrected column)
SOURCES = [
    ('adapter-v1',  'adapter_binary_prauc_v1.json',        'lat'),
    ('adapter-v17', 'adapter_binary_prauc_v17.json',       'lat'),
    ('finetuned',   'clf_finetuned_extracted_binary.json', 'ft'),
]


def stats_for(rows, key, rng):
    base = np.array([r['base']['ap'] for r in rows], float)
    corr = np.array([r[key]['ap'] for r in rows], float)
    ceil = np.array([r['gt']['ap'] for r in rows], float)
    n = len(rows)
    base_m, corr_m, ceil_m = base.mean(), corr.mean(), ceil.mean()
    gain = corr_m - base_m
    gap = gain / (ceil_m - base_m)
    # bootstrap over recordings (same resample for gain and gap_recovered)
    gains, gaps = np.empty(N_BOOT), np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        bb, cc, gg = base[idx].mean(), corr[idx].mean(), ceil[idx].mean()
        gains[b] = cc - bb
        gaps[b] = (cc - bb) / (gg - bb)
    gain_ci = [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]
    gap_ci = [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]
    p = float(wilcoxon(corr, base, alternative='greater').pvalue)
    return {
        'n': n, 'base': float(base_m), 'corrected': float(corr_m), 'ceiling': float(ceil_m),
        'gain': float(gain), 'gain_ci': gain_ci,
        'gap_recovered': float(gap), 'gap_ci': gap_ci,
        'wilcoxon_p': round(p, 4), 'n_improved': int((corr > base).sum()),
    }


def main():
    rng = np.random.default_rng(SEED)
    out = []
    for st in SETS:
        for name, fname, key in SOURCES:
            data = json.load(open(os.path.join(RES, fname)))
            rows = [r for r in data if r['set'] == st]
            row = {'set': st, 'name': name}
            row.update(stats_for(rows, key, rng))
            out.append(row)
            print(f"{st:<9} {name:<12} n={row['n']:>2} base={row['base']:.3f} "
                  f"corr={row['corrected']:.3f} gain={row['gain']:+.3f} "
                  f"CI=[{row['gain_ci'][0]:.3f},{row['gain_ci'][1]:.3f}] "
                  f"p={row['wilcoxon_p']} improved={row['n_improved']}/{row['n']}", flush=True)
    dst = os.path.join(RES, 'correction_stats.json')
    json.dump(out, open(dst, 'w'), indent=2)
    print('\nsaved:', dst, flush=True)


if __name__ == '__main__':
    main()
