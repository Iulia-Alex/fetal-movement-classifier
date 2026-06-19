"""
Smoke test before the big job:
  - load EACH extraction model (gate: load OK?)
  - extract 1 signal -> R-peak F1 vs GT (gate: extraction sane?)
  - classify 6 channels x 3 clf (timing a full cycle -> wall-time projection)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import numpy as np
import torch
torch.set_num_threads(1)

import pipeline_registry as R
import pipeline_clf as C
import pretrained_clf as E
from eval_qrs_f1_v3 import compute_qrs_f1

from config import NPY
TEST = 'Test_db_87_SNRmn=15dB_SNRfm=0dB_SNRfn=15dB'   # easy
N_SIGNALS = 122


def load_sig(sub, name):
    d = os.path.join(NPY, sub, name)
    return np.stack([np.load(os.path.join(d, [x for x in os.listdir(d) if f'_ch{c}.npy' in x][0])).astype(np.float32)
                     for c in range(1, 7)])


def main():
    mixture = load_sig('mixture', TEST)
    gt = load_sig('signals', TEST)
    print(f"Smoke pe: {TEST}\n", flush=True)
    print("Incarc clasificatoarele Edward...", flush=True)
    clf = {mid: E.load_model(mid) for mid in (14, 15, 18)}

    # classification time: computed once on GT (independent of extraction model)
    t = time.time()
    _ = C.classify_channel(gt[0], clf)
    t_clf_1ch = time.time() - t
    t_clf_sig = t_clf_1ch * 6
    print(f"  classification 1 channel: {t_clf_1ch:.1f}s -> 6 channels/signal: {t_clf_sig:.1f}s\n", flush=True)

    print(f"{'model':<6}{'load':<7}{'R-peak F1':<11}{'t_extract':<11}{'verif':<8}", flush=True)
    print("-"*45, flush=True)
    proj = 0.0
    ok_models = []
    for v in R.MODEL_ORDER:
        verified = R.REGISTRY[v][5]
        try:
            m = R.load_extractor(v)
        except Exception as e:
            print(f"{v:<6}{'FAIL':<7} load: {str(e)[:50]}", flush=True)
            continue
        try:
            t = time.time()
            ext = R.infer(v, m, mixture)
            t_ext = time.time() - t
            n = min(ext.shape[1], gt.shape[1])
            f1 = float(np.mean([r['f1'] for r in compute_qrs_f1(ext[:, :n], gt[:, :n])]))
            flag = '' if verified else 'UNVERIF'
            print(f"{v:<6}{'ok':<7}{f1:<11.3f}{t_ext:<11.1f}{flag:<8}", flush=True)
            per_model = N_SIGNALS * (t_ext + t_clf_sig)
            proj += per_model
            ok_models.append((v, f1))
        except Exception as e:
            print(f"{v:<6}{'ok':<7} infer EROARE: {str(e)[:50]}", flush=True)

    print("\n" + "="*50, flush=True)
    n_ok = len(ok_models)
    print(f"Modele OK: {n_ok}/{len(R.MODEL_ORDER)}", flush=True)
    print(f"Total sequential cost (all models x 122 signals): {proj/3600:.1f} h", flush=True)
    for w in (12, 14, 16):
        print(f"  ~wall-time cu {w} workers: {proj/3600/min(w,n_ok if n_ok else 1):.1f} h", flush=True)
    bad = [v for v, f1 in ok_models if f1 < 0.3]
    if bad:
        print(f"\nWARNING F1 R-peak < 0.3 (suspicious extraction): {bad}", flush=True)


if __name__ == '__main__':
    main()
