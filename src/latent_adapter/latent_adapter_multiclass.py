"""
ADAPTER LATENT MULTICLASA (4 clase de miscare) — varianta scalata pe date, paralela
cu latent_adapter_binary.py. Acelasi principiu ca latent_adapter.py (corectie la
bottleneck-ul clasificatorului M15, LABEL-ONLY, prin decoderul INGHETAT) DAR:
  - antrenat pe 113 semnale (Final_Test_DB 122 minus cele 9 held-out), nu pe DB_1;
  - extractiile DEJA in cache (inferred_{VARIANT}/) — fara re-inferenta;
  - parametrizat pe varianta de extractie: v1 (Baseline) sau v17 (Attention-supervised).

Loss = CrossEntropy (ponderi de clasa inv-freq temperate, contra dominantei no-move)
prin DECODER-UL INGHETAT vs masca categoriala (0..3). Nu se foloseste semnal GT nicaieri.

Eval pe DOUA seturi disjuncte de train: held-out (9) + Test_DB Sem1..Sem11.
Raporteaza acc + macro-F1 (+ F1 per-clasa) pt gt(M15 curat)/base(M15 extras)/lat(M15+A).
Probe: `python3 latent_adapter_multiclass.py probe [v17]`.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import numpy as np
import torch, torch.nn as nn
os.environ.setdefault('LA_THREADS', '8')
torch.set_num_threads(int(os.environ['LA_THREADS']))
import scipy.io as sio
from sklearn.metrics import accuracy_score, f1_score

from diag_info_ceiling import load_sig, load_mask
import pipeline_registry as R
import pretrained_clf as E
from latent_adapter import (encode, decode, LatentAdapter, heldout_signals,
                            predict_dense_latent, N_CLASS)
from movement_dataset import _extract_fecg

from config import NPY, FECG_ROOT as ROOT, TEST_DIR, RESULTS_DIR as RES
WIN = E.WIN
SEMS = [f'Sem{i}' for i in range(1, 12)]

# extraction variant (source fECG data used to train/eval the adapter)
VARIANT = next((a for a in sys.argv[1:] if a.startswith('v') and a[1:].isdigit()), 'v1')
INFERRED = os.path.join(ROOT, f'inferred_{VARIANT}')
CKPT = os.path.join(RES, f'adapter_latent_multiclass_m15_{VARIANT}.pt')
OUT = os.path.join(RES, f'adapter_latent_multiclass_{VARIANT}.json')


# ── cache (b_ext, skips, 4-class label) from extractions already in inferred_{VARIANT} ──
def build_cache(names, net, n_windows, seed=0):
    rng = np.random.default_rng(seed)
    per_ch = max(1, n_windows // (len(names) * 6))
    B, S, Y = [], [[], [], [], []], []
    t0 = time.time()
    for fi, nm in enumerate(names):
        ext = np.load(os.path.join(INFERRED, nm + '.npy'))          # extras (cache)
        mc = load_mask('mc_masks', nm).astype(int)                  # categorial 0..3
        N = min(ext.shape[1], len(mc)); max_start = N - WIN
        for ch in range(6):
            starts = rng.integers(0, max_start, size=per_ch)
            feats = np.stack([E.features_5(ext[ch, s0:s0 + WIN]) for s0 in starts])
            with torch.no_grad():
                b, skips = encode(net, torch.from_numpy(feats))
            B.append(b.half().numpy())
            for j in range(4):
                S[j].append(skips[j].half().numpy())
            Y.append(np.stack([mc[s0:s0 + WIN] for s0 in starts]).astype(np.int8))
        if (fi + 1) % 10 == 0:
            print(f'  cache [{fi+1}/{len(names)}] {(time.time()-t0)/60:.1f}min', flush=True)
    B = np.concatenate(B); S = [np.concatenate(s) for s in S]; Y = np.concatenate(Y)
    freq = np.bincount(Y.ravel(), minlength=N_CLASS)
    print(f'  cache: {len(B)} windows, ~{(B.nbytes+sum(s.nbytes for s in S))/1e9:.1f}GB '
          f'({(time.time()-t0)/60:.1f}min); class freq={freq.tolist()}', flush=True)
    return B, S, Y


def train(net, cache, epochs=12, bs=32, lr=1e-3, class_weight=None):
    B, S, Y = cache; n = len(B); A = LatentAdapter()
    opt = torch.optim.Adam(A.parameters(), lr=lr)
    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)
    for p in net.parameters():
        p.requires_grad_(False)
    for ep in range(epochs):
        perm = np.random.permutation(n); tot = 0.0; t0 = time.time()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            b = torch.from_numpy(B[idx].astype(np.float32))
            skips = tuple(torch.from_numpy(S[j][idx].astype(np.float32)) for j in range(4))
            y = torch.from_numpy(Y[idx].astype(np.int64))            # (bs,WIN)
            opt.zero_grad()
            logits = decode(net, A(b), skips)                        # (bs,4,WIN)
            loss = lossf(logits, y)
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f'  ep{ep+1}/{epochs} loss={tot/n:.4f} ({time.time()-t0:.0f}s)', flush=True)
    return A.eval()


def _mc_metrics(mc, pred):
    """acc, macro-F1, [F1 per-class 0..3]."""
    fpc = f1_score(mc, pred, labels=list(range(N_CLASS)), average=None, zero_division=0)
    return [accuracy_score(mc, pred), float(np.mean(fpc))] + fpc.tolist()


def eval_signal(net15, A, fecg, ext, mc):
    """Per-channel: base(M15 on extracted), lat(M15+A on extracted), gt(M15 on clean)."""
    N = min(fecg.shape[1], ext.shape[1], len(mc)); mc = mc[:N]
    out = {k: [] for k in ('gt', 'base', 'lat')}
    for ch in range(6):
        pg = E.run_model(15, fecg[ch, :N], net15, stride=3840)[0][:N]
        pb = E.run_model(15, ext[ch, :N], net15, stride=3840)[0][:N]
        pl = predict_dense_latent(net15, A, ext[ch, :N], stride=3840)[:N]
        out['gt'].append(_mc_metrics(mc, pg))
        out['base'].append(_mc_metrics(mc, pb))
        out['lat'].append(_mc_metrics(mc, pl))
    return {k: np.array(v).mean(0).tolist() for k, v in out.items()}   # [acc,mF1,F1_0..3]


def heldout_fn(nm):
    fecg = load_sig('signals', nm)
    ext = np.load(os.path.join(INFERRED, nm + '.npy'))
    mc = load_mask('mc_masks', nm).astype(int)
    return fecg, ext, mc


def sem_fn(EXT, stem):
    mat = sio.loadmat(os.path.join(TEST_DIR, stem + '.mat')); o = mat['out']
    mix = o['mixture'][0][0].astype(np.float32); fecg = _extract_fecg(o)
    if fecg.shape[0] < mix.shape[0]:
        fecg = np.repeat(fecg, mix.shape[0] // fecg.shape[0], axis=0)
    mc = o['category_mask'][0][0].ravel().astype(int); del mat
    return fecg, R.infer(VARIANT, EXT, mix), mc


def main():
    probe = 'probe' in sys.argv
    n_windows = 600 if probe else 9000
    epochs = 2 if probe else 12

    net15 = E.load_model(15)
    heldout = set(heldout_signals())
    all_sigs = sorted(os.listdir(os.path.join(NPY, 'mixture')))
    train_names = [s for s in all_sigs
                   if os.path.exists(os.path.join(INFERRED, s + '.npy')) and s not in heldout]
    if probe:
        train_names = train_names[::len(train_names) // 4][:4]
    print(f'{"PROBE " if probe else ""}[{VARIANT}] Train on {len(train_names)} signals (122-9 held-out), '
          f'target {n_windows} windows, extractions from {INFERRED.split("/")[-1]}.', flush=True)

    t = time.time()
    cache = build_cache(train_names, net15, n_windows)
    # class weights (inv-freq, tempered) to counter no-move class dominance
    freq = np.bincount(cache[2].ravel(), minlength=N_CLASS) + 1
    cw = (freq.sum() / freq); cw = (cw / cw.mean()) ** 0.5
    print(f'  class weights {np.round(cw, 2).tolist()}', flush=True)

    A = train(net15, cache, epochs=epochs, class_weight=cw)
    print(f'  training done ({(time.time()-t)/60:.1f}min)', flush=True)
    if probe:
        print('PROBE done (no eval/save).'); return
    torch.save(A.state_dict(), CKPT)

    EXT = R.load_extractor(VARIANT)
    allrows = []
    print('=== EVAL held-out (9) ===', flush=True)
    for nm in sorted(heldout):
        fecg, ext, mc = heldout_fn(nm)
        m = eval_signal(net15, A, fecg, ext, mc); m['set'] = 'HELD-OUT'; m['name'] = nm.split('_SNR')[0]
        allrows.append(m)
        print(f"  {m['name']:<12} mF1 base={m['base'][1]:.3f} lat={m['lat'][1]:.3f} gt={m['gt'][1]:.3f}"
              f" | acc base={m['base'][0]:.3f} lat={m['lat'][0]:.3f} gt={m['gt'][0]:.3f}", flush=True)
    print('=== EVAL Test_DB (Sem1..Sem11) ===', flush=True)
    for stem in SEMS:
        fecg, ext, mc = sem_fn(EXT, stem)
        m = eval_signal(net15, A, fecg, ext, mc); m['set'] = 'TEST_DB'; m['name'] = stem
        allrows.append(m)
        print(f"  {m['name']:<12} mF1 base={m['base'][1]:.3f} lat={m['lat'][1]:.3f} gt={m['gt'][1]:.3f}"
              f" | acc base={m['base'][0]:.3f} lat={m['lat'][0]:.3f} gt={m['gt'][0]:.3f}", flush=True)

    json.dump(allrows, open(OUT, 'w'), indent=2)
    print('\n=== SUMMARY (mean per set) — [acc, macroF1, F1_nomove, F1_lin, F1_spline, F1_helix] ===', flush=True)
    for label in ('HELD-OUT', 'TEST_DB'):
        sub = [r for r in allrows if r['set'] == label]
        for k in ('gt', 'base', 'lat'):
            v = np.mean([r[k] for r in sub], axis=0)
            print(f'  {label:<9} {k:<5} acc={v[0]:.3f} mF1={v[1]:.3f} '
                  f'| perclasa {np.round(v[2:], 3).tolist()}', flush=True)
    print(f'\nGata ({(time.time()-t)/60:.1f}min). json: {OUT}', flush=True)


if __name__ == '__main__':
    main()
