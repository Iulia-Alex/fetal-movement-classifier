"""
BINARY LATENT ADAPTER (movement yes/no) — the direct variant on the robust task.
Same principle as latent_adapter.py (correction at the bottleneck, LABEL-ONLY, through
the FROZEN decoder) but on the BINARY classifier M14 (1-channel out, BCE), trained on
MORE data: 113 signals (Final_Test_DB 122 minus the 9 held-out), using the v1 extractions
already cached (inferred_v1/) — no re-inference.

Motivation (see ch.4 §4.3): the extraction injects spurious variation in the still
intervals -> over-detection. The binary task fails precisely because of this, so a binary
adapter that suppresses false positives attacks exactly what broke. Real headroom: binary-F1
base ~0.35-0.56, ceiling ~0.65.

Eval on TWO disjoint sets: held-out (9) + Test_DB Sem1..Sem11.
Reports F1 + PRECISION + RECALL (movement class) to check that the gain comes from
precision (fewer false positives), NOT from accuracy on no-move.
Probe: `python3 latent_adapter_binary.py probe`.
"""
import sys, os, glob, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import numpy as np
import torch, torch.nn as nn
os.environ.setdefault('LA_THREADS', '8')
torch.set_num_threads(int(os.environ['LA_THREADS']))
import scipy.io as sio
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from diag_info_ceiling import load_sig, load_mask
from build_adapter import snr_cat
import pipeline_registry as R
import pretrained_clf as E
from latent_adapter import encode, decode, LatentAdapter, heldout_signals
from movement_dataset import _extract_fecg

from config import NPY, FECG_ROOT as ROOT, TEST_DIR, RESULTS_DIR as RES
WIN = E.WIN
SEMS = [f'Sem{i}' for i in range(1, 12)]

# extraction variant (source fECG data used to train/eval the binary adapter)
VARIANT = next((a for a in sys.argv[1:] if a.startswith('v') and a[1:].isdigit()), 'v1')
INFERRED = os.path.join(ROOT, f'inferred_{VARIANT}')
CKPT = os.path.join(RES, f'adapter_latent_binary_m14_{VARIANT}.pt')
OUT = os.path.join(RES, f'adapter_latent_binary_{VARIANT}.json')


# ── cache (b_ext, skips, binary label) from extractions already in inferred_v1 ─
def build_cache(names, net, n_windows, seed=0):
    rng = np.random.default_rng(seed)
    per_ch = max(1, n_windows // (len(names) * 6))
    B, S, Y = [], [[], [], [], []], []
    t0 = time.time()
    for fi, nm in enumerate(names):
        ext = np.load(os.path.join(INFERRED, nm + '.npy'))        # v1 extracted (cache)
        mb = (load_mask('mc_masks', nm).astype(int) > 0).astype(np.int8)   # binary
        N = min(ext.shape[1], len(mb)); max_start = N - WIN
        for ch in range(6):
            starts = rng.integers(0, max_start, size=per_ch)
            feats = np.stack([E.features_5(ext[ch, s0:s0 + WIN]) for s0 in starts])
            with torch.no_grad():
                b, skips = encode(net, torch.from_numpy(feats))
            B.append(b.half().numpy())
            for j in range(4):
                S[j].append(skips[j].half().numpy())
            Y.append(np.stack([mb[s0:s0 + WIN] for s0 in starts]))
        if (fi + 1) % 10 == 0:
            print(f'  cache [{fi+1}/{len(names)}] {(time.time()-t0)/60:.1f}min', flush=True)
    B = np.concatenate(B); S = [np.concatenate(s) for s in S]; Y = np.concatenate(Y)
    print(f'  cache: {len(B)} windows, ~{(B.nbytes+sum(s.nbytes for s in S))/1e9:.1f}GB '
          f'({(time.time()-t0)/60:.1f}min); positive(movement)={Y.mean():.3f}', flush=True)
    return B, S, Y


def train(net, cache, epochs=12, bs=32, lr=1e-3, pos_weight=3.0):
    B, S, Y = cache; n = len(B); A = LatentAdapter()
    opt = torch.optim.Adam(A.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    for p in net.parameters():
        p.requires_grad_(False)
    for ep in range(epochs):
        perm = np.random.permutation(n); tot = 0.0; t0 = time.time()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            b = torch.from_numpy(B[idx].astype(np.float32))
            skips = tuple(torch.from_numpy(S[j][idx].astype(np.float32)) for j in range(4))
            y = torch.from_numpy(Y[idx].astype(np.float32))           # (bs,WIN)
            opt.zero_grad()
            logits = decode(net, A(b), skips).squeeze(1)              # (bs,WIN)
            loss = lossf(logits, y)
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f'  ep{ep+1}/{epochs} loss={tot/n:.4f} ({time.time()-t0:.0f}s)', flush=True)
    return A.eval()


def predict_dense_binary(net, A, sig1d, stride=3840, batch=32, return_prob=False):
    N = len(sig1d); L = max(N, WIN)
    starts = list(range(0, L - WIN + 1, stride))
    if not starts or starts[-1] != L - WIN:
        starts.append(L - WIN)
    psum = np.zeros(L); cnt = np.zeros(L); buf, bs = [], []

    def flush():
        nonlocal buf, bs
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf))
        with torch.no_grad():
            b, skips = encode(net, x)
            prob = torch.sigmoid(decode(net, A(b), skips).squeeze(1)).numpy()  # (k,WIN)
        for bi, s0 in enumerate(bs):
            psum[s0:s0 + WIN] += prob[bi]; cnt[s0:s0 + WIN] += 1
        buf, bs = [], []

    for s0 in starts:
        buf.append(E.features_5(sig1d[s0:s0 + WIN])); bs.append(s0)
        if len(buf) >= batch:
            flush()
    flush()
    cnt[cnt == 0] = 1
    prob = (psum / cnt)[:N]
    return prob if return_prob else (prob >= 0.5).astype(int)


def _bin_metrics(mb, pred):
    return (accuracy_score(mb, pred),
            f1_score(mb, pred, zero_division=0),
            precision_score(mb, pred, zero_division=0),
            recall_score(mb, pred, zero_division=0))


def eval_signal(net14, A, fecg, ext, mb):
    """Per-channel: base(M14 on extracted), lat(M14+A on extracted), gt(M14 on clean)."""
    N = min(fecg.shape[1], ext.shape[1], len(mb)); mb = mb[:N]
    out = {k: [] for k in ('gt', 'base', 'lat')}
    for ch in range(6):
        pg = E.run_model(14, fecg[ch, :N], net14, stride=3840)[0][:N]
        pb = E.run_model(14, ext[ch, :N], net14, stride=3840)[0][:N]
        pl = predict_dense_binary(net14, A, ext[ch, :N], stride=3840)[:N]
        out['gt'].append(_bin_metrics(mb, pg))
        out['base'].append(_bin_metrics(mb, pb))
        out['lat'].append(_bin_metrics(mb, pl))
    return {k: np.array(v).mean(0).tolist() for k, v in out.items()}   # [acc,f1,prec,rec]


def heldout_fn(nm):
    fecg = load_sig('signals', nm)
    ext = np.load(os.path.join(INFERRED, nm + '.npy'))
    mb = (load_mask('mc_masks', nm).astype(int) > 0).astype(int)
    return fecg, ext, mb


def sem_fn(EXT, stem):
    mat = sio.loadmat(os.path.join(TEST_DIR, stem + '.mat')); o = mat['out']
    mix = o['mixture'][0][0].astype(np.float32); fecg = _extract_fecg(o)
    if fecg.shape[0] < mix.shape[0]:
        fecg = np.repeat(fecg, mix.shape[0] // fecg.shape[0], axis=0)
    mb = (o['category_mask'][0][0].ravel().astype(int) > 0).astype(int); del mat
    return fecg, R.infer(VARIANT, EXT, mix), mb


def main():
    probe = 'probe' in sys.argv
    n_windows = 600 if probe else 9000
    epochs = 2 if probe else 12

    net14 = E.load_model(14)
    heldout = set(heldout_signals())
    all_sigs = sorted(os.listdir(os.path.join(NPY, 'mixture')))
    train_names = [s for s in all_sigs
                   if os.path.exists(os.path.join(INFERRED, s + '.npy')) and s not in heldout]
    if probe:
        train_names = train_names[::len(train_names) // 4][:4]
    print(f'{"PROBE " if probe else ""}[{VARIANT}] Train on {len(train_names)} signals (122-9 held-out), '
          f'target {n_windows} windows, extractions from {INFERRED.split("/")[-1]}.', flush=True)

    t = time.time()
    cache = build_cache(train_names, net14, n_windows)
    A = train(net14, cache, epochs=epochs)
    print(f'  training done ({(time.time()-t)/60:.1f}min)', flush=True)
    if probe:
        print('PROBE done (no eval/save).'); return
    torch.save(A.state_dict(), CKPT)

    EXT = R.load_extractor(VARIANT)
    allrows = []
    print('=== EVAL held-out (9) ===', flush=True)
    for nm in sorted(heldout):
        fecg, ext, mb = heldout_fn(nm)
        m = eval_signal(net14, A, fecg, ext, mb); m['set'] = 'HELD-OUT'; m['name'] = nm.split('_SNR')[0]
        allrows.append(m)
        print(f"  {m['name']:<12} binF1 base={m['base'][1]:.3f} lat={m['lat'][1]:.3f} gt={m['gt'][1]:.3f}"
              f" | prec base={m['base'][2]:.3f} lat={m['lat'][2]:.3f} | rec base={m['base'][3]:.3f} lat={m['lat'][3]:.3f}", flush=True)
    print('=== EVAL Test_DB (Sem1..Sem11) ===', flush=True)
    for stem in SEMS:
        fecg, ext, mb = sem_fn(EXT, stem)
        m = eval_signal(net14, A, fecg, ext, mb); m['set'] = 'TEST_DB'; m['name'] = stem
        allrows.append(m)
        print(f"  {m['name']:<12} binF1 base={m['base'][1]:.3f} lat={m['lat'][1]:.3f} gt={m['gt'][1]:.3f}"
              f" | prec base={m['base'][2]:.3f} lat={m['lat'][2]:.3f} | rec base={m['base'][3]:.3f} lat={m['lat'][3]:.3f}", flush=True)

    json.dump(allrows, open(OUT, 'w'), indent=2)
    print('\n=== SUMMARY (mean per set) — [acc, F1, precision, recall] movement class ===', flush=True)
    for label in ('HELD-OUT', 'TEST_DB'):
        sub = [r for r in allrows if r['set'] == label]
        for k in ('gt', 'base', 'lat'):
            a, f, p, rc = np.mean([r[k] for r in sub], axis=0)
            print(f'  {label:<9} {k:<5} acc={a:.3f} F1={f:.3f} prec={p:.3f} rec={rc:.3f}', flush=True)
    print(f'\nDone ({(time.time()-t)/60:.1f}min). json: {OUT}', flush=True)


if __name__ == '__main__':
    main()
