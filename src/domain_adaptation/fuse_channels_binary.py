"""
ITERATION 3 — CHANNEL FUSION (binary). The fetus projects differently onto the 6
abdominal leads; the fine-tuned classifier (on extracted) runs per-channel, but
the current pipeline AVERAGES AP across channels. Here we combine the 6 streams into
a single per-sample decision -> should beat the mean (headroom mean->best-channel:
clean 0.65 mean vs 0.74 best-channel).

Small fusion head (Conv1d 6->16->1) on top of the FROZEN fine-tuned classifier,
trained on labels (113 signals). Eval on held-out(9)+Sem(11), AP + best-F1:
  per-chan mean (current pipeline) | fused mean/max (untrained) | fused learned |
  ceiling: clean classifier on clean, fused.
Probe: `python3 fuse_channels_binary.py probe`.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import numpy as np
import torch, torch.nn as nn
os.environ.setdefault('LA_THREADS', '8')
torch.set_num_threads(int(os.environ['LA_THREADS']))

import pretrained_clf as E
import pipeline_registry as R
from latent_adapter_binary import heldout_fn, sem_fn, SEMS, RES, INFERRED, NPY
from latent_adapter import heldout_signals
from eval_adapter_binary_prauc import ap, best_f1

WIN = E.WIN
FT_CKPT = os.path.join(RES, 'clf_finetuned_extracted_binary.pt')   # iteration 2 (ep10)
CKPT = os.path.join(RES, 'fusion_head_binary.pt')
OUT = os.path.join(RES, 'fuse_channels_binary.json')
N_CAP = int(os.environ.get('N_CAP', '200000'))   # cap lungime/semnal la build (eval = full)


def load_ft():
    net = E.AttUNet1D(5, 1, False)
    net.load_state_dict(torch.load(FT_CKPT, map_location='cpu'))
    return net.eval()


def ch_probs(net, sig6, N):
    """6 per-sample probability streams (binary classifier run on each channel)."""
    return np.stack([E.run_model(14, sig6[ch, :N], net, stride=3840)[1][:N] for ch in range(6)])


class FusionHead(nn.Module):
    def __init__(self, h=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6, h, 15, padding=7), nn.GroupNorm(4, h), nn.ReLU(),
            nn.Conv1d(h, 1, 1))

    def forward(self, p):                      # p: (B,6,L) probabilitati -> (B,L) logit
        return self.net(p).squeeze(1)


def build_cache(names, net_ft, n_windows, seed=0):
    rng = np.random.default_rng(seed)
    per_sig = max(1, n_windows // len(names))
    P, Y = [], []
    t0 = time.time()
    for fi, nm in enumerate(names):
        ext = np.load(os.path.join(INFERRED, nm + '.npy'))
        from diag_info_ceiling import load_mask
        mb = (load_mask('mc_masks', nm).astype(int) > 0).astype(np.int8)
        N = min(ext.shape[1], len(mb))
        if N_CAP:
            N = min(N, N_CAP)
        probs = ch_probs(net_ft, ext, N)                     # (6,N) fluxuri ft pe extras
        max_start = N - WIN
        for s0 in rng.integers(0, max_start, size=per_sig):
            P.append(probs[:, s0:s0 + WIN].astype(np.float16))
            Y.append(mb[s0:s0 + WIN])
        if (fi + 1) % 20 == 0:
            print(f'  cache [{fi+1}/{len(names)}] {(time.time()-t0)/60:.1f}min', flush=True)
    P = np.stack(P); Y = np.stack(Y)
    print(f'  cache: {len(P)} windows, ~{P.nbytes/1e9:.1f}GB ({(time.time()-t0)/60:.1f}min); '
          f'positive={Y.mean():.3f}', flush=True)
    return P, Y


def train(cache, epochs=20, bs=64, lr=1e-3, pos_weight=3.0):
    P, Y = cache; n = len(P); head = FusionHead()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    for ep in range(epochs):
        perm = np.random.permutation(n); tot = 0.0; t0 = time.time()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            p = torch.from_numpy(P[idx].astype(np.float32))
            y = torch.from_numpy(Y[idx].astype(np.float32))
            opt.zero_grad()
            loss = lossf(head(p), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'  ep{ep+1}/{epochs} loss={tot/n:.4f} ({time.time()-t0:.0f}s)', flush=True)
    return head.eval()


def fused_learned(head, probs6):
    with torch.no_grad():
        logit = head(torch.from_numpy(probs6.astype(np.float32))[None])[0].numpy()
    return 1 / (1 + np.exp(-logit))


def eval_signal(net_ft, net_base, head, fecg, ext, mb):
    N = min(fecg.shape[1], ext.shape[1], len(mb)); mb = mb[:N]
    pe = ch_probs(net_ft, ext, N)                 # ft pe extras, 6 canale
    pc = ch_probs(net_base, fecg, N)              # clasificator curat pe curat, 6 canale (plafon)
    out = {
        'ft_perchan': [np.mean([ap(mb, pe[c]) for c in range(6)]),
                       np.mean([best_f1(mb, pe[c]) for c in range(6)])],
        'ft_mean':  [ap(mb, pe.mean(0)),       best_f1(mb, pe.mean(0))],
        'ft_max':   [ap(mb, pe.max(0)),        best_f1(mb, pe.max(0))],
        'ft_learn': [ap(mb, fused_learned(head, pe)), best_f1(mb, fused_learned(head, pe))],
        'ceil_perchan': [np.mean([ap(mb, pc[c]) for c in range(6)]),
                         np.mean([best_f1(mb, pc[c]) for c in range(6)])],
        'ceil_mean': [ap(mb, pc.mean(0)),      best_f1(mb, pc.mean(0))],
    }
    return out


def main():
    probe = 'probe' in sys.argv
    n_windows = 600 if probe else 6000
    epochs = 3 if probe else 20

    net_ft = load_ft(); net_base = E.load_model(14)
    heldout = set(heldout_signals())
    all_sigs = sorted(os.listdir(os.path.join(NPY, 'mixture')))
    train_names = [s for s in all_sigs
                   if os.path.exists(os.path.join(INFERRED, s + '.npy')) and s not in heldout]
    if probe:
        train_names = train_names[::len(train_names) // 4][:4]
    print(f'{"PROBE " if probe else ""}Channel fusion binary: {len(train_names)} signals, '
          f'{n_windows} windows (cap {N_CAP}).', flush=True)

    t = time.time()
    cache = build_cache(train_names, net_ft, n_windows)
    head = train(cache, epochs=epochs)
    print(f'  training done ({(time.time()-t)/60:.1f}min)', flush=True)
    if probe:
        print('PROBE done (no eval/save).'); return
    torch.save(head.state_dict(), CKPT)

    EXT = R.load_extractor('v1')
    rows = []
    print('=== EVAL held-out (9) ===', flush=True)
    for nm in sorted(heldout):
        fecg, ext, mb = heldout_fn(nm)
        m = eval_signal(net_ft, net_base, head, fecg, ext, mb); m['set'] = 'HELD-OUT'; m['name'] = nm.split('_SNR')[0]
        rows.append(m)
        print(f"  {m['name']:<12} AP perchan={m['ft_perchan'][0]:.3f} mean={m['ft_mean'][0]:.3f} "
              f"max={m['ft_max'][0]:.3f} learn={m['ft_learn'][0]:.3f} | ceil mean={m['ceil_mean'][0]:.3f}", flush=True)
    print('=== EVAL Test_DB (Sem1..Sem11) ===', flush=True)
    for stem in SEMS:
        fecg, ext, mb = sem_fn(EXT, stem)
        m = eval_signal(net_ft, net_base, head, fecg, ext, mb); m['set'] = 'TEST_DB'; m['name'] = stem
        rows.append(m)
        print(f"  {m['name']:<12} AP perchan={m['ft_perchan'][0]:.3f} mean={m['ft_mean'][0]:.3f} "
              f"max={m['ft_max'][0]:.3f} learn={m['ft_learn'][0]:.3f} | ceil mean={m['ceil_mean'][0]:.3f}", flush=True)

    json.dump(rows, open(OUT, 'w'), indent=2)
    print('\n=== SUMMARY (mean per set) — AP [and best-F1] ===', flush=True)
    for label in ('HELD-OUT', 'TEST_DB'):
        sub = [r for r in rows if r['set'] == label]
        for k in ('ft_perchan', 'ft_mean', 'ft_max', 'ft_learn', 'ceil_perchan', 'ceil_mean'):
            a = np.mean([r[k][0] for r in sub]); f = np.mean([r[k][1] for r in sub])
            print(f'  {label:<9} {k:<12} AP={a:.3f}  bestF1={f:.3f}', flush=True)
    print(f'\nDone ({(time.time()-t)/60:.1f}min). json: {OUT}', flush=True)


if __name__ == '__main__':
    main()
