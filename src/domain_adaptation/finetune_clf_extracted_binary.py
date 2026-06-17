"""
FINE-TUNE BINARY classifier on the EXTRACTED DOMAIN (main domain-adaptation lever).
The dense classifier (AttUNet1D, M14) was trained on CLEAN signal, but we run it on
EXTRACTED signal -> train/test mismatch = over-detection. Here we adapt the classifier
TO the extracted domain: start from M14 weights and re-train on extracted signals
(inferred_v1) with binary masks, on the same 113 signals (122 - 9 held-out).

Unlike the latent adapter (small correction, label-only, frozen decoder),
here we train the ENTIRE network -> full domain-adaptation capacity. Creates a
NEW classifier (to be presented in the thesis as domain adaptation, not the fixed model).

Eval on held-out (9) + Test_DB Sem1..11, THRESHOLD-INDEPENDENT metrics (AP + best-F1):
  base = M14 (clean) on extracted | ft = M14 fine-tuned on extracted | gt = M14 on clean (ceiling)
Probe: `python3 finetune_clf_extracted_binary.py probe`.
"""
import sys, os, time, json, copy
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
from diag_info_ceiling import load_mask
from eval_adapter_binary_prauc import ap, best_f1

WIN = E.WIN
CKPT = os.path.join(RES, 'clf_finetuned_extracted_binary.pt')
OUT = os.path.join(RES, 'clf_finetuned_extracted_binary.json')


# ── cache: features_5(extracted) + binary label, from inferred_v1 ─────────────
def build_cache(names, n_windows, seed=0):
    rng = np.random.default_rng(seed)
    per_ch = max(1, n_windows // (len(names) * 6))
    X, Y = [], []
    t0 = time.time()
    for fi, nm in enumerate(names):
        ext = np.load(os.path.join(INFERRED, nm + '.npy'))
        mb = (load_mask('mc_masks', nm).astype(int) > 0).astype(np.int8)
        N = min(ext.shape[1], len(mb)); max_start = N - WIN
        for ch in range(6):
            for s0 in rng.integers(0, max_start, size=per_ch):
                X.append(E.features_5(ext[ch, s0:s0 + WIN]).astype(np.float16))
                Y.append(mb[s0:s0 + WIN])
        if (fi + 1) % 20 == 0:
            print(f'  cache [{fi+1}/{len(names)}] {(time.time()-t0)/60:.1f}min', flush=True)
    X = np.stack(X); Y = np.stack(Y)
    print(f'  cache: {len(X)} windows, ~{X.nbytes/1e9:.1f}GB ({(time.time()-t0)/60:.1f}min); '
          f'positive(movement)={Y.mean():.3f}', flush=True)
    return X, Y


def train(cache, epochs=10, bs=32, lr=3e-4, pos_weight=3.0, init_ckpt=None):
    X, Y = cache; n = len(X)
    if init_ckpt:                                # warm-start de la un checkpoint existent
        net = E.AttUNet1D(5, 1, False)
        net.load_state_dict(torch.load(init_ckpt, map_location='cpu'))
        print(f'  resume from {os.path.basename(init_ckpt)} (lr={lr})', flush=True)
    else:
        net = E.load_model(14)                   # init from M14 (fine-tune, not from scratch)
    net.train()
    for p in net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    for ep in range(epochs):
        perm = np.random.permutation(n); tot = 0.0; t0 = time.time()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x = torch.from_numpy(X[idx].astype(np.float32))          # (bs,5,WIN)
            y = torch.from_numpy(Y[idx].astype(np.float32))          # (bs,WIN)
            opt.zero_grad()
            logits = net(x).squeeze(1)                               # (bs,WIN)
            loss = lossf(logits, y)
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f'  ep{ep+1}/{epochs} loss={tot/n:.4f} ({time.time()-t0:.0f}s)', flush=True)
    return net.eval()


def eval_signal(net_base, net_ft, fecg, ext, mb):
    """Per-channel AP + best-F1: base(M14 clean on extracted)/ft(M14 fine-tuned on extracted)/gt(M14 on clean)."""
    N = min(fecg.shape[1], ext.shape[1], len(mb)); mb = mb[:N]
    out = {k: {'ap': [], 'bf1': []} for k in ('gt', 'base', 'ft')}
    for ch in range(6):
        pg = E.run_model(14, fecg[ch, :N], net_base, stride=3840)[1][:N]
        pb = E.run_model(14, ext[ch, :N],  net_base, stride=3840)[1][:N]
        pf = E.run_model(14, ext[ch, :N],  net_ft,   stride=3840)[1][:N]
        for k, pr in (('gt', pg), ('base', pb), ('ft', pf)):
            out[k]['ap'].append(ap(mb, pr)); out[k]['bf1'].append(best_f1(mb, pr))
    return {k: {'ap': float(np.nanmean(v['ap'])), 'bf1': float(np.nanmean(v['bf1']))}
            for k, v in out.items()}


def main():
    probe = 'probe' in sys.argv
    resume = 'resume' in sys.argv                 # continue from ep10 checkpoint
    tag = os.environ.get('TAG', '')               # suffix for alternative runs (e.g. morewin)
    n_windows = 600 if probe else int(os.environ.get('NW', '9000'))
    epochs = 2 if probe else int(os.environ.get('EPOCHS', '10'))
    if resume:
        out_ck = os.path.join(RES, 'clf_finetuned_extracted_binary_more.pt')
        out_js = os.path.join(RES, 'clf_finetuned_extracted_binary_more.json')
        init_ck = CKPT; lr = 2e-4                  # gentler refinement on resume
    else:
        suf = f'_{tag}' if tag else ''
        out_ck = os.path.join(RES, f'clf_finetuned_extracted_binary{suf}.pt')
        out_js = os.path.join(RES, f'clf_finetuned_extracted_binary{suf}.json')
        init_ck = None; lr = 3e-4

    heldout = set(heldout_signals())
    all_sigs = sorted(os.listdir(os.path.join(NPY, 'mixture')))
    train_names = [s for s in all_sigs
                   if os.path.exists(os.path.join(INFERRED, s + '.npy')) and s not in heldout]
    if probe:
        train_names = train_names[::len(train_names) // 4][:4]
    print(f'{"PROBE " if probe else ""}{"RESUME " if resume else ""}Binary fine-tune on extracted: '
          f'{len(train_names)} signals, {n_windows} windows, {epochs} epochs.', flush=True)

    t = time.time()
    cache = build_cache(train_names, n_windows)
    net_ft = train(cache, epochs=epochs, lr=lr, init_ckpt=init_ck)
    print(f'  training done ({(time.time()-t)/60:.1f}min)', flush=True)
    if probe:
        print('PROBE done (no eval/save).'); return
    torch.save(net_ft.state_dict(), out_ck)

    net_base = E.load_model(14)
    EXT = R.load_extractor('v1')
    allrows = []
    print('=== EVAL held-out (9) ===', flush=True)
    for nm in sorted(heldout):
        fecg, ext, mb = heldout_fn(nm)
        m = eval_signal(net_base, net_ft, fecg, ext, mb); m['set'] = 'HELD-OUT'; m['name'] = nm.split('_SNR')[0]
        allrows.append(m)
        print(f"  {m['name']:<12} AP base={m['base']['ap']:.3f} ft={m['ft']['ap']:.3f} gt={m['gt']['ap']:.3f}"
              f" | bestF1 base={m['base']['bf1']:.3f} ft={m['ft']['bf1']:.3f} gt={m['gt']['bf1']:.3f}", flush=True)
    print('=== EVAL Test_DB (Sem1..Sem11) ===', flush=True)
    for stem in SEMS:
        fecg, ext, mb = sem_fn(EXT, stem)
        m = eval_signal(net_base, net_ft, fecg, ext, mb); m['set'] = 'TEST_DB'; m['name'] = stem
        allrows.append(m)
        print(f"  {m['name']:<12} AP base={m['base']['ap']:.3f} ft={m['ft']['ap']:.3f} gt={m['gt']['ap']:.3f}"
              f" | bestF1 base={m['base']['bf1']:.3f} ft={m['ft']['bf1']:.3f} gt={m['gt']['bf1']:.3f}", flush=True)

    json.dump(allrows, open(out_js, 'w'), indent=2)
    print('\n=== SUMMARY (mean per set) — AP and best-F1 (threshold-independent) ===', flush=True)
    for label in ('HELD-OUT', 'TEST_DB'):
        sub = [r for r in allrows if r['set'] == label]
        for k in ('gt', 'base', 'ft'):
            mAP = np.mean([r[k]['ap'] for r in sub]); mBF = np.mean([r[k]['bf1'] for r in sub])
            print(f'  {label:<9} {k:<5} AP={mAP:.3f}  bestF1={mBF:.3f}', flush=True)
    print(f'\nDone ({(time.time()-t)/60:.1f}min). json: {out_js}', flush=True)


if __name__ == '__main__':
    main()
