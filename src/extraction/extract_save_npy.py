"""
Extrage v1 si v17 pe toate cele 122 semnale Test_DB (toate 6 canalele) si
salveaza cate un .npy per semnal, forma (6, N), in doua foldere separate:
  inferred_v1/<semnal>.npy   si   inferred_v17/<semnal>.npy
Resumabil (sare peste ce exista). Paralel moderat (nu sufoca jobul mare).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pathsetup  # noqa
import numpy as np
import multiprocessing as mp

from config import NPY, FECG_ROOT as OUT_BASE
MODELS = ['v1', 'v17']
NPROC = 6


def out_dir(v):
    return os.path.join(OUT_BASE, f'inferred_{v}')


def load_mixture(name):
    d = os.path.join(NPY, 'mixture', name)
    return np.stack([np.load(os.path.join(d, [x for x in os.listdir(d) if f'_ch{c}.npy' in x][0])).astype(np.float32)
                     for c in range(1, 7)])


_EXT = {}


def _get(v):
    if v not in _EXT:
        import pipeline_registry as R
        _EXT[v] = R.load_extractor(v)
    return _EXT[v]


def worker(task):
    v, name = task
    import torch
    torch.set_num_threads(1)
    import pipeline_registry as R
    path = os.path.join(out_dir(v), f'{name}.npy')
    if os.path.exists(path):
        return (v, name, 'skip')
    try:
        ext = R.infer(v, _get(v), load_mixture(name)).astype(np.float32)
        np.save(path, ext)
        return (v, name, f'ok {ext.shape}')
    except Exception as e:
        return (v, name, f'EROARE: {e}')


def main():
    for v in MODELS:
        os.makedirs(out_dir(v), exist_ok=True)
    signals = sorted(os.listdir(os.path.join(NPY, 'mixture')))
    tasks = [(v, s) for v in MODELS for s in signals
             if not os.path.exists(os.path.join(out_dir(v), f'{s}.npy'))]
    tasks.sort()
    total = len(MODELS) * len(signals)
    print(f'Total {total} (model x signal) | to run: {len(tasks)} | NPROC={NPROC}', flush=True)
    print(f'Folders: {out_dir("v1")}  and  {out_dir("v17")}', flush=True)
    t0 = time.time(); n = 0; err = 0
    with mp.Pool(NPROC) as pool:
        for v, name, msg in pool.imap_unordered(worker, tasks, chunksize=2):
            n += 1
            if 'EROARE' in msg:
                err += 1; print(f'  [{v}] {name[:40]}: {msg}', flush=True)
            if n % 20 == 0 or n == len(tasks):
                el = time.time() - t0
                print(f'  [{n}/{len(tasks)}] {el/60:.0f}min, ETA ~{el/n*(len(tasks)-n)/60:.0f}min', flush=True)
    # raport
    for v in MODELS:
        cnt = len([f for f in os.listdir(out_dir(v)) if f.endswith('.npy')])
        print(f'  inferred_{v}: {cnt} fisiere .npy', flush=True)
    print(f'TERMINAT. {n} taskuri, {err} erori, {(time.time()-t0)/60:.0f}min.', flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
