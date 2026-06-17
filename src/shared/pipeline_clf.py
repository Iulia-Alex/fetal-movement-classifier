"""
Dense 1D classification using all three pre-trained classifiers, sharing features:
  - features_5 (per-window) computed ONCE -> run through M14 and M15.
  - features_3_global computed once -> M18.
Overlap-add across windows. Returns dense predictions (length = len(sig)).
"""
import numpy as np
import torch
import pretrained_clf as E

WIN = E.WIN          # 3840
DEFAULT_STRIDE = 1024


def _starts(L, win, stride):
    if L <= win:
        return [0]
    s = list(range(0, L - win + 1, stride))
    if s[-1] != L - win:
        s.append(L - win)
    return s


def _dense(model, feat_windows, starts, L, win, n_out, binary, batch=64, device='cpu'):
    psum = np.zeros((max(n_out, 1), max(L, win)), dtype=np.float64)
    cnt = np.zeros(max(L, win), dtype=np.float64)
    for i in range(0, len(feat_windows), batch):
        xb = torch.from_numpy(feat_windows[i:i+batch]).to(device)
        with torch.no_grad():
            o = model(xb)
            p = (torch.sigmoid(o) if binary else torch.softmax(o, dim=1)).cpu().numpy()
        for bi in range(p.shape[0]):
            s0 = starts[i + bi]
            psum[:, s0:s0+win] += p[bi]
            cnt[s0:s0+win] += 1
    cnt[cnt == 0] = 1
    prob = psum / cnt
    if binary:
        return (prob[0] > 0.5).astype(int)[:L]
    return np.argmax(prob, axis=0)[:L]


def classify_channel(sig1d, clf, stride=DEFAULT_STRIDE, device='cpu'):
    """clf = {14:model,15:model,18:model}. Returns (p14,p15,p18) dense arrays."""
    L = len(sig1d)
    win = WIN
    starts = _starts(L, win, stride)

    # features_5 (per-window) -> shared M14 + M15
    f5 = np.stack([E.features_5(sig1d[s0:s0+win]) for s0 in starts]).astype(np.float32)
    p14 = _dense(clf[14], f5, starts, L, win, 1, True,  device=device)
    p15 = _dense(clf[15], f5, starts, L, win, 4, False, device=device)

    # features_3 at signal level -> M18
    f3 = E.features_3_global(sig1d)
    f3w = np.stack([f3[:, s0:s0+win] for s0 in starts]).astype(np.float32)
    p18 = _dense(clf[18], f3w, starts, L, win, 1, True, device=device)

    return p14, p15, p18
