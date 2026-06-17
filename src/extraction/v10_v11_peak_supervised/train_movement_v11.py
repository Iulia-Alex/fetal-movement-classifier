"""
Training script v11 — ComplexUNetV11 (v9 arch, 1.87M) + direct prediction.

Loss = SignalMSE + ALPHA * PeakMSE(fqrs)

Fara ComplexMSE (cauza principala de suprimare a amplitudinilor in v9).
Fara soft mask (direct prediction ca v1/v8 — modelul poate prezice orice amplitudine).

Motivatie:
  v1 (direct + SignalMSE) → amplitudini ~90-95% GT, arhitectura slaba (0.59M)
  v9 (mask + Sig+Cpl)    → amplitudini suprimate, ComplexMSE interfereaza
  v11 = arhitectura v9 (1.87M) + obiectiv simplu (SignalMSE + PeakMSE)
       → sanse bune de amplitudini corecte cu capacitate mai mare
"""

import os, sys, json, time, datetime
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import pathsetup  # noqa

from complex_network_v11 import ComplexUNetV11
from movement_dataset_v10 import MovementECGDatasetV10
from movement_dataset import (
    TARGET_SIZE_F, TARGET_SIZE_T,
    NFFT, HOP_LENGTH, WIN_LENGTH, FS,
)

# ---------------------------------------------------------------------------
DATA_DIR        = '../data/movement_ecg'
MODEL_SAVE_PATH = '../models/movement_CUNet_v11_direct_peak.pth'
HISTORY_PATH    = '../models/movement_CUNet_v11_direct_peak_history.json'
LOG_DIR         = '../logs'

LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
BATCH_SIZE      = 16      # 1.87M model — BS=16 ca v9
MAX_EPOCHS      = 300
PATIENCE        = 20
VAL_SPLIT       = 0.15
NUM_WORKERS     = 2
PRINT_EVERY     = 20
SEED            = 42
ALPHA           = 3.0     # greutatea PeakMSE

IN_CHANNELS     = 6
DIMENSION       = TARGET_SIZE_F * TARGET_SIZE_T
WINDOW_SAMPLES  = 4 * FS
ORIG_F          = NFFT // 2 + 1
ORIG_T          = 1 + WINDOW_SAMPLES // HOP_LENGTH

_DEVICE_HANN: dict = {}


def _get_hann(device):
    if str(device) not in _DEVICE_HANN:
        _DEVICE_HANN[str(device)] = torch.hann_window(WIN_LENGTH, device=device)
    return _DEVICE_HANN[str(device)]


def signal_mse(pred_spec, fecg_time):
    B, C, H, W = pred_spec.shape
    real = F.interpolate(
        pred_spec.real.reshape(B * C, 1, H, W),
        size=(ORIG_F, ORIG_T), mode='bilinear', align_corners=False,
    ).squeeze(1)
    imag = F.interpolate(
        pred_spec.imag.reshape(B * C, 1, H, W),
        size=(ORIG_F, ORIG_T), mode='bilinear', align_corners=False,
    ).squeeze(1)
    spec      = torch.complex(real, imag)
    window    = _get_hann(spec.device)
    pred_time = torch.istft(
        spec, n_fft=NFFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=window, length=WINDOW_SAMPLES,
    ).reshape(B, C, WINDOW_SAMPLES)
    return F.mse_loss(pred_time, fecg_time), pred_time


def peak_mse(pred_time, fecg_time, peak_mask):
    mask = peak_mask.unsqueeze(1)                          # (B, 1, T)
    n    = mask.sum() * pred_time.shape[1] + 1e-8
    return ((pred_time - fecg_time) ** 2 * mask).sum() / n


def composed_loss(pred_spec, fecg_time, peak_mask):
    sig_loss, pred_time = signal_mse(pred_spec, fecg_time)
    pk_loss             = peak_mse(pred_time, fecg_time, peak_mask)
    total               = sig_loss + ALPHA * pk_loss
    return total, sig_loss.item(), pk_loss.item()


# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience):
        self.patience   = patience
        self.counter    = 0
        self.best_loss  = float('inf')
        self.best_epoch = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'    [checkpoint] val_loss={val_loss:.6f} -> saved')
        else:
            self.counter += 1
            print(f'    [early stop] no improvement '
                  f'{self.counter}/{self.patience} '
                  f'(best={self.best_loss:.6f} @ ep {self.best_epoch + 1})')
        return self.counter >= self.patience


_current_epoch = 0


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = total_sig = total_pk = 0.0
    n_batches  = len(loader)
    ctx        = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch_idx, (x, y, y_time, peak_mask) in enumerate(loader):
            x         = x.to(device)
            y_time    = y_time.to(device)
            peak_mask = peak_mask.to(device)

            pred = model(x)
            loss, sig_l, pk_l = composed_loss(pred, y_time, peak_mask)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.clip_weights()

            total_loss += loss.item()
            total_sig  += sig_l
            total_pk   += pk_l

            if (batch_idx + 1) % PRINT_EVERY == 0 or (batch_idx + 1) == n_batches:
                avg = total_loss / (batch_idx + 1)
                tag = 'train' if train else 'val'
                print(f'  [{tag}] ep {_current_epoch}/{MAX_EPOCHS} '
                      f'| batch {batch_idx + 1}/{n_batches} '
                      f'| loss={loss.item():.4f} '
                      f'(sig={sig_l:.4f} pk={pk_l:.4f}) '
                      f'| avg={avg:.4f}')

    n = n_batches
    return total_loss / n, total_sig / n, total_pk / n


def main():
    global _current_epoch

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, 'movement_CUNet_v11_direct_peak.log')
    import sys as _sys
    _sys.stdout = open(log_path, 'w', buffering=1)
    _sys.stderr = _sys.stdout

    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Start : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'Alpha (peak loss weight): {ALPHA}')

    dataset = MovementECGDatasetV10(DATA_DIR)
    n_val   = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f'Train: {n_train}  Val: {n_val}')

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model  = ComplexUNetV11(DIMENSION, in_channels=IN_CHANNELS).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params / 1e6:.3f} M')

    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True,
    )
    early_stop = EarlyStopping(patience=PATIENCE)

    history = {'train_loss': [], 'val_loss': [],
               'train_sig': [], 'train_pk': [],
               'val_sig':   [], 'val_pk':   [], 'lr': []}

    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        _current_epoch = epoch + 1
        t_ep = time.time()

        lr_now = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch + 1}/{MAX_EPOCHS}  |  lr={lr_now:.2e}')
        print('-' * 60)

        tr_loss, tr_sig, tr_pk = run_epoch(
            model, train_loader, optimizer, device, train=True)
        vl_loss, vl_sig, vl_pk = run_epoch(
            model, val_loader,   optimizer, device, train=False)

        elapsed = time.time() - t0
        ep_time = time.time() - t_ep
        print(f'\n  >> Epoch {epoch + 1}: '
              f'train={tr_loss:.4f} (sig={tr_sig:.4f} pk={tr_pk:.4f})  '
              f'val={vl_loss:.4f} (sig={vl_sig:.4f} pk={vl_pk:.4f})  '
              f't={ep_time:.1f}s  total={elapsed/60:.1f}min')

        for k, v in [('train_loss', tr_loss), ('val_loss', vl_loss),
                     ('train_sig', tr_sig),   ('train_pk', tr_pk),
                     ('val_sig',   vl_sig),   ('val_pk',   vl_pk),
                     ('lr', lr_now)]:
            history[k].append(v)
        json.dump(history, open(HISTORY_PATH, 'w'), indent=2)

        scheduler.step(vl_loss)

        if early_stop.step(vl_loss, model, epoch):
            print(f'\nEarly stopping after {epoch + 1} epochs.')
            break

    print(f'\n{"=" * 70}')
    print(f'Training finished: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'Best val_loss    : {early_stop.best_loss:.6f} at epoch {early_stop.best_epoch + 1}')
    print(f'Model saved to   : {MODEL_SAVE_PATH}')
    print('=' * 70)


if __name__ == '__main__':
    main()
