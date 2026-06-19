"""
Training script v12 — ComplexUNetV12 (v9 arch, 1.87M) + gain mask (1.5×sigmoid).

Loss = SignalMSE + ALPHA * QRSwideMSE
  - QRSwideMSE: peak_mask ±100 samples (vs ±30 in v11) → covers PQRS
  - No ComplexMSE (the main cause of amplitude suppression in v9)
  - Gain mask 1.5×sigmoid → can exceed the mixture magnitude at antiphase bins

Motivation:
  v9  (mask 1×sig + Sig+Cpl)    → suppressed amplitudes (ComplexMSE 20× larger)
  v11 (direct + Sig+Peak ±30)   → rising amplitudes, no hard ceiling
  v12 (mask 1.5×sig + Sig+Peak ±100) → energy recovery + full PQRS focus
"""

import os, sys, json, time, datetime
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import pathsetup  # noqa

from complex_network_v12 import ComplexUNetV12
from movement_dataset_v12 import MovementECGDatasetV12
from movement_dataset import (
    TARGET_SIZE_F, TARGET_SIZE_T,
    NFFT, HOP_LENGTH, WIN_LENGTH, FS,
)

# ---------------------------------------------------------------------------
DATA_DIR        = '../data/movement_ecg'
MODEL_SAVE_PATH = '../models/movement_CUNet_v12_gainmask.pth'
HISTORY_PATH    = '../models/movement_CUNet_v12_gainmask_history.json'
CKPT_PATH       = '../models/movement_CUNet_v12_gainmask_ckpt.pth'
LOG_DIR         = '../logs'

LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
BATCH_SIZE      = 16
MAX_EPOCHS      = 300
PATIENCE        = 20
VAL_SPLIT       = 0.15
NUM_WORKERS     = 2
PRINT_EVERY     = 20
SEED            = 42
ALPHA           = 3.0     # greutatea QRSwideMSE

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
    """QRSwideMSE — peak_mask ±100 samples din dataset."""
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
    log_path = os.path.join(LOG_DIR, 'movement_CUNet_v12_gainmask.log')
    import sys as _sys
    _sys.stdout = open(log_path, 'w', buffering=1)
    _sys.stderr = _sys.stdout

    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Start : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'Alpha (QRSwideMSE weight): {ALPHA}')
    print(f'Peak mask dilation: ±100 samples (±100ms at 1kHz)')

    dataset = MovementECGDatasetV12(DATA_DIR)
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

    model  = ComplexUNetV12(DIMENSION, in_channels=IN_CHANNELS).to(device)
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
    start_epoch = 0

    # --- Resume logic ---
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch            = ckpt['epoch']
        history                = ckpt['history']
        early_stop.best_loss   = ckpt['best_loss']
        early_stop.best_epoch  = ckpt['best_epoch']
        early_stop.counter     = ckpt['es_counter']
        print(f'Resumed from full checkpoint at epoch {start_epoch}')
    elif os.path.exists(MODEL_SAVE_PATH) and os.path.exists(HISTORY_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        history = json.load(open(HISTORY_PATH))
        start_epoch            = len(history['val_loss'])
        early_stop.best_loss   = min(history['val_loss'])
        early_stop.best_epoch  = history['val_loss'].index(early_stop.best_loss)
        print(f'Warm start from best model (ep {early_stop.best_epoch + 1}), '
              f'continuing from ep {start_epoch + 1}')

    t0 = time.time()
    for epoch in range(start_epoch, MAX_EPOCHS):
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

        torch.save({
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'epoch':      epoch + 1,
            'history':    history,
            'best_loss':  early_stop.best_loss,
            'best_epoch': early_stop.best_epoch,
            'es_counter': early_stop.counter,
        }, CKPT_PATH)

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
