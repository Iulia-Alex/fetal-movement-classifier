"""
Training script v6 — fine-tune from v1 best (ep199) with baseline-aware loss.

Architecture : identical to v1 (ComplexUNet, 0.59M, 128×400, 1000 Hz)
Init         : warm start from models/movement_CUNet_128x400_composed.pth (v1 ep199)

Loss: SignalMSE + LAMBDA_AMP * AmpWeightedMSE + LAMBDA_BASE * BaselinePenalty

  SignalMSE         : full time-domain MSE (global fidelity anchor)
  AmpWeightedMSE    : MSE * w,       w = (|target| / max|target|)^2 per channel
                      → extra emphasis on R-peaks
  BaselinePenalty   : pred^2 * (1-w)
                      → penalises non-zero predictions during baseline periods
                        (complementary to AmpWeightedMSE)

Together:
  at QRS peak  → loss = (1 + LAMBDA_AMP) * (pred-target)^2   (peaks must match)
  at baseline  → loss = (1 + LAMBDA_BASE) * pred^2           (output must be zero)
"""

import os
import sys
import json
import time
import datetime
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import pathsetup  # noqa

from complex_network import ComplexUNet
from movement_dataset import (
    MovementECGDataset, TARGET_SIZE_F, TARGET_SIZE_T,
    NFFT, HOP_LENGTH, WIN_LENGTH, FS,
)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
DATA_DIR          = '../data/movement_ecg'
WARM_START_PATH   = '../models/movement_CUNet_128x400_composed.pth'   # v1 ep199
MODEL_SAVE_PATH   = '../models/movement_CUNet_128x400_v6_baseline.pth'
HISTORY_PATH      = '../models/movement_CUNet_128x400_v6_baseline_history.json'
LOG_DIR           = '../logs'

LEARNING_RATE     = 5e-5      # half of v1/v5 — fine-tuning from warm start
WEIGHT_DECAY      = 1e-5
BATCH_SIZE        = 32
MAX_EPOCHS        = 200
PATIENCE          = 20        # extra patience for fine-tuning
VAL_SPLIT         = 0.15
NUM_WORKERS       = 2
PRINT_EVERY       = 20
SEED              = 42

IN_CHANNELS       = 6
DIMENSION         = TARGET_SIZE_F * TARGET_SIZE_T   # 128 × 400

WINDOW_SAMPLES    = 4 * FS                          # 4000
ORIG_F            = NFFT // 2 + 1                   # 129
ORIG_T            = 1 + WINDOW_SAMPLES // HOP_LENGTH  # 401

LAMBDA_AMP        = 1.0   # amplitude-weighted peak emphasis
LAMBDA_BASE       = 1.0   # baseline suppression penalty

_DEVICE_HANN: dict = {}


def _get_hann(device):
    key = str(device)
    if key not in _DEVICE_HANN:
        _DEVICE_HANN[key] = torch.hann_window(WIN_LENGTH, device=device)
    return _DEVICE_HANN[key]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def _istft_pred(pred_spec: torch.Tensor) -> torch.Tensor:
    B, C, H, W = pred_spec.shape
    real = F.interpolate(
        pred_spec.real.reshape(B * C, 1, H, W),
        size=(ORIG_F, ORIG_T), mode='bilinear', align_corners=False,
    ).squeeze(1)
    imag = F.interpolate(
        pred_spec.imag.reshape(B * C, 1, H, W),
        size=(ORIG_F, ORIG_T), mode='bilinear', align_corners=False,
    ).squeeze(1)
    spec = torch.complex(real, imag)
    window = _get_hann(spec.device)
    pred_time = torch.istft(
        spec, n_fft=NFFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=window, length=WINDOW_SAMPLES,
    )
    return pred_time.reshape(B, C, WINDOW_SAMPLES)


def baseline_aware_loss(pred_spec: torch.Tensor, fecg_time: torch.Tensor) -> torch.Tensor:
    """
    SignalMSE + LAMBDA_AMP * AmpWeightedMSE + LAMBDA_BASE * BaselinePenalty

    w = (|target| / max|target|)^2  per channel — in [0,1]

    AmpWeightedMSE  = mean( (pred-target)^2 * w )        high near R-peaks
    BaselinePenalty = mean( pred^2 * (1-w) )             high during baseline
    """
    pred_time = _istft_pred(pred_spec)

    # 1. Global MSE anchor
    sig_mse = F.mse_loss(pred_time, fecg_time)

    # amplitude weights (per channel, broadcast over time)
    abs_target = fecg_time.abs()
    max_amp = abs_target.amax(dim=2, keepdim=True).clamp(min=1e-8)
    w = (abs_target / max_amp).pow(2)   # (B, C, T)

    # 2. Peak emphasis
    amp_mse = ((pred_time - fecg_time).pow(2) * w).mean()

    # 3. Baseline suppression
    base_penalty = (pred_time.pow(2) * (1.0 - w)).mean()

    return sig_mse + LAMBDA_AMP * amp_mse + LAMBDA_BASE * base_penalty


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience, save_path):
        self.patience   = patience
        self.save_path  = save_path
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_epoch = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            print(f'    [checkpoint] val_loss={val_loss:.6f} -> saved')
        else:
            self.counter += 1
            print(f'    [early stop] no improvement {self.counter}/{self.patience} '
                  f'(best={self.best_loss:.6f} @ ep {self.best_epoch + 1})')
            if self.counter >= self.patience:
                return True
        return False


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, device, train, epoch_num, total_epochs):
    model.train() if train else model.eval()
    phase     = 'train' if train else 'val'
    total_loss = 0.0
    n_batches  = len(loader)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, (x, y, y_time) in enumerate(loader):
            x      = x.to(device)
            y_time = y_time.to(device)

            pred = model(x)
            loss = baseline_aware_loss(pred, y_time)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % PRINT_EVERY == 0 or (batch_idx + 1) == n_batches:
                avg = total_loss / (batch_idx + 1)
                print(f'  [{phase}] ep {epoch_num}/{total_epochs} '
                      f'| batch {batch_idx + 1}/{n_batches} '
                      f'| loss={loss.item():.6f} | avg={avg:.6f}')
                sys.stdout.flush()

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_name(0)}')
    print()

    print(f'Loading dataset from {DATA_DIR} ...')
    full_dataset = MovementECGDataset(DATA_DIR)
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val
    print(f'Total windows: {n_total}  (train={n_train}, val={n_val})')
    print()

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(full_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=(device == 'cuda'))
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=(device == 'cuda'))

    model = ComplexUNet(DIMENSION, in_channels=IN_CHANNELS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params / 1e6:.2f} M')

    # Resume from v6 checkpoint if exists, else warm start from v1
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f'Resumed from v6 checkpoint: {MODEL_SAVE_PATH}')
    else:
        print(f'Loading warm start from {WARM_START_PATH} ...')
        model.load_state_dict(torch.load(WARM_START_PATH, map_location=device))
        print('Warm start loaded OK')
    print()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True,
    )
    early_stop = EarlyStopping(patience=PATIENCE, save_path=MODEL_SAVE_PATH)

    # Resume history if exists
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            history = json.load(f)
        early_stop.best_loss  = min(history['val_loss'])
        early_stop.best_epoch = history['val_loss'].index(early_stop.best_loss)
        print(f'History loaded: {len(history["val_loss"])} epochs, '
              f'best val={early_stop.best_loss:.6f} @ ep {early_stop.best_epoch + 1}')
    else:
        history = {'train_loss': [], 'val_loss': [], 'lr': []}

    epochs_done = len(history['val_loss'])
    t_start     = time.time()

    print('=' * 70)
    print(f'Resumed          : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Continuing from ep {epochs_done + 1}/{epochs_done + MAX_EPOCHS}')
    print(f'Loss             : SignalMSE + {LAMBDA_AMP}*AmpWeightedMSE + {LAMBDA_BASE}*BaselinePenalty')
    print(f'LR={LEARNING_RATE}, WD={WEIGHT_DECAY}, BS={BATCH_SIZE}, patience={PATIENCE}')
    print('=' * 70)

    for epoch in range(epochs_done, epochs_done + MAX_EPOCHS):
        epoch_start = time.time()
        current_lr  = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch + 1}/{epochs_done + MAX_EPOCHS}  |  lr={current_lr:.2e}')
        print('-' * 50)

        train_loss = run_epoch(model, train_loader, optimizer, device,
                               train=True,  epoch_num=epoch + 1, total_epochs=epochs_done + MAX_EPOCHS)
        val_loss   = run_epoch(model, val_loader,   optimizer, device,
                               train=False, epoch_num=epoch + 1, total_epochs=epochs_done + MAX_EPOCHS)

        elapsed       = time.time() - epoch_start
        total_elapsed = time.time() - t_start
        print(f'\n  >> Epoch {epoch + 1}: '
              f'train={train_loss:.6f}  val={val_loss:.6f}  '
              f'time={elapsed:.1f}s  total={total_elapsed/60:.1f}min')

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)

        scheduler.step(val_loss)

        if early_stop.step(val_loss, model, epoch):
            print(f'\nEarly stopping after {epoch + 1} epochs.')
            break

    print('\n' + '=' * 70)
    print(f'Training finished: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Best val_loss    : {early_stop.best_loss:.6f} at epoch {early_stop.best_epoch + 1}')
    print(f'Model saved to   : {MODEL_SAVE_PATH}')
    print('=' * 70)


if __name__ == '__main__':
    main()
