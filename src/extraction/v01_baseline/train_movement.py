"""
Training script for fECG extraction from movement ECG data using ComplexUNet.

Input  : out.mixture  — abdominal signal (mECG + fECG + movement noise), 6 channels
Output : out.fecg     — clean fetal ECG, 6 channels

Run in the background (survives SSH disconnect):
    nohup /home/iulia.orvas/miniconda3/envs/ecg/bin/python train_movement.py \
        > logs/movement_CUNet.log 2>&1 &
    echo "PID: $!"
"""

import os
import sys
import json
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add project root to path so imports work regardless of cwd
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
DATA_DIR        = '../data/movement_ecg'
MODEL_SAVE_PATH = '../models/movement_CUNet_128x400_composed.pth'
HISTORY_PATH    = '../models/movement_CUNet_128x400_composed_history.json'
LOG_DIR         = '../logs'

LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
BATCH_SIZE      = 32
MAX_EPOCHS      = 200
PATIENCE        = 15        # early-stopping patience (epochs without improvement)
VAL_SPLIT       = 0.15
NUM_WORKERS     = 2         # 2 workers for CPU/GPU overlap (STFT on CPU while GPU trains)
PRINT_EVERY     = 20        # print batch loss every N batches
SEED            = 42

IN_CHANNELS     = 6                           # number of ECG channels in the dataset
DIMENSION       = TARGET_SIZE_F * TARGET_SIZE_T  # 128 × 400 = 51200

# Time-domain loss constants
WINDOW_SAMPLES  = 4 * FS                     # 4000 samples per window
ORIG_F          = NFFT // 2 + 1              # 129  (original STFT freq bins)
ORIG_T          = 1 + WINDOW_SAMPLES // HOP_LENGTH  # 401 (original STFT time frames)
LAMBDA_TIME     = 1.0                        # weight of time-domain loss vs spec loss

# Hann window cached per device (GPU iSTFT is 7× faster than CPU)
_DEVICE_HANN: dict = {}

def _get_hann(device):
    key = str(device)
    if key not in _DEVICE_HANN:
        _DEVICE_HANN[key] = torch.hann_window(WIN_LENGTH, device=device)
    return _DEVICE_HANN[key]

# ---------------------------------------------------------------------------
# Loss: spectrogram MSE + time-domain MSE  (ComposedLoss from the paper)
# ---------------------------------------------------------------------------
def complex_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on real + imaginary spectrogram parts."""
    diff = pred - target
    return diff.real.pow(2).mean() + diff.imag.pow(2).mean()


def signal_mse(pred_spec: torch.Tensor, fecg_time: torch.Tensor) -> torch.Tensor:
    """
    Time-domain MSE: resize predicted spectrogram back to original STFT shape,
    apply iSTFT, then compare with ground-truth time-domain fECG.

    pred_spec : (B, C, 128, 400) complex  — model output
    fecg_time : (B, C, 4000)    float32  — ground-truth normalised fECG
    """
    B, C, H, W = pred_spec.shape

    # Resize back to original STFT shape  (B*C, 1, H, W) → (B*C, ORIG_F, ORIG_T)
    real = F.interpolate(
        pred_spec.real.reshape(B * C, 1, H, W),
        size=(ORIG_F, ORIG_T), mode='bilinear', align_corners=False,
    ).squeeze(1)   # (B*C, 129, 401)
    imag = F.interpolate(
        pred_spec.imag.reshape(B * C, 1, H, W),
        size=(ORIG_F, ORIG_T), mode='bilinear', align_corners=False,
    ).squeeze(1)

    # iSTFT on GPU (7× faster than CPU on M4000)
    spec = torch.complex(real, imag)   # (B*C, 129, 401) — stays on GPU
    window = _get_hann(spec.device)
    pred_time_bc = torch.istft(
        spec, n_fft=NFFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=window, length=WINDOW_SAMPLES,
    )  # (B*C, 4000)

    pred_time = pred_time_bc.reshape(B, C, WINDOW_SAMPLES)

    return F.mse_loss(pred_time, fecg_time)


# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int, save_path: str):
        self.patience = patience
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            print(f'    [checkpoint] val_loss={val_loss:.6f} → saved to {self.save_path}')
        else:
            self.counter += 1
            print(f'    [early stop] no improvement for {self.counter}/{self.patience} epochs '
                  f'(best={self.best_loss:.6f} @ epoch {self.best_epoch + 1})')
            if self.counter >= self.patience:
                return True
        return False


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, device, train: bool, epoch_num: int, total_epochs: int):
    model.train() if train else model.eval()
    phase = 'train' if train else 'val'
    total_loss = 0.0
    n_batches = len(loader)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, (x, y, y_time) in enumerate(loader):
            x      = x.to(device)
            y      = y.to(device)
            y_time = y_time.to(device)

            pred      = model(x)
            loss      = signal_mse(pred, y_time)   # time-domain MSE (as in paper)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % PRINT_EVERY == 0 or (batch_idx + 1) == n_batches:
                avg = total_loss / (batch_idx + 1)
                print(f'  [{phase}] epoch {epoch_num}/{total_epochs} '
                      f'| batch {batch_idx + 1}/{n_batches} '
                      f'| batch_loss={loss.item():.6f} '
                      f'| running_avg={avg:.6f}')
                sys.stdout.flush()

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)

    # Directories
    os.makedirs('models', exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_name(0)}')
    print()

    # Dataset
    print(f'Loading dataset from {DATA_DIR} ...')
    full_dataset = MovementECGDataset(DATA_DIR)
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val
    print(f'Total windows : {n_total}  (train={n_train}, val={n_val})')
    print()

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(full_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=(device == 'cuda'))
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=(device == 'cuda'))

    # Model
    model = ComplexUNet(DIMENSION, in_channels=IN_CHANNELS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params / 1e6:.2f} M')
    print()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    early_stop = EarlyStopping(patience=PATIENCE, save_path=MODEL_SAVE_PATH)

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    t_start = time.time()

    print('=' * 70)
    print(f'Training started: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch + 1}/{MAX_EPOCHS}  |  lr={current_lr:.2e}')
        print('-' * 50)

        train_loss = run_epoch(model, train_loader, optimizer, device,
                               train=True, epoch_num=epoch + 1, total_epochs=MAX_EPOCHS)
        val_loss   = run_epoch(model, val_loader,   optimizer, device,
                               train=False, epoch_num=epoch + 1, total_epochs=MAX_EPOCHS)

        elapsed = time.time() - epoch_start
        total_elapsed = time.time() - t_start

        print(f'\n  >> Epoch {epoch + 1} summary: '
              f'train={train_loss:.6f}  val={val_loss:.6f}  '
              f'time={elapsed:.1f}s  total={total_elapsed/60:.1f}min')

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # Save history after every epoch
        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)

        scheduler.step(val_loss)

        if early_stop.step(val_loss, model, epoch):
            print(f'\nEarly stopping triggered after {epoch + 1} epochs.')
            break

    print('\n' + '=' * 70)
    print(f'Training finished: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Best val_loss   : {early_stop.best_loss:.6f} at epoch {early_stop.best_epoch + 1}')
    print(f'Model saved to  : {MODEL_SAVE_PATH}')
    print(f'History saved to: {HISTORY_PATH}')
    print('=' * 70)


if __name__ == '__main__':
    main()
