"""
Training script v17 — ComplexAttentionUNet with attention supervision.

Compared to v16:
  - Dataset: movement_dataset_v17.py (returns att_target_T128)
  - Loss: L_total = SignalMSE + λ_att * L_att
      L_att = MSE(alpha_AG1_predicted, att_target_AG1_resolution)
      alpha_AG1 capturat prin forward hook pe model.ag1.psi (sigmoid output)
      att_target: (B,128) → avg_pool →64 → broadcast la (B,1,64,64)
  - λ_att = 0.1 (both components logged separately in history)
  - Restul identic cu v16: SignalMSE, AdamW, ReduceLROnPlateau, PATIENCE=20

Arhitectura: ComplexAttentionUNet din complex_network_v16.py (identic cu v16).
Dataset: movement_dataset_v17.py (500Hz, STFT 128×128, fqrs attention target)
"""

import os, sys, json, time, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import pathsetup  # noqa

from complex_network_v16 import ComplexAttentionUNet
from movement_dataset_v17 import (
    MovementECGDatasetV17, NFFT, HOP_LENGTH, WIN_LENGTH,
    FS, WINDOW, TARGET_SIZE_F, TARGET_SIZE_T,
)

# ---------------------------------------------------------------------------
DATA_DIR        = '../data/movement_ecg'
MODEL_SAVE_PATH = '../models/movement_CUNet_v17_attsup.pth'
HISTORY_PATH    = '../models/movement_CUNet_v17_attsup_history.json'
LOG_DIR         = '../logs'

LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
BATCH_SIZE      = 8
MAX_EPOCHS      = 300
PATIENCE        = 20
VAL_SPLIT       = 0.15
NUM_WORKERS     = 2
PRINT_EVERY     = 20
SEED            = 42
LAMBDA_ATT      = 0.1    # pondere L_att vs SignalMSE

IN_CHANNELS     = 6
DIMENSION       = TARGET_SIZE_F * TARGET_SIZE_T   # 16384
ORIG_F          = NFFT // 2 + 1                   # 129
ORIG_T          = 128
WINDOW_SAMPLES  = WINDOW

_DEVICE_HANN: dict = {}


def _get_hann(device):
    if str(device) not in _DEVICE_HANN:
        _DEVICE_HANN[str(device)] = torch.hann_window(WIN_LENGTH, device=device)
    return _DEVICE_HANN[str(device)]


def signal_mse(pred_spec: torch.Tensor, fecg_time: torch.Tensor) -> torch.Tensor:
    B, C, Fq, T = pred_spec.shape
    zeros = torch.zeros(B, C, 1, T, dtype=pred_spec.dtype, device=pred_spec.device)
    pred_full = torch.cat([pred_spec, zeros], dim=2)
    pred_flat = pred_full.reshape(B * C, ORIG_F, ORIG_T)
    window    = _get_hann(pred_flat.device)
    pred_time = torch.istft(
        pred_flat, n_fft=NFFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=window,
        center=True, length=WINDOW_SAMPLES,
    ).reshape(B, C, WINDOW_SAMPLES)
    return F.mse_loss(pred_time, fecg_time)


def att_loss(alpha_ag1: torch.Tensor, att_target_128: torch.Tensor) -> torch.Tensor:
    """
    Computes the MSE between alpha AG1 and the attention target at 64x64 resolution.

    alpha_ag1      : (B, 1, 64, 64)  — sigmoid output din AG1
    att_target_128 : (B, 128)        — binary mask at T=128
    """
    B = att_target_128.shape[0]
    # Downsample mask 128 → 64 with avg_pool (preserving soft values)
    target_64 = F.avg_pool1d(
        att_target_128.unsqueeze(1), kernel_size=2, stride=2
    ).squeeze(1)   # (B, 64)
    # Broadcast over the F dimension (dim=2): (B, 1, 64, 64)
    target_2d = target_64.unsqueeze(1).unsqueeze(2).expand(-1, 1, 64, -1)  # (B,1,64,64)
    return F.mse_loss(alpha_ag1, target_2d)


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


def run_epoch(model, loader, optimizer, device, train=True, alphas_store=None):
    """
    alphas_store : dict that receives alpha_ag1 from the forward hook.
                   The hook must be registered before calling this function.
    """
    model.train() if train else model.eval()
    total_loss     = 0.0
    total_sig_mse  = 0.0
    total_l_att    = 0.0
    n_batches      = len(loader)
    ctx            = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch_idx, (x, _, y_time, att_t) in enumerate(loader):
            x      = x.to(device)
            y_time = y_time.to(device)
            att_t  = att_t.to(device)   # (B, 128)

            pred = model(x)   # forward hook captureaza alpha_ag1

            # Obtine alpha capturata de hook
            alpha_ag1 = alphas_store.get('AG1')  # (B, 1, 64, 64)

            sig_mse_val = signal_mse(pred, y_time)
            l_att_val   = att_loss(alpha_ag1, att_t) if alpha_ag1 is not None else torch.tensor(0.0, device=device)
            loss        = sig_mse_val + LAMBDA_ATT * l_att_val

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.apply(model.W_clipper)

            total_loss    += loss.item()
            total_sig_mse += sig_mse_val.item()
            total_l_att   += l_att_val.item()

            if (batch_idx + 1) % PRINT_EVERY == 0 or (batch_idx + 1) == n_batches:
                avg      = total_loss     / (batch_idx + 1)
                avg_sig  = total_sig_mse  / (batch_idx + 1)
                avg_att  = total_l_att    / (batch_idx + 1)
                tag      = 'train' if train else 'val'
                print(f'  [{tag}] ep {_current_epoch}/{MAX_EPOCHS} '
                      f'| batch {batch_idx+1}/{n_batches} '
                      f'| sig_mse={avg_sig:.6f} | l_att={avg_att:.6f} '
                      f'| total={avg:.6f}')

    n = n_batches
    return total_loss / n, total_sig_mse / n, total_l_att / n


def main():
    global _current_epoch

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, 'movement_CUNet_v17_attsup.log')
    import sys as _sys
    _sys.stdout = open(log_path, 'w', buffering=1)
    _sys.stderr = _sys.stdout

    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Start : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')

    dataset = MovementECGDatasetV17(DATA_DIR)
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

    model  = ComplexAttentionUNet(DIMENSION, in_channels=IN_CHANNELS).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params / 1e6:.3f} M')
    print(f'Arhitectura: ComplexAttentionUNet (v16) + attention supervision L_att')
    print(f'λ_att = {LAMBDA_ATT}')

    # ── Forward hook for alpha AG1 ──────────────────────────────────────────
    # Registered once — captures the sigmoid output of AG1.psi
    alphas_store: dict = {}

    def _hook_ag1(module, inp, out):
        alphas_store['AG1'] = out   # (B, 1, 64, 64)

    model.ag1.psi.register_forward_hook(_hook_ag1)
    # ────────────────────────────────────────────────────────────────────────

    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True,
    )
    early_stop = EarlyStopping(patience=PATIENCE)

    epochs_done = 0
    history     = {'train_loss': [], 'val_loss': [],
                   'train_sig_mse': [], 'val_sig_mse': [],
                   'train_l_att': [], 'val_l_att': [], 'lr': []}
    if os.path.exists(HISTORY_PATH):
        history = json.load(open(HISTORY_PATH))
        if os.path.exists(MODEL_SAVE_PATH) and history['val_loss']:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            epochs_done = len(history['val_loss'])
            early_stop.best_loss  = min(history['val_loss'])
            early_stop.best_epoch = history['val_loss'].index(early_stop.best_loss)
            print(f'Resumed: {epochs_done} epochs, '
                  f'best val={early_stop.best_loss:.6f} @ ep {early_stop.best_epoch + 1}')

    t0 = time.time()
    for epoch in range(epochs_done, MAX_EPOCHS):
        _current_epoch = epoch + 1
        t_ep = time.time()

        lr_now = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch + 1}/{MAX_EPOCHS}  |  lr={lr_now:.2e}')
        print('-' * 50)

        tr_loss, tr_sig, tr_att = run_epoch(model, train_loader, optimizer, device,
                                            train=True,  alphas_store=alphas_store)
        va_loss, va_sig, va_att = run_epoch(model, val_loader,   optimizer, device,
                                            train=False, alphas_store=alphas_store)

        elapsed  = time.time() - t0
        ep_time  = time.time() - t_ep
        print(f'\n  >> Epoch {epoch+1}: '
              f'train={tr_loss:.6f} (sig={tr_sig:.6f}, att={tr_att:.6f})  '
              f'val={va_loss:.6f} (sig={va_sig:.6f}, att={va_att:.6f})  '
              f't={ep_time:.1f}s  total={elapsed/60:.1f}min')

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_sig_mse'].append(tr_sig)
        history['val_sig_mse'].append(va_sig)
        history['train_l_att'].append(tr_att)
        history['val_l_att'].append(va_att)
        history['lr'].append(lr_now)
        json.dump(history, open(HISTORY_PATH, 'w'), indent=2)

        scheduler.step(va_loss)

        if early_stop.step(va_loss, model, epoch):
            print(f'\nEarly stopping after {epoch + 1} epochs.')
            break

    print(f'\n{"=" * 70}')
    print(f'Training finished: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'Best val_loss    : {early_stop.best_loss:.6f} at epoch {early_stop.best_epoch + 1}')
    print(f'Model saved to   : {MODEL_SAVE_PATH}')
    print('=' * 70)


if __name__ == '__main__':
    main()
