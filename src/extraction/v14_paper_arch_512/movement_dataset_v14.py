"""
Dataset v14 — same as movement_dataset.py but with:
  - Downsample 1000Hz → 500Hz (factor 2, simple subsampling)
  - STFT params from paper: n_fft=256, hop=15, win=100 → natural 129×128
  - Resize to 128×128 (drop 1 freq bin, exactly as in paper)
  - Window = 3830 samples @ 1000Hz = 1915 @ 500Hz = 3.83s
  - fecg_time at 500Hz (1915 samples), for iSTFT in loss

Compatible with signal_mse in train_movement_v14.py:
  ORIG_F=129, ORIG_T=128, iSTFT with n_fft=256, hop=15, win=100, length=1915.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import librosa
import torch.nn.functional as F

# Parametri dataset
FS_ORIG         = 1000     # original recording sampling rate
DOWNSAMPLE      = 2        # downsample factor: 1000 → 500 Hz
FS              = 500      # after downsampling
WINDOW_ORIG     = 3830     # window @ 1000 Hz (= 3.83 s)
WINDOW_FS       = 1915     # window @ 500 Hz  (= 3.83 s)
STRIDE_ORIG     = 3830     # stride @ 1000 Hz (no overlap)
N_SAMPLES_TOTAL = 600_000  # total length of each recording @ 1000 Hz

# STFT params — exact ca paper
NFFT        = 256
HOP_LENGTH  = 15
WIN_LENGTH  = 100
TARGET_SIZE_F = 128   # 129 → 128 (drop ultimul bin)
TARGET_SIZE_T = 128   # 128 (natural cu 1915 samples, hop=15, center=True)


def _extract_fecg(out):
    field = out['fecg'][0][0]
    if field.dtype == object:
        return field.flat[0].astype(np.float32)
    return field.astype(np.float32)


def _load_mat(path):
    mat = sio.loadmat(path)
    mixture = mat['out']['mixture'][0][0].astype(np.float32)
    fecg    = _extract_fecg(mat['out'])
    del mat
    if fecg.shape[0] < mixture.shape[0]:
        fecg = np.repeat(fecg, mixture.shape[0] // fecg.shape[0], axis=0)
    return mixture, fecg


def _npz_path(mat_path):
    base = os.path.splitext(mat_path)[0]
    return base + '_signals.npz'


def _load_signals(mat_path):
    npz = _npz_path(mat_path)
    if os.path.exists(npz):
        data = np.load(npz)
        return data['mixture'], data['fecg']
    mixture, fecg = _load_mat(mat_path)
    try:
        np.savez_compressed(npz, mixture=mixture, fecg=fecg)
    except Exception as e:
        print(f'[WARN] Could not write cache {npz}: {e}')
    return mixture, fecg


def _stft_multichannel(signal, nfft, hop_length, win_length):
    specs = []
    for ch in range(signal.shape[0]):
        S = librosa.stft(signal[ch], n_fft=nfft,
                         hop_length=hop_length, win_length=win_length)
        specs.append(S)
    return np.stack(specs, axis=0)   # (C, F, T)


def _to_resized_complex_tensor(spec, target_h, target_w):
    real = torch.from_numpy(np.real(spec))
    imag = torch.from_numpy(np.imag(spec))
    real = F.interpolate(real.unsqueeze(0), size=(target_h, target_w),
                         mode='bilinear', align_corners=False).squeeze(0)
    imag = F.interpolate(imag.unsqueeze(0), size=(target_h, target_w),
                         mode='bilinear', align_corners=False).squeeze(0)
    return torch.complex(real, imag)


class MovementECGDatasetV14(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith('.mat')
        )

        # Window index @ 1000 Hz
        self.windows = []
        for file_idx in range(len(self.files)):
            for start in range(0, N_SAMPLES_TOTAL - WINDOW_ORIG + 1, STRIDE_ORIG):
                self.windows.append((file_idx, start))

        self._wins_per_file = (N_SAMPLES_TOTAL - WINDOW_ORIG) // STRIDE_ORIG + 1
        self._cache: dict   = {}
        self._bad_files: set = set()

    def _get_signals(self, file_idx):
        if file_idx in self._bad_files:
            raise ValueError(f'Bad file: {self.files[file_idx]}')
        if file_idx not in self._cache:
            try:
                mixture, fecg = _load_signals(self.files[file_idx])
                self._cache[file_idx] = (mixture, fecg)
            except Exception as e:
                print(f'\n[WARN] Skipping {os.path.basename(self.files[file_idx])}: {e}')
                self._bad_files.add(file_idx)
                raise
        return self._cache[file_idx]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        for attempt in range(20):
            try:
                candidate = (idx + attempt * self._wins_per_file) % len(self.windows)
                file_idx, start = self.windows[candidate]
                mixture, fecg = self._get_signals(file_idx)
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f'Could not load valid data after 20 attempts (idx={idx})')

        end = start + WINDOW_ORIG
        mix_win  = mixture[:, start:end].copy()   # (C, 3830) @ 1000Hz
        fecg_win = fecg[:,   start:end].copy()

        # Downsample 1000 → 500 Hz
        mix_win  = mix_win[:,  ::DOWNSAMPLE]   # (C, 1915)
        fecg_win = fecg_win[:, ::DOWNSAMPLE]   # (C, 1915)

        # Per-channel normalisation using the mixture std
        stds = mix_win.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        mix_win  = mix_win  / stds
        fecg_win = fecg_win / stds

        # STFT → (C, 129, 128) natural
        mix_spec  = _stft_multichannel(mix_win,  NFFT, HOP_LENGTH, WIN_LENGTH)
        fecg_spec = _stft_multichannel(fecg_win, NFFT, HOP_LENGTH, WIN_LENGTH)

        # Resize la 128×128
        x = _to_resized_complex_tensor(mix_spec,  TARGET_SIZE_F, TARGET_SIZE_T)
        y = _to_resized_complex_tensor(fecg_spec, TARGET_SIZE_F, TARGET_SIZE_T)

        # fecg_time @ 500 Hz (for SignalMSE)
        fecg_time = torch.from_numpy(fecg_win).float()   # (C, 1915)

        return x, y, fecg_time
