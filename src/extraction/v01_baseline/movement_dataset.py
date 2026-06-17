"""
Dataset for movement_ecg .mat files.

Each .mat file contains:
  out.mixture : (6, 600000) float64  — abdominal mixture (mECG + fECG + movement noise)
  out.fecg    : cell{1}(6, 600000)  — clean fetal ECG projected on 6 electrodes

Strategy:
  - Window the 600 s signal into 4 s segments (4000 samples at 1000 Hz), stride 4000.
  - Compute STFT per channel → complex spectrogram (129, ~401).
  - Resize to (128, 400) and return as complex tensor (6, 128, 400).
    Frequency axis: 129→128 (negligible loss).
    Time axis: 401→400 (10 ms/frame → QRS fits 6-8 frames, preserves peaks).
  - On first access a .npz cache is written next to the .mat file to avoid
    reloading the full 423 MB file on subsequent epochs/runs.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import librosa
import torch.nn.functional as F


FS = 1000          # sampling rate assumed from file context (600 000 samples / 600 s)
WINDOW_SEC = 4     # seconds per training sample
STRIDE_SEC = 4     # stride in seconds (no overlap → ~half the windows, faster epochs)
NFFT = 256
HOP_LENGTH = 10
WIN_LENGTH = 128
TARGET_SIZE_F = 128   # frequency bins  (129 → 128, negligible loss)
TARGET_SIZE_T = 400   # time frames     (401 → 400, preserves 10 ms/frame resolution)
TARGET_SIZE   = TARGET_SIZE_F  # kept for backward compat (single-dim references)
N_SAMPLES_TOTAL = 600_000


def _extract_fecg(out):
    """Robustly extract fecg array from MATLAB struct.

    Old .mat: out['fecg'][0][0] is the numeric array directly.
    New .mat: out['fecg'][0][0] is a MATLAB cell array (dtype=object),
              and the actual signal is one level deeper.
    """
    field = out['fecg'][0][0]
    if field.dtype == object:
        # cell array — take the first (and only) fetus
        return field.flat[0].astype(np.float32)
    return field.astype(np.float32)


def _load_mat(path):
    """Load mixture and fecg from a .mat file, return (mixture, fecg) as float32 arrays."""
    mat = sio.loadmat(path)
    mixture = mat['out']['mixture'][0][0].astype(np.float32)   # (6, N)
    fecg = _extract_fecg(mat['out'])                            # (6, N) or (1, N)
    del mat
    # If fecg has fewer channels than mixture, broadcast to match
    if fecg.shape[0] < mixture.shape[0]:
        fecg = np.repeat(fecg, mixture.shape[0] // fecg.shape[0], axis=0)
    return mixture, fecg


def _npz_path(mat_path):
    base = os.path.splitext(mat_path)[0]
    return base + '_signals.npz'


def _load_signals(mat_path):
    """Load from .npz cache if available, else load .mat and create cache."""
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
    """Compute STFT for each channel; returns complex array (C, F, T)."""
    specs = []
    for ch in range(signal.shape[0]):
        S = librosa.stft(signal[ch], n_fft=nfft,
                         hop_length=hop_length, win_length=win_length)
        specs.append(S)
    return np.stack(specs, axis=0)   # (C, F, T)


def _to_resized_complex_tensor(spec, target_h, target_w):
    """Convert (C, F, T) complex numpy array to complex torch tensor (C, H, W)."""
    real = torch.from_numpy(np.real(spec))   # (C, F, T)
    imag = torch.from_numpy(np.imag(spec))
    # F.interpolate expects (N, C, H, W); treat channels as batch
    real = F.interpolate(real.unsqueeze(0), size=(target_h, target_w),
                         mode='bilinear', align_corners=False).squeeze(0)
    imag = F.interpolate(imag.unsqueeze(0), size=(target_h, target_w),
                         mode='bilinear', align_corners=False).squeeze(0)
    return torch.complex(real, imag)


class MovementECGDataset(Dataset):
    """
    Parameters
    ----------
    data_dir   : folder with movement_ecg .mat files
    window_sec : length of each training window in seconds (default 4 s)
    stride_sec : stride between windows in seconds (default 2 s → 50 % overlap)
    fs         : assumed sampling rate of the recordings (default 1000 Hz)
    """

    def __init__(self, data_dir,
                 window_sec=WINDOW_SEC,
                 stride_sec=STRIDE_SEC,
                 fs=FS,
                 nfft=NFFT,
                 hop_length=HOP_LENGTH,
                 win_length=WIN_LENGTH,
                 target_h=TARGET_SIZE_F,
                 target_w=TARGET_SIZE_T):

        self.data_dir = data_dir
        self.window = int(window_sec * fs)
        self.stride = int(stride_sec * fs)
        self.nfft = nfft
        self.hop_length = hop_length
        self.win_length = win_length
        self.target_h = target_h
        self.target_w = target_w

        self.files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith('.mat')
        )

        # Build index: list of (file_idx, start_sample)
        n = N_SAMPLES_TOTAL
        self.windows = []
        for file_idx in range(len(self.files)):
            for start in range(0, n - self.window + 1, self.stride):
                self.windows.append((file_idx, start))

        # Number of windows per file — used to skip to a different file on fallback
        self._wins_per_file = (n - self.window) // self.stride + 1

        # In-memory cache: {file_idx: (mixture, fecg)}
        self._cache: dict = {}
        # Files that failed to load (skipped silently)
        self._bad_files: set = set()

    # ------------------------------------------------------------------
    def _get_signals(self, file_idx):
        if file_idx in self._bad_files:
            raise ValueError(f'Bad file: {self.files[file_idx]}')
        if file_idx not in self._cache:
            try:
                mixture, fecg = _load_signals(self.files[file_idx])
                self._cache[file_idx] = (mixture, fecg)
            except Exception as e:
                print(f'\n[WARN] Skipping corrupted file '
                      f'{os.path.basename(self.files[file_idx])}: {e}')
                self._bad_files.add(file_idx)
                raise
        return self._cache[file_idx]

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.windows)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        # On fallback, jump by _wins_per_file each time → guaranteed different file
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

        end = start + self.window
        mix_win = mixture[:, start:end].copy()   # (C, window)
        fecg_win = fecg[:, start:end].copy()      # (C, window)

        # Per-channel normalisation using mixture std (same scale for both)
        stds = mix_win.std(axis=1, keepdims=True)   # (C, 1)
        stds = np.where(stds < 1e-8, 1.0, stds)
        mix_win = mix_win / stds
        fecg_win = fecg_win / stds   # preserve relative amplitude

        # STFT
        mix_spec = _stft_multichannel(mix_win, self.nfft,
                                      self.hop_length, self.win_length)
        fecg_spec = _stft_multichannel(fecg_win, self.nfft,
                                       self.hop_length, self.win_length)

        # Resize and convert to complex tensors
        x = _to_resized_complex_tensor(mix_spec, self.target_h, self.target_w)
        y = _to_resized_complex_tensor(fecg_spec, self.target_h, self.target_w)

        # Time-domain fECG (for time-domain loss component)
        fecg_time = torch.from_numpy(fecg_win).float()   # (C, window) float32

        return x, y, fecg_time


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    data_dir = 'data/movement_ecg'
    print(f'Building dataset from {data_dir} ...')
    dset = MovementECGDataset(data_dir)
    print(f'Total windows: {len(dset)}')
    print('Loading first sample (triggers .mat → .npz conversion) ...')
    x, y, fecg_time = dset[0]
    print(f'  x         : {x.shape}  dtype={x.dtype}')
    print(f'  y         : {y.shape}  dtype={y.dtype}')
    print(f'  fecg_time : {fecg_time.shape}  dtype={fecg_time.dtype}')
    print(f'  x real range : [{x.real.min():.3f}, {x.real.max():.3f}]')
    print(f'  y real range : [{y.real.min():.3f}, {y.real.max():.3f}]')
    print('OK')
