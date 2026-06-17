"""
Dataset for movement_ecg — paper-faithful version.

Resamples 1000 Hz → 500 Hz, then uses the paper's STFT parameters:
  n_fft=256, win=100, hop=15 (85% overlap)
  3.83 s windows → 1915 samples at 500 Hz
  STFT output: (6, 129, 128) → drop last freq bin → (6, 128, 128)
  No resize needed — spectrogram is naturally 128×128.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import scipy.signal
import librosa

# Paper STFT parameters
FS_ORIG    = 1000       # original sampling rate of movement_ecg data
FS         = 500        # paper's target sampling rate
WINDOW_SEC = 3.83       # seconds per window (chosen so STFT gives exactly 128 time frames)
WINDOW     = int(WINDOW_SEC * FS)  # 1915 samples
STRIDE     = WINDOW     # no overlap
NFFT       = 256
HOP_LENGTH = 15
WIN_LENGTH = 100
TARGET_SIZE_F = 128     # freq bins after dropping last (129 → 128)
TARGET_SIZE_T = 128     # time frames (naturally 128)
N_SAMPLES_ORIG = 600_000  # samples per file at 1000 Hz


def _extract_fecg(out):
    field = out['fecg'][0][0]
    if field.dtype == object:
        return field.flat[0].astype(np.float32)
    return field.astype(np.float32)


def _load_mat(path):
    mat = sio.loadmat(path)
    mixture = mat['out']['mixture'][0][0].astype(np.float32)
    fecg = _extract_fecg(mat['out'])
    del mat
    if fecg.shape[0] < mixture.shape[0]:
        fecg = np.repeat(fecg, mixture.shape[0] // fecg.shape[0], axis=0)
    return mixture, fecg


def _npz_path_500(mat_path):
    base = os.path.splitext(mat_path)[0]
    return base + '_signals_500hz.npz'


def _load_signals_500hz(mat_path):
    """Load signals resampled to 500 Hz, with .npz cache."""
    npz = _npz_path_500(mat_path)
    if os.path.exists(npz):
        data = np.load(npz)
        return data['mixture'], data['fecg']
    # Load original 1000 Hz
    mixture, fecg = _load_mat(mat_path)
    # Resample 1000 → 500 Hz (decimate by 2 with anti-aliasing)
    mix_500 = scipy.signal.decimate(mixture, 2, axis=1).astype(np.float32)
    fecg_500 = scipy.signal.decimate(fecg, 2, axis=1).astype(np.float32)
    try:
        np.savez_compressed(npz, mixture=mix_500, fecg=fecg_500)
    except Exception as e:
        print(f'[WARN] Could not write cache {npz}: {e}')
    return mix_500, fecg_500


def _stft_multichannel(signal, nfft, hop_length, win_length):
    """STFT per channel → complex array (C, F, T)."""
    specs = []
    for ch in range(signal.shape[0]):
        S = librosa.stft(signal[ch], n_fft=nfft,
                         hop_length=hop_length, win_length=win_length,
                         center=True)
        specs.append(S)
    return np.stack(specs, axis=0)


class MovementECGDatasetPaper(Dataset):
    """
    Paper-faithful dataset: 500 Hz, 3.83 s windows, natural 128×128 spectrograms.
    Returns (mix_spec, fecg_spec, fecg_time) where:
      mix_spec, fecg_spec: (6, 128, 128) complex tensors (last freq bin dropped)
      fecg_time: (6, 1915) float32 — time-domain target for SignalMSE
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith('.mat')
        )

        # Build index: (file_idx, start_sample) at 500 Hz
        n_500 = N_SAMPLES_ORIG // 2  # 300,000 samples at 500 Hz
        self.windows = []
        for file_idx in range(len(self.files)):
            for start in range(0, n_500 - WINDOW + 1, STRIDE):
                self.windows.append((file_idx, start))

        self._wins_per_file = (n_500 - WINDOW) // STRIDE + 1
        self._cache: dict = {}
        self._bad_files: set = set()

    def _get_signals(self, file_idx):
        if file_idx in self._bad_files:
            raise ValueError(f'Bad file: {self.files[file_idx]}')
        if file_idx not in self._cache:
            try:
                mixture, fecg = _load_signals_500hz(self.files[file_idx])
                self._cache[file_idx] = (mixture, fecg)
            except Exception as e:
                print(f'\n[WARN] Skipping corrupted file '
                      f'{os.path.basename(self.files[file_idx])}: {e}')
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

        end = start + WINDOW
        mix_win = mixture[:, start:end].copy()
        fecg_win = fecg[:, start:end].copy()

        # Per-channel normalization by mixture std
        stds = mix_win.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        mix_win = mix_win / stds
        fecg_win = fecg_win / stds

        # STFT → (6, 129, 128) — naturally 128 time frames at 500 Hz
        mix_spec = _stft_multichannel(mix_win, NFFT, HOP_LENGTH, WIN_LENGTH)
        fecg_spec = _stft_multichannel(fecg_win, NFFT, HOP_LENGTH, WIN_LENGTH)

        # Drop last freq bin: (6, 129, 128) → (6, 128, 128)
        mix_spec = mix_spec[:, :-1, :]
        fecg_spec = fecg_spec[:, :-1, :]

        # Convert to complex tensors
        x = torch.complex(
            torch.from_numpy(np.real(mix_spec).copy()),
            torch.from_numpy(np.imag(mix_spec).copy()),
        )
        y = torch.complex(
            torch.from_numpy(np.real(fecg_spec).copy()),
            torch.from_numpy(np.imag(fecg_spec).copy()),
        )

        fecg_time = torch.from_numpy(fecg_win).float()  # (6, 1915)

        return x, y, fecg_time


if __name__ == '__main__':
    data_dir = '../data/movement_ecg'
    print(f'Building paper-style dataset from {data_dir} ...')
    dset = MovementECGDatasetPaper(data_dir)
    print(f'Total windows: {len(dset)}')
    x, y, fecg_time = dset[0]
    print(f'  x         : {x.shape}  dtype={x.dtype}')
    print(f'  y         : {y.shape}  dtype={y.dtype}')
    print(f'  fecg_time : {fecg_time.shape}  dtype={fecg_time.dtype}')
    print('OK')
