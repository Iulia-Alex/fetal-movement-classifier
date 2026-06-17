"""
MovementECGDatasetV10 — extinde dataset-ul de baza cu fqrs peak mask.

Returnează (x, y, fecg_time, peak_mask) unde:
  peak_mask : (WINDOW_SAMPLES,) float32 — 1.0 în jurul fiecare R-peak fetal,
              0.0 în rest. Folosit pentru peak loss la pozițiile exacte fqrs.

Peak mask: dilatare ±DILATION samples (30ms la 1kHz) în jurul fqrs.
"""

import os
import numpy as np
import torch
import scipy.io as sio

from movement_dataset import (
    MovementECGDataset, _npz_path,
    NFFT, HOP_LENGTH, WIN_LENGTH, TARGET_SIZE_F, TARGET_SIZE_T, FS,
    WINDOW_SEC, STRIDE_SEC,
)

DILATION = 30   # ±30 samples = ±30ms around each R-peak


def _fqrs_npz_path(mat_path):
    base = os.path.splitext(mat_path)[0]
    return base + '_fqrs.npz'


def _load_fqrs(mat_path):
    """Load fqrs from .mat or from .npz cache."""
    cache = _fqrs_npz_path(mat_path)
    if os.path.exists(cache):
        return np.load(cache)['fqrs']
    mat = sio.loadmat(mat_path)
    raw = mat['out']['fqrs'][0][0]
    fqrs = raw.flat[0].astype(np.int32).flatten()
    del mat
    try:
        np.savez_compressed(cache, fqrs=fqrs)
    except Exception as e:
        print(f'[WARN] Could not write fqrs cache {cache}: {e}')
    return fqrs


def _make_peak_mask(fqrs, start, window, dilation=DILATION):
    """Build a peak mask of shape (window,) with 1.0 around each R-peak."""
    mask = np.zeros(window, dtype=np.float32)
    # Select peaks that fall inside the current window
    local = fqrs[(fqrs >= start) & (fqrs < start + window)] - start
    for p in local:
        lo = max(0, p - dilation)
        hi = min(window, p + dilation + 1)
        mask[lo:hi] = 1.0
    return mask


class MovementECGDatasetV10(MovementECGDataset):
    """
    Ca MovementECGDataset dar returnează și fqrs peak mask.

    Returns: (x, y, fecg_time, peak_mask)
      x         : (6, 128, 400) complex tensor — mixture spectrogram
      y         : (6, 128, 400) complex tensor — fecg spectrogram
      fecg_time : (6, window)   float32       — fecg time domain
      peak_mask : (window,)     float32       — 1 langa R-peaks, 0 in rest
    """

    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir, **kwargs)
        # Cache fqrs per file: {file_idx: fqrs_array}
        self._fqrs_cache: dict = {}

    def _get_fqrs(self, file_idx):
        if file_idx not in self._fqrs_cache:
            try:
                fqrs = _load_fqrs(self.files[file_idx])
                self._fqrs_cache[file_idx] = fqrs
            except Exception as e:
                print(f'[WARN] Could not load fqrs for '
                      f'{os.path.basename(self.files[file_idx])}: {e}')
                self._fqrs_cache[file_idx] = np.array([], dtype=np.int32)
        return self._fqrs_cache[file_idx]

    def __getitem__(self, idx):
        # Obtine x, y, fecg_time de la parinte
        # Need file_idx and start to build the peak_mask
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

        # Reimplementam getitem ca sa avem file_idx si start
        import numpy as np
        from movement_dataset import _stft_multichannel, _to_resized_complex_tensor
        import torch

        end = start + self.window
        mix_win  = mixture[:, start:end].copy()
        fecg_win = fecg[:,   start:end].copy()

        stds = mix_win.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        mix_win  = mix_win  / stds
        fecg_win = fecg_win / stds

        mix_spec  = _stft_multichannel(mix_win,  self.nfft, self.hop_length, self.win_length)
        fecg_spec = _stft_multichannel(fecg_win, self.nfft, self.hop_length, self.win_length)

        x = _to_resized_complex_tensor(mix_spec,  self.target_h, self.target_w)
        y = _to_resized_complex_tensor(fecg_spec, self.target_h, self.target_w)

        fecg_time = torch.from_numpy(fecg_win).float()

        # Peak mask
        fqrs      = self._get_fqrs(file_idx)
        peak_mask = _make_peak_mask(fqrs, start, self.window, DILATION)
        peak_mask = torch.from_numpy(peak_mask)   # (window,)

        return x, y, fecg_time, peak_mask


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    dset = MovementECGDatasetV10('../data/movement_ecg')
    print(f'Total windows: {len(dset)}')
    x, y, ft, pm = dset[0]
    print(f'  x         : {x.shape}')
    print(f'  y         : {y.shape}')
    print(f'  fecg_time : {ft.shape}')
    print(f'  peak_mask : {pm.shape}  n_peaks={pm.sum():.0f}  '
          f'coverage={pm.mean()*100:.1f}%')
