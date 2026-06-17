"""
MovementECGDatasetV12 — ca V10 dar cu DILATION=100 pentru QRSwideMSE.

Dilatare ±100 samples (±100ms la 1kHz) acopera complexul PQRS complet,
inclusiv undele Q si S laterale (nu doar R-peak-ul ±30ms).
"""

import os
import numpy as np
import torch

from movement_dataset_v10 import (
    MovementECGDatasetV10,
    _make_peak_mask,
    _load_fqrs,
)
from movement_dataset import (
    _stft_multichannel, _to_resized_complex_tensor,
)

DILATION = 100   # ±100ms — acopera Q, R, S (vs ±30ms in v10)


class MovementECGDatasetV12(MovementECGDatasetV10):
    """
    Ca MovementECGDatasetV10 dar cu peak_mask dilata la ±100 samples.

    Returns: (x, y, fecg_time, peak_mask)  — identic cu V10 ca forma
    """

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

        end      = start + self.window
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

        fqrs      = self._get_fqrs(file_idx)
        peak_mask = _make_peak_mask(fqrs, start, self.window, DILATION)
        peak_mask = torch.from_numpy(peak_mask)   # (window,)

        return x, y, fecg_time, peak_mask


if __name__ == '__main__':
    dset = MovementECGDatasetV12('../data/movement_ecg')
    print(f'Total windows: {len(dset)}')
    x, y, ft, pm = dset[0]
    print(f'  x         : {x.shape}')
    print(f'  y         : {y.shape}')
    print(f'  fecg_time : {ft.shape}')
    print(f'  peak_mask : {pm.shape}  n_peaks={pm.sum():.0f}  '
          f'coverage={pm.mean()*100:.1f}%  (dilation={DILATION})')
