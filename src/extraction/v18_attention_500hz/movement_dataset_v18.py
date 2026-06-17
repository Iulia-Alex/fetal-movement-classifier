"""
Dataset v18 — 1 kHz, 4-second windows, STFT (6, 128, 400).

Re-exports movement_dataset.py with explicit constants for v18.
Differences from v15/v16/v17 (500 Hz, 128×128):
  - No decimation: signal stays at 1000 Hz.
  - STFT: NFFT=256, HOP=10, WIN=128 → (129, ~401) → bilinear resize → (128, 400).
  - Temporal resolution: 10 ms/frame (vs 30 ms/frame at 500 Hz).
  - QRS complex (~80-100 ms) = ~8-10 frames (vs 2-3 at 500 Hz).
"""

from movement_dataset import (
    MovementECGDataset          as MovementECGDatasetV18,
    FS, WINDOW_SEC,
    NFFT, HOP_LENGTH, WIN_LENGTH,
    TARGET_SIZE_F, TARGET_SIZE_T,
    N_SAMPLES_TOTAL,
    _extract_fecg, _stft_multichannel, _to_resized_complex_tensor,
)
import numpy as np

WINDOW = int(WINDOW_SEC * FS)          # 4000 samples
ORIG_F = NFFT // 2 + 1                 # 129 freq bins before resize

__all__ = [
    'MovementECGDatasetV18',
    'FS', 'WINDOW_SEC', 'WINDOW',
    'NFFT', 'HOP_LENGTH', 'WIN_LENGTH',
    'TARGET_SIZE_F', 'TARGET_SIZE_T', 'ORIG_F',
]

if __name__ == '__main__':
    import os
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'movement_ecg')
    dset = MovementECGDatasetV18(data_dir)
    print(f'Total windows: {len(dset)}')
    x, y, t = dset[0]
    print(f'  x: {x.shape}  y: {y.shape}  t: {t.shape}')
    print('OK')
