"""
ComplexUNet v11 — v9 architecture (1.87M) + direct prediction (fara masca).

Diferenta fata de v9:
  - forward() returneaza x direct dupa diag_out, fara sigmoid mask
  - Nu e constrans la [0,1] per bin — poate prezice amplitudini corecte
  - Antrenat cu SignalMSE + PeakMSE(fqrs), fara ComplexMSE

Motivatie: v9 cu ComplexMSE suprimatamplitudinile (ComplexMSE ~10x > SignalMSE
in loss → modelul optimizeaza mai mult structura spectrala decat amplitudinile).
v11 = arhitectura puternica (1.87M) + obiectiv direct pe amplitudini.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import all building blocks from v9 — identical architecture
from complex_network_v9 import (
    ComplexReLU, GKActivation, RoActivation,
    ComplexConvLayer, DownBlock, UpBlock, Diag, WeightClipper,
)


class ComplexUNetV11(nn.Module):
    """
    Identic cu ComplexUNetV9 dar forward() returneaza predictie directa
    (fara soft mask). Acelasi numar de parametri: ~1.87M.
    """
    def __init__(self, dimension, in_channels=6):
        super().__init__()
        self.W_clipper = WeightClipper(1.0)

        self.diag_in    = Diag(dimension)
        self.conv1      = ComplexConvLayer(in_channels, 16)
        self.down1      = DownBlock(16,  32)
        self.down2      = DownBlock(32,  64)
        self.down3      = DownBlock(64, 128)

        self.bottleneck = nn.Sequential(
            ComplexConvLayer(128, 256),
            ComplexConvLayer(256, 128),
        )

        self.up1     = UpBlock(128 + 128, 64)
        self.up2     = UpBlock( 64 +  64, 32)
        self.up3     = UpBlock( 32 +  32, 16)
        self.conv_out = ComplexConvLayer(16, in_channels, kernel_size=1, padding=0)
        self.diag_out = Diag(dimension)

    def clip_weights(self):
        self.apply(self.W_clipper)

    def forward(self, x):
        x = self.diag_in(x)

        x    = self.conv1(x)
        res1 = self.down1(x)
        res2 = self.down2(res1)
        res3 = self.down3(res2)

        x = self.bottleneck(res3)

        x = self.up1(torch.cat([x,   res3], dim=1))
        x = self.up2(torch.cat([x,   res2], dim=1))
        x = self.up3(torch.cat([x,   res1], dim=1))

        x = self.conv_out(x)
        x = self.diag_out(x)
        return x   # direct prediction — no mask


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, c, h, w = 2, 6, 128, 400
    x = torch.randn(b, c, h, w, dtype=torch.complex64).to(device)
    model = ComplexUNetV11(h * w, in_channels=c).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params / 1e6:.3f} M')
    y = model(x)
    print(f'Output shape: {y.shape}')
