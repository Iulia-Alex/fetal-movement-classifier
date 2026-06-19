"""
ComplexUNet v10 — v1 architecture (0.59M) + soft mask output.

Differences from v9 (1.87M, RoActivation, concat skip):
  - Shared conv weights for Re and Im (sameW=True) → fewer parameters
  - LeakyReLU(0.2) instead of RoActivation → simpler, converges more easily
  - Skip connections via ADDITION (not concatenation) → decoder input = same dimension
  - Diag: magnitude scaling exp(β) (not phase rotation)
  - ~0.59M parameters

Why we start from v1:
  v1 was already good visually. v10 keeps exactly v1's architecture but adds
  peak loss at the fqrs positions in the training script.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
class ComplexLeakyReLU(nn.Module):
    """LeakyReLU(0.2) applied independently on Re and Im."""
    def __init__(self):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        return self.act(x.real) + 1j * self.act(x.imag)


class ComplexConvLayer(nn.Module):
    """
    Complex convolution with SHARED weights for Re and Im (sameW=True from v1).
    The same Conv2d applied to Re(x) and Im(x) separately.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = ComplexLeakyReLU()

    def forward(self, x):
        r = self.bn(self.conv(x.real))
        i = self.bn(self.conv(x.imag))
        return self.act(r + 1j * i)


class ComplexMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(x.real) + 1j * self.pool(x.imag)


class ComplexUpsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.up(x.real) + 1j * self.up(x.imag)


# ---------------------------------------------------------------------------
class DiagMag(nn.Module):
    """
    Diag din v1: scalare magnitudine per element.
    x * exp(β) — zero-init → identitate la start.
    """
    def __init__(self, dimension):
        super().__init__()
        self.betas = nn.Parameter(torch.zeros(dimension))

    def forward(self, x):
        b, c, h, w = x.shape
        scale = torch.exp(self.betas)              # (F*T,)
        xr = x.real.reshape(b * c, h * w) * scale
        xi = x.imag.reshape(b * c, h * w) * scale
        return (xr + 1j * xi).reshape(b, c, h, w)


# ---------------------------------------------------------------------------
class WeightClipper:
    def __init__(self, clip_value=1.0):
        self.clip_value = clip_value

    def __call__(self, module):
        if hasattr(module, 'weight'):
            module.weight.data.clamp_(-self.clip_value, self.clip_value)


# ---------------------------------------------------------------------------
class ComplexUNetV10(nn.Module):
    """
    v1-style ComplexUNet cu:
      - Shared weights Re/Im
      - LeakyReLU(0.2)
      - Skip via ADUNARE (nu concat)
      - Soft mask output: sigmoid(logits) * mixture_input
      - ~0.59M parametri
    """
    def __init__(self, dimension, in_channels=6):
        super().__init__()
        self.W_clipper = WeightClipper(1.0)

        self.diag_in = DiagMag(dimension)

        # Encoder
        self.conv1 = ComplexConvLayer(in_channels, 32)      # (6→32, full res)
        self.pool1 = ComplexMaxPool()                        # /2
        self.conv2 = ComplexConvLayer(32, 64)               # (32→64, /2)
        self.pool2 = ComplexMaxPool()                        # /4
        self.conv3 = ComplexConvLayer(64, 128)              # (64→128, /4)
        self.pool3 = ComplexMaxPool()                        # /8

        # Bottleneck
        self.bottleneck = ComplexConvLayer(128, 128)         # (128→128, /8)

        # Decoder — skip via addition (input = up(x) + skip, same dim)
        self.up3   = ComplexUpsample()                       # /8 → /4
        self.conv4 = ComplexConvLayer(128, 64)               # 128→64
        self.up2   = ComplexUpsample()                       # /4 → /2
        self.conv5 = ComplexConvLayer(64, 32)                # 64→32
        self.up1   = ComplexUpsample()                       # /2 → full
        self.conv6 = ComplexConvLayer(32, in_channels, kernel_size=1, padding=0)

        self.diag_out = DiagMag(dimension)

    def clip_weights(self):
        self.apply(self.W_clipper)

    def forward(self, x):
        x_in = x                    # save input for masking

        x = self.diag_in(x)

        # Encoder
        s1 = self.conv1(x)          # (32, full)
        s2 = self.conv2(self.pool1(s1))   # (64, /2)
        s3 = self.conv3(self.pool2(s2))   # (128, /4)

        # Bottleneck
        x = self.bottleneck(self.pool3(s3))   # (128, /8)

        # Decoder with skip via addition
        x = self.conv4(self.up3(x) + s3)      # (64, /4)
        x = self.conv5(self.up2(x) + s2)      # (32, /2)
        x = self.conv6(self.up1(x) + s1)      # (in_ch, full)

        x = self.diag_out(x)

        # Soft mask: sigmoid(logits) × mixture_input
        mask = torch.sigmoid(x.real) + 1j * torch.sigmoid(x.imag)
        return mask * x_in


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, c, h, w = 2, 6, 128, 400
    x = torch.randn(b, c, h, w, dtype=torch.complex64).to(device)
    model = ComplexUNetV10(h * w, in_channels=c).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params / 1e6:.3f} M')
    y = model(x)
    print(f'Output shape: {y.shape}')
    print(f'Output real range: [{y.real.min().item():.3f}, {y.real.max().item():.3f}]')
