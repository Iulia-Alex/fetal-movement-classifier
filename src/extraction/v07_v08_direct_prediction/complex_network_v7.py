"""
ComplexUNet v7 — paper architecture adapted for 128×400, direct output (no sigmoid masking).

Differences vs complex_network_paper.py:
  - NO sigmoid mask at output (direct prediction, like v1)
  - NO per-sample normalize/denormalize (dataset normalizes by mixture std)
  - Adapted for 128×400 input (not 128×128)

Paper features kept:
  - Separate conv_real / conv_imag with cross-mixing combine: (r-i) + j(r+i)
  - BatchNorm after each conv (separate for real and imag)
  - RoActivation: learnable convex mix of CReLU + GK + GroupSort
  - Skip connections via CONCATENATION (not addition)
  - Diagonal layers at input/output
  - Weight clipping

Channels (further reduced to allow BS=32 at 128×400):
  6 → 16 → 32 → 64 → 128, bottleneck 128 → 256 → 128
  Original paper: 6 → 64 → 128 → 256 → 512, bottleneck 512 → 1024 → 512
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
class ComplexReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x.real) + 1j * torch.relu(x.imag)


class GKActivation(nn.Module):
    """Phase-preserving: z / (1 + |z|)"""
    def forward(self, x):
        return x / (1.0 + torch.abs(x))


class GroupSortActivation(nn.Module):
    """min(Re, Im) + j·max(Re, Im)"""
    def forward(self, x):
        r, i = x.real, x.imag
        return torch.min(r, i) + 1j * torch.max(r, i)


class RoActivation(nn.Module):
    """Learnable convex combination of CReLU + GK + GroupSort (paper Eq. 6)."""
    def __init__(self):
        super().__init__()
        self.mu_logits = nn.Parameter(torch.randn(3))
        self.crelu = ComplexReLU()
        self.gk    = GKActivation()
        self.gs    = GroupSortActivation()

    def forward(self, x):
        mu = torch.softmax(self.mu_logits, dim=0)
        return mu[0] * self.crelu(x) + mu[1] * self.gk(x) + mu[2] * self.gs(x)


# ---------------------------------------------------------------------------
# Complex conv: separate weights for real/imag, split-complex (no cross-mixing)
# Paper Eq. (1): Y = R_i(W_Re * Re(x) + i * W_Im * Im(x))
# ---------------------------------------------------------------------------
class ComplexConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_real = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn_real   = nn.BatchNorm2d(out_ch)
        self.bn_imag   = nn.BatchNorm2d(out_ch)
        self.act       = RoActivation()

    def forward(self, x):
        r = self.bn_real(self.conv_real(x.real))
        i = self.bn_imag(self.conv_imag(x.imag))
        # Split-complex: real and imag processed independently (paper Eq. 1)
        y = r + 1j * i
        return self.act(y)


# ---------------------------------------------------------------------------
# Diagonal layer
# Paper Eq. (2): Λ = Diag(e^{iβ_1}, ..., e^{iβ_{F×T}}) — phase rotation
# ---------------------------------------------------------------------------
class Diag(nn.Module):
    """Learnable phase rotation: x * e^{iβ} per element. Zeros init → identity."""
    def __init__(self, dimension):
        super().__init__()
        self.betas = nn.Parameter(torch.zeros(dimension))

    def forward(self, x):
        b, c, h, w = x.shape
        beta  = self.betas                        # (F*T,)
        cos_b = torch.cos(beta)
        sin_b = torch.sin(beta)
        xr = x.real.reshape(b * c, h * w)
        xi = x.imag.reshape(b * c, h * w)
        # x * e^{iβ} = (xr·cos - xi·sin) + j*(xr·sin + xi·cos)
        r_out = xr * cos_b - xi * sin_b
        i_out = xr * sin_b + xi * cos_b
        return (r_out + 1j * i_out).reshape(b, c, h, w)


# ---------------------------------------------------------------------------
# Down / Up sampling
# ---------------------------------------------------------------------------
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
# Down / Up blocks
# ---------------------------------------------------------------------------
class DownBlock(nn.Module):
    """Conv then pool."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ComplexConvLayer(in_ch, out_ch)
        self.pool = ComplexMaxPool()

    def forward(self, x):
        return self.pool(self.conv(x))


class UpBlock(nn.Module):
    """Conv then upsample."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ComplexConvLayer(in_ch, out_ch)
        self.up   = ComplexUpsample()

    def forward(self, x):
        return self.up(self.conv(x))


# ---------------------------------------------------------------------------
# Weight clipper
# ---------------------------------------------------------------------------
class WeightClipper:
    def __init__(self, clip_value=1.0):
        self.clip_value = clip_value

    def __call__(self, module):
        if hasattr(module, 'weight'):
            module.weight.data.clamp_(-self.clip_value, self.clip_value)


# ---------------------------------------------------------------------------
# ComplexUNet v7
# ---------------------------------------------------------------------------
class ComplexUNetV7(nn.Module):
    def __init__(self, dimension, in_channels=6):
        super().__init__()
        self.W_clipper = WeightClipper(1.0)

        self.diag_in = Diag(dimension)

        # Encoder
        self.conv1 = ComplexConvLayer(in_channels, 16)   # full res, 16 ch
        self.down1 = DownBlock(16,  32)                  # /2,  32 ch
        self.down2 = DownBlock(32,  64)                  # /4,  64 ch
        self.down3 = DownBlock(64, 128)                  # /8, 128 ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ComplexConvLayer(128, 256),
            ComplexConvLayer(256, 128),
        )

        # Decoder (cat doubles input channels)
        self.up1 = UpBlock(128 + 128, 64)   # cat(bottleneck_128, res3_128) → 64,  /4
        self.up2 = UpBlock( 64 +  64, 32)   # cat(up1_64,  res2_64)         → 32,  /2
        self.up3 = UpBlock( 32 +  32, 16)   # cat(up2_32,  res1_32)         → 16,  /1

        # Output head
        self.conv_out = ComplexConvLayer(16, in_channels, kernel_size=1, padding=0)

        self.diag_out = Diag(dimension)

    def clip_weights(self):
        self.apply(self.W_clipper)

    def forward(self, x):
        x = self.diag_in(x)

        # Encoder — save skip connections
        x    = self.conv1(x)         # 32 ch, full res
        res1 = self.down1(x)         # 64 ch, /2
        res2 = self.down2(res1)      # 128 ch, /4
        res3 = self.down3(res2)      # 256 ch, /8

        # Bottleneck
        x = self.bottleneck(res3)    # 256 ch, /8

        # Decoder with concatenation skips
        x = self.up1(torch.cat([x,    res3], dim=1))   # 512 → 128, → /4
        x = self.up2(torch.cat([x,    res2], dim=1))   # 256 →  64, → /2
        x = self.up3(torch.cat([x,    res1], dim=1))   # 128 →  32, → /1

        x = self.conv_out(x)         # 32 → in_channels, full res
        x = self.diag_out(x)
        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, c, h, w = 2, 6, 128, 400
    x = torch.randn(b, c, h, w, dtype=torch.complex64).to(device)
    model = ComplexUNetV7(h * w, in_channels=c).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params / 1e6:.2f} M')
    y = model(x)
    print(f'Output shape: {y.shape}')
