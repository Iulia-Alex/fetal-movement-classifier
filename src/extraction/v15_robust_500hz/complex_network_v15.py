"""
Paper-faithful ComplexUNet adapted for 6-channel input.

Key features matching src/network.py:
  - sameW=False: separate conv_real / conv_imag with cross-mixing combine
  - BatchNorm after each conv
  - 1 conv layer per DownBlock / UpBlock (reduced from 3 to fit M4000 8GB)
  - Skip connections via concatenation (not addition)
  - RoActivation: learnable convex mix of CReLU + GK + GroupSort
  - Diagonal layer with separate real/imag betas
  - Sigmoid mask × normalised input at output
  - Per-sample normalize / denormalize
  - Weight clipping

Channels (reduced to fit M4000 8GB):
  6→32→64→128→256, bottleneck 256→512→256
  Original paper: 6→64→128→256→512, bottleneck 512→1024→512
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Activation functions (paper Section 2.2.4)
# ---------------------------------------------------------------------------
class ComplexReLU(nn.Module):
    """Split-complex ReLU: ReLU applied independently to real and imag."""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x.real) + 1j * self.relu(x.imag)


class GKActivation(nn.Module):
    """Georgiou-Katsageras: phase-preserving, z / (1 + |z|)."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (1 + torch.abs(x))


class GroupSortActivation(nn.Module):
    """GroupSort: min(Re,Im) + 1j * max(Re,Im)."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        r, i = x.real, x.imag
        return torch.min(r, i) + 1j * torch.max(r, i)


class RoActivation(nn.Module):
    """Learnable convex combination of CReLU + GK + GroupSort (Eq. 6)."""
    def __init__(self):
        super().__init__()
        self.mu_logits = nn.Parameter(torch.randn(3))
        self.crelu = ComplexReLU()
        self.gk = GKActivation()
        self.gs = GroupSortActivation()

    def forward(self, x):
        mu = torch.softmax(self.mu_logits, dim=0)
        return mu[0] * self.crelu(x) + mu[1] * self.gk(x) + mu[2] * self.gs(x)


# ---------------------------------------------------------------------------
# Complex convolution with separate real/imag weights + cross-mixing
# ---------------------------------------------------------------------------
class ComplexConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, activation=RoActivation):
        super().__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm_real = nn.BatchNorm2d(out_channels)
        self.norm_imag = nn.BatchNorm2d(out_channels)
        self.activation = activation()

    def forward(self, x):
        r = self.norm_real(self.conv_real(x.real))
        i = self.norm_imag(self.conv_imag(x.imag))
        # Paper's complex combine: (r - i) + 1j*(r + i)
        y = (r - i) + 1j * (r + i)
        return self.activation(y)


# ---------------------------------------------------------------------------
# Diagonal layer (efficient element-wise version of paper's Diag)
# ---------------------------------------------------------------------------
class Diag(nn.Module):
    """Learnable diagonal scaling, separate for real and imag."""
    def __init__(self, dimension):
        super().__init__()
        self.betas_real = nn.Parameter(torch.ones(dimension))
        self.betas_imag = nn.Parameter(torch.ones(dimension))

    def forward(self, x):
        b, c, h, w = x.shape
        scale_r = torch.exp(self.betas_real)
        scale_i = torch.exp(self.betas_imag)
        r = x.real.reshape(b * c, h * w) * scale_r
        i = x.imag.reshape(b * c, h * w) * scale_i
        return (r + 1j * i).reshape(b, c, h, w)


# ---------------------------------------------------------------------------
# Down/Up sampling
# ---------------------------------------------------------------------------
class ComplexDownSample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.pool = nn.MaxPool2d(scale_factor)

    def forward(self, x):
        return self.pool(x.real) + 1j * self.pool(x.imag)


class ComplexUpSample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')

    def forward(self, x):
        return self.up(x.real) + 1j * self.up(x.imag)


# ---------------------------------------------------------------------------
# Down/Up blocks with 3 conv layers each (paper architecture)
# ---------------------------------------------------------------------------
class ComplexDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ComplexConvLayer(in_ch, out_ch)
        self.down = ComplexDownSample()

    def forward(self, x):
        return self.down(self.conv(x))


class ComplexUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ComplexConvLayer(in_ch, out_ch)
        self.up = ComplexUpSample()

    def forward(self, x):
        return self.up(self.conv(x))


# ---------------------------------------------------------------------------
# Weight clipper (as in paper)
# ---------------------------------------------------------------------------
class WeightClipper:
    def __init__(self, clip_value=1.0):
        self.clip_value = clip_value

    def __call__(self, module):
        if hasattr(module, 'weight'):
            module.weight.data.clamp_(-self.clip_value, self.clip_value)


# ---------------------------------------------------------------------------
# ComplexUNet — paper architecture adapted for 6-channel input
# ---------------------------------------------------------------------------
class ComplexUNet(nn.Module):
    def __init__(self, dimension, in_channels=6):
        super().__init__()
        self.W_clipper = WeightClipper(1.0)

        # Input diagonal layer
        self.diag_in = Diag(dimension)

        # Encoder (halved channels to fit M4000 8GB)
        self.conv1 = ComplexConvLayer(in_channels, 32)
        self.down1 = ComplexDownBlock(32, 64)
        self.down2 = ComplexDownBlock(64, 128)
        self.down3 = ComplexDownBlock(128, 256)

        # Bottleneck (no downsampling)
        self.bottleneck = nn.Sequential(
            ComplexConvLayer(256, 512),
            ComplexConvLayer(512, 256),
        )

        # Decoder — input channels are doubled due to concatenation
        self.up1 = ComplexUpBlock(512, 128)    # cat(256 + 256) → 128
        self.up2 = ComplexUpBlock(256, 64)     # cat(128 + 128) → 64
        self.up3 = ComplexUpBlock(128, 32)     # cat(64 + 64) → 32

        # Output head
        self.conv2 = ComplexConvLayer(32, 16)
        self.conv3 = ComplexConvLayer(16, in_channels, kernel_size=1, padding=0)

        # Output diagonal layer
        self.diag_out = Diag(dimension)

    def normalize(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        self._mean = mean
        self._std = std
        return (x - mean) / (std + 1e-6)

    def denormalize(self, x):
        return x * self._std + self._mean

    def forward(self, x):
        # Per-sample normalization (paper approach)
        x = self.normalize(x)
        init = x

        x = self.diag_in(x)

        # Encoder
        x = self.conv1(x)
        res1 = self.down1(x)     # 64 ch, /2 spatial
        res2 = self.down2(res1)  # 128 ch, /4 spatial
        res3 = self.down3(res2)  # 256 ch, /8 spatial

        # Bottleneck
        x = self.bottleneck(res3)  # 256 ch, /8 spatial

        # Decoder with concatenation skip connections
        x = self.up1(torch.cat([x, res3], dim=1))   # 512→128, /4
        x = self.up2(torch.cat([x, res2], dim=1))    # 256→64,  /2
        x = self.up3(torch.cat([x, res1], dim=1))    # 128→32,  /1

        x = self.conv2(x)   # 32→16
        x = self.conv3(x)   # 16→in_channels

        x = self.diag_out(x)

        # Sigmoid mask × normalised input (paper approach)
        mask = torch.sigmoid(x.real) + 1j * torch.sigmoid(x.imag)
        x = mask * init

        x = self.denormalize(x)
        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, c, h, w = 2, 6, 128, 128
    x = torch.randn(b, c, h, w) + 1j * torch.randn(b, c, h, w)
    x = x.to(device)
    model = ComplexUNet(h * w, in_channels=c).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params / 1e6:.2f} M')
    y = model(x)
    print(f'Output: {y.shape}')
