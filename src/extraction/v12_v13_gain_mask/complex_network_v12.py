"""
ComplexUNet v12 — v9 architecture + gain mask (1.5 × sigmoid).

Difference from v9:
  - mask = 1.5 * sigmoid(logits)  ∈ [0, 1.5]  (not [0, 1])
  - Allows energy recovery at bins with destructive interference
    (where fECG/mixture_spec > 1 — a pure sigmoid cannot reach there)

Loss (in train_movement_v12.py):
  - SignalMSE + ALPHA * QRSwideMSE (fqrs ±100 samples)
  - No ComplexMSE (the main cause of amplitude suppression in v9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
class ComplexReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x.real) + 1j * torch.relu(x.imag)


class GKActivation(nn.Module):
    def forward(self, x):
        return x / (1.0 + torch.abs(x))


class GroupSortActivation(nn.Module):
    def forward(self, x):
        r, i = x.real, x.imag
        return torch.min(r, i) + 1j * torch.max(r, i)


class RoActivation(nn.Module):
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
# Complex conv
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
        return self.act(r + 1j * i)


# ---------------------------------------------------------------------------
# Diagonal layer
# ---------------------------------------------------------------------------
class Diag(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.betas = nn.Parameter(torch.zeros(dimension))

    def forward(self, x):
        b, c, h, w = x.shape
        cos_b = torch.cos(self.betas)
        sin_b = torch.sin(self.betas)
        xr = x.real.reshape(b * c, h * w)
        xi = x.imag.reshape(b * c, h * w)
        r_out = xr * cos_b - xi * sin_b
        i_out = xr * sin_b + xi * cos_b
        return (r_out + 1j * i_out).reshape(b, c, h, w)


# ---------------------------------------------------------------------------
# Down / Up
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


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ComplexConvLayer(in_ch, out_ch)
        self.pool = ComplexMaxPool()

    def forward(self, x):
        return self.pool(self.conv(x))


class UpBlock(nn.Module):
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
# ComplexUNet v12 — v9 arch + gain mask 1.5 × sigmoid
# ---------------------------------------------------------------------------
GAIN = 1.5   # mask range [0, GAIN] — allows energy recovery in anti-phase bins


class ComplexUNetV12(nn.Module):
    def __init__(self, dimension, in_channels=6):
        super().__init__()
        self.W_clipper = WeightClipper(1.0)

        self.diag_in = Diag(dimension)

        # Encoder
        self.conv1 = ComplexConvLayer(in_channels, 16)
        self.down1 = DownBlock(16,  32)
        self.down2 = DownBlock(32,  64)
        self.down3 = DownBlock(64, 128)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ComplexConvLayer(128, 256),
            ComplexConvLayer(256, 128),
        )

        # Decoder
        self.up1 = UpBlock(128 + 128, 64)
        self.up2 = UpBlock( 64 +  64, 32)
        self.up3 = UpBlock( 32 +  32, 16)

        # Output head
        self.conv_out = ComplexConvLayer(16, in_channels, kernel_size=1, padding=0)
        self.diag_out = Diag(dimension)

    def clip_weights(self):
        self.apply(self.W_clipper)

    def forward(self, x):
        x_in = x

        x = self.diag_in(x)

        # Encoder
        x    = self.conv1(x)
        res1 = self.down1(x)
        res2 = self.down2(res1)
        res3 = self.down3(res2)

        # Bottleneck
        x = self.bottleneck(res3)

        # Decoder
        x = self.up1(torch.cat([x,    res3], dim=1))
        x = self.up2(torch.cat([x,    res2], dim=1))
        x = self.up3(torch.cat([x,    res1], dim=1))

        x = self.conv_out(x)
        x = self.diag_out(x)

        # Gain mask: 1.5 * sigmoid(logits) ∈ [0, 1.5]
        mask = GAIN * torch.sigmoid(x.real) + 1j * GAIN * torch.sigmoid(x.imag)
        return mask * x_in


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, c, h, w = 2, 6, 128, 400
    x = torch.randn(b, c, h, w, dtype=torch.complex64).to(device)
    model = ComplexUNetV12(h * w, in_channels=c).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params / 1e6:.2f} M')
    y = model(x)
    print(f'Output shape: {y.shape}')
    print(f'Output real range: [{y.real.min().item():.3f}, {y.real.max().item():.3f}]')
    print(f'Mask range (should be ~[0, 1.5]): gain={GAIN}')
