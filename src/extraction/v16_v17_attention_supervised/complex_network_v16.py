"""
ComplexAttentionUNet (v16) — paper backbone (v15) + Attention Gates pe skip connections.

Architectural inspiration:
  - Backbone: our paper (v15): cross-mixing ComplexConvLayer, exp(β) Diag,
    RoActivation, sigmoid mask output, per-sample normalize/denormalize.
  - Attention Gates: Schlemper et al. 2019 "Attention U-Net: Learning Where to Look
    for the Pancreas" (https://github.com/ozan-oktay/Attention-Gated-Networks).
    Adapted for complex-valued features: α computed on the magnitudes (|x|),
    applied to the complex signal: attended_x = α ⊙ x.

Differences from v15:
  - Skip connections are filtered through an Attention Gate before concatenation.
  - The AG suppresses irrelevant skip features using the decoder signal (gating) as reference.
  - Magnitude-based attention — rationale: the magnitude captures "how much" of
    a feature is present, the attention decides whether it is relevant (not polarity/phase).
  - ~200K additional parameters (AG3+AG2+AG1) vs v15 (~7.13M → ~7.33M total).

Architecture:
  Input (B,6,128,128) complex
    │
    ├─ normalize + diag_in
    │
    ├─ conv1: 6→32  @128            [first feature extraction]
    ├─ down1: 32→64  @64   = res1   [skip1]
    ├─ down2: 64→128 @32   = res2   [skip2]
    ├─ down3: 128→256@16   = res3   [skip3]
    │
    ├─ bottleneck: 256→512→256 @16
    │
    ├─ AG3(skip=res3, gate=bottleneck) → attended_res3
    ├─ up1: cat(bottleneck+attended_res3)=512 → 128 @32
    │
    ├─ AG2(skip=res2, gate=up1) → attended_res2
    ├─ up2: cat(up1+attended_res2)=256 → 64 @64
    │
    ├─ AG1(skip=res1, gate=up2) → attended_res1
    ├─ up3: cat(up2+attended_res1)=128 → 32 @128
    │
    ├─ conv2: 32→16, conv3: 16→6
    ├─ diag_out
    └─ sigmoid mask × normalized_input + denormalize
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Activations (same as v15 / paper)
# ---------------------------------------------------------------------------
class ComplexReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x.real) + 1j * self.relu(x.imag)


class GKActivation(nn.Module):
    def forward(self, x):
        return x / (1 + torch.abs(x))


class GroupSortActivation(nn.Module):
    def forward(self, x):
        r, i = x.real, x.imag
        return torch.min(r, i) + 1j * torch.max(r, i)


class RoActivation(nn.Module):
    """Learnable convex mix of CReLU + GK + GroupSort (paper Eq. 6)."""
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
# Complex convolution cu cross-mixing (paper Section 2.2.3)
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
        # Paper cross-mixing: (r - i) + 1j*(r + i)
        y = (r - i) + 1j * (r + i)
        return self.activation(y)


# ---------------------------------------------------------------------------
# Diag layer cu exp(β) scaling (paper Section 2.2.2)
# ---------------------------------------------------------------------------
class Diag(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.betas_real = nn.Parameter(torch.ones(dimension))
        self.betas_imag = nn.Parameter(torch.ones(dimension))

    def forward(self, x):
        b, c, h, w = x.shape
        r = x.real.reshape(b * c, h * w) * torch.exp(self.betas_real)
        i = x.imag.reshape(b * c, h * w) * torch.exp(self.betas_imag)
        return (r + 1j * i).reshape(b, c, h, w)


# ---------------------------------------------------------------------------
# Down/Up sampling (same as v15)
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
        self.up   = ComplexUpSample()

    def forward(self, x):
        return self.up(self.conv(x))


# ---------------------------------------------------------------------------
# Attention Gate (new in v16)
#
# Inspiration: Schlemper et al. 2019 "Attention U-Net"
#   https://github.com/ozan-oktay/Attention-Gated-Networks
#
# Complex adaptation: α computed on magnitudes |skip| and |gate|,
# applied as a real scalar to the complex skip signal:
#   attended_x = α(|skip|, |gate|) ⊙ skip
#
# Rationale:
#   - |z| = amplitude (how strong the feature is)
#   - attention decides if the amplitude is useful, not the phase
#   - phase (QRS morphology, direction) is preserved unchanged
# ---------------------------------------------------------------------------
class ComplexAttentionGate(nn.Module):
    """
    Additive attention gate for complex features.
    skip and gate are at the same spatial resolution.

    Args:
        F_x   : channels in skip connection (encoder)
        F_g   : channels in gating signal (decoder)
        F_int : intermediate channels (usually max(F_x,F_g)//2)
    """
    def __init__(self, F_x: int, F_g: int, F_int: int):
        super().__init__()
        # θ_x: projects skip magnitude onto the intermediate space
        self.theta_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        # θ_g: projects gating-signal magnitude onto the intermediate space
        self.theta_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        # ψ: attention coefficient in [0,1] per spatial location (1 channel — channel-agnostic)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_skip: torch.Tensor, x_gate: torch.Tensor) -> torch.Tensor:
        """
        x_skip : complex (B, F_x, H, W) — skip from encoder
        x_gate : complex (B, F_g, H, W) — gating signal from decoder (same resolution)
        Returns: complex (B, F_x, H, W) — attended skip
        """
        # Real-valued magnitudes for attention computation
        mag_s = torch.abs(x_skip)   # (B, F_x, H, W)
        mag_g = torch.abs(x_gate)   # (B, F_g, H, W)

        # Independent projections + additive combination (Eq. from paper)
        theta_s = self.theta_x(mag_s)          # (B, F_int, H, W)
        theta_g = self.theta_g(mag_g)          # (B, F_int, H, W)
        alpha   = self.psi(self.relu(theta_s + theta_g))  # (B, 1, H, W), in [0,1]

        # Apply attention to the complex signal — broadcast over channels
        return alpha * x_skip


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
# ComplexAttentionUNet — v16
# ---------------------------------------------------------------------------
class ComplexAttentionUNet(nn.Module):
    """
    Paper-faithful ComplexUNet (v15 backbone) with Attention Gates
    on all 3 skip connections.

    Parameters ~ 7.33M (v15: 7.13M + ~200K for 3 AGs).
    """
    def __init__(self, dimension: int, in_channels: int = 6):
        super().__init__()
        self.W_clipper = WeightClipper(1.0)

        self.diag_in  = Diag(dimension)
        self.diag_out = Diag(dimension)

        # Encoder (same as v15)
        self.conv1 = ComplexConvLayer(in_channels, 32)
        self.down1 = ComplexDownBlock(32,  64)    # → res1: 64ch  @H/2
        self.down2 = ComplexDownBlock(64,  128)   # → res2: 128ch @H/4
        self.down3 = ComplexDownBlock(128, 256)   # → res3: 256ch @H/8

        # Bottleneck (same as v15)
        self.bottleneck = nn.Sequential(
            ComplexConvLayer(256, 512),
            ComplexConvLayer(512, 256),
        )

        # ── Attention Gates (NOVITATE v16) ──────────────────────────────────
        # AG3: skip=res3 (256ch), gate=bottleneck (256ch), intermediate=128
        self.ag3 = ComplexAttentionGate(F_x=256, F_g=256, F_int=128)
        # AG2: skip=res2 (128ch), gate=up1_out (128ch), intermediate=64
        self.ag2 = ComplexAttentionGate(F_x=128, F_g=128, F_int=64)
        # AG1: skip=res1 (64ch),  gate=up2_out (64ch),  intermediate=32
        self.ag1 = ComplexAttentionGate(F_x=64,  F_g=64,  F_int=32)
        # ────────────────────────────────────────────────────────────────────

        # Decoder (same structure as v15, but receives attended skip)
        self.up1 = ComplexUpBlock(512, 128)   # cat(bottleneck_256 + attended_res3_256)
        self.up2 = ComplexUpBlock(256, 64)    # cat(up1_128 + attended_res2_128)
        self.up3 = ComplexUpBlock(128, 32)    # cat(up2_64 + attended_res1_64)

        # Output head (same as v15)
        self.conv2 = ComplexConvLayer(32, 16)
        self.conv3 = ComplexConvLayer(16, in_channels, kernel_size=1, padding=0)

    def normalize(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std  = x.std(dim=(1, 2, 3), keepdim=True)
        self._mean = mean
        self._std  = std
        return (x - mean) / (std + 1e-6)

    def denormalize(self, x):
        return x * self._std + self._mean

    def forward(self, x):
        # Per-sample normalization (paper)
        x    = self.normalize(x)
        init = x

        x = self.diag_in(x)

        # ── Encoder ──────────────────────────────────────────────────────────
        x    = self.conv1(x)      # (B, 32, H,   W)
        res1 = self.down1(x)      # (B, 64, H/2, W/2)
        res2 = self.down2(res1)   # (B, 128, H/4, W/4)
        res3 = self.down3(res2)   # (B, 256, H/8, W/8)

        # ── Bottleneck ───────────────────────────────────────────────────────
        x = self.bottleneck(res3)  # (B, 256, H/8, W/8)

        # ── Decoder with Attention Gates ─────────────────────────────────────
        # Level 3: gate = bottleneck output, skip = res3
        res3_att = self.ag3(x_skip=res3, x_gate=x)
        x = self.up1(torch.cat([x, res3_att], dim=1))   # (B, 128, H/4, W/4)

        # Level 2: gate = up1 output, skip = res2
        res2_att = self.ag2(x_skip=res2, x_gate=x)
        x = self.up2(torch.cat([x, res2_att], dim=1))   # (B, 64, H/2, W/2)

        # Level 1: gate = up2 output, skip = res1
        res1_att = self.ag1(x_skip=res1, x_gate=x)
        x = self.up3(torch.cat([x, res1_att], dim=1))   # (B, 32, H, W)

        # ── Output head ──────────────────────────────────────────────────────
        x = self.conv2(x)   # (B, 16, H, W)
        x = self.conv3(x)   # (B, in_ch, H, W)
        x = self.diag_out(x)

        # Sigmoid mask × normalized input (paper output strategy)
        mask = torch.sigmoid(x.real) + 1j * torch.sigmoid(x.imag)
        x = mask * init

        return self.denormalize(x)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, c, h, w = 2, 6, 128, 128
    x = (torch.randn(b, c, h, w) + 1j * torch.randn(b, c, h, w)).to(device)
    model = ComplexAttentionUNet(h * w, in_channels=c).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params / 1e6:.3f} M')
    y = model(x)
    print(f'Input:  {x.shape}, Output: {y.shape}')
    # Verifica dimensiunile attention maps
    print('Forward pass OK.')
