"""Member 2 — T1c unimodal segmentation model.

3D Attention-Gated U-Net for contrast-enhanced T1 (T1c) brain tumour
segmentation. Predicts the three overlapping BraTS sub-regions
(WT, TC, ET) as raw logits.

Why attention gates for T1c
---------------------------
T1c produces sparse, high-intensity enhancing tumour on a comparatively
dark background — the signal of interest is small, bright, and
localised. A plain U-Net's skip connections pass the entire encoder
feature map to the decoder, including large regions of irrelevant
background. Attention gates (Oktay et al. 2018) insert a learned
soft gate on every skip: the decoder's gating signal queries the
encoder feature and produces a per-voxel attention map alpha. The
encoder feature is multiplied by alpha before being concatenated,
which soft-suppresses anatomy that the decoder does not need and
amplifies the bright enhancing-tumour voxels. This is exactly the
inductive bias T1c rewards, and is why M2 is expected to land the
best Dice_ET in the team.

No deep supervision
-------------------
Unlike M1 (which uses deep supervision to force every decoder level
to produce a semantically meaningful mask for the ET-imbalanced T1n
task), M2 does NOT use deep supervision. The attention gates already
provide a strong implicit regulariser by forcing each decoder level
to learn what to keep from its skip. Stacking deep supervision on
top would muddy that signal and add ~3× the auxiliary heads of M1
for no expected gain on a modality where the relevant signal is
already concentrated.

Forward contract
----------------
- ``forward(x)`` returns a single tensor of shape (B, 3, H, W, D)
  containing raw logits for (WT, TC, ET). The mode (train/eval) does
  not change the return type — M2 has no auxiliary outputs.
- ``forward_features(x)`` returns the bottleneck feature map at shape
  (B, 256, H/16, W/16, D/16). This is the binding contract from
  claude.md §2.6b that lets M5's Latent Fusion Head consume M2's
  features. M2's native bottleneck is 256 channels so no projection
  conv is needed.

Output convention
-----------------
The 3-channel output is RAW LOGITS (no sigmoid). ``dice_bce_loss``
applies ``binary_cross_entropy_with_logits`` internally for numerical
stability; sub-regions are overlapping so sigmoid-per-channel — not
softmax — is the correct activation at metric / visualisation time.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionGate3D(nn.Module):
    """3D attention gate (Oktay et al. 2018) used on every U-Net skip.

    Inputs at forward time:
        g — gating signal from the decoder, channels ``F_g``
        x — skip-connection feature from the encoder, channels ``F_l``

    Computation:
        alpha = sigmoid( psi( ReLU( W_g(g) + W_x(x) ) ) )
        out   = x * alpha

    The encoder skip is multiplied by the per-voxel attention map
    before being concatenated, so background activations get
    soft-suppressed before they reach the decoder convs.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm3d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm3d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        alpha = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        return x * alpha


class ConvBlock3D(nn.Module):
    """Two 3×3×3 convs with batch-norm + ReLU. No residual skip — M2 is
    a standard U-Net, not residual (residual blocks were M1's choice for
    the T1n task)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionUNet3D(nn.Module):
    """3D Attention-Gated U-Net for T1c whole-tumour / sub-region seg."""

    def __init__(self, in_channels: int = 1, out_channels: int = 3) -> None:
        super().__init__()

        # Encoder — 4 levels, channels 1 → 32 → 64 → 128 → 256.
        self.enc1 = ConvBlock3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ConvBlock3D(128, 256)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck (256 channels, no projection — already 256).
        self.bottleneck = ConvBlock3D(256, 256)

        # Decoder — 4 levels, channels 256 → 128 → 64 → 32 → 32.
        # Each level: ConvTranspose3d → AttentionGate3D on the skip → concat → ConvBlock3D.
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.att1 = AttentionGate3D(F_g=128, F_l=256, F_int=128)
        self.dec1 = ConvBlock3D(128 + 256, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionGate3D(F_g=64, F_l=128, F_int=64)
        self.dec2 = ConvBlock3D(64 + 128, 64)

        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.att3 = AttentionGate3D(F_g=32, F_l=64, F_int=32)
        self.dec3 = ConvBlock3D(32 + 64, 32)

        self.up4 = nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2)
        self.att4 = AttentionGate3D(F_g=32, F_l=32, F_int=16)
        self.dec4 = ConvBlock3D(32 + 32, 32)

        # Main classifier head — 3-channel logits at full resolution.
        self.out_head = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- Encoder ----
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))

        # ---- Bottleneck ----
        b = self.bottleneck(self.pool4(s4))

        # ---- Decoder (each level: upsample → attention-gate skip → concat → conv) ----
        g1 = self.up1(b)
        s4_att = self.att1(g=g1, x=s4)
        d1 = self.dec1(torch.cat([g1, s4_att], dim=1))

        g2 = self.up2(d1)
        s3_att = self.att2(g=g2, x=s3)
        d2 = self.dec2(torch.cat([g2, s3_att], dim=1))

        g3 = self.up3(d2)
        s2_att = self.att3(g=g3, x=s2)
        d3 = self.dec3(torch.cat([g3, s2_att], dim=1))

        g4 = self.up4(d3)
        s1_att = self.att4(g=g4, x=s1)
        d4 = self.dec4(torch.cat([g4, s1_att], dim=1))

        return self.out_head(d4)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the bottleneck feature map at shape (B, 256, H/16, W/16, D/16).

        Binding contract from claude.md §2.6b — required so M5's Latent
        Fusion Head can consume M2's features. M2's native bottleneck is
        256 channels so no projection is needed.
        """
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        return self.bottleneck(self.pool4(s4))


if __name__ == "__main__":
    torch.manual_seed(42)

    model = AttentionUNet3D(in_channels=1, out_channels=3)
    x = torch.randn(2, 1, 64, 64, 64)

    model.train()
    out = model(x)
    assert isinstance(out, torch.Tensor), f"Expected single tensor, got {type(out)}"
    assert tuple(out.shape) == (2, 3, 64, 64, 64), f"shape {tuple(out.shape)}"

    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert tuple(out_eval.shape) == (2, 3, 64, 64, 64)

    features = model.forward_features(x)
    assert tuple(features.shape) == (2, 256, 4, 4, 4), f"features shape {tuple(features.shape)}"

    print(f"output shape:               {tuple(out.shape)}")
    print(f"forward_features shape:     {tuple(features.shape)}")
    print(f"trainable parameters:       {sum(p.numel() for p in model.parameters()):,}")
    print("smoke test: PASSED")
