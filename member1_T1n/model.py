"""Member 1 — T1n unimodal segmentation model.

3D Residual U-Net with deep supervision for binary whole-tumour segmentation
on the T1n (native T1) modality.

Why residual blocks for T1n
---------------------------
T1n provides smooth multi-scale anatomical contrast but no contrast-agent
enhancement. A deep encoder is needed to pick up subtle tissue boundaries.
In a 4-level 3D net trained on patches with <1 % positive voxels, gradient
vanishing in the deeper layers is the typical failure mode. Identity skip
connections inside each block let gradients flow through the addition path
even when the conv path is saturated, so the deeper levels actually learn.
The 1×1×1 channel-matching projection on the skip keeps the addition
shape-safe whenever the block changes channel count.

Why deep supervision
--------------------
Binary masks where positive voxels are <0.5 % of the volume produce weak
gradients at the deepest decoder layers. Deep supervision attaches
auxiliary 1×1×1 classifiers to decoder levels 2, 3, 4 and folds them into
the loss with weights [0.5, 0.25, 0.125] (see plan.md). Each decoder level
is then forced to produce a semantically meaningful mask, which acts as a
strong implicit regulariser and prevents the network from collapsing to
the trivial all-background solution.

Forward contract
----------------
- Training mode (``self.training == True``): returns the 4-tuple
  ``(main_pred, aux_level2, aux_level3, aux_level4)``. All four tensors
  share shape ``(B, 1, D, H, W)`` matching the input — auxiliary heads are
  upsampled trilinearly to the input resolution before the 1×1×1 head so
  the loss sees a homogeneous batch and the team's existing
  ``dice_bce_loss`` can be reused per output without per-level resampling.
- Eval mode: returns ``main_pred`` only, as a single tensor.

Output convention
-----------------
The four outputs are RAW LOGITS (no sigmoid in the architecture). This
matches the shared ``dice_bce_loss`` (``trainer.py``) which applies
``binary_cross_entropy_with_logits`` and an internal sigmoid for the Dice
term — both for numerical stability. The design.md "+ sigmoid" notation
is the conceptual final activation; in practice sigmoid is applied at
inference / metric computation, never inside the model. Apply
``torch.sigmoid`` to the eval-mode output when thresholding to a binary
mask for visualisation or evaluation.
"""
from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """Two 3×3×3 convs with batch-norm + ReLU and an identity skip.

    When ``in_channels != out_channels`` a 1×1×1 projection on the skip
    path matches the channel count so the addition is well-defined.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResidualUNet3D(nn.Module):
    """3D Residual U-Net with deep supervision for T1n whole-tumour seg."""

    def __init__(self, in_channels: int = 1, out_channels: int = 3) -> None:
        super().__init__()

        # Encoder — 4 levels, channels 1 → 32 → 64 → 128 → 256.
        self.enc1 = ResidualBlock3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ResidualBlock3D(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ResidualBlock3D(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ResidualBlock3D(128, 256)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck (256 channels).
        self.bottleneck = ResidualBlock3D(256, 256)

        # Decoder — 4 levels, channels 256 → 128 → 64 → 32 → 32.
        # Each level: ConvTranspose3d (×2 stride) → concat skip → ResidualBlock3D.
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock3D(128 + 256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock3D(64 + 128, 64)
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock3D(32 + 64, 32)
        self.up4 = nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock3D(32 + 32, 32)

        # Main classifier head (full resolution).
        self.out_head = nn.Conv3d(32, out_channels, kernel_size=1)

        # Deep supervision heads at decoder levels 2, 3, 4.
        # Numbering matches design.md / plan.md:
        #   level 2 = dec2 output (16³ at 64 ch for 64³ input)
        #   level 3 = dec3 output (32³ at 32 ch)
        #   level 4 = dec4 output (full res at 32 ch — same input as main)
        self.aux2_head = nn.Conv3d(64, out_channels, kernel_size=1)
        self.aux3_head = nn.Conv3d(32, out_channels, kernel_size=1)
        self.aux4_head = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # ---- Encoder ----
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))

        # ---- Bottleneck ----
        b = self.bottleneck(self.pool4(s4))

        # ---- Decoder (each level: upsample → concat skip → residual) ----
        d1 = self.dec1(torch.cat([self.up1(b), s4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), s3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), s2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), s1], dim=1))

        # ---- Main head (full resolution) ----
        main_pred = self.out_head(d4)

        if not self.training:
            return main_pred

        # ---- Auxiliary heads at decoder levels 2, 3, 4 ----
        # Each is a 1×1×1 logit map at the level's native spatial size;
        # trilinearly upsample to the input shape so loss inputs are uniform.
        target_shape = main_pred.shape[2:]
        aux2 = F.interpolate(self.aux2_head(d2), size=target_shape, mode="trilinear", align_corners=False)
        aux3 = F.interpolate(self.aux3_head(d3), size=target_shape, mode="trilinear", align_corners=False)
        aux4 = self.aux4_head(d4)
        if aux4.shape[2:] != target_shape:
            aux4 = F.interpolate(aux4, size=target_shape, mode="trilinear", align_corners=False)

        return main_pred, aux2, aux3, aux4


if __name__ == "__main__":
    torch.manual_seed(42)

    model = ResidualUNet3D()
    x = torch.randn(2, 1, 64, 64, 64)

    # --- Training-mode forward: tuple of four tensors ---
    model.train()
    outputs = model(x)
    assert isinstance(outputs, tuple) and len(outputs) == 4, (
        f"Expected 4-tuple in training mode, got {type(outputs).__name__} of length "
        f"{len(outputs) if isinstance(outputs, tuple) else 'N/A'}"
    )
    main_pred, aux2, aux3, aux4 = outputs
    expected = (2, 3, 64, 64, 64)
    for name, t in (("main_pred", main_pred), ("aux2", aux2), ("aux3", aux3), ("aux4", aux4)):
        assert tuple(t.shape) == expected, f"{name} shape {tuple(t.shape)} != {expected}"

    # --- Eval-mode forward: single tensor ---
    model.eval()
    with torch.no_grad():
        eval_out = model(x)
    assert isinstance(eval_out, torch.Tensor), (
        f"Expected single tensor in eval mode, got {type(eval_out).__name__}"
    )
    assert tuple(eval_out.shape) == expected, (
        f"Eval output shape {tuple(eval_out.shape)} != {expected}"
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"train output shape (main):  {tuple(main_pred.shape)}")
    print(f"train output shape (aux2):  {tuple(aux2.shape)}")
    print(f"train output shape (aux3):  {tuple(aux3.shape)}")
    print(f"train output shape (aux4):  {tuple(aux4.shape)}")
    print(f"eval output shape:          {tuple(eval_out.shape)}")
    print(f"trainable parameters:       {n_params:,}")
    print("smoke test: PASSED")
