"""Member 5 — Multimodal naive 4-channel baseline (Pipeline A).

Owner: Member 5
Modality: all four modalities stacked as input channels (in this order:
t1c, t1n, t2f, t2w — matches shared.preprocessing.preprocess_patient_multimodal).

This is the simplest credible multimodal model — M1's ResidualUNet3D
architecture with in_channels=4 and no deep supervision. It serves as
the numerical floor that the XAI-guided Latent Fusion Head (Pipeline C)
must beat to validate the research thesis.

Forward contract
----------------
Returns a SINGLE TENSOR in BOTH training and eval mode:
    shape (B, 3, H, W, D), raw logits

Unlike M1 (which returns a 4-tuple in training mode for deep supervision),
M5 has no auxiliary heads — the contract is just "tensor in, tensor out."

Output convention
-----------------
Raw logits (no sigmoid). Sigmoid is applied inside the shared
``dice_bce_loss`` and at metric time, never inside the model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


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


class MultimodalUNet3D(nn.Module):
    """3D Residual U-Net for naive 4-channel multimodal segmentation.

    Same residual skeleton as M1's ResidualUNet3D but with
    ``in_channels=4`` (stacked T1c/T1n/T2f/T2w) and no deep supervision.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3) -> None:
        super().__init__()

        # Encoder — 4 levels, channels 4 → 32 → 64 → 128 → 256.
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
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock3D(128 + 256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock3D(64 + 128, 64)
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock3D(32 + 64, 32)
        self.up4 = nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock3D(32 + 32, 32)

        # Main classifier head (full resolution). No auxiliary heads.
        self.out_head = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the bottleneck feature map at shape
        ``(B, 256, H/16, W/16, D/16)``. Identical contract to M1-M4.

        Used here for completeness/symmetry — M5 fusion (Pipeline C)
        does NOT consume M5's own features; it consumes M1-M4's features.
        """
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        return self.bottleneck(self.pool4(s4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- Encoder ----
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))

        # ---- Bottleneck ----
        b = self.bottleneck(self.pool4(s4))

        # ---- Decoder ----
        d1 = self.dec1(torch.cat([self.up1(b), s4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), s3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), s2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), s1], dim=1))

        # ---- Main head (full resolution) — single tensor, both modes ----
        return self.out_head(d4)


if __name__ == "__main__":
    import torch
    torch.manual_seed(42)

    model = MultimodalUNet3D(in_channels=4, out_channels=3)
    x = torch.randn(2, 4, 64, 64, 64)   # 4-channel multimodal input

    expected = (2, 3, 64, 64, 64)

    # Training mode → single tensor (NOT a tuple)
    model.train()
    out_train = model(x)
    assert isinstance(out_train, torch.Tensor), \
        f"Expected single tensor in training mode, got {type(out_train).__name__}"
    assert tuple(out_train.shape) == expected, \
        f"Train output shape {tuple(out_train.shape)} != {expected}"

    # Eval mode → single tensor
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert isinstance(out_eval, torch.Tensor)
    assert tuple(out_eval.shape) == expected

    # forward_features → (B, 256, H/16, W/16, D/16)
    features = model.forward_features(x)
    assert tuple(features.shape) == (2, 256, 4, 4, 4), \
        f"forward_features shape {tuple(features.shape)} != (2,256,4,4,4)"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"input shape:             {tuple(x.shape)}")
    print(f"train output shape:      {tuple(out_train.shape)}")
    print(f"eval output shape:       {tuple(out_eval.shape)}")
    print(f"forward_features shape:  {tuple(features.shape)}")
    print(f"trainable parameters:    {n_params:,}")
    print("smoke test: PASSED")
