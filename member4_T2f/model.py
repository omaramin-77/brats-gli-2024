"""Member 4 — T2f (FLAIR) segmentation models.

TWO architectures compared side-by-side:
  1. T2fSegModel   — 3D Residual U-Net  (original, ~9.9 M params)
  2. T2fSegResNet  — SegResNet via MONAI (~4.7 M params, faster, competitive)

Both expose:
  • forward(x)                   → raw logits (sigmoid applied externally)
  • forward(x, return_features)  → (logits, bottleneck_features)
  • get_encoder_layers()         → list of encoder blocks for Grad-CAM hooks

Run directly to sanity-check both:
  python model.py
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# Model 1 — 3D Residual U-Net (original plan)
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock3D(nn.Module):
    """Two 3×3×3 convs + BN + ReLU + residual skip."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_ch),
            )
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class T2fSegModel(nn.Module):
    """3D Residual U-Net for single-channel FLAIR segmentation.

    Input:  (B, 1, D, H, W)
    Output: (B, 1, D, H, W)  — raw logits
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32):
        super().__init__()
        self.arch_name = "ResUNet3D"

        # Encoder
        self.enc1 = ResBlock3D(in_channels, base_ch)
        self.enc2 = ResBlock3D(base_ch,     base_ch * 2)
        self.enc3 = ResBlock3D(base_ch * 2, base_ch * 4)
        self.enc4 = ResBlock3D(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool3d(kernel_size=2)

        # Bottleneck
        self.bottleneck = ResBlock3D(base_ch * 8, base_ch * 8)

        # Decoder
        self.up4  = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec4 = ResBlock3D(base_ch * 4 + base_ch * 8, base_ch * 4)   # 128+256 → 128

        self.up3  = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec3 = ResBlock3D(base_ch * 2 + base_ch * 4, base_ch * 2)   # 64+128  → 64

        self.up2  = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec2 = ResBlock3D(base_ch + base_ch * 2, base_ch)            # 32+64   → 32

        self.up1  = nn.ConvTranspose3d(base_ch, base_ch, kernel_size=2, stride=2)
        self.dec1 = ResBlock3D(base_ch + base_ch, base_ch)                # 32+32   → 32

        self.out_conv = nn.Conv3d(base_ch, 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.out_conv(d1)
        if return_features:
            return out, b
        return out

    def get_encoder_layers(self) -> list:
        return [self.enc1, self.enc2, self.enc3, self.enc4]


# ══════════════════════════════════════════════════════════════════════════════
# Model 2 — SegResNet (MONAI) wrapper
#   Lighter, trains faster on Kaggle P100/T4, competitive Dice
#   Paper: Myronenko 2018 — "3D MRI Brain Tumor Segmentation Using
#          Autoencoder Regularization"
# ══════════════════════════════════════════════════════════════════════════════

class T2fSegResNet(nn.Module):
    """MONAI SegResNet wrapper — identical interface to T2fSegModel.

    Why SegResNet suits FLAIR:
      • Residual blocks at every encoder/decoder level (same as ResUNet3D)
      • Uses instance normalisation instead of batch norm — better for
        small batch sizes common in 3D medical imaging
      • ~4.7 M params vs 9.9 M for ResUNet3D → faster epoch on Kaggle
      • Competitive with full U-Net on BraTS whole-tumour Dice
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super().__init__()
        self.arch_name = "SegResNet"

        try:
            from monai.networks.nets import SegResNet as _SegResNet
        except ImportError as e:
            raise ImportError(
                "MONAI not installed. Run: pip install monai"
            ) from e

        self._net = _SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            dropout_prob=0.1,
        )

        # We'll extract features from the deepest encoder block
        # SegResNet's encoder is self._net.encoder (list of ResBlocks)
        self._bottleneck_features: torch.Tensor | None = None
        self._hook_handle = None

    def _register_feature_hook(self):
        """Register a hook to capture the deepest encoder output."""
        # The last layer before the decoder in MONAI SegResNet
        try:
            target = self._net.encoder[-1]
        except (AttributeError, IndexError):
            target = self._net.down_layers[-1] if hasattr(self._net, 'down_layers') else None

        if target is not None:
            def hook(m, inp, out):
                self._bottleneck_features = out
            self._hook_handle = target.register_forward_hook(hook)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_features and self._hook_handle is None:
            self._register_feature_hook()

        out = self._net(x)

        if return_features:
            feats = self._bottleneck_features
            return out, feats
        return out

    def get_encoder_layers(self) -> list:
        """Return encoder blocks for Grad-CAM targeting."""
        try:
            return list(self._net.encoder)
        except AttributeError:
            try:
                return list(self._net.down_layers)
            except AttributeError:
                return []

    def __del__(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()


# ══════════════════════════════════════════════════════════════════════════════
# Factory helper
# ══════════════════════════════════════════════════════════════════════════════

def build_model(arch: str = "resunet", **kwargs) -> nn.Module:
    """Return the requested model by name.

    Args:
        arch: 'resunet' or 'segresnet'
        **kwargs: forwarded to the model constructor
    """
    arch = arch.lower()
    if arch == "resunet":
        return T2fSegModel(**kwargs)
    elif arch in ("segresnet", "seg_resnet"):
        return T2fSegResNet(**kwargs)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose 'resunet' or 'segresnet'.")


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  M4 model self-test")
    print("=" * 55)
    dummy = torch.randn(1, 1, 64, 64, 64)

    for arch in ("resunet", "segresnet"):
        model = build_model(arch, in_channels=1)
        out   = model(dummy)
        feat_out, feats = model(dummy, return_features=True)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n[{model.arch_name}]")
        print(f"  Input shape   : {dummy.shape}")
        print(f"  Output shape  : {out.shape}")
        print(f"  Features shape: {feats.shape if feats is not None else 'N/A'}")
        print(f"  Trainable params: {params:,}")
        assert out.shape == dummy.shape, "Output shape mismatch!"
        print("  ✓ shape OK")
