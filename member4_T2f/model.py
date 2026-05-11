"""Member 4 — T2f (FLAIR) segmentation model.

Architecture: 3D Residual U-Net
Modality:     T2f (FLAIR) — single channel input

Why residual connections?
    Each ResBlock adds the input directly to the output (identity shortcut).
    Gradients flow backward unchanged through the skip, solving the vanishing
    gradient problem that makes plain deep CNNs hard to train on 3D volumes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
    """Two 3x3x3 convs + BN + ReLU with a residual skip connection."""

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
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class T2fSegModel(nn.Module):
    """3D Residual U-Net for single-channel FLAIR segmentation.

    Input shape:  (B, 1, D, H, W)
    Output shape: (B, 1, D, H, W)  raw logits — sigmoid applied externally
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32):
        super().__init__()

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
        self.dec4 = ResBlock3D(base_ch * 4 + base_ch * 8, base_ch * 4)   # 128+256=384 → 128
        self.up3  = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec3 = ResBlock3D(base_ch * 2 + base_ch * 4, base_ch * 2)   # 64+128=192  → 64

        self.up2  = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec2 = ResBlock3D(base_ch + base_ch * 2, base_ch)            # 32+64=96    → 32

        self.up1  = nn.ConvTranspose3d(base_ch, base_ch, kernel_size=2, stride=2)
        self.dec1 = ResBlock3D(base_ch + base_ch, base_ch)                # 32+32=64    → 32

        # Output
        self.out_conv = nn.Conv3d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

    def get_encoder_layers(self) -> list:
        """Expose encoder blocks for Grad-CAM hook registration."""
        return [self.enc1, self.enc2, self.enc3, self.enc4]


if __name__ == "__main__":
    model = T2fSegModel(in_channels=1, base_ch=32)
    dummy = torch.randn(1, 1, 64, 64, 64)
    out   = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {params:,}")
