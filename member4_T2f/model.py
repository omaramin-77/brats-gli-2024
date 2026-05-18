"""Member 4 — T2f (FLAIR) segmentation model.

Modality: T2f / FLAIR (Fluid Attenuated Inversion Recovery)
Output:   3-channel logits (WT, TC, ET) — sigmoid applied externally, NEVER inside model.

Architecture:
  T2fSegModel  — 3D Residual U-Net  (~9.9 M params)

LATENT FUSION CONTRACT (binding — see design.md §9.2):
  Model exposes:
      forward_features(x) → (B, 256, H/16, W/16, D/16)
"""
from __future__ import annotations

import torch
import torch.nn as nn

BOTTLENECK_CHANNELS = 256   # shared contract — do not change


# ══════════════════════════════════════════════════════════════════════════════
# Shared building block
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.skip = (
            nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, bias=False), nn.BatchNorm3d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


# ══════════════════════════════════════════════════════════════════════════════
# Model — 3D Residual U-Net
# ══════════════════════════════════════════════════════════════════════════════

class T2fSegModel(nn.Module):
    """3D Residual U-Net for single-channel FLAIR segmentation.

    Encoder:     1→32→64→128→256 (4 MaxPool steps → spatial /16)
    Bottleneck:  256 ch
    Decoder:     256→128→64→32→32
    Head:        Conv3d(32, 3, 1) → 3-channel logits (WT, TC, ET)
    """

    arch_name = "ResUNet3D"

    def __init__(self, in_channels: int = 1, base_ch: int = 32, out_channels: int = 3):
        super().__init__()
        self.enc1 = ResBlock3D(in_channels, base_ch)
        self.enc2 = ResBlock3D(base_ch,     base_ch * 2)
        self.enc3 = ResBlock3D(base_ch * 2, base_ch * 4)
        self.enc4 = ResBlock3D(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ResBlock3D(base_ch * 8, base_ch * 8)
        self.proj = (
            nn.Conv3d(base_ch * 8, BOTTLENECK_CHANNELS, 1, bias=False)
            if base_ch * 8 != BOTTLENECK_CHANNELS else nn.Identity()
        )

        self.up4  = nn.ConvTranspose3d(BOTTLENECK_CHANNELS, base_ch * 4, 2, stride=2)
        self.dec4 = ResBlock3D(base_ch * 4 + base_ch * 8, base_ch * 4)
        self.up3  = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec3 = ResBlock3D(base_ch * 2 + base_ch * 4, base_ch * 2)
        self.up2  = nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2)
        self.dec2 = ResBlock3D(base_ch + base_ch * 2, base_ch)
        self.up1  = nn.ConvTranspose3d(base_ch, base_ch, 2, stride=2)
        self.dec1 = ResBlock3D(base_ch + base_ch, base_ch)
        self.out_conv = nn.Conv3d(base_ch, out_channels, 1)

    def _encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.proj(self.bottleneck(self.pool(e4)))
        return b, [e1, e2, e3, e4]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 256, D/16, H/16, W/16) — encoder bottleneck only."""
        b, _ = self._encode(x)
        return b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, (e1, e2, e3, e4) = self._encode(x)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)
    # Final shape: (1, 3, 64, 64, 64)  [3 tumor classes!]

    def get_encoder_layers(self) -> list[nn.Module]:
        return [self.enc1, self.enc2, self.enc3, self.enc4]


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_model(arch: str = "resunet", **kwargs) -> nn.Module:
    """Build model (only ResUNet3D supported)."""
    key = arch.lower().replace("_", "").replace("-", "")
    if key == "resunet":
        return T2fSegModel(**kwargs)
    raise ValueError(f"Unknown arch '{arch}'. Choose 'resunet'.")


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  M4 model smoke test")
    print("=" * 60)

    dummy         = torch.randn(2, 1, 64, 64, 64)
    expected_out  = (2, 3, 64, 64, 64)
    expected_feat = (2, BOTTLENECK_CHANNELS, 4, 4, 4)   # 64/16 = 4

    model  = build_model("resunet", in_channels=1, out_channels=3)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    out  = model(dummy)
    feat = model.forward_features(dummy)

    assert tuple(out.shape)  == expected_out, \
        f"output {out.shape} ≠ {expected_out}"
    assert tuple(feat.shape) == expected_feat, \
        f"features {feat.shape} ≠ {expected_feat}"

    print(f"\n[{model.arch_name}]")
    print(f"  forward()          : {tuple(out.shape)}   ✓")
    print(f"  forward_features() : {tuple(feat.shape)}   ✓")
    print(f"  Params             : {params:,}")

    print("\nAll assertions passed ✓")