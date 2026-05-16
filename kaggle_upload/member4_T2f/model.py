"""Member 4 — T2f (FLAIR) segmentation models.

Modality: T2f / FLAIR (Fluid Attenuated Inversion Recovery)
Output:   3-channel logits (WT, TC, ET) — sigmoid applied externally, NEVER inside model.

TWO architectures compared:
  1. T2fSegModel  — 3D Residual U-Net  (~9.9 M params)
  2. T2fSegResNet — SegResNet via MONAI (~4.7 M params, faster on Kaggle)

LATENT FUSION CONTRACT (binding — see design.md §9.2):
  Both models expose:
      forward_features(x) → (B, 256, H/16, W/16, D/16)

CHANNEL ORDER (frozen — claude.md §3.4):
  Output channels: index 0=WT, 1=TC, 2=ET
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
# Model 1 — 3D Residual U-Net
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
        b, _ = self._encode(x)
        return b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, (e1, e2, e3, e4) = self._encode(x)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

    def get_encoder_layers(self) -> list[nn.Module]:
        return [self.enc1, self.enc2, self.enc3, self.enc4]


# ══════════════════════════════════════════════════════════════════════════════
# Model 2 — SegResNet (MONAI) wrapper
# ══════════════════════════════════════════════════════════════════════════════

class T2fSegResNet(nn.Module):
    """MONAI SegResNet wrapper with version-agnostic forward_features.

    Probe findings (MONAI on this machine):
      - down_layers.3 is the last encoder group: spatial=D/16, channels=256
      - up_samples.0 starts the decoder at the same spatial size with 128ch
      - The hook must target down_layers.3 specifically (encoder, not decoder)

    forward_features() hooks down_layers.3 (or equivalent) by finding the
    last 'down_layers' group, which is always the encoder bottleneck.
    No extra pooling or projection needed when native ch=256 and spatial=D/16.
    """

    arch_name = "SegResNet"

    def __init__(self, in_channels: int = 1, out_channels: int = 3,
                 init_filters: int = 32):
        super().__init__()
        try:
            from monai.networks.nets import SegResNet as _SegResNet
        except ImportError as exc:
            raise ImportError("pip install monai") from exc

        self._net = _SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            dropout_prob=0.1,
        )

        # ── Probe: find native bottleneck channel count ───────────────────────
        # We hook ONLY down_layers groups (encoder), never up_samples (decoder).
        # This avoids confusing decoder tensors at the same spatial size.
        probe   = torch.zeros(1, in_channels, 32, 32, 32)
        enc_out: list[tuple[int, int]] = []   # (spatial, channels) encoder only

        handles = []
        named = dict(self._net.named_modules())
        for name, mod in named.items():
            # Only hook top-level down_layers groups (e.g. "down_layers.3")
            # Skip sub-layers like "down_layers.3.1.conv1"
            if name.startswith("down_layers.") and name.count(".") == 1:
                def _h(m, i, o, _enc=enc_out):
                    if isinstance(o, torch.Tensor) and o.ndim == 5:
                        _enc.append((int(o.shape[2]), int(o.shape[1])))
                handles.append(mod.register_forward_hook(_h))

        with torch.no_grad():
            self._net(probe)

        for h in handles:
            h.remove()

        if not enc_out:
            # Fallback: down_layers not found, use init_filters * 8
            native_s  = 32 // 8
            native_ch = init_filters * 8
        else:
            # Last encoder group = smallest spatial = true bottleneck
            native_s  = min(s for s, _ in enc_out)
            native_ch = max(c for s, c in enc_out if s == native_s)

        # Extra pooling: how many MaxPool3d(2) to go from native_s → 32/16=2
        target_s    = 32 // 16   # = 2  (spatial for 32^3 input at /16)
        extra_pools = 0
        s = native_s
        while s > target_s:
            s //= 2
            extra_pools += 1

        self.extra_pool = (
            nn.Sequential(*[nn.MaxPool3d(2) for _ in range(extra_pools)])
            if extra_pools > 0 else nn.Identity()
        )

        # Projection: native_ch → 256
        self.proj = (
            nn.Conv3d(native_ch, BOTTLENECK_CHANNELS, 1, bias=False)
            if native_ch != BOTTLENECK_CHANNELS else nn.Identity()
        )

        self._native_s    = native_s
        self._cache       = None
        self._hook_handle = None

    def _attach_hook(self, x: torch.Tensor) -> None:
        """Hook the last down_layers group (encoder bottleneck only)."""
        named = dict(self._net.named_modules())

        # Find all top-level down_layers groups
        dl_names = sorted(
            [n for n in named if n.startswith("down_layers.") and n.count(".") == 1],
            key=lambda n: int(n.split(".")[1])
        )

        if dl_names:
            # Last encoder group = deepest = bottleneck
            target_name = dl_names[-1]
            target_mod  = named[target_name]
            self._hook_handle = target_mod.register_forward_hook(
                lambda m, i, o: setattr(self, '_cache', o)
            )
        else:
            # Fallback: find the module whose output spatial == input/16
            target_s = x.shape[2] // 16
            seen: dict = {}
            handles = []
            for depth, (name, mod) in enumerate(self._net.named_modules()):
                def _mk(n, d):
                    def _h(m, i, o):
                        if isinstance(o, torch.Tensor) and o.ndim == 5:
                            seen[n] = (d, int(o.shape[2]), int(o.shape[1]))
                    return _h
                handles.append(mod.register_forward_hook(_mk(name, depth)))
            with torch.no_grad():
                self._net(x)
            for h in handles:
                h.remove()
            # Among modules at target spatial, pick max channels (encoder, not decoder)
            candidates = {n: (d, s, c) for n, (d, s, c) in seen.items() if s == target_s}
            if candidates:
                best = max(candidates.items(), key=lambda kv: kv[1][2])  # max channels
                target_mod = named[best[0]]
                self._hook_handle = target_mod.register_forward_hook(
                    lambda m, i, o: setattr(self, '_cache', o)
                )
            else:
                raise RuntimeError(
                    f"Cannot find SegResNet bottleneck at spatial {target_s}. "
                    f"Available: {sorted(set(v[1] for v in seen.values()))}."
                )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 256, D/16, H/16, W/16) — encoder bottleneck only."""
        if self._hook_handle is None:
            self._attach_hook(x)
        self._cache = None
        self._net(x)
        if self._cache is None:
            raise RuntimeError("forward_features hook did not fire.")
        return self.proj(self.extra_pool(self._cache))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)

    def get_encoder_layers(self) -> list[nn.Module]:
        named = dict(self._net.named_modules())
        dl_names = sorted(
            [n for n in named if n.startswith("down_layers.") and n.count(".") == 1],
            key=lambda n: int(n.split(".")[1])
        )
        if dl_names:
            return [named[n] for n in dl_names]
        if hasattr(self._net, 'encoder'):
            return list(self._net.encoder)
        return []

    def __del__(self):
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_model(arch: str = "resunet", **kwargs) -> nn.Module:
    key = arch.lower().replace("_", "").replace("-", "")
    if key == "resunet":
        return T2fSegModel(**kwargs)
    if key == "segresnet":
        return T2fSegResNet(**kwargs)
    raise ValueError(f"Unknown arch '{arch}'. Choose 'resunet' or 'segresnet'.")


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

    for arch in ("resunet", "segresnet"):
        model  = build_model(arch, in_channels=1, out_channels=3)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        out  = model(dummy)
        feat = model.forward_features(dummy)

        assert tuple(out.shape)  == expected_out, \
            f"{arch}: output {out.shape} ≠ {expected_out}"
        assert tuple(feat.shape) == expected_feat, \
            f"{arch}: features {feat.shape} ≠ {expected_feat}"

        print(f"\n[{model.arch_name}]")
        print(f"  forward()          : {tuple(out.shape)}   ✓")
        print(f"  forward_features() : {tuple(feat.shape)}   ✓")
        print(f"  Params             : {params:,}")

    print("\nAll assertions passed ✓")
