"""Latent Space Fusion Head (Pipeline C) — the project's novel contribution.

Three nn.Modules:
  - ModalityGatedFusion: per-voxel cross-modality attention over 4 bottlenecks
  - LatentFusionDecoder: 8^3 -> 128^3 segmentation decoder
  - LatentFusionEnsemble: wraps both

Plus helper build_gate_bias() that converts the ablation-derived modality
importance scores into an attention bias prior. The fusion module starts
with zero conv weights and this bias, so at training step 0 the
per-voxel attention is uniform across voxels and equal to softmax(bias).
As training proceeds, the conv weights learn spatial structure.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ModalityGatedFusion
# ---------------------------------------------------------------------------
class ModalityGatedFusion(nn.Module):
    """Per-voxel softmax attention over four 256-channel bottleneck maps."""

    def __init__(
        self,
        channels: int = 256,
        num_modalities: int = 4,
        gate_bias_init: torch.Tensor | None = None,
    ):
        """
        gate_bias_init: optional tensor of shape (num_modalities,) containing
            the initial bias for each modality channel of the attention conv.
            This is what carries the XAI-informed prior into the model. If
            None, the conv is initialised with the PyTorch default (zero bias).
        """
        super().__init__()
        self.num_modalities = num_modalities
        self.attention_conv = nn.Conv3d(
            channels * num_modalities, num_modalities, kernel_size=1, bias=True
        )
        if gate_bias_init is not None:
            if gate_bias_init.shape != (num_modalities,):
                raise ValueError(
                    f"gate_bias_init must be shape ({num_modalities},); "
                    f"got {tuple(gate_bias_init.shape)}"
                )
            with torch.no_grad():
                self.attention_conv.bias.copy_(gate_bias_init)
                # Zero the weights so the first forward pass produces
                # attention determined entirely by the bias prior. After
                # training, the conv learns spatial structure.
                self.attention_conv.weight.zero_()

    def forward(self, f_t1c, f_t1n, f_t2f, f_t2w):
        """Each f_* has shape (B, 256, D, H, W).

        Returns:
            fused:     (B, 256, D, H, W) — softmax-weighted sum
            attention: (B, 4, D, H, W) — per-voxel modality weights
        """
        # Stack ORDER: (t1c, t1n, t2f, t2w) — matches MODALITIES in config.
        stack = [f_t1c, f_t1n, f_t2f, f_t2w]
        concat = torch.cat(stack, dim=1)                        # (B, 1024, D, H, W)
        attention_logits = self.attention_conv(concat)          # (B, 4, D, H, W)
        attention = F.softmax(attention_logits, dim=1)          # softmax across modalities
        fused = sum(
            attention[:, i:i + 1] * stack[i]
            for i in range(self.num_modalities)
        )
        return fused, attention


# ---------------------------------------------------------------------------
# LatentFusionDecoder
# ---------------------------------------------------------------------------
class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class LatentFusionDecoder(nn.Module):
    """(B, 256, 8, 8, 8) -> (B, 3, 128, 128, 128)."""

    def __init__(self, in_channels: int = 256, out_channels: int = 3):
        super().__init__()
        # Three 2x upsamples bring 8 -> 16 -> 32 -> 64. Then a final 2x
        # trilinear upsample to 128 keeps the decoder small and avoids
        # ConvTranspose checkerboard artefacts at the final resolution.
        self.up1 = _UpBlock(in_channels, 128)   # 8 -> 16
        self.up2 = _UpBlock(128, 64)            # 16 -> 32
        self.up3 = _UpBlock(64, 32)             # 32 -> 64
        self.head = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        # Final upsample 64 -> 128 via trilinear interpolation.
        x = F.interpolate(
            x, size=(128, 128, 128), mode="trilinear", align_corners=False
        )
        return self.head(x)


# ---------------------------------------------------------------------------
# LatentFusionEnsemble
# ---------------------------------------------------------------------------
class LatentFusionEnsemble(nn.Module):
    def __init__(self, gate_bias_init: torch.Tensor | None = None):
        super().__init__()
        self.fusion = ModalityGatedFusion(
            channels=256, num_modalities=4, gate_bias_init=gate_bias_init,
        )
        self.decoder = LatentFusionDecoder(in_channels=256, out_channels=3)

    def forward(self, features: dict):
        """features: {"t1c","t1n","t2f","t2w"} each (B, 256, 8, 8, 8)."""
        fused, attention = self.fusion(
            features["t1c"], features["t1n"], features["t2f"], features["t2w"],
        )
        logits = self.decoder(fused)
        return logits, attention   # logits: (B, 3, 128, 128, 128)


# ---------------------------------------------------------------------------
# Gate-bias initialisation from ablation importance scores
# ---------------------------------------------------------------------------
def build_gate_bias(importance_json_path: str | Path) -> torch.Tensor:
    """Read modality_importance_scores.json and produce a (4,) tensor
    of attention-bias initial values.

    Strategy: average the per-sub-region importance drops across WT/TC/ET
    to get a single per-modality "general importance" score. Take the
    log of the softmax of these averages — this gives a bias vector
    such that softmax(0 + bias) reproduces the desired attention prior.

    The bias is fed to ModalityGatedFusion with the attention conv's
    weights zeroed; at training step 0, attention(any voxel) = softmax(bias).
    """
    data = json.loads(Path(importance_json_path).read_text(encoding="utf-8"))
    # Order MUST match ModalityGatedFusion.forward: (t1c, t1n, t2f, t2w).
    mods = ("t1c", "t1n", "t2f", "t2w")
    avg = torch.tensor(
        [(data["wt"][m] + data["tc"][m] + data["et"][m]) / 3.0 for m in mods],
        dtype=torch.float32,
    )
    # Add a small epsilon so the log doesn't blow up on a 0 score.
    avg = avg + 1e-3
    # Normalise so the softmax of the bias reproduces the importance ratio.
    bias = torch.log(avg / avg.sum())
    return bias
