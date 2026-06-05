"""Latent Space Fusion Head (Pipeline C) — the project's novel contribution.

Modules:
  - ModalityGatedFusion: per-voxel cross-modality attention over N (default 4)
    256-channel bottleneck maps, with per-modality LayerNorm-style normalisation
    applied BEFORE concatenation so transformer / residual-conv encoders'
    differing logit scales stop fighting the attention conv.
  - LatentFusionDecoder: 8^3 -> 128^3 segmentation decoder. Now all four 2x
    upsamples are learned ConvTranspose blocks (no trailing trilinear leap).
  - LatentFusionEnsemble: wraps both, accepts an arbitrary modality subset
    (e.g. drop t2w from the original 4 -> run a 3-modality controlled ablation
    against the 4-modality variant).

The ablation-derived gate bias prior comes from member5_multimodal/ablation.py
(modality_importance_scores.json). build_gate_bias() reads it and produces a
tensor of shape (num_modalities,) that initialises the attention conv's bias.
The conv weights are zeroed at init so step-0 attention is exactly softmax(bias).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# Canonical ordered modality list (matches shared.config.MODALITIES). Any subset
# the LatentFusionEnsemble is instantiated with must be a contiguous-or-not
# subset of this tuple — but the order within the subset must match this tuple,
# so weight columns line up across runs.
ALL_MODALITIES: tuple[str, ...] = ("t1c", "t1n", "t2f", "t2w")


# ---------------------------------------------------------------------------
# Per-modality feature normalisation
# ---------------------------------------------------------------------------
def _normalise_modality_features(t: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm-style normalisation of one modality's (B, 256, D, H, W) tensor.

    Subtracts per-sample mean and divides by per-sample std over
    (channel, spatial) dims. No learnable affine — we deliberately do not add
    parameters here so the attention conv keeps full control.

    Why this exists: M1/M2/M4 are residual U-Nets, M3 is a Swin Transformer,
    M5 is a residual U-Net with 4-channel input. Each architecture's bottleneck
    distribution has a different scale and shift; concatenating raw features
    forces the attention conv to learn modality identity from magnitude alone,
    which is fragile. Normalising puts all modalities on the same footing.
    """
    mean = t.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = t.std(dim=(1, 2, 3, 4), keepdim=True) + eps
    return (t - mean) / std


# ---------------------------------------------------------------------------
# ModalityGatedFusion
# ---------------------------------------------------------------------------
class ModalityGatedFusion(nn.Module):
    """Per-voxel softmax attention over N normalised 256-channel bottleneck maps."""

    def __init__(
        self,
        channels: int = 256,
        modalities: tuple[str, ...] = ALL_MODALITIES,
        gate_bias_init: torch.Tensor | None = None,
        normalise_inputs: bool = True,
    ):
        """
        modalities: ordered subset of ALL_MODALITIES. The default fuses
            all four; passing ("t1c", "t1n", "t2f") runs a 3-modality
            controlled ablation that excludes t2w. Changing this changes
            the attention conv's input shape (channels * len(modalities))
            and output shape (len(modalities)), so checkpoints are NOT
            interchangeable between modality subsets.

        gate_bias_init: optional tensor of shape (len(modalities),) containing
            the initial bias for each modality channel of the attention conv.
            This is what carries the XAI-informed prior into the model. If
            None, the conv is initialised with the PyTorch default (zero bias).

        normalise_inputs: when True, each modality's tensor is per-sample
            LayerNorm'd before concat. Default True for Tier-1 fix. Set False
            only for ablation runs that isolate the normalisation effect.
        """
        super().__init__()
        unknown = [m for m in modalities if m not in ALL_MODALITIES]
        if unknown:
            raise ValueError(
                f"Unknown modalities {unknown}; must be subset of {ALL_MODALITIES}"
            )
        if len(modalities) < 2:
            raise ValueError(
                f"Need at least 2 modalities for fusion; got {modalities}"
            )

        self.modalities = tuple(modalities)
        self.num_modalities = len(self.modalities)
        self.normalise_inputs = bool(normalise_inputs)
        self.attention_conv = nn.Conv3d(
            channels * self.num_modalities, self.num_modalities,
            kernel_size=1, bias=True,
        )
        if gate_bias_init is not None:
            if gate_bias_init.shape != (self.num_modalities,):
                raise ValueError(
                    f"gate_bias_init must be shape ({self.num_modalities},); "
                    f"got {tuple(gate_bias_init.shape)}"
                )
            with torch.no_grad():
                self.attention_conv.bias.copy_(gate_bias_init)
                # Zero the weights so the first forward pass produces
                # attention determined entirely by the bias prior. After
                # training, the conv learns spatial structure.
                self.attention_conv.weight.zero_()

    def forward(self, features: dict[str, torch.Tensor]):
        """features: {mod: (B, 256, D, H, W)} for each mod in self.modalities.

        Returns:
            fused:     (B, 256, D, H, W) — softmax-weighted sum
            attention: (B, N, D, H, W) — per-voxel modality weights
        """
        # Pull tensors in the canonical self.modalities order so weight
        # columns are stable across runs.
        ordered = [features[m] for m in self.modalities]
        if self.normalise_inputs:
            ordered = [_normalise_modality_features(t) for t in ordered]
        concat = torch.cat(ordered, dim=1)                  # (B, C*N, D, H, W)
        attention_logits = self.attention_conv(concat)      # (B, N, D, H, W)
        attention = F.softmax(attention_logits, dim=1)
        fused = sum(
            attention[:, i:i + 1] * ordered[i]
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
    """(B, 256, 8, 8, 8) -> (B, 3, 128, 128, 128).

    Four learned ×2 ConvTranspose stages (8→16→32→64→128) — every stage now
    has learned parameters, removing the original trailing trilinear leap
    from 64³ to 128³ that contributed zero capacity at the highest
    resolution where small structures (TC/ET) are decided.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 3):
        super().__init__()
        self.up1 = _UpBlock(in_channels, 128)   # 8  -> 16
        self.up2 = _UpBlock(128, 64)            # 16 -> 32
        self.up3 = _UpBlock(64, 32)             # 32 -> 64
        self.up4 = _UpBlock(32, 16)             # 64 -> 128
        self.head = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# LatentFusionEnsemble
# ---------------------------------------------------------------------------
class LatentFusionEnsemble(nn.Module):
    def __init__(
        self,
        gate_bias_init: torch.Tensor | None = None,
        modalities: tuple[str, ...] = ALL_MODALITIES,
        normalise_inputs: bool = True,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.fusion = ModalityGatedFusion(
            channels=256,
            modalities=self.modalities,
            gate_bias_init=gate_bias_init,
            normalise_inputs=normalise_inputs,
        )
        self.decoder = LatentFusionDecoder(in_channels=256, out_channels=3)

    def forward(self, features: dict):
        """features: dict containing at least the modalities this ensemble
        was constructed with. Extra keys are ignored, which lets the
        FeaturesDataset return all 4 modalities even when we run the
        3-modality ablation variant.
        """
        # ModalityGatedFusion picks the right subset internally.
        fused, attention = self.fusion(features)
        logits = self.decoder(fused)
        return logits, attention   # logits: (B, 3, 128, 128, 128)


# ---------------------------------------------------------------------------
# Gate-bias initialisation from ablation importance scores
# ---------------------------------------------------------------------------
def build_gate_bias(
    importance_json_path: str | Path,
    modalities: tuple[str, ...] = ALL_MODALITIES,
) -> torch.Tensor:
    """Read modality_importance_scores.json and produce a (N,) tensor
    of attention-bias initial values, ordered to match ``modalities``.

    Strategy: average the per-sub-region importance drops across WT/TC/ET
    to get a single per-modality "general importance" score. Take the
    log of the softmax of these averages — this gives a bias vector
    such that softmax(0 + bias) reproduces the desired attention prior.

    When ``modalities`` is a strict subset of ALL_MODALITIES, the bias is
    re-normalised across the surviving modalities so the prior probabilities
    still sum to 1. This is what makes the 3-modality controlled ablation
    comparable to the 4-modality run: the surviving modalities split the
    full prior mass between them, rather than each carrying their absolute
    4-modality share.

    The bias is fed to ModalityGatedFusion with the attention conv's
    weights zeroed; at training step 0, attention(any voxel) = softmax(bias).
    """
    data = json.loads(Path(importance_json_path).read_text(encoding="utf-8"))
    avg = torch.tensor(
        [(data["wt"][m] + data["tc"][m] + data["et"][m]) / 3.0 for m in modalities],
        dtype=torch.float32,
    )
    # Add a small epsilon so the log doesn't blow up on a 0 score.
    avg = avg + 1e-3
    # Normalise so softmax of the bias reproduces the importance ratio
    # within whichever modality subset we are training over.
    bias = torch.log(avg / avg.sum())
    return bias
