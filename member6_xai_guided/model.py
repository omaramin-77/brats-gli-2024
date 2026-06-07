"""Member 6 — XAI-guided multi-task M5 (Proposal C).

Architecture summary
--------------------
Shared encoder/decoder identical to M5's MultimodalUNet3D (4-channel residual
U-Net), but the single output head is replaced with three sub-region-specialised
heads. Each head fuses the shared decoder feature map (32 channels at full
resolution) with a per-sub-region XAI-conditioned re-injection of the raw input.

Per-sub-region re-injection
~~~~~~~~~~~~~~~~~~~~~~~~~~~
For each sub-region sr in (WT, TC, ET) we hold a tiny per-modality scalar gate
of shape (4,) — one weight per input modality. It is normalised by softmax so
the four weights always sum to 1, then applied to the raw 4-channel input to
produce a single-channel "XAI-conditioned mixture":

    sr_input = sum_m  softmax(sr_gate)[m] * x[:, m]            (B, 1, H, W, D)

The gate logits are initialised from the per-sub-region ablation drops in
results/tables/modality_importance_scores.json so that at step 0:
- WT's head sees a mix dominated by T2-FLAIR
- TC's head sees a mix dominated by T1c
- ET's head sees a mix dominated by T1c

The gates are FULLY TRAINABLE — the XAI scores are a prior, not a clamp. We
ALSO log the post-training gate values so the report can show whether the
network kept the XAI guidance or drifted away from it.

Per-sub-region head
~~~~~~~~~~~~~~~~~~~
Each head takes the shared decoder features (32 ch) concatenated with the
XAI-conditioned single-channel mixture (1 ch) and produces a single-channel
logit map. This means every voxel of every sub-region's prediction sees:
- The joint multimodal feature representation learned by the shared backbone
- A sub-region-specific weighted view of the raw modalities, biased by XAI

Output contract
---------------
forward(x) -> (B, 3, H, W, D) raw logits in (WT, TC, ET) order.
forward_features(x) -> (B, 256, H/16, W/16, D/16) bottleneck — same contract
                       as every other member, so this model is ensemblable.

Why this is a legitimate paper contribution
------------------------------------------
The naive M5 baseline already implicitly learns "T2-FLAIR matters for WT" via
its first conv kernel. What this architecture adds is *per-sub-region*
modality bias at the decoder head — a structural inductive bias the naive
baseline cannot express because it shares a single head across sub-regions.
The XAI initialisation tells the network *which* modalities each sub-region
should look at; the training data tells it *how much* to weight that prior.
The natural ablation control is the same model with random gate init.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks (copied from M5 / M1)
# ---------------------------------------------------------------------------
class ResidualBlock3D(nn.Module):
    """Two 3x3x3 convs with BN + ReLU and an identity skip.

    A 1x1x1 projection matches channel count when in_channels != out_channels
    so the addition stays shape-safe.
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


# ---------------------------------------------------------------------------
# Per-sub-region head
# ---------------------------------------------------------------------------
class XAIGuidedSubregionHead(nn.Module):
    """One sub-region's specialised decoder head.

    State:
        modality_gate : nn.Parameter of shape (4,) — softmax-normalised weights
            that combine the 4 input modalities into a single sub-region
            mixture. Initialised from XAI ablation scores so at step 0 the
            head sees the XAI-prescribed modality emphasis.

    Forward:
        shared_features : (B, 32, H, W, D)
        raw_input       : (B, 4, H, W, D)
        => logits       : (B, 1, H, W, D)
    """

    def __init__(
        self,
        shared_channels: int = 32,
        num_modalities: int = 4,
        hidden_channels: int = 16,
        gate_init_logits: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if gate_init_logits is None:
            gate_init_logits = torch.zeros(num_modalities)
        if gate_init_logits.shape != (num_modalities,):
            raise ValueError(
                f"gate_init_logits must be ({num_modalities},); "
                f"got {tuple(gate_init_logits.shape)}"
            )
        # Trainable per-modality logits. Softmax in forward keeps them
        # interpretable as probabilities that sum to 1.
        self.modality_gate = nn.Parameter(gate_init_logits.clone().float())

        # Refinement conv on (shared + 1) channels -> hidden -> 1 logit.
        self.refine = nn.Sequential(
            nn.Conv3d(shared_channels + 1, hidden_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv3d(hidden_channels, 1, kernel_size=1)

    @property
    def gate_probabilities(self) -> torch.Tensor:
        """(4,) softmax-normalised gate, useful for logging."""
        return F.softmax(self.modality_gate, dim=0)

    def forward(self, shared_features: torch.Tensor,
                raw_input: torch.Tensor) -> torch.Tensor:
        # Build the per-sub-region XAI-conditioned single-channel mixture.
        gate = self.gate_probabilities                          # (4,)
        gate = gate.view(1, -1, 1, 1, 1)                        # (1, 4, 1, 1, 1)
        sr_mixture = (raw_input * gate).sum(dim=1, keepdim=True)  # (B, 1, H, W, D)

        # Concatenate with the shared decoder features and refine to a logit.
        combined = torch.cat([shared_features, sr_mixture], dim=1)  # (B, 33, H, W, D)
        h = self.refine(combined)
        return self.head(h)                                     # (B, 1, H, W, D)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class XAIGuidedMultimodalUNet3D(nn.Module):
    """4-channel ResUNet shared backbone + 3 XAI-guided sub-region heads.

    Parameters mostly mirror MultimodalUNet3D. The new arguments are:
        gate_init_logits : (3, 4) tensor of softmax-input logits, one row per
            sub-region (WT, TC, ET) in that fixed order, columns ordered to
            match shared.config.MODALITIES = (t1c, t1n, t2f, t2w). If None,
            all gates start at zero (uniform softmax = no XAI prior — useful
            as the ablation control variant).
    """

    SUBREGIONS: tuple[str, str, str] = ("wt", "tc", "et")

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        head_hidden: int = 16,
        gate_init_logits: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        assert out_channels == 3, "This model always predicts 3 sub-regions"
        self.num_modalities = in_channels

        # ---- Encoder (identical to M5) ----------------------------------
        self.enc1 = ResidualBlock3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ResidualBlock3D(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ResidualBlock3D(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ResidualBlock3D(128, 256)
        self.pool4 = nn.MaxPool3d(2)

        # ---- Bottleneck (identical) -------------------------------------
        self.bottleneck = ResidualBlock3D(256, 256)

        # ---- Decoder (identical) ----------------------------------------
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock3D(128 + 256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock3D(64 + 128, 64)
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock3D(32 + 64, 32)
        self.up4 = nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock3D(32 + 32, 32)

        # ---- XAI-guided per-sub-region heads (the new part) ------------
        if gate_init_logits is None:
            gate_init_logits = torch.zeros(3, in_channels)
        if tuple(gate_init_logits.shape) != (3, in_channels):
            raise ValueError(
                f"gate_init_logits must be (3, {in_channels}); "
                f"got {tuple(gate_init_logits.shape)}"
            )
        self.heads = nn.ModuleList([
            XAIGuidedSubregionHead(
                shared_channels=32,
                num_modalities=in_channels,
                hidden_channels=head_hidden,
                gate_init_logits=gate_init_logits[i],
            )
            for i in range(3)
        ])

    # ------------------------------------------------------------------
    # Shared trunk (encoder + decoder)
    # ------------------------------------------------------------------
    def _shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the M5-identical backbone, returning the (32, H, W, D) feature map."""
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        b = self.bottleneck(self.pool4(s4))
        d1 = self.dec1(torch.cat([self.up1(b), s4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), s3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), s2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), s1], dim=1))
        return d4

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self._shared_forward(x)
        head_logits = [head(shared, x) for head in self.heads]    # 3 x (B, 1, H, W, D)
        return torch.cat(head_logits, dim=1)                      # (B, 3, H, W, D)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Bottleneck features at (B, 256, H/16, W/16, D/16) — same contract
        as M1-M5 so this model is ensemblable via the latent-fusion head.
        """
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        return self.bottleneck(self.pool4(s4))

    # ------------------------------------------------------------------
    # Interpretation helpers
    # ------------------------------------------------------------------
    def gate_probabilities(self) -> torch.Tensor:
        """Return the current (3, 4) softmax-normalised modality gates.

        Rows ordered (WT, TC, ET); columns ordered to match the input channels
        the model was constructed with (typically t1c, t1n, t2f, t2w from
        preprocess_patient_multimodal).
        """
        return torch.stack([h.gate_probabilities for h in self.heads], dim=0)


# ---------------------------------------------------------------------------
# XAI initialisation helper
# ---------------------------------------------------------------------------
# Per shared.config.MODALITIES, input channel order is (t1c, t1n, t2f, t2w).
_MODALITY_ORDER = ("t1c", "t1n", "t2f", "t2w")
_SUBREGIONS = ("wt", "tc", "et")


def build_xai_gate_init(
    importance_json_path: str | Path,
    eps: float = 1e-3,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Build the (3, 4) gate-init-logits tensor from XAI ablation scores.

    Strategy:
        - Read per-sub-region per-modality importance from
          modality_importance_scores.json (Dice drops when each modality
          is zeroed at M5 inference).
        - Add eps to every drop so log(0) doesn't appear.
        - Normalise into a per-sub-region probability vector.
        - Take log to get softmax-input logits (so softmax(logits) reproduces
          the desired probability at training step 0).
        - Optional ``temperature`` < 1.0 sharpens the prior, > 1.0 softens it.

    Returns
    -------
    torch.Tensor shape (3, 4), with rows in (wt, tc, et) order and columns
    in (t1c, t1n, t2f, t2w) order — matches shared.config.MODALITIES.
    """
    data = json.loads(Path(importance_json_path).read_text(encoding="utf-8"))
    if temperature <= 0:
        raise ValueError(f"temperature must be positive; got {temperature}")

    out = torch.zeros(len(_SUBREGIONS), len(_MODALITY_ORDER), dtype=torch.float32)
    for i, sr in enumerate(_SUBREGIONS):
        scores = torch.tensor(
            [data[sr][m] + eps for m in _MODALITY_ORDER],
            dtype=torch.float32,
        )
        probs = scores / scores.sum()
        out[i] = torch.log(probs) / temperature
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

    torch.manual_seed(42)

    # Random-init variant (control)
    model_random = XAIGuidedMultimodalUNet3D(in_channels=4, out_channels=3)
    x = torch.randn(2, 4, 64, 64, 64)
    out = model_random(x)
    print(f"[random init]  output shape: {tuple(out.shape)}")
    print(f"[random init]  gate probs (uniform expected):")
    print(model_random.gate_probabilities().detach().numpy().round(3))
    assert tuple(out.shape) == (2, 3, 64, 64, 64)

    # XAI-init variant (the headline configuration)
    importance_path = _Path(__file__).resolve().parent.parent / \
        "results" / "tables" / "modality_importance_scores.json"
    if importance_path.exists():
        gate_init = build_xai_gate_init(importance_path)
        model_xai = XAIGuidedMultimodalUNet3D(
            in_channels=4, out_channels=3, gate_init_logits=gate_init,
        )
        out_xai = model_xai(x)
        print(f"\n[XAI init]     output shape: {tuple(out_xai.shape)}")
        print(f"[XAI init]     gate probs at init (rows: wt/tc/et, "
              f"cols: t1c/t1n/t2f/t2w):")
        print(model_xai.gate_probabilities().detach().numpy().round(3))
        assert tuple(out_xai.shape) == (2, 3, 64, 64, 64)

        # Backward smoke
        target = (torch.rand_like(out_xai) > 0.7).float()
        loss = F.binary_cross_entropy_with_logits(out_xai, target)
        loss.backward()
        # Gate gradients should be nonzero — confirms the XAI parameters
        # actually receive training signal.
        for i, sr in enumerate(("wt", "tc", "et")):
            g = model_xai.heads[i].modality_gate.grad
            print(f"[XAI init]     {sr} gate.grad norm: {g.norm().item():.6f}")

        # forward_features contract
        feats = model_xai.forward_features(x)
        print(f"\n[XAI init]     forward_features shape: {tuple(feats.shape)}")
        assert tuple(feats.shape) == (2, 256, 4, 4, 4)
    else:
        print(f"\n[XAI init]     {importance_path} not found — skipping XAI-init test")

    n_params = sum(p.numel() for p in model_random.parameters())
    print(f"\ntotal params: {n_params:,}")
    print("smoke test: PASSED")
