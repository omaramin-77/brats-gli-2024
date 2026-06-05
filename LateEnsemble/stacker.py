"""PerSubregionStacker — late-fusion weighted ensemble across M1–M5.

For each sub-region (WT, TC, ET) the final logit is a per-member-weighted
sum of the five members' logits for that sub-region. Weights are
non-negative and sum to 1 per sub-region (softmax over a length-5 vector).
This gives 5 × 3 = 15 trainable scalars.

Initialisation strategy
-----------------------
M5 (naive 4-channel baseline) is the consensus base — its TC/ET predictions
are roughly 2× any unimodal model's. We seed it with a fixed prior weight
``m5_prior`` (default 0.5). The remaining 1 − m5_prior is distributed
across M1–M4 in proportion to the per-sub-region modality importance
scores from member5_multimodal/ablation.py. Inverting the softmax then
gives the logit values that reproduce these target probabilities at
training step 0.

Stacking-only consumer: this module operates on cached logit tensors of
shape ``(B, 5, 3, 128, 128, 128)``. It does not see the original images.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SUBREGIONS = ("wt", "tc", "et")


class PerSubregionStacker(nn.Module):
    """5 members × 3 sub-regions = 15 softmax-input parameters.

    Forward expects ``member_logits`` of shape (B, 5, 3, H, W, D); returns
    the weighted sum (B, 3, H, W, D). Weights along the member axis are a
    softmax of ``self.logits`` for each sub-region independently.
    """

    def __init__(
        self,
        num_members: int = 5,
        num_subregions: int = 3,
        init_logits: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if init_logits is None:
            init_logits = torch.zeros(num_members, num_subregions)
        if tuple(init_logits.shape) != (num_members, num_subregions):
            raise ValueError(
                f"init_logits must be {(num_members, num_subregions)}, "
                f"got {tuple(init_logits.shape)}"
            )
        self.logits = nn.Parameter(init_logits.clone().float())

    @property
    def weights(self) -> torch.Tensor:
        """Softmax over members per sub-region → (M, S) probabilities."""
        return F.softmax(self.logits, dim=0)

    def forward(self, member_logits: torch.Tensor) -> torch.Tensor:
        """member_logits: (B, M, S, H, W, D) → (B, S, H, W, D)."""
        if member_logits.ndim != 6:
            raise ValueError(
                "Expected member_logits with shape (B, M, S, H, W, D); "
                f"got {tuple(member_logits.shape)}"
            )
        w = self.weights                                      # (M, S)
        w = w.view(1, w.shape[0], w.shape[1], 1, 1, 1)        # broadcast
        return (member_logits * w).sum(dim=1)


# ---------------------------------------------------------------------------
# Weight initialisation helpers
# ---------------------------------------------------------------------------
def _probs_to_logits(probs: torch.Tensor) -> torch.Tensor:
    """Convert a probability vector to softmax-equivalent logits.

    log(p) recovers p under softmax up to an additive constant, which
    cancels in softmax. We add a small epsilon so log(0) never appears.
    """
    eps = 1e-8
    return torch.log(probs.clamp_min(eps))


def build_uniform_logits(num_members: int = 5, num_subregions: int = 3) -> torch.Tensor:
    """All-zero logits → uniform softmax (control variant)."""
    return torch.zeros(num_members, num_subregions)


def build_xai_init_logits(
    importance_json: str | Path,
    members_short: list[str],
    ablation_keys: dict,
    m5_prior: float = 0.5,
) -> torch.Tensor:
    """Build per-sub-region softmax-input logits from XAI ablation scores.

    Strategy:
      For each sub-region the target softmax probability vector p has:
        - p[m5]  = m5_prior                                            (consensus base)
        - p[mi]  = (1 − m5_prior) · imp[sr][mod_of(mi)] / Σ_unimodal   (specialists)
      We return ``log(p)`` so softmax(log(p)) = p at step 0.

    Args:
        importance_json: path to results/tables/modality_importance_scores.json
        members_short:   ordered list of short names — must match
                         members.MEMBER_SPECS positions exactly.
        ablation_keys:   mapping short-name → modality key in the JSON
                         (e.g. {"m1": "t1n", ...}). M5 is NOT in this map
                         — it is recognised separately and given m5_prior.
        m5_prior:        fixed probability mass reserved for M5.
    """
    if not 0.0 < m5_prior < 1.0:
        raise ValueError(f"m5_prior must be in (0, 1); got {m5_prior}")

    data = json.loads(Path(importance_json).read_text(encoding="utf-8"))
    eps = 1e-3   # keep log(p) finite when a modality's drop is exactly 0

    num_members = len(members_short)
    num_subregions = len(SUBREGIONS)
    probs = torch.zeros(num_members, num_subregions)

    for sr_idx, sr in enumerate(SUBREGIONS):
        # First pass: collect unimodal importance scores for normalisation.
        unimodal_scores = {}
        for m_idx, short in enumerate(members_short):
            if short == "m5":
                continue
            mod_key = ablation_keys[short]
            unimodal_scores[m_idx] = float(data[sr][mod_key]) + eps
        unimodal_total = sum(unimodal_scores.values())

        # Second pass: assign probability mass.
        for m_idx, short in enumerate(members_short):
            if short == "m5":
                probs[m_idx, sr_idx] = m5_prior
            else:
                share = unimodal_scores[m_idx] / unimodal_total
                probs[m_idx, sr_idx] = (1.0 - m5_prior) * share

        # Sanity: should sum to 1 per sub-region.
        s = float(probs[:, sr_idx].sum())
        if not math.isclose(s, 1.0, abs_tol=1e-5):
            raise RuntimeError(
                f"XAI-init probs for sr={sr} did not sum to 1 (got {s:.6f})"
            )

    return _probs_to_logits(probs)


def format_weight_table(weights: torch.Tensor, members_short: list[str]) -> str:
    """Pretty-print a (M, S) weight table for logging / report."""
    header = f"{'member':<8} " + " ".join(f"{sr.upper():>8}" for sr in SUBREGIONS)
    lines = [header, "-" * len(header)]
    for m_idx, short in enumerate(members_short):
        row = f"{short:<8} " + " ".join(
            f"{float(weights[m_idx, sr_idx]):>8.4f}"
            for sr_idx in range(len(SUBREGIONS))
        )
        lines.append(row)
    return "\n".join(lines)
