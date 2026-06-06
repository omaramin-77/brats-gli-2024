"""nn.Module wrappers that present an ensemble as a single forward-callable model.

The two trained ensemble models — Pipeline C (Latent-Space Fusion) and Pipeline D
(LateEnsemble stacker) — compose multiple unimodal models. Their forward pass is
not "one tensor in, one tensor out"; the natural call signatures are
``fusion({"t1c": ..., "t1n": ...})`` and ``stacker(stacked_logits)``. Both are
incompatible with MONAI's ``sliding_window_inference(predictor=model)``.

These wrappers expose the standard signature ``forward(x) -> logits`` so the
existing inference path in ``interface/backend/inference.py`` works unchanged:

  - Input  : (B, 4, P, P, P), channel order (t1c, t1n, t2f, t2w) — matches
             ``shared.config.MODALITIES`` and what ``preprocess_patient_multimodal``
             returns.
  - Output : (B, 3, P, P, P) raw logits for (WT, TC, ET).

These wrappers are self-loading: ``__init__`` materialises every required
checkpoint. The backend registry skips its usual ``load_state_dict`` step for
entries whose ``is_self_loading`` flag is set.

Memory note: each wrapper holds independent copies of all submodel weights, so
loading both at once is heavy. The registry's LRU eviction (cap 3 by default)
keeps the working set bounded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from shared.config import CHECKPOINT_DIR, MODALITIES, TABLES_DIR
from interface.backend.config import DEVICE


# Channel index of each modality in the (B, 4, ...) multimodal input tensor.
# Must match shared.config.MODALITIES = ("t1c", "t1n", "t2f", "t2w").
_MOD_CHANNEL = {mod: idx for idx, mod in enumerate(MODALITIES)}

# Which member's encoder handles which modality.
_MOD_TO_MEMBER = {
    "t1c": "m2",
    "t1n": "m1",
    "t2f": "m4",
    "t2w": "m3",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _load_state_into(model: nn.Module, ckpt_path: Path, label: str) -> None:
    """Load model weights from a checkpoint, surfacing the load diff."""
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{label}: checkpoint not found at {ckpt_path}"
        )
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload.get("model_state", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # Strict=False because some checkpoints carry extra training-only
        # heads (e.g. M1's deep supervision auxiliaries). A large diff would
        # indicate a real architecture mismatch — log so it's visible.
        print(
            f"[ensemble-wrap] {label}: load_state_dict diff "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )


def _build_m1() -> nn.Module:
    from member1_T1n.model import ResidualUNet3D
    return ResidualUNet3D(in_channels=1, out_channels=3)


def _build_m2() -> nn.Module:
    from member2_T1c.model import AttentionUNet3D
    return AttentionUNet3D(in_channels=1, out_channels=3)


def _build_m3() -> nn.Module:
    from member3_T2w.model import M3T2wSwinUNETR
    # img_size=PATCH_SIZE matches the trained checkpoint. We always call this
    # via sliding_window_inference, which feeds (B, 1, P, P, P) patches.
    from shared.config import PATCH_SIZE
    return M3T2wSwinUNETR(
        img_size=PATCH_SIZE, in_channels=1, out_channels=3, feature_size=24,
    )


def _build_m4() -> nn.Module:
    from member4_T2f.model import build_model
    return build_model(arch="resunet", in_channels=1, out_channels=3)


def _build_m5() -> nn.Module:
    from member5_multimodal.model import MultimodalUNet3D
    return MultimodalUNet3D(in_channels=4, out_channels=3)


_MEMBER_BUILDERS = {
    "m1": (_build_m1, "M1_T1n_ResUNet_best.pt"),
    "m2": (_build_m2, "M2_T1c_AttUNet_best.pt"),
    "m3": (_build_m3, "M3_T2w_SwinUNETR_best.pt"),
    "m4": (_build_m4, "M4_T2f_resunet_best.pt"),
    "m5": (_build_m5, "M5_Multimodal_4ch_best.pt"),
}


def _build_and_load_member(short: str) -> nn.Module:
    builder, ckpt_name = _MEMBER_BUILDERS[short]
    model = builder()
    _load_state_into(model, CHECKPOINT_DIR / ckpt_name, short.upper())
    return model


def _eval_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Single forward pass that normalises away M1's training-mode tuple."""
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


# ---------------------------------------------------------------------------
# Pipeline C — Latent Space Fusion Ensemble
# ---------------------------------------------------------------------------
class PipelineCInferenceWrapper(nn.Module):
    """Wraps Pipeline C (LatentFusionEnsemble) for end-to-end inference.

    Composition at forward time:
        1. Pull each modality channel out of the (B, 4, ...) multimodal patch.
        2. Run the corresponding member's ``forward_features`` to get a
           (B, 256, P/16, P/16, P/16) bottleneck.
        3. Hand the dict of bottlenecks to the trained LatentFusionEnsemble.
        4. Return its (B, 3, P, P, P) logits.

    The 3-modality ablation variant (``modalities=("t1c","t1n","t2f")``) skips
    the M3/T2w encoder entirely — saves ~13 s/inference on a small GPU and
    matches the training-time architecture for that variant.
    """

    def __init__(
        self,
        fusion_ckpt_name: str = "M5_LatentFusion_XAI_best.pt",
        modalities: Sequence[str] = MODALITIES,
        normalise_inputs: bool = True,
    ) -> None:
        super().__init__()
        # Lazy import — keeps the backend importable when ensemble code paths
        # have an unrelated transient error.
        from ensemble.fusion import LatentFusionEnsemble, build_gate_bias

        self.modalities = tuple(modalities)
        unknown = [m for m in self.modalities if m not in MODALITIES]
        if unknown:
            raise ValueError(
                f"Unknown fusion modalities {unknown}; must be subset of {MODALITIES}"
            )

        # Build + load only the encoders we actually need (3-mod runs skip one).
        self._encoders: dict[str, nn.Module] = {}
        for mod in self.modalities:
            member_short = _MOD_TO_MEMBER[mod]
            enc = _build_and_load_member(member_short)
            enc.eval()
            # Freeze — encoders are inference-only in the ensemble path.
            for p in enc.parameters():
                p.requires_grad_(False)
            # Register as a submodule so .to(device) carries it along.
            self.add_module(f"enc_{mod}", enc)
            self._encoders[mod] = enc

        # Build the fusion head with the same architecture options used at
        # train time, then load its trained weights.
        importance_path = TABLES_DIR / "modality_importance_scores.json"
        gate_bias = build_gate_bias(importance_path, modalities=self.modalities)
        self.fusion = LatentFusionEnsemble(
            gate_bias_init=gate_bias,
            modalities=self.modalities,
            normalise_inputs=normalise_inputs,
        )
        _load_state_into(
            self.fusion,
            CHECKPOINT_DIR / fusion_ckpt_name,
            f"PipelineC ({fusion_ckpt_name})",
        )
        self.fusion.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 4, P, P, P) raw multimodal patch.
        Returns (B, 3, P, P, P) logits.
        """
        features: dict[str, torch.Tensor] = {}
        for mod in self.modalities:
            channel = _MOD_CHANNEL[mod]
            xi = x[:, channel:channel + 1]
            enc = self._encoders[mod]
            # forward_features returns (B, 256, P/16, P/16, P/16)
            features[mod] = enc.forward_features(xi)
        logits, _attn = self.fusion(features)
        return logits


# ---------------------------------------------------------------------------
# Pipeline D — LateEnsemble stacker
# ---------------------------------------------------------------------------
class PipelineDInferenceWrapper(nn.Module):
    """Wraps the LateEnsemble PerSubregionStacker for end-to-end inference.

    Composition at forward time:
        1. Run each of M1–M5 on its modality (M1–M4 single-channel slices,
           M5 the full 4-channel input).
        2. Stack the five logit tensors to (B, 5, 3, P, P, P).
        3. Pass through the 15-scalar stacker → weighted (B, 3, P, P, P) logits.

    The stacker checkpoint records ``members_short`` order; we assert it
    matches the position order used here so weight columns never get
    mis-assigned.
    """

    # The stacker expects member features in this fixed order — see
    # LateEnsemble/members.py:MEMBER_SPECS. Do not reorder.
    _MEMBER_ORDER: tuple[str, ...] = ("m1", "m2", "m3", "m4", "m5")

    def __init__(
        self,
        stacker_ckpt_name: str = "LateEnsemble_stacker_best.pt",
    ) -> None:
        super().__init__()
        from LateEnsemble.stacker import PerSubregionStacker

        # Build + load each submodel, frozen for inference.
        for short in self._MEMBER_ORDER:
            model = _build_and_load_member(short)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.add_module(f"member_{short}", model)

        # Load the 15-scalar stacker checkpoint and validate member order.
        ckpt_path = CHECKPOINT_DIR / stacker_ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Stacker checkpoint not found at {ckpt_path}. "
                "Run python LateEnsemble/train_stacker.py first."
            )
        payload = torch.load(ckpt_path, map_location="cpu")
        ckpt_members = payload.get("members_short")
        if ckpt_members is not None and tuple(ckpt_members) != self._MEMBER_ORDER:
            raise RuntimeError(
                f"LateEnsemble stacker checkpoint was saved with member order "
                f"{ckpt_members}, but the inference wrapper expects "
                f"{self._MEMBER_ORDER}. Either the checkpoint or "
                "LateEnsemble/members.MEMBER_SPECS has drifted."
            )
        self.stacker = PerSubregionStacker(
            num_members=len(self._MEMBER_ORDER), num_subregions=3,
        )
        self.stacker.load_state_dict(payload["stacker_state"])
        self.stacker.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 4, P, P, P) raw multimodal patch.
        Returns (B, 3, P, P, P) logits.
        """
        # Slice out each modality. Channel mapping matches MODALITIES order.
        t1c = x[:, _MOD_CHANNEL["t1c"]:_MOD_CHANNEL["t1c"] + 1]
        t1n = x[:, _MOD_CHANNEL["t1n"]:_MOD_CHANNEL["t1n"] + 1]
        t2f = x[:, _MOD_CHANNEL["t2f"]:_MOD_CHANNEL["t2f"] + 1]
        t2w = x[:, _MOD_CHANNEL["t2w"]:_MOD_CHANNEL["t2w"] + 1]

        l1 = _eval_logits(getattr(self, "member_m1"), t1n)   # M1 expects T1n
        l2 = _eval_logits(getattr(self, "member_m2"), t1c)   # M2 expects T1c
        l3 = _eval_logits(getattr(self, "member_m3"), t2w)   # M3 expects T2w
        l4 = _eval_logits(getattr(self, "member_m4"), t2f)   # M4 expects T2f
        l5 = _eval_logits(getattr(self, "member_m5"), x)     # M5 expects all 4

        # Stack to (B, M=5, S=3, P, P, P) in MEMBER_ORDER.
        stacked = torch.stack([l1, l2, l3, l4, l5], dim=1)
        return self.stacker(stacked)


# ---------------------------------------------------------------------------
# Convenience builders for the registry catalogue
# ---------------------------------------------------------------------------
def build_pipeline_c_4mod() -> nn.Module:
    return PipelineCInferenceWrapper(
        fusion_ckpt_name="M5_LatentFusion_XAI_best.pt",
        modalities=MODALITIES,
    )


def build_pipeline_c_3mod() -> nn.Module:
    return PipelineCInferenceWrapper(
        fusion_ckpt_name="M5_LatentFusion_XAI_only_t1ct1nt2f_best.pt",
        modalities=("t1c", "t1n", "t2f"),
    )


def build_pipeline_d() -> nn.Module:
    return PipelineDInferenceWrapper(
        stacker_ckpt_name="LateEnsemble_stacker_best.pt",
    )
