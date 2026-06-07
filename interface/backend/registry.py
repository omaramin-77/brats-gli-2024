"""Model registry — declarative entries + lazy loading.

Why a registry instead of importing models on app startup
----------------------------------------------------------
Several 3D U-Nets resident on a single GPU at once OOMs anything under 16 GB.
We declare every model as data here, then materialise weights only when an
inference call actually needs them. An LRU cap evicts the least-recently-used
weights when the cache fills.

Catalogue (as of 2026-06)
-------------------------
- m1_t1n, m2_t1c, m3_t2w, m4_t2f       — the four unimodal specialists.
- m5_baseline                          — naive 4-channel multimodal U-Net.
- pipeline_c_xai_4mod                  — XAI-initialised Latent Space Fusion
                                         over M1–M4 bottlenecks.
- pipeline_c_xai_3mod                  — Pipeline C controlled ablation
                                         (drops T2w/M3).
- pipeline_d_late_ensemble             — XAI-weighted late ensemble (the
                                         LateEnsemble PerSubregionStacker).

The three ensemble entries are *self-loading*: their builders compose
multiple submodels and load every required state_dict in __init__. The
registry skips its standard state_dict path for those, marked by
``ModelEntry.is_self_loading=True``. Each ensemble also declares
``extra_required_checkpoints`` so the UI can show a precise reason when an
ensemble is disabled (e.g. "missing M3_T2w_SwinUNETR_best.pt").
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn

from interface.backend.config import CHECKPOINTS, DEVICE, MODEL_CACHE_SIZE


# ---------------------------------------------------------------------------
# Entry definition
# ---------------------------------------------------------------------------
@dataclass
class ModelEntry:
    key: str                    # stable identifier used in URLs
    display_name: str           # what the UI shows
    short_name: str             # compact label for comparison tables
    modality: str               # "t1n" | "t1c" | "t2w" | "t2f" | "multimodal"
    in_channels: int            # 1 for unimodal, 4 for multimodal
    architecture: str           # human-readable architecture name
    checkpoint_name: str        # filename stem under CHECKPOINTS used for the
                                # enabled/disabled probe. For ensembles whose
                                # weights are spread across multiple files this
                                # is the *gating* checkpoint — typically the
                                # fusion head or stacker.
    builder: Callable[[], nn.Module]
    description: str            # short blurb shown on the model card
    badge: str = ""             # optional UI badge ("Ensemble", "Baseline", "Coming soon")

    # When True, the registry calls ``builder()`` and trusts it to have already
    # loaded every required state_dict (this is how the ensemble wrappers
    # work — they materialise 4-5 unimodal checkpoints + the fusion head in
    # their own __init__). When False (the default for unimodal entries),
    # the registry loads the matching checkpoint into the model itself.
    is_self_loading: bool = False

    # Optional list of extra checkpoint stems that must exist on disk for this
    # entry to be considered enabled. For ensembles, these are the submodel
    # checkpoints the wrapper will try to load at build time. Spotting a
    # missing submodel here means the UI can flag the entry without paying
    # the import-and-fail cost when the user clicks it.
    extra_required_checkpoints: tuple[str, ...] = ()

    # populated by ModelRegistry
    enabled: bool = field(init=False, default=False)
    load_error: Optional[str] = field(init=False, default=None)

    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINTS / f"{self.checkpoint_name}.pt"

    def missing_checkpoints(self) -> list[Path]:
        """Return every required checkpoint that doesn't exist on disk."""
        missing: list[Path] = []
        if not self.checkpoint_path.exists():
            missing.append(self.checkpoint_path)
        for stem in self.extra_required_checkpoints:
            p = CHECKPOINTS / f"{stem}.pt"
            if not p.exists():
                missing.append(p)
        return missing


# ---------------------------------------------------------------------------
# Builders (factories — return a fresh nn.Module on each call)
# ---------------------------------------------------------------------------
# Builders are wrapped in lambdas so the model classes are NOT imported until
# a build is requested. This keeps `import registry` cheap and means a broken
# member module does not crash the whole app — only that one entry.

def _build_m1() -> nn.Module:
    from member1_T1n.model import ResidualUNet3D
    return ResidualUNet3D(in_channels=1, out_channels=3)


def _build_m2() -> nn.Module:
    from member2_T1c.model import AttentionUNet3D
    return AttentionUNet3D(in_channels=1, out_channels=3)


def _build_m3() -> nn.Module:
    from member3_T2w.model import M3T2wSwinUNETR
    # feature_size matches the checkpoint that was trained (see member3 train.py).
    return M3T2wSwinUNETR(
        img_size=(64, 64, 64),
        in_channels=1,
        out_channels=3,
        feature_size=24,
    )


def _build_m4() -> nn.Module:
    from member4_T2f.model import build_model
    return build_model(arch="resunet", in_channels=1, out_channels=3)


def _build_m5() -> nn.Module:
    from member5_multimodal.model import MultimodalUNet3D
    return MultimodalUNet3D(in_channels=4, out_channels=3)


def _build_m6_xai_guided() -> nn.Module:
    """XAI-guided multi-task M5. Builder must reproduce the architecture the
    checkpoint was trained against — including the (3, 4) gate init shape —
    so we always use the XAI-init variant here. The trained gate values
    are loaded from the checkpoint via the standard load_state_dict path.
    """
    from shared.config import TABLES_DIR
    from member6_xai_guided.model import (
        XAIGuidedMultimodalUNet3D, build_xai_gate_init,
    )
    importance_path = TABLES_DIR / "modality_importance_scores.json"
    gate_init = build_xai_gate_init(importance_path) if importance_path.exists() else None
    return XAIGuidedMultimodalUNet3D(
        in_channels=4, out_channels=3, gate_init_logits=gate_init,
    )


def _build_m6_xai_guided_random() -> nn.Module:
    """The 'random init' ablation control variant of M6."""
    from member6_xai_guided.model import XAIGuidedMultimodalUNet3D
    return XAIGuidedMultimodalUNet3D(
        in_channels=4, out_channels=3, gate_init_logits=None,
    )


def _build_pipeline_c_4mod() -> nn.Module:
    from interface.backend.ensemble_wrappers import build_pipeline_c_4mod
    return build_pipeline_c_4mod()


def _build_pipeline_c_3mod() -> nn.Module:
    from interface.backend.ensemble_wrappers import build_pipeline_c_3mod
    return build_pipeline_c_3mod()


def _build_pipeline_d() -> nn.Module:
    from interface.backend.ensemble_wrappers import build_pipeline_d
    return build_pipeline_d()


# ---------------------------------------------------------------------------
# Catalog — order here is the order the UI shows
# ---------------------------------------------------------------------------
_CATALOG: tuple[ModelEntry, ...] = (
    ModelEntry(
        key="m1_t1n",
        display_name="T1n — Residual U-Net",
        short_name="M1 T1n",
        modality="t1n",
        in_channels=1,
        architecture="3D Residual U-Net (deep supervision)",
        checkpoint_name="M1_T1n_ResUNet_best",
        builder=_build_m1,
        description="Native T1 expert. Residual blocks with deep supervision "
                    "to handle the sparse-positive ET sub-region.",
        badge="Modality expert",
    ),
    ModelEntry(
        key="m2_t1c",
        display_name="T1c — Attention U-Net",
        short_name="M2 T1c",
        modality="t1c",
        in_channels=1,
        architecture="3D Attention-Gated U-Net",
        checkpoint_name="M2_T1c_AttUNet_best",
        builder=_build_m2,
        description="Contrast-enhanced T1 expert. Attention gates suppress "
                    "irrelevant background so the bright enhancing tumour wins.",
        badge="Modality expert",
    ),
    ModelEntry(
        key="m3_t2w",
        display_name="T2w — Swin UNETR",
        short_name="M3 T2w",
        modality="t2w",
        in_channels=1,
        architecture="Swin-UNETR (MONAI)",
        checkpoint_name="M3_T2w_SwinUNETR_best",
        builder=_build_m3,
        description="T2-weighted expert. Shifted-window self-attention captures "
                    "long-range oedema context that local convs miss.",
        badge="Modality expert",
    ),
    ModelEntry(
        key="m4_t2f",
        display_name="T2-FLAIR — Residual U-Net",
        short_name="M4 T2f",
        modality="t2f",
        in_channels=1,
        architecture="3D Residual U-Net",
        checkpoint_name="M4_T2f_resunet_best",
        builder=_build_m4,
        description="FLAIR expert. Standardised modality + residual U-Net is the "
                    "strongest single-modality baseline.",
        badge="Modality expert",
    ),
    ModelEntry(
        key="m5_baseline",
        display_name="Naive Baseline — 4-channel",
        short_name="M5 Baseline",
        modality="multimodal",
        in_channels=4,
        architecture="3D Residual U-Net, 4-channel input",
        checkpoint_name="M5_Multimodal_4ch_best",
        builder=_build_m5,
        description="Stack all four modalities at the input. The numerical floor "
                    "any principled fusion model has to beat.",
        badge="Baseline",
    ),
    # ------------------------------------------------------------------
    # M6 — XAI-guided multi-task M5 (Proposal C)
    # ------------------------------------------------------------------
    ModelEntry(
        key="m6_xai_guided",
        display_name="XAI-Guided Multi-Task M5",
        short_name="M6 XAI-M5",
        modality="multimodal",
        in_channels=4,
        architecture="ResUNet + 3 sub-region-specialised heads with XAI-init modality gates",
        checkpoint_name="M6_XAIGuided_Multimodal_best",
        builder=_build_m6_xai_guided,
        description="M5's backbone with three sub-region-specialised heads. "
                    "Each head holds a (4,)-vector of per-modality gate weights "
                    "initialised from the XAI ablation scores — so at training "
                    "step 0 the WT head emphasises T2-FLAIR and the TC/ET heads "
                    "emphasise T1c. Gates are fully trainable; the prior is a "
                    "starting point, not a clamp.",
        badge="XAI-guided",
    ),
    ModelEntry(
        key="m6_xai_guided_random",
        display_name="XAI-Guided Multi-Task M5 (random init, ablation)",
        short_name="M6 random",
        modality="multimodal",
        in_channels=4,
        architecture="ResUNet + 3 sub-region-specialised heads with uniform gate init",
        checkpoint_name="M6_XAIGuided_Multimodal_random_init_best",
        builder=_build_m6_xai_guided_random,
        description="Same architecture as M6 but with uniform (random) gate "
                    "initialisation. Ablation control that isolates the value "
                    "of the XAI prior — if this matches M6 by the end of "
                    "training, the prior didn't earn its keep.",
        badge="Ablation control",
    ),
    # ------------------------------------------------------------------
    # Pipeline C — XAI-initialised Latent-Space Fusion (the headline ensemble)
    # ------------------------------------------------------------------
    ModelEntry(
        key="pipeline_c_xai_4mod",
        display_name="Pipeline C — Latent Fusion (4-mod, XAI-init)",
        short_name="PC-4mod",
        modality="multimodal",
        in_channels=4,
        architecture="LatentFusionEnsemble (LayerNorm + deep decoder) over M1–M4 bottlenecks",
        checkpoint_name="M5_LatentFusion_XAI_best",
        builder=_build_pipeline_c_4mod,
        description="The novel contribution. XAI ablation scores initialise the "
                    "per-voxel modality attention bias; the LayerNorm-equalised "
                    "M1–M4 bottlenecks are fused and decoded back to 128³.",
        badge="Ensemble",
        is_self_loading=True,
        extra_required_checkpoints=(
            "M1_T1n_ResUNet_best",
            "M2_T1c_AttUNet_best",
            "M3_T2w_SwinUNETR_best",
            "M4_T2f_resunet_best",
        ),
    ),
    ModelEntry(
        key="pipeline_c_xai_3mod",
        display_name="Pipeline C — Latent Fusion (3-mod, drop T2w)",
        short_name="PC-3mod",
        modality="multimodal",
        in_channels=4,
        architecture="LatentFusionEnsemble over M1, M2, M4 (M3/T2w excluded)",
        checkpoint_name="M5_LatentFusion_XAI_only_t1ct1nt2f_best",
        builder=_build_pipeline_c_3mod,
        description="Controlled ablation against the 4-modality fusion. Drops "
                    "the Swin/T2w branch — the modality the ablation study "
                    "ranked least important and whose bottleneck distribution "
                    "is least compatible with the residual-net features.",
        badge="Ensemble (ablation)",
        is_self_loading=True,
        extra_required_checkpoints=(
            "M1_T1n_ResUNet_best",
            "M2_T1c_AttUNet_best",
            "M4_T2f_resunet_best",
        ),
    ),
    # ------------------------------------------------------------------
    # Pipeline D — XAI-weighted late ensemble (LateEnsemble stacker)
    # ------------------------------------------------------------------
    ModelEntry(
        key="pipeline_d_late_ensemble",
        display_name="Pipeline D — XAI-Weighted Late Ensemble",
        short_name="PD-Stacker",
        modality="multimodal",
        in_channels=4,
        architecture="PerSubregionStacker (15 trainable weights, softmax over M1–M5)",
        checkpoint_name="LateEnsemble_stacker_best",
        builder=_build_pipeline_d,
        description="XAI ablation scores initialise per-sub-region softmax "
                    "weights over M1–M5's full-resolution logit predictions; "
                    "weights are fit on the validation split. Best safety net "
                    "if Pipeline C trails the naive baseline.",
        badge="Ensemble",
        is_self_loading=True,
        extra_required_checkpoints=(
            "M1_T1n_ResUNet_best",
            "M2_T1c_AttUNet_best",
            "M3_T2w_SwinUNETR_best",
            "M4_T2f_resunet_best",
            "M5_Multimodal_4ch_best",
        ),
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class _LRUWeights:
    """Tiny LRU around loaded model state.

    Storing the entire ``nn.Module`` (not just the state_dict) means
    repeated inference calls skip the re-instantiation cost. Eviction
    releases CUDA memory by deleting the module reference and calling
    ``torch.cuda.empty_cache()``.
    """

    def __init__(self, cap: int) -> None:
        self._cap = max(1, int(cap))
        self._store: "OrderedDict[str, nn.Module]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[nn.Module]:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
            return None

    def put(self, key: str, model: nn.Module) -> None:
        with self._lock:
            self._store[key] = model
            self._store.move_to_end(key)
            while len(self._store) > self._cap:
                evicted_key, evicted_model = self._store.popitem(last=False)
                del evicted_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class ModelRegistry:
    """Singleton-style registry. Use the module-level ``registry`` instance."""

    def __init__(self, catalog: tuple[ModelEntry, ...] = _CATALOG) -> None:
        self._entries: dict[str, ModelEntry] = {e.key: e for e in catalog}
        self._cache = _LRUWeights(MODEL_CACHE_SIZE)
        self._refresh_enabled_flags()

    # ----- catalog inspection -----
    def list(self) -> list[ModelEntry]:
        # Recheck flags on every list call so the frontend sees the late-fusion
        # entry flip to enabled as soon as the user drops the checkpoint in,
        # without restarting the server.
        self._refresh_enabled_flags()
        return list(self._entries.values())

    def get(self, key: str) -> ModelEntry:
        if key not in self._entries:
            raise KeyError(f"Unknown model '{key}'")
        return self._entries[key]

    def _refresh_enabled_flags(self) -> None:
        for entry in self._entries.values():
            missing = entry.missing_checkpoints()
            if missing:
                entry.enabled = False
                names = ", ".join(p.name for p in missing)
                entry.load_error = f"missing checkpoint(s): {names}"
                continue
            entry.enabled = True
            entry.load_error = None

    # ----- weight loading -----
    def load(self, key: str) -> nn.Module:
        """Return a ready-to-infer module on DEVICE. Cached after first call."""
        entry = self.get(key)
        if not entry.enabled:
            raise RuntimeError(
                f"Model '{key}' is not available: {entry.load_error or 'not enabled'}"
            )

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        model = entry.builder()

        if not entry.is_self_loading:
            # Standard unimodal/baseline path: the builder returns a blank model
            # and we splice the matching checkpoint into it here.
            payload = torch.load(entry.checkpoint_path, map_location="cpu")
            state = payload.get("model_state", payload)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                # Strict=False is intentional (some checkpoints have extra heads),
                # but we surface the diff so a real architecture mismatch shows up
                # in logs rather than silently corrupting predictions.
                print(
                    f"[registry] {key}: load_state_dict diff — "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )
        else:
            # Ensemble path: the builder (see interface/backend/ensemble_wrappers.py)
            # has already loaded every submodel + the fusion head from disk.
            # The registry's job here is just to move it to DEVICE and cache.
            print(f"[registry] {key}: self-loading ensemble materialised")

        model.to(DEVICE)
        model.eval()
        self._cache.put(key, model)
        return model

    def loaded_keys(self) -> list[str]:
        return self._cache.keys()

    def evict_all(self) -> None:
        self._cache.clear()


registry = ModelRegistry()
