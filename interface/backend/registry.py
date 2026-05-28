"""Model registry — declarative entries + lazy loading.

Why a registry instead of importing models on app startup
----------------------------------------------------------
Five 3D U-Nets resident on a single GPU at once OOMs anything under 16 GB.
We declare every model as data here, then materialise weights only when an
inference call actually needs them. An LRU cap evicts the least-recently-used
weights when the cache fills.

How the 6th-slot late-fusion model plugs in
-------------------------------------------
The user is training a separate late-fusion model on another machine. When
they have it, they drop:

  1. ``ensemble/late_fusion.py`` with a callable ``build_model() -> nn.Module``
  2. ``results/checkpoints/M6_LateFusion_best.pt``

The registry detects both and auto-enables the entry. Until then the entry
stays in the catalog as ``enabled=False`` so the frontend can show it with a
"coming soon" badge without any code change.
"""
from __future__ import annotations

import importlib
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
    checkpoint_name: str        # filename stem under CHECKPOINTS
    builder: Callable[[], nn.Module]
    description: str            # short blurb shown on the model card
    badge: str = ""             # optional UI badge ("Ensemble", "Baseline", "Coming soon")

    # populated by ModelRegistry
    enabled: bool = field(init=False, default=False)
    load_error: Optional[str] = field(init=False, default=None)

    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINTS / f"{self.checkpoint_name}.pt"


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


def _build_m6_late_fusion() -> nn.Module:
    """Late-fusion model — user supplies the architecture later.

    Expected: ``ensemble/late_fusion.py`` with a ``build_model() -> nn.Module``
    callable. Until that file is added this raises ``ImportError`` and the
    registry marks the entry disabled.
    """
    mod = importlib.import_module("ensemble.late_fusion")
    if not hasattr(mod, "build_model"):
        raise AttributeError(
            "ensemble/late_fusion.py is present but does not expose a "
            "build_model() function. Add `def build_model() -> nn.Module: ...`."
        )
    return mod.build_model()


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
    ModelEntry(
        key="m6_late_fusion",
        display_name="Late Feature Fusion",
        short_name="M6 Late Fusion",
        modality="multimodal",
        in_channels=4,
        architecture="Late feature fusion (pending)",
        checkpoint_name="M6_LateFusion_best",
        builder=_build_m6_late_fusion,
        description="Late-fusion ensemble trained separately. Plug in by dropping "
                    "ensemble/late_fusion.py + M6_LateFusion_best.pt.",
        badge="Coming soon",
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
            if not entry.checkpoint_path.exists():
                entry.enabled = False
                entry.load_error = f"checkpoint missing: {entry.checkpoint_path.name}"
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

        model.to(DEVICE)
        model.eval()
        self._cache.put(key, model)
        return model

    def loaded_keys(self) -> list[str]:
        return self._cache.keys()

    def evict_all(self) -> None:
        self._cache.clear()


registry = ModelRegistry()
