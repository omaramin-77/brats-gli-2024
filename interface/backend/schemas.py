"""Pydantic request / response models.

Keep these tight: only the fields the frontend actually needs. Anything that
should be a download (mask, slice PNG) is delivered via a separate endpoint,
not embedded as base64.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health + catalog
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    device: str
    cuda_available: bool
    vram_gb: float
    models_total: int
    models_enabled: int
    models_loaded: list[str]


class ModelInfo(BaseModel):
    key: str
    display_name: str
    short_name: str
    modality: str
    in_channels: int
    architecture: str
    description: str
    badge: str
    enabled: bool
    load_error: Optional[str] = None


class ModelCatalog(BaseModel):
    models: list[ModelInfo]


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
class SessionInfo(BaseModel):
    sid: str
    created_at: float
    updated_at: float
    uploads: dict[str, str]
    predictions: list[str]


class UploadAck(BaseModel):
    sid: str
    kind: str
    filename: str
    size_bytes: int


# ---------------------------------------------------------------------------
# Predictions / metrics
# ---------------------------------------------------------------------------
class PerSubregionMetrics(BaseModel):
    """Metrics for a single model on a session."""
    model_config = {"protected_namespaces": ()}

    model_key: str
    model_display_name: str
    model_short_name: str
    architecture: str
    modality: str

    dice_wt: float
    dice_tc: float
    dice_et: float
    iou_wt: float
    iou_tc: float
    iou_et: float
    hd95_wt: float
    hd95_tc: float
    hd95_et: float

    volume_mm3_wt: float
    volume_mm3_tc: float
    volume_mm3_et: float

    inference_time_s: float


class PredictRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_key: str = Field(..., description="A key from /models")


class CompareRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_keys: list[str] = Field(..., min_length=2)


class ComparisonResponse(BaseModel):
    sid: str
    results: list[PerSubregionMetrics]
    skipped: list[dict]   # entries that failed with reason
