"""Prediction endpoints:

POST /sessions/{sid}/predict                          run a single model
GET  /sessions/{sid}/predictions/{model_key}/mask.nii.gz   download NIfTI mask
GET  /sessions/{sid}/predictions/{model_key}/slice         overlay PNG slice
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

import numpy as np

from interface.backend.config import GPU_LOCK
from interface.backend.inference import get_or_run_prediction, missing_modalities
from interface.backend.registry import registry
from interface.backend.schemas import PerSubregionMetrics, PredictRequest
from interface.backend.sessions import sessions
from interface.backend.slices import slice_to_png_bytes, take_slice

router = APIRouter(tags=["predictions"])


# ---------------------------------------------------------------------------
@router.post(
    "/sessions/{sid}/predict",
    response_model=PerSubregionMetrics,
)
async def run_predict(sid: str, body: PredictRequest) -> PerSubregionMetrics:
    if not sessions.exists(sid):
        raise HTTPException(404, f"session {sid} not found")
    try:
        entry = registry.get(body.model_key)
    except KeyError:
        raise HTTPException(404, f"unknown model '{body.model_key}'")
    if not entry.enabled:
        raise HTTPException(503, f"model {body.model_key} not available: {entry.load_error}")

    missing = missing_modalities(body.model_key, sid)
    if missing:
        raise HTTPException(
            400,
            f"missing uploads for {body.model_key}: {missing}. "
            "Upload all required modalities (and seg.nii.gz for ground truth metrics).",
        )

    async with GPU_LOCK:
        # Run synchronously inside the GPU lock; the heavy lifting is the
        # forward pass, which is already a single contiguous CPU/GPU operation.
        # FastAPI runs sync handlers in a threadpool so we don't block the loop.
        from anyio import to_thread
        metrics = await to_thread.run_sync(get_or_run_prediction, body.model_key, sid)

    return PerSubregionMetrics(**_pick_metric_fields(metrics))


def _pick_metric_fields(metrics: dict) -> dict:
    """Drop any extra keys that aren't part of the schema so Pydantic's
    strict-by-default behaviour doesn't choke."""
    keys = (
        "model_key", "model_display_name", "model_short_name", "architecture", "modality",
        "dice_wt", "dice_tc", "dice_et",
        "iou_wt", "iou_tc", "iou_et",
        "hd95_wt", "hd95_tc", "hd95_et",
        "volume_mm3_wt", "volume_mm3_tc", "volume_mm3_et",
        "inference_time_s",
    )
    return {k: metrics.get(k) for k in keys}


# ---------------------------------------------------------------------------
@router.get("/sessions/{sid}/predictions/{model_key}/mask.nii.gz")
def download_mask(sid: str, model_key: str) -> FileResponse:
    if not sessions.has_prediction(sid, model_key):
        raise HTTPException(404, "no prediction; POST /predict first")
    path = sessions.prediction_dir(sid, model_key) / "mask.nii.gz"
    return FileResponse(
        path=path,
        media_type="application/gzip",
        filename=f"{sid}_{model_key}_mask.nii.gz",
    )


@router.get("/sessions/{sid}/predictions/{model_key}/metrics")
def get_metrics(sid: str, model_key: str) -> dict:
    if not sessions.has_prediction(sid, model_key):
        raise HTTPException(404, "no prediction; POST /predict first")
    return sessions.load_prediction_metrics(sid, model_key)


# ---------------------------------------------------------------------------
@router.get("/sessions/{sid}/predictions/{model_key}/slice")
def overlay_slice(
    sid: str,
    model_key: str,
    view: Literal["axial", "coronal", "sagittal"] = Query("axial"),
    index: int = Query(..., ge=0),
    region: Literal["wt", "tc", "et", "all"] = Query("all"),
    backdrop: Literal["t1n", "t1c", "t2w", "t2f"] = Query("t1c"),
) -> Response:
    """Render an MRI slice with the predicted overlay on top.

    backdrop : which modality to show beneath the overlay. T1c is the default
               because the contrast-enhanced image is the most readable for
               most viewers.
    """
    if not sessions.has_prediction(sid, model_key):
        raise HTTPException(404, "no prediction; POST /predict first")

    path = sessions.upload_path(sid, backdrop)
    if path is None:
        raise HTTPException(400, f"backdrop modality '{backdrop}' not uploaded")

    from shared.preprocessing import preprocess_patient
    vol, _ = preprocess_patient(str(sessions.upload_dir(sid)), backdrop)
    volume = vol[0]

    img_slice = take_slice(volume, view, index)
    mask = sessions.load_prediction_mask(sid, model_key)   # (3, H, W, D)
    mask_slice = np.stack([
        take_slice(mask[i].astype(np.uint8), view, index) for i in range(3)
    ], axis=0)

    png = slice_to_png_bytes(img_slice, mask_slice, region=region)
    return Response(content=png, media_type="image/png")
