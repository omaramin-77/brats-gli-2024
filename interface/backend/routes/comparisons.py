"""POST /sessions/{sid}/compare        run-or-reuse a list of models
POST /sessions/{sid}/compare-all    same but for every enabled model"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from interface.backend.config import GPU_LOCK
from interface.backend.inference import get_or_run_prediction, missing_modalities
from interface.backend.registry import registry
from interface.backend.schemas import (
    CompareRequest,
    ComparisonResponse,
    PerSubregionMetrics,
)
from interface.backend.sessions import sessions

router = APIRouter(tags=["comparisons"])


async def _run_one(model_key: str, sid: str) -> tuple[dict | None, dict | None]:
    """Wrap a single inference call with the right error reporting shape."""
    try:
        entry = registry.get(model_key)
    except KeyError:
        return None, {"model_key": model_key, "reason": "unknown model"}
    if not entry.enabled:
        return None, {
            "model_key": model_key,
            "reason": entry.load_error or "model disabled",
        }
    missing = missing_modalities(model_key, sid)
    if missing:
        return None, {
            "model_key": model_key,
            "reason": f"missing uploads: {missing}",
        }

    async with GPU_LOCK:
        from anyio import to_thread
        try:
            metrics = await to_thread.run_sync(get_or_run_prediction, model_key, sid)
        except Exception as exc:  # pragma: no cover — surfacing the real reason
            return None, {"model_key": model_key, "reason": f"inference error: {exc}"}
    return metrics, None


def _pick(metrics: dict) -> dict:
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
@router.post("/sessions/{sid}/compare", response_model=ComparisonResponse)
async def compare(sid: str, body: CompareRequest) -> ComparisonResponse:
    if not sessions.exists(sid):
        raise HTTPException(404, f"session {sid} not found")

    results: list[PerSubregionMetrics] = []
    skipped: list[dict] = []
    for key in body.model_keys:
        metrics, error = await _run_one(key, sid)
        if metrics is not None:
            results.append(PerSubregionMetrics(**_pick(metrics)))
        elif error is not None:
            skipped.append(error)

    return ComparisonResponse(sid=sid, results=results, skipped=skipped)


@router.post("/sessions/{sid}/compare-all", response_model=ComparisonResponse)
async def compare_all(sid: str) -> ComparisonResponse:
    if not sessions.exists(sid):
        raise HTTPException(404, f"session {sid} not found")

    all_keys = [e.key for e in registry.list()]
    results: list[PerSubregionMetrics] = []
    skipped: list[dict] = []
    for key in all_keys:
        metrics, error = await _run_one(key, sid)
        if metrics is not None:
            results.append(PerSubregionMetrics(**_pick(metrics)))
        elif error is not None:
            skipped.append(error)

    return ComparisonResponse(sid=sid, results=results, skipped=skipped)
