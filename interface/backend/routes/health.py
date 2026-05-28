"""GET /health — small introspection endpoint useful for liveness probes
and for the frontend to know which device the backend is using."""
from __future__ import annotations

from fastapi import APIRouter

import torch

from interface.backend.config import DEVICE
from interface.backend.registry import registry
from interface.backend.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    vram = 0.0
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    entries = registry.list()
    return HealthResponse(
        status="ok",
        device=str(DEVICE),
        cuda_available=torch.cuda.is_available(),
        vram_gb=round(vram, 2),
        models_total=len(entries),
        models_enabled=sum(1 for e in entries if e.enabled),
        models_loaded=registry.loaded_keys(),
    )
