"""GET /models — list every registered model with its current enabled state."""
from __future__ import annotations

from fastapi import APIRouter

from interface.backend.registry import registry
from interface.backend.schemas import ModelCatalog, ModelInfo

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelCatalog)
def list_models() -> ModelCatalog:
    items = []
    for e in registry.list():
        items.append(ModelInfo(
            key=e.key,
            display_name=e.display_name,
            short_name=e.short_name,
            modality=e.modality,
            in_channels=e.in_channels,
            architecture=e.architecture,
            description=e.description,
            badge=e.badge,
            enabled=e.enabled,
            load_error=e.load_error,
        ))
    return ModelCatalog(models=items)
