"""FastAPI entry point.

Run from the repo root:

    uvicorn interface.backend.main:app --reload --port 8000

The shared/ folder is auto-added to sys.path by ``config.py`` so the app
imports preprocessing/metrics/etc. without any further setup.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from interface.backend.config import CORS_ORIGINS, DEVICE, SESSION_TTL_HOURS
from interface.backend.routes import (
    comparisons as comparisons_route,
    health as health_route,
    models as models_route,
    predictions as predictions_route,
    reports as reports_route,
    sessions as sessions_route,
)
from interface.backend.sessions import sessions

log = logging.getLogger("brats.backend")
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Periodic session sweep
# ---------------------------------------------------------------------------
async def _session_sweeper() -> None:
    """Once an hour, delete sessions older than SESSION_TTL_HOURS."""
    while True:
        try:
            removed = sessions.sweep_expired()
            if removed:
                log.info("session sweep: removed %d expired sessions", removed)
        except Exception as exc:  # pragma: no cover
            log.exception("session sweep failed: %s", exc)
        await asyncio.sleep(3600.0)


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("brats backend starting on %s (session TTL %sh)", DEVICE, SESSION_TTL_HOURS)
    sweeper_task = asyncio.create_task(_session_sweeper())
    try:
        yield
    finally:
        sweeper_task.cancel()
        try:
            await sweeper_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Brain MRI Tumour Segmentation — BraTS GLI-2024",
    version="0.1.0",
    description=(
        "Inference + comparison backend for the team's 5 trained segmentation "
        "models (with a 6th slot reserved for a separately trained late-fusion "
        "model). Research prototype — not for clinical use."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(health_route.router)
app.include_router(models_route.router)
app.include_router(sessions_route.router)
app.include_router(predictions_route.router)
app.include_router(comparisons_route.router)
app.include_router(reports_route.router)


# ---------------------------------------------------------------------------
# Exception handlers — return clean JSON instead of the default HTML 500 page,
# which is what the Loveable frontend will expect.
# ---------------------------------------------------------------------------
@app.exception_handler(KeyError)
def _key_error(_, exc: KeyError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
def _file_not_found(_, exc: FileNotFoundError):
    # Most commonly: a model can't run because a required modality wasn't
    # uploaded. The shared preprocessing raises FileNotFoundError with a
    # readable message; re-surface it.
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ValueError)
def _value_error(_, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "brats-backend",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }
