"""Backend-side configuration.

The repo's ``shared/config.py`` is the source of truth for everything that
affects training and inference numerics (PATCH_SIZE, CHECKPOINT_DIR, modality
order, target channel names). This module only adds web-layer concerns:
session storage, CORS origins, GPU concurrency limit.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Ensure the repo root is importable so ``from shared.config import ...`` works
# regardless of where uvicorn is launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from shared.config import CHECKPOINT_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parent

# Session storage: uploaded NIfTI volumes + cached prediction outputs.
# Override with $BRATS_SESSION_DIR if you want it elsewhere (e.g. on a fast SSD).
SESSION_ROOT = Path(os.environ.get("BRATS_SESSION_DIR", BACKEND_ROOT / "sessions"))
SESSION_ROOT.mkdir(parents=True, exist_ok=True)

# Where trained checkpoints live. Re-exported here so routes don't need to
# import from shared/ themselves.
CHECKPOINTS = CHECKPOINT_DIR

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One inference at a time. With BATCH_SIZE=1 and 3D U-Nets at 128**3 a second
# concurrent request OOMs even an 8 GB card. The lock is held around the
# actual forward pass only, not around upload or preprocessing.
GPU_LOCK = asyncio.Lock()

# How many model weights to keep resident in memory before evicting the LRU.
# Compare-all touches 5+ models; 3 is the floor that keeps a single user warm.
MODEL_CACHE_SIZE = int(os.environ.get("BRATS_MODEL_CACHE", "3"))

# Hours of idle time before a session is eligible for cleanup. A background
# sweep in main.py honours this.
SESSION_TTL_HOURS = float(os.environ.get("BRATS_SESSION_TTL_HOURS", "6"))

# ---------------------------------------------------------------------------
# CORS — Vite default + CRA default. Add your deployment origin via the env var.
# ---------------------------------------------------------------------------
_DEFAULT_ORIGINS = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("BRATS_CORS_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if o.strip()
]

# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------
# BraTS NIfTI volumes are typically 5-30 MB compressed. 200 MB is generous and
# still blocks accidental dataset uploads.
MAX_UPLOAD_BYTES = int(os.environ.get("BRATS_MAX_UPLOAD_MB", "200")) * 1024 * 1024

# Modalities the user can upload. The seg key is optional at upload time but
# required for ground-truth metrics; the routes enforce that per-endpoint.
UPLOAD_KINDS = ("t1n", "t1c", "t2w", "t2f", "seg")
