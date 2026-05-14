"""Single source of truth for hyperparameters and environment configuration.

Members import names from this module. Local override is fine inside a member's
own train.py (e.g. ``LR = 5e-5``), but this file should not be edited on main
unless the change has been agreed by the whole team.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

GLOBAL_SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Repo root = parent of this file's parent (.../brats-gli-2024/shared/config.py)
REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits"
DATA_PATH_FILE = REPO_ROOT / "data" / "raw" / "DATA_PATH.txt"
DATA_PATH_LOCAL_FILE = REPO_ROOT / "data" / "raw" / "DATA_PATH.local.txt"
RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MLRUNS_DIR = REPO_ROOT / "experiments" / "mlruns"


def _read_first_nonempty_line(path: Path) -> str:
    """Return the first non-empty stripped line of ``path``, or '' if none."""
    if not path.exists():
        return ""
    raw = path.read_text(encoding="utf-8").strip().splitlines()
    return next((line.strip() for line in raw if line.strip()), "")


def get_data_root() -> str:
    """Read the absolute path to the raw BraTS folder.

    Resolution order:
        1. ``data/raw/DATA_PATH.local.txt`` (gitignored, per-machine override).
        2. ``data/raw/DATA_PATH.txt`` (committed fallback).

    Members create ``DATA_PATH.local.txt`` on their own machine and never push
    it; ``DATA_PATH.txt`` stays unchanged for the primary workstation.
    """
    candidate = _read_first_nonempty_line(DATA_PATH_LOCAL_FILE)
    if not candidate:
        candidate = _read_first_nonempty_line(DATA_PATH_FILE)

    if not candidate:
        raise FileNotFoundError(
            "Create data/raw/DATA_PATH.local.txt and put the absolute path to "
            "your local BraTS root on a single line. Do not commit this file."
        )

    if not os.path.isdir(candidate):
        raise FileNotFoundError(
            f"DATA_ROOT does not exist on this machine: {candidate}. "
            "Edit data/raw/DATA_PATH.local.txt with the path that is valid here."
        )
    return candidate


# ---------------------------------------------------------------------------
# GPU-aware defaults (auto-detected at import time)
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    _props = torch.cuda.get_device_properties(0)
    DEVICE_NAME = _props.name
    VRAM_GB = _props.total_memory / 1e9
else:
    DEVICE_NAME = "cpu"
    VRAM_GB = 0.0

# 14 GB cutoff: the 4080 (16 GB) gets 96**3 patches; 12 GB cards (3060/5070)
# fall back to 64**3 to leave headroom for AMP buffers and validation.
PATCH_SIZE = (96, 96, 96) if VRAM_GB >= 14 else (64, 64, 64)
BATCH_SIZE = 1
# Windows lacks a usable fork(), so DataLoader workers re-import the parent
# script and frequently deadlock. Keep 0 on Windows; bump to 4 on Linux/WSL2.
NUM_WORKERS = 0

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50  # 50 chosen over 100 to fit team hardware budget;
                # rely on EarlyStopper(patience=15) to halt earlier
                # if val_dice_wt plateaus.
PATIENCE = 15
VAL_EVERY_N_EPOCHS = 5
GRAD_CLIP_NORM = 1.0
AMP_ENABLED = True

# ---------------------------------------------------------------------------
# BraTS GLI-2024 labels (used inside dataset.py to build WT/TC/ET):
#   0 = background
#   1 = necrotic core (NCR)
#   2 = peritumoural oedema (ED)
#   3 = enhancing tumour (ET)
#   4 = resection cavity (RC, excluded from all sub-regions)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Modality ordering — fixed across the whole project so member5's stacked
# multimodal tensors line up with member1-4's per-channel checkpoints.
# ---------------------------------------------------------------------------
MODALITIES = ("t1c", "t1n", "t2f", "t2w")

# ---- 3-channel target convention (BraTS protocol) ----
# All models output 3 channels in this exact order.
TARGET_CHANNELS = 3
TARGET_CHANNEL_NAMES = ("wt", "tc", "et")

# ---- Best-metric for early stopping / checkpointing ----
# WT is the most stable signal early in training; using ET would
# stop training prematurely while ET is still warming up.
BEST_METRIC = "dice_wt"

print(
    f"[config] {DEVICE_NAME} | {VRAM_GB:.1f} GB | "
    f"patch={PATCH_SIZE} | AMP={AMP_ENABLED}"
)
