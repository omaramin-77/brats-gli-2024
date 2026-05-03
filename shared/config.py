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
RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MLRUNS_DIR = REPO_ROOT / "experiments" / "mlruns"


def get_data_root() -> str:
    """Read the absolute path to the raw BraTS folder from DATA_PATH.txt.

    Each team machine writes its own DATA_PATH.txt the first time it clones.
    The file is committed but contains a local absolute path; members update
    it on their own machine and never push the change.
    """
    if not DATA_PATH_FILE.exists():
        raise FileNotFoundError(
            f"DATA_PATH.txt missing at {DATA_PATH_FILE}. "
            "Write the absolute path to the BraTS root into this file."
        )
    raw = DATA_PATH_FILE.read_text(encoding="utf-8").strip().splitlines()
    candidate = next((line.strip() for line in raw if line.strip()), "")
    if not candidate:
        raise ValueError(f"DATA_PATH.txt at {DATA_PATH_FILE} is empty.")
    if not os.path.isdir(candidate):
        raise FileNotFoundError(
            f"DATA_ROOT does not exist on this machine: {candidate}. "
            "Edit data/raw/DATA_PATH.txt with the path that is valid here."
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
BATCH_SIZE = 2
# Windows lacks a usable fork(), so DataLoader workers re-import the parent
# script and frequently deadlock. Keep 0 on Windows; bump to 4 on Linux/WSL2.
NUM_WORKERS = 0

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
PATIENCE = 15
VAL_EVERY_N_EPOCHS = 5
GRAD_CLIP_NORM = 1.0
AMP_ENABLED = True

# ---------------------------------------------------------------------------
# BraTS GLI-2024 segmentation labels
# ---------------------------------------------------------------------------
LABEL_BACKGROUND = 0
LABEL_TC = 2          # Tumour core / necrosis
LABEL_ET = 4          # Enhancing tumour
LABEL_WT_VALUES = [2, 4]  # Whole tumour = TC + ET (label 1 is unused in GLI-2024)

# ---------------------------------------------------------------------------
# Modality ordering — fixed across the whole project so member5's stacked
# multimodal tensors line up with member1-4's per-channel checkpoints.
# ---------------------------------------------------------------------------
MODALITIES = ("t1c", "t1n", "t2f", "t2w")

print(
    f"[config] {DEVICE_NAME} | {VRAM_GB:.1f} GB | "
    f"patch={PATCH_SIZE} | AMP={AMP_ENABLED}"
)
