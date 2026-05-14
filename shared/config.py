"""Single source of truth for hyperparameters and environment configuration.

Modified for Kaggle Compatibility & Fusion Strategy:
1. Detects Kaggle environment and specific dataset paths.
2. Automatically handles directory creation for writeable outputs.
3. Points to the correct nested data root for BraTS-subset.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import torch

GLOBAL_SEED = 42

# --- Environment Detection ---
IS_KAGGLE = os.path.exists('/kaggle/working')

# --- 1. Paths Management ---
if IS_KAGGLE:
    INPUT_DIR = Path('/kaggle/input')
    
    # Based on your listing: /kaggle/input/brats-code/
    REPO_ROOT = INPUT_DIR / "brats-code"
    
    # Path to the patient folders: /kaggle/input/brats-subset/subset/
    DATA_PATH_KAGGLE = INPUT_DIR / "brats-subset/subset"
    
    # Results must go to /kaggle/working to be writeable
    RESULTS_DIR = Path('/kaggle/working/results')
    
    # Add shared to sys.path so 'from shared.config' works from any script
    # Path: /kaggle/input/brats-code/shared
    SHARED_LIB_PATH = REPO_ROOT / "shared"
    if str(SHARED_LIB_PATH) not in sys.path:
        sys.path.insert(0, str(SHARED_LIB_PATH))
else:
    # Local Logic
    REPO_ROOT = Path(__file__).resolve().parent.parent
    RESULTS_DIR = REPO_ROOT / "results"
    DATA_PATH_KAGGLE = None

# Standard Repo Sub-dirs
SPLITS_DIR = REPO_ROOT / "data" / "splits"
DATA_PATH_FILE = REPO_ROOT / "data" / "raw" / "DATA_PATH.txt"
DATA_PATH_LOCAL_FILE = REPO_ROOT / "data" / "raw" / "DATA_PATH.local.txt"

# Output Dirs (Member 4 T2f Specific)
CHECKPOINT_DIR = RESULTS_DIR / "T2F/checkpoints"
FIGURES_DIR    = RESULTS_DIR / "T2F/figures"
TABLES_DIR     = RESULTS_DIR / "T2F/tables"
FEATURE_DIR    = RESULTS_DIR / "T2F/features"  # Added for Fusion Strategy
MLRUNS_DIR     = RESULTS_DIR / "mlruns"

# Ensure output directories exist (required for Kaggle /working space)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

def _read_first_nonempty_line(path: Path) -> str:
    """Return the first non-empty stripped line of path, or '' if none."""
    if not path.exists():
        return ""
    raw = path.read_text(encoding="utf-8").strip().splitlines()
    return next((line.strip() for line in raw if line.strip()), "")

def get_data_root() -> str:
    """Read the absolute path to the raw BraTS folder."""
    if IS_KAGGLE:
        return str(DATA_PATH_KAGGLE)

    candidate = _read_first_nonempty_line(DATA_PATH_LOCAL_FILE)
    if not candidate:
        candidate = _read_first_nonempty_line(DATA_PATH_FILE)

    if not candidate:
        raise FileNotFoundError("Create data/raw/DATA_PATH.local.txt with local path.")

    if not os.path.isdir(candidate):
        raise FileNotFoundError(f"DATA_ROOT does not exist: {candidate}")
    return candidate

# --- 2. GPU-aware defaults ---
if torch.cuda.is_available():
    _props = torch.cuda.get_device_properties(0)
    DEVICE_NAME = _props.name
    VRAM_GB = _props.total_memory / 1e9
else:
    DEVICE_NAME = "cpu"
    VRAM_GB = 0.0

# Dynamic Patch Size based on VRAM
PATCH_SIZE = (96, 96, 96) if VRAM_GB >= 14 else (64, 64, 64)
BATCH_SIZE = 2
NUM_WORKERS = 4 if not sys.platform.startswith('win') else 0

# --- 3. Training defaults ---
LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
PATIENCE = 15
VAL_EVERY_N_EPOCHS = 5
GRAD_CLIP_NORM = 1.0
AMP_ENABLED = True

# --- 4. Labels & Modalities ---
LABEL_BACKGROUND = 0
LABEL_TC = 2
LABEL_ET = 4
LABEL_WT_VALUES = [2, 4]

MODALITIES = ("t1c", "t1n", "t2f", "t2w")

print(
    f"[config] {DEVICE_NAME} | {VRAM_GB:.1f} GB | "
    f"patch={PATCH_SIZE} | Kaggle={IS_KAGGLE}"
)