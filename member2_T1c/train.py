"""Member 2 — T1c training entry point.

See member1_T1n/train.py for the full reference template. Copy that loop and
swap MODALITY = "t1c" + your own model.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed  # noqa: E402
set_global_seed()

from shared.config import (  # noqa: E402
    AMP_ENABLED, BATCH_SIZE, LR, NUM_EPOCHS, NUM_WORKERS, PATCH_SIZE,
    PATIENCE, VAL_EVERY_N_EPOCHS, WEIGHT_DECAY,
    CHECKPOINT_DIR, FIGURES_DIR, MLRUNS_DIR, SPLITS_DIR, TABLES_DIR,
    TARGET_CHANNELS, TARGET_CHANNEL_NAMES, BEST_METRIC, GLOBAL_SEED,
    get_data_root,
)
from shared.dataset import BraTSDataset, get_dataloader, load_splits  # noqa: E402,F401


MEMBER_NAME = "member2_T1c"
MODALITY = "t1c"


def main() -> None:
    # TODO: build datasets/loaders, model, optimizer, scaler
    # TODO: implement train+validate loop (see member1_T1n/train.py)
    # TODO: log to MLflow under experiment name MEMBER_NAME
    raise NotImplementedError


if __name__ == "__main__":
    main()
