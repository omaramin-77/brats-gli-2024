"""Member 5 — Multimodal training entry point.

See member1_T1n/train.py for the full reference template. Set MODALITY = "multimodal".
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed  # noqa: E402
set_global_seed()

from shared.config import *  # noqa: E402,F401,F403
from shared.dataset import BraTSDataset, get_dataloader, load_splits  # noqa: E402,F401


MEMBER_NAME = "member5_multimodal"
MODALITY = "multimodal"


def main() -> None:
    # TODO: implement train+validate loop (mirror member1_T1n/train.py)
    # Multimodal input has 4 channels — make sure the model's first conv layer
    # is constructed with in_channels=4.
    raise NotImplementedError


if __name__ == "__main__":
    main()
