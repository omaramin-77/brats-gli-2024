"""Member 3 — T2w training entry point.

See member1_T1n/train.py for the full reference template. Copy that loop and
set MODALITY = "t2w".
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


MEMBER_NAME = "member3_T2w"
MODALITY = "t2w"


def main() -> None:
    # TODO: implement train+validate loop (mirror member1_T1n/train.py)
    raise NotImplementedError


if __name__ == "__main__":
    main()
