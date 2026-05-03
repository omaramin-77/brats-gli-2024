"""Member 1 — T1n test-set evaluation.

Loads the best checkpoint produced by train.py and reports Dice/IoU/HD95 on
the held-out test split. Saves a CSV row to results/tables/.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed  # noqa: E402
set_global_seed()


def main() -> None:
    # TODO: load test split, instantiate model, restore best checkpoint
    # TODO: run validate_one_epoch on the test loader
    # TODO: write a row to results/tables/test_metrics.csv
    raise NotImplementedError


if __name__ == "__main__":
    main()
