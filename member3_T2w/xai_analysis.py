"""Member 3 — T2w explainability via 3D Grad-CAM."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed  # noqa: E402
set_global_seed()


def main() -> None:
    # TODO: produce Grad-CAM overlays for a few test patients
    raise NotImplementedError


if __name__ == "__main__":
    main()
