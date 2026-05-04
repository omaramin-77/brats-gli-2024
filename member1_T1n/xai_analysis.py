"""Member 1 — T1n explainability via 3D Grad-CAM.

Loads the trained model and produces Grad-CAM heatmaps for selected test
cases, then writes overlay PNGs into results/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed  # noqa: E402
set_global_seed()

from shared.config import TABLES_DIR  # noqa: E402

# ── test-set contamination guard ─────────────────────────────────────────────
# XAI runs on test patients. The test split must never be examined before
# evaluate.py has produced its results row. This assertion enforces that order.
_metrics_csv = TABLES_DIR / "test_metrics.csv"
if not _metrics_csv.exists():
    raise RuntimeError(
        "\n\ntest_metrics.csv not found.\n"
        "You must run evaluate.py and obtain your test results BEFORE\n"
        "running xai_analysis.py. This prevents accidental test-set\n"
        "peeking during development.\n"
        f"Expected file: {_metrics_csv}\n"
    )
# ─────────────────────────────────────────────────────────────────────────────

# from shared.grad_cam_3d import GradCAM3D
# from shared.visualization import plot_gradcam_overlay


def main() -> None:
    # TODO: restore best checkpoint
    # TODO: for a few test patients, generate cam and call plot_gradcam_overlay
    # ── GradCAM usage template ──────────────────────────────────────────────────
    # target_layer = model.<your_bottleneck_layer>   # replace with actual layer
    # with GradCAM3D(model, target_layer) as cam:
    #     heatmap = cam.generate(input_tensor)        # shape: (H, W, D)
    #     plot_gradcam_overlay(mri_slice, heatmap_slice, save_path=...)
    # ────────────────────────────────────────────────────────────────────────────
    raise NotImplementedError


if __name__ == "__main__":
    main()
