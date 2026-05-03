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

# from shared.grad_cam_3d import GradCAM3D
# from shared.visualization import plot_gradcam_overlay


def main() -> None:
    # TODO: restore best checkpoint
    # TODO: cam = GradCAM3D(model, target_layer=model.encoder.bottleneck)
    # TODO: for a few test patients, generate cam and call plot_gradcam_overlay
    # TODO: cam.remove_hooks()
    raise NotImplementedError


if __name__ == "__main__":
    main()
