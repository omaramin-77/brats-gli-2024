"""Member 1 — T1n explainability via 3D Grad-CAM.

Loads the trained checkpoint and produces Grad-CAM heatmaps for three test
patients, saving axial / coronal / sagittal overlays under
``results/figures/``. M1 uses Grad-CAM only — no attention maps; the
attention-based variants are M2's and M3's responsibilities.

Order rule (claude.md §2.4 / §4):
    train.py  →  evaluate.py  →  xai_analysis.py
The ``test_metrics.csv`` guard inside ``main()`` enforces this — the script
refuses to run before evaluate.py has produced its row, which is what
prevents test-set peeking during development.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed  # noqa: E402
set_global_seed()

import torch  # noqa: E402
import numpy as np  # noqa: E402

from shared.config import (  # noqa: E402
    CHECKPOINT_DIR,
    FIGURES_DIR,
    SPLITS_DIR,
    TABLES_DIR,
    get_data_root,
)
from shared.preprocessing import preprocess_patient  # noqa: E402
from shared.trainer import CheckpointManager  # noqa: E402
from shared.grad_cam_3d import GradCAM3D  # noqa: E402
from shared.visualization import plot_gradcam_overlay  # noqa: E402

from member1_T1n.model import ResidualUNet3D  # noqa: E402


MEMBER_NAME = "M1_T1n_ResUNet"
MODALITY = "t1n"
N_PATIENTS = 3
TARGET_LAYER_NAME = "enc4"  # last encoder ResidualBlock3D — design.md M1 §XAI


def _resolve_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Locate the Grad-CAM target via named_modules() and print confirmation.

    design.md M1 calls for "Grad-CAM on bottleneck + last encoder layers".
    For a single hook we use the LAST encoder block (``enc4``) — its output
    is the deepest spatially-resolved feature in the encoder path,
    immediately before the bottleneck pool.
    """
    named = dict(model.named_modules())
    if TARGET_LAYER_NAME not in named:
        available = [n for n in named if n]
        raise RuntimeError(
            f"Target layer '{TARGET_LAYER_NAME}' not found in model. "
            f"Available top-level modules: {available[:8]} ..."
        )
    target = named[TARGET_LAYER_NAME]
    print(
        f"[xai] target layer for Grad-CAM: '{TARGET_LAYER_NAME}' "
        f"-> {type(target).__name__}"
    )
    return target


def main() -> None:
    # ── test-set contamination guard (claude.md §4) ──────────────────────────
    # XAI runs on test patients; the test split must never be examined before
    # evaluate.py has produced its results row. Run from main() rather than
    # module-load so a plain ``import member1_T1n.xai_analysis`` stays clean.
    metrics_csv = TABLES_DIR / "test_metrics.csv"
    if not metrics_csv.exists():
        raise RuntimeError(
            "\n\ntest_metrics.csv not found.\n"
            "Run member1_T1n/evaluate.py BEFORE running xai_analysis.py.\n"
            "This guard prevents accidental test-set peeking during "
            "development.\n"
            f"Expected file: {metrics_csv}\n"
        )

    DATA_ROOT = get_data_root()

    # Deterministic pick: first, middle, last from the test split. Three
    # patients give us three different tumour sizes / locations without
    # any random sampling (which would change figure provenance between runs).
    test_ids_all = [
        line.strip()
        for line in (SPLITS_DIR / "test_ids.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(test_ids_all) < N_PATIENTS:
        raise RuntimeError(
            f"Need {N_PATIENTS} test IDs but only {len(test_ids_all)} available."
        )
    mid = len(test_ids_all) // 2
    test_ids = [test_ids_all[0], test_ids_all[mid], test_ids_all[-1]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet3D(in_channels=1, out_channels=3).to(device)
    ckpt = CheckpointManager(CHECKPOINT_DIR, MEMBER_NAME)
    model, _, _, _ = ckpt.load_best(model)
    model = model.to(device)
    model.eval()

    target_layer = _resolve_target_layer(model)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for pid in test_ids:
        print(f"[xai] patient {pid}")
        patient_dir = str(Path(DATA_ROOT) / pid)
        vol, seg = preprocess_patient(patient_dir, MODALITY)
        x = torch.from_numpy(vol).float().unsqueeze(0).to(device)

        # Build per-subregion GT masks from the raw integer seg for visualisation.
        wt_gt = ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.uint8)
        tc_gt = ((seg == 1) | (seg == 3)).astype(np.uint8)
        et_gt = (seg == 3).astype(np.uint8)
        subregion_gts = {"wt": wt_gt, "tc": tc_gt, "et": et_gt}

        with GradCAM3D(model, target_layer) as cam:
            for sr in ("wt", "tc", "et"):
                heatmap = cam.generate(x, target_channel=sr)
                for view_name, axis in (("axial", 2), ("coronal", 1), ("sagittal", 0)):
                    sr_seg = subregion_gts[sr]
                    if sr_seg.sum() > 0:
                        other = tuple(a for a in (0, 1, 2) if a != axis)
                        idx = int(np.argmax(sr_seg.sum(axis=other)))
                    else:
                        idx = sr_seg.shape[axis] // 2
                    if axis == 0:
                        vs, cs, ss = vol[0][idx], heatmap[idx], sr_seg[idx]
                    elif axis == 1:
                        vs, cs, ss = vol[0][:, idx], heatmap[:, idx], sr_seg[:, idx]
                    else:
                        vs, cs, ss = vol[0][:, :, idx], heatmap[:, :, idx], sr_seg[:, :, idx]
                    save_path = FIGURES_DIR / f"gradcam_M1_{pid}_{sr}_{view_name}.png"
                    plot_gradcam_overlay(
                        vs, cs, ss,
                        title=f"M1 T1n — {pid} — {sr} — {view_name} (slice {idx})",
                        save_path=str(save_path),
                    )
                    print(f"[xai] saved {save_path.name}")

    print()
    print("TODO: Per-subregion Grad-CAM analysis (manual, fill in for report):")
    print("  WT — does the heatmap match whole-tumour extent?")
    print("  TC — does it focus on necrotic/enhancing core?")
    print("  ET — does it focus on enhancing tumour only?")
    print("If yes -> model attends correctly per sub-region.")
    print("If WT looks right but TC/ET drift -> deep features are not channel-specific.")


if __name__ == "__main__":
    main()