"""Member 2 — T1c explainability via 3D Grad-CAM + Occlusion Sensitivity.

Loads the trained M2 checkpoint and produces:

1. Per-sub-region 3D Grad-CAM overlays (27 PNGs) — 3 test patients
   (first / middle / last from test_ids.txt) × 3 sub-regions (WT/TC/ET)
   × 3 views (axial / coronal / sagittal). Filename:
   ``gradcam_M2_{pid}_{sr}_{view}.png``.

2. **Occlusion Sensitivity** for the MIDDLE test patient only — M2's
   second XAI method (design.md §4 M2 / plan.md Phase 3). An 8³
   zero-mask slides over the input volume with stride 4. For each
   position we measure the drop in the model's mean predicted
   probability for each sub-region. The drop tensor is the heatmap:
   high drop = the model needed that region to produce its
   prediction. Saved as ``occlusion_M2_{pid}_{sr}_axial.png`` (axial
   view only — coronal/sagittal would triple compute cost for little
   additional signal).

Cost note
---------
Occlusion is O((volume/stride)**3) forward passes. For a 128³ volume
with stride 4 that is 32**3 = 32,768 passes per sub-region (sub-regions
share forward passes — three drops are computed per pass). Inference
is batched (8 occluded volumes per forward) under ``torch.no_grad()``
to amortise GPU overhead. Expect ~30 minutes total on a single GPU
for one patient at 128³. A reader committing to run this script
should budget accordingly.

Order rule (claude.md §2.4)
---------------------------
    train.py  →  evaluate.py  →  xai_analysis.py
The ``test_metrics.csv`` guard inside ``main()`` enforces this — the
script refuses to run before evaluate.py has produced its row, which
is what prevents test-set peeking during development.
"""
from shared.seed import set_global_seed
set_global_seed()

from pathlib import Path

import numpy as np
import torch
import argparse
from shared.config import PATCH_SIZE
from shared.config import (
    CHECKPOINT_DIR,
    FIGURES_DIR,
    SPLITS_DIR,
    TABLES_DIR,
    get_data_root,
)
from shared.trainer import CheckpointManager
from shared.grad_cam_3d import GradCAM3D
from shared.visualization import plot_gradcam_overlay
from shared.preprocessing import preprocess_patient

from member2_T1c.model import AttentionUNet3D


MEMBER_NAME = "M2_T1c_AttUNet_focal"
MODALITY = "t1c"
N_PATIENTS = 3
TARGET_LAYER_NAME = "enc4"          # last encoder ConvBlock3D
OCCLUSION_WINDOW = 8
OCCLUSION_STRIDE = 8
OCCLUSION_BATCH = 1
SUBREGIONS = ("wt", "tc", "et")
CHANNEL_INDEX = {"wt": 0, "tc": 1, "et": 2}


def _resolve_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Locate the Grad-CAM target via named_modules() and print confirmation.

    For M2 we use the LAST encoder block (``enc4``) — its output is the
    deepest spatially-resolved feature in the encoder path, immediately
    before the bottleneck pool. Same convention as M1.
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


def _peak_slice(seg: np.ndarray, axis: int) -> int:
    """Pick the slice along ``axis`` with the most positive voxels of ``seg``.

    Used for per-sub-region peak-slice selection — falls back to the
    middle slice when the sub-region is empty.
    """
    if seg.sum() > 0:
        other = tuple(a for a in (0, 1, 2) if a != axis)
        return int(np.argmax(seg.sum(axis=other)))
    return seg.shape[axis] // 2


def _gradcam_pass(model, target_layer, x, vol, subregion_gts, pid):
    """Run Grad-CAM over (wt, tc, et) × (axial, coronal, sagittal) for one patient."""
    with GradCAM3D(model, target_layer) as cam:
        for sr in SUBREGIONS:
            heatmap = cam.generate(x, target_channel=sr)
            sr_seg = subregion_gts[sr]
            for view_name, axis in (("axial", 2), ("coronal", 1), ("sagittal", 0)):
                idx = _peak_slice(sr_seg, axis)
                if axis == 0:
                    vs, cs, ss = vol[0][idx], heatmap[idx], sr_seg[idx]
                elif axis == 1:
                    vs, cs, ss = vol[0][:, idx], heatmap[:, idx], sr_seg[:, idx]
                else:
                    vs, cs, ss = vol[0][:, :, idx], heatmap[:, :, idx], sr_seg[:, :, idx]
                save_path = FIGURES_DIR / f"gradcam_M2_{pid}_{sr}_{view_name}.png"
                plot_gradcam_overlay(
                    vs, cs, ss,
                    title=f"M2 T1c — {pid} — {sr} — {view_name} (slice {idx})",
                    save_path=str(save_path),
                )
                print(f"[xai] saved {save_path.name}")


def _occlusion_sensitivity(model, x, device, pid, vol, subregion_gts):
    """Compute and save per-sub-region occlusion sensitivity heatmaps for one patient.

    For each window position p (stride OCCLUSION_STRIDE), produce an
    occluded copy of ``x`` with an OCCLUSION_WINDOW³ zero patch at p,
    forward the model, compute mean predicted probability per channel,
    and record the drop relative to the un-occluded baseline. Drops
    are accumulated into a (3, D, H, W) heatmap by summing each
    window's contribution over the window's voxels and dividing by
    the per-voxel coverage count (since stride < window the windows
    overlap and each voxel is touched multiple times).
    """
    model.eval()

    _, _, D, H, W = x.shape
    w = OCCLUSION_WINDOW
    s = OCCLUSION_STRIDE

    # Baseline mean predicted probability per channel.
    with torch.no_grad():
        baseline_logits = model(x)
        baseline_probs = torch.sigmoid(baseline_logits)
        baseline_mean = baseline_probs.mean(dim=(2, 3, 4)).clone()  # (1, 3)
    del baseline_logits, baseline_probs
    torch.cuda.empty_cache()

    # Grid of valid window starts on each axis.
    starts_d = list(range(0, D - w + 1, s))
    starts_h = list(range(0, H - w + 1, s))
    starts_w = list(range(0, W - w + 1, s))
    positions = [(d, h, ww) for d in starts_d for h in starts_h for ww in starts_w]
    total = len(positions)
    print(f"[occlusion] {pid}: {total} positions  (window={w}, stride={s})")

    # Per-channel accumulators.
    drop_sum = torch.zeros(3, D, H, W, dtype=torch.float32, device=device)
    cover = torch.zeros(D, H, W, dtype=torch.float32, device=device)

    seen = 0
    with torch.no_grad():
        for batch_start in range(0, total, OCCLUSION_BATCH):
            batch_positions = positions[batch_start:batch_start + OCCLUSION_BATCH]
            B = len(batch_positions)
            batch = x.expand(B, -1, -1, -1, -1).clone()
            for b, (dd, hh, ww0) in enumerate(batch_positions):
                batch[b, :, dd:dd + w, hh:hh + w, ww0:ww0 + w] = 0.0

            logits = model(batch)
            probs = torch.sigmoid(logits)
            occ_mean = probs.mean(dim=(2, 3, 4))               # (B, 3)
            drops = (baseline_mean - occ_mean).clamp(min=0.0)  # (B, 3)

            for b, (dd, hh, ww0) in enumerate(batch_positions):
                d = drops[b]                                    # (3,)
                drop_sum[:, dd:dd + w, hh:hh + w, ww0:ww0 + w] += d.view(3, 1, 1, 1)
                cover[dd:dd + w, hh:hh + w, ww0:ww0 + w] += 1.0

            seen += B
            if seen % 1000 < OCCLUSION_BATCH:
                print(f"[occlusion] {pid}: {seen}/{total} positions")

    cover_safe = cover.clamp(min=1.0)                          # avoid div-by-zero outside grid
    heatmap = (drop_sum / cover_safe.unsqueeze(0)).cpu().numpy()  # (3, D, H, W)

    # Normalise per sub-region to [0, 1] for visualisation consistency.
    for sr in SUBREGIONS:
        ch = CHANNEL_INDEX[sr]
        ch_map = heatmap[ch]
        ch_min, ch_max = ch_map.min(), ch_map.max()
        if ch_max - ch_min > 1e-8:
            ch_map = (ch_map - ch_min) / (ch_max - ch_min)
        sr_seg = subregion_gts[sr]
        idx = _peak_slice(sr_seg, axis=2)                       # axial
        vs = vol[0][:, :, idx]
        cs = ch_map[:, :, idx]
        ss = sr_seg[:, :, idx]
        save_path = FIGURES_DIR / f"occlusion_M2_{pid}_{sr}_axial.png"
        plot_gradcam_overlay(
            vs, cs, ss,
            title=f"M2 T1c — {pid} — {sr} — occlusion (axial slice {idx})",
            save_path=str(save_path),
        )
        print(f"[occlusion] saved {save_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "patch"], default="full")
    args = parser.parse_args()
    PATCH_SIDE = PATCH_SIZE[0] if isinstance(PATCH_SIZE, tuple) else PATCH_SIZE

    # ── test-set contamination guard (claude.md §4) ──────────────────────────
    # XAI runs on test patients; the test split must never be examined before
    # evaluate.py has produced its results row. Run from main() rather than
    # module-load so a plain ``import member2_T1c.xai_analysis`` stays clean.
    metrics_csv = TABLES_DIR / "test_metrics.csv"
    if not metrics_csv.exists():
        raise RuntimeError(
            "\n\ntest_metrics.csv not found.\n"
            "Run member2_T1c/evaluate.py BEFORE running xai_analysis.py.\n"
            "This guard prevents accidental test-set peeking during "
            "development.\n"
            f"Expected file: {metrics_csv}\n"
        )

    DATA_ROOT = get_data_root()

    all_test_ids = [
        line.strip()
        for line in (SPLITS_DIR / "test_ids.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(all_test_ids) < N_PATIENTS:
        raise RuntimeError(
            f"Need at least {N_PATIENTS} test IDs but only "
            f"{len(all_test_ids)} available."
        )
    # Deterministic selection: first, middle, last.
    middle_idx = len(all_test_ids) // 2
    test_ids = [all_test_ids[0], all_test_ids[middle_idx], all_test_ids[-1]]
    middle_pid = all_test_ids[middle_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet3D(in_channels=1, out_channels=3).to(device)
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

        # Per-sub-region GT masks built from the raw integer seg.
        wt_gt = ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.uint8)
        tc_gt = ((seg == 1) | (seg == 3)).astype(np.uint8)
        et_gt = (seg == 3).astype(np.uint8)
        subregion_gts = {"wt": wt_gt, "tc": tc_gt, "et": et_gt}

        if args.mode == "patch":
            # Centre a PATCH_SIDE**3 cube on the WT centroid so XAI runs
            # at the same resolution the model was trained on — much cheaper
            # than running on the full 128**3 volume, especially for the
            # occlusion-sensitivity pass.
            if wt_gt.sum() > 0:
                centre = np.argwhere(wt_gt > 0).mean(axis=0).astype(int)
            else:
                centre = np.array(wt_gt.shape) // 2
            half = PATCH_SIDE // 2
            centre = np.clip(centre, half, np.array(wt_gt.shape) - half)
            s = tuple(slice(int(c - half), int(c + half)) for c in centre)

            # Replace vol, seg, sub-region GTs with their patched versions.
            vol = vol[(slice(None),) + s]                        # (1, P, P, P)
            seg = seg[s]                                          # (P, P, P)
            wt_gt = ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.uint8)
            tc_gt = ((seg == 1) | (seg == 3)).astype(np.uint8)
            et_gt = (seg == 3).astype(np.uint8)
            subregion_gts = {"wt": wt_gt, "tc": tc_gt, "et": et_gt}

            x = torch.from_numpy(vol).float().unsqueeze(0).to(device)
            print(f"[xai] patch mode: centred at {tuple(centre)}, shape {tuple(vol.shape)}")

        _gradcam_pass(model, target_layer, x, vol, subregion_gts, pid)

        # Occlusion sensitivity runs ONLY for the middle test patient
        # (compute cost ~30 min per patient — three patients would not
        # add value relative to Grad-CAM coverage we already have).
        if pid == middle_pid:
            print(f"[xai] occlusion sensitivity on middle patient {pid}")
            _occlusion_sensitivity(model, x, device, pid, vol, subregion_gts)

    print()
    print("TODO: Per-subregion Grad-CAM analysis (manual, fill in for report):")
    print("  WT — does the heatmap match whole-tumour extent on T1c?")
    print("  TC — does it focus on necrotic/enhancing core?")
    print("  ET — does it concentrate on the bright enhancing voxels?")
    print("Cross-check vs Occlusion: if Grad-CAM and Occlusion both highlight")
    print("the same region, Grad-CAM is validated as trustworthy for M2.")


if __name__ == "__main__":
    main()
