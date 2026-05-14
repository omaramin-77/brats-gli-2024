"""End-to-end Phase 0 sanity check.

Run after `pip install -r requirements.txt` and after `shared/create_splits.py`
has produced data/splits/. Exits non-zero on the first failed assertion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed  # noqa: E402
set_global_seed()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from shared.config import (  # noqa: E402
    DEVICE_NAME,
    FIGURES_DIR,
    PATCH_SIZE,
    SPLITS_DIR,
    VRAM_GB,
    get_data_root,
)
from shared.dataset import BraTSDataset, load_splits  # noqa: E402
from shared.preprocessing import preprocess_patient, preprocess_patient_multimodal  # noqa: E402
from shared.trainer import dice_bce_loss  # noqa: E402
from shared.visualization import plot_patient_overview  # noqa: E402


def fail(msg: str) -> None:
    print(f"\nFAILED: {msg}")
    sys.exit(1)


def main() -> None:
    print(f"[sanity] device      : {DEVICE_NAME} ({VRAM_GB:.1f} GB)")
    data_root = get_data_root()
    print(f"[sanity] DATA_ROOT   : {data_root}")

    # --- Step 3: confirm split sizes match split_stats.json ----------------
    splits = load_splits(str(SPLITS_DIR))
    stats = json.loads((SPLITS_DIR / "split_stats.json").read_text(encoding="utf-8"))
    if len(splits["train"]) != stats["train"]:
        fail(f"train count mismatch: txt={len(splits['train'])} json={stats['train']}")
    if len(splits["val"]) != stats["val"]:
        fail(f"val count mismatch: txt={len(splits['val'])} json={stats['val']}")
    if len(splits["test"]) != stats["test"]:
        fail(f"test count mismatch: txt={len(splits['test'])} json={stats['test']}")
    print(f"[sanity] splits      : train={stats['train']} val={stats['val']} test={stats['test']}  OK")

    # --- Step 4-8: preprocess one patient ---------------------------------
    patient_id = splits["train"][0]
    patient_dir = Path(data_root) / patient_id
    print(f"[sanity] testing on  : {patient_id}")

    unimodal = {}
    seg_ref = None
    for mod in ("t1c", "t1n", "t2f", "t2w"):
        vol, seg = preprocess_patient(str(patient_dir), mod)
        if vol.shape != (1, 128, 128, 128):
            fail(f"{mod} unimodal shape {vol.shape} != (1,128,128,128)")
        if seg.shape != (128, 128, 128):
            fail(f"seg shape {seg.shape} != (128,128,128)")
        if np.isnan(vol).any() or np.isinf(vol).any():
            fail(f"{mod} contains NaN/Inf")
        unimodal[mod] = vol[0]
        seg_ref = seg

    multi_vol, multi_seg = preprocess_patient_multimodal(str(patient_dir))
    if multi_vol.shape != (4, 128, 128, 128):
        fail(f"multimodal vol shape {multi_vol.shape} != (4,128,128,128)")
    if multi_seg.shape != (128, 128, 128):
        fail(f"multimodal seg shape {multi_seg.shape} != (128,128,128)")
    if np.isnan(multi_vol).any() or np.isinf(multi_vol).any():
        fail("multimodal vol contains NaN/Inf")

    unique_labels = sorted(np.unique(multi_seg).tolist())
    allowed = {0, 1, 2, 3, 4}
    if not set(unique_labels).issubset(allowed):
        fail(f"unexpected seg labels {unique_labels}")
    print(f"[sanity] seg labels  : {unique_labels}  OK")

    # --- Step 9-10: BraTSDataset patches ----------------------------------
    patch_side = PATCH_SIZE[0] if isinstance(PATCH_SIZE, tuple) else PATCH_SIZE

    ds_uni = BraTSDataset(
        data_root=data_root,
        split_file=str(SPLITS_DIR / "train_ids.txt"),
        modality="t1n",
        patch_size=patch_side,
        augment=False,
        patches_per_volume=1,
    )
    item = ds_uni[0]
    expected_uni = (1, patch_side, patch_side, patch_side)
    if tuple(item["image"].shape) != expected_uni:
        fail(f"unimodal patch shape {tuple(item['image'].shape)} != {expected_uni}")
    if tuple(item["label"].shape) != (3, patch_side, patch_side, patch_side):
        fail(f"unimodal LABEL shape {tuple(item['label'].shape)} != (3,P,P,P)")

    ds_multi = BraTSDataset(
        data_root=data_root,
        split_file=str(SPLITS_DIR / "train_ids.txt"),
        modality="multimodal",
        patch_size=patch_side,
        augment=False,
        patches_per_volume=1,
    )
    item_m = ds_multi[0]
    expected_multi = (4, patch_side, patch_side, patch_side)
    if tuple(item_m["image"].shape) != expected_multi:
        fail(f"multimodal patch shape {tuple(item_m['image'].shape)} != {expected_multi}")
    if tuple(item_m["label"].shape) != (3, patch_side, patch_side, patch_side):
        fail(f"multimodal LABEL shape {tuple(item_m['label'].shape)} != (3,P,P,P)")
    print(f"[sanity] patch unimod: {expected_uni}  OK")
    print(f"[sanity] patch multim: {expected_multi}  OK")

    # --- Step 11: loss smoke test -----------------------------------------
    pred = torch.randn(1, 3, patch_side, patch_side, patch_side)
    target = (torch.rand(1, 3, patch_side, patch_side, patch_side) > 0.7).float()
    loss = dice_bce_loss(pred, target)
    if not torch.isfinite(loss):
        fail(f"dice_bce_loss returned non-finite value: {loss.item()}")
    print(f"[sanity] loss        : {loss.item():.4f}  OK")

    # --- Step 11a: 9-key metrics smoke test --------------------------------
    from shared.metrics import compute_all_metrics
    m = compute_all_metrics(pred, target)
    expected_keys = {"dice_wt", "dice_tc", "dice_et",
                    "iou_wt", "iou_tc", "iou_et",
                    "hd95_wt", "hd95_tc", "hd95_et"}
    missing = expected_keys - set(m.keys())
    if missing:
        fail(f"compute_all_metrics missing keys: {missing}")
    print(f"[sanity] metrics     : 9 keys present OK")

    # --- Step 11b: GradCAM3D context-manager smoke test --------------------
    import torch.nn as nn
    from shared.grad_cam_3d import GradCAM3D

    class _SanityTinyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Conv3d(1, 4, 3, padding=1)
            self.head = nn.Conv3d(4, 3, 1)

        def forward(self, x):
            return self.head(self.encoder(x))

    cam_net = _SanityTinyNet()
    cam_net.eval()
    target_layer = cam_net.encoder
    hooks_before = len(target_layer._forward_hooks) + len(target_layer._backward_hooks)
    cam_input = torch.randn(1, 1, 16, 16, 16)
    with GradCAM3D(cam_net, target_layer) as cam:
        heatmap = cam.generate(cam_input)
    hooks_after = len(target_layer._forward_hooks) + len(target_layer._backward_hooks)
    if heatmap.shape != (16, 16, 16):
        fail(f"GradCAM3D heatmap shape {heatmap.shape} != (16,16,16)")
    if heatmap.min() < 0.0 or heatmap.max() > 1.0 + 1e-5:
        fail(f"GradCAM3D heatmap range [{heatmap.min():.3f},{heatmap.max():.3f}] outside [0,1]")
    if hooks_after != hooks_before:
        fail("GradCAM3D context manager leaked hooks on exit")
    print(f"[sanity] gradcam3d   : shape={heatmap.shape} range=[{heatmap.min():.3f},{heatmap.max():.3f}]  OK")

    for ch in ("wt", "tc", "et"):
        with GradCAM3D(cam_net, target_layer) as cam:
            hmap = cam.generate(cam_input, target_channel=ch)
        if hmap.shape != (16, 16, 16):
            fail(f"GradCAM3D(target_channel={ch}) heatmap shape {hmap.shape}")
        if hmap.min() < 0.0 or hmap.max() > 1.0 + 1e-5:
            fail(f"GradCAM3D(target_channel={ch}) range out of [0,1]")
    print(f"[sanity] gradcam3d   : per-channel (wt/tc/et) OK")

    # --- Step 12: figure --------------------------------------------------
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "sanity_check_patient.png"
    plot_patient_overview(unimodal, seg_ref, patient_id, save_path=str(fig_path))
    if not fig_path.exists():
        fail(f"figure {fig_path} was not created")

    # --- Final summary ----------------------------------------------------
    print()
    print("==============================================")
    print(" PHASE 0 SANITY CHECK - ALL TESTS PASSED")
    print("==============================================")
    print(f" Patient tested   : {patient_id}")
    print(f" Unimodal shapes  : (1, 128, 128, 128) OK")
    print(f" Multimodal shape : (4, 128, 128, 128) OK")
    print(f" Seg unique labels: {unique_labels} OK")
    print(f" Patch shape (T1n): {expected_uni} OK")
    print(f" NaN/Inf check    : CLEAN OK")
    print(f" Figure saved     : {fig_path.relative_to(REPO_ROOT)} OK")
    print("==============================================")
    print(" Phase 0 complete. All members can begin their pipelines.")


if __name__ == "__main__":
    main()
