"""Member 3 — T2w test-set evaluation.

Run ONCE at the very end of the project — never during hyperparameter tuning.
Tune on the validation set only (train.py). Test is sacred.

Usage:
    python member3_T2w/evaluate.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed
set_global_seed()

import torch

from shared.config import (
    CHECKPOINT_DIR,
    NUM_WORKERS,
    PATCH_SIZE,
    SPLITS_DIR,
    TABLES_DIR,
    get_data_root,
)
from shared.dataset import BraTSDataset, get_dataloader
from shared.trainer import CheckpointManager, dice_bce_loss, validate_one_epoch

from model import build_model

MEMBER_NAME  = "member3_T2w"
MODALITY     = "t2w"
PATCH_SIZE   = (64, 64, 64)   # must match train.py
FEATURE_SIZE = 24


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{MEMBER_NAME}/evaluate] device={device}")

    data_root = get_data_root()

    # Test split — ONLY here, never in train.py
    test_ds = BraTSDataset(
        data_root=data_root,
        split_file=str(SPLITS_DIR / "test_ids.txt"),
        modality=MODALITY,
        patch_size=PATCH_SIZE[0],
        augment=False,
        patches_per_volume=1,
    )
    test_loader = get_dataloader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    print(f"[{MEMBER_NAME}/evaluate] test patients: {len(test_ds)}")

    # Restore best checkpoint
    ckpt = CheckpointManager(save_dir=str(CHECKPOINT_DIR), member_name=MEMBER_NAME)
    model = build_model(img_size=PATCH_SIZE, feature_size=FEATURE_SIZE).to(device)
    model, _, epoch, val_dice = ckpt.load_best(model, optimizer=None)
    model.eval()
    print(f"[{MEMBER_NAME}/evaluate] loaded checkpoint epoch={epoch}  val_dice={val_dice:.4f}")

    # Evaluate
    print(f"[{MEMBER_NAME}/evaluate] running sliding-window inference ...")
    metrics = validate_one_epoch(model, test_loader, device, dice_bce_loss)

    dice = metrics.get("dice", float("nan"))
    iou  = metrics.get("iou",  float("nan"))
    hd95 = metrics.get("hd95", float("nan"))
    loss = metrics.get("loss", float("nan"))

    print(f"\n[{MEMBER_NAME}/evaluate] ── Test Results ────────────────────")
    print(f"  Dice  : {dice:.4f}")
    print(f"  IoU   : {iou:.4f}")
    print(f"  HD95  : {hd95:.2f} mm")
    print(f"  Loss  : {loss:.4f}")
    print("─────────────────────────────────────────────────────────────")

    # Write CSV row
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "test_metrics.csv"
    fieldnames = ["member", "modality", "architecture",
                  "dice", "iou", "hd95", "val_dice_best", "checkpoint_epoch"]
    row = {
        "member":           "M3",
        "modality":         MODALITY,
        "architecture":     "SwinUNETR",
        "dice":             f"{dice:.4f}",
        "iou":              f"{iou:.4f}",
        "hd95":             f"{hd95:.2f}",
        "val_dice_best":    f"{val_dice:.4f}",
        "checkpoint_epoch": epoch,
    }
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[{MEMBER_NAME}/evaluate] result appended → {csv_path}")


if __name__ == "__main__":
    main()
