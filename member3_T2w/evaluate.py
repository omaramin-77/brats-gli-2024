# member3_T2w/evaluate.py
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed
set_global_seed()

import torch

from shared.config import CHECKPOINT_DIR, GLOBAL_SEED, NUM_WORKERS, PATCH_SIZE, SPLITS_DIR, TABLES_DIR, get_data_root
from shared.dataset import BraTSDataset, get_dataloader
from shared.trainer import CheckpointManager, dice_bce_loss, validate_one_epoch

from model import build_model


MEMBER_NAME = "M3_T2w_SwinUNETR"
MODALITY = "t2w"
ARCHITECTURE = "SwinUNETR"
FEATURE_SIZE = 24


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] device={device}")

    data_root = get_data_root()

    test_ds = BraTSDataset(
        data_root=data_root,
        split_file=str(SPLITS_DIR / "test_ids.txt"),
        modality=MODALITY,
        patch_size=PATCH_SIZE[0],
        augment=False,
        patches_per_volume=1,
        full_volume=True,
    )

    test_loader = get_dataloader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = build_model(img_size=PATCH_SIZE, feature_size=FEATURE_SIZE).to(device)
    ckpt = CheckpointManager(save_dir=str(CHECKPOINT_DIR), member_name=MEMBER_NAME)
    model, _, epoch, best_val_dice = ckpt.load_best(model, optimizer=None)
    model.eval()

    print(f"[evaluate] loaded best checkpoint epoch={epoch}, best_val_dice_wt={best_val_dice:.4f}")
    print("[evaluate] running test-set evaluation...")

    metrics = validate_one_epoch(
        model=model,
        loader=test_loader,
        device=device,
        loss_fn=dice_bce_loss,
    )

    meta_path = CHECKPOINT_DIR / f"{MEMBER_NAME}_train_meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    row = {
        "member": "M3",
        "modality": MODALITY,
        "architecture": ARCHITECTURE,
        "seed": GLOBAL_SEED,
        "dice_wt": metrics.get("dice_wt", float("nan")),
        "dice_tc": metrics.get("dice_tc", float("nan")),
        "dice_et": metrics.get("dice_et", float("nan")),
        "iou_wt": metrics.get("iou_wt", float("nan")),
        "iou_tc": metrics.get("iou_tc", float("nan")),
        "iou_et": metrics.get("iou_et", float("nan")),
        "hd95_wt": metrics.get("hd95_wt", float("nan")),
        "hd95_tc": metrics.get("hd95_tc", float("nan")),
        "hd95_et": metrics.get("hd95_et", float("nan")),
        "training_time_hrs": meta.get("training_time_hrs", float("nan")),
        "gpu_memory_gb": meta.get("gpu_memory_gb", float("nan")),
    }

    print("=" * 70)
    for k, v in row.items():
        print(f"{k}: {v}")
    print("=" * 70)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "test_metrics.csv"

    fieldnames = list(row.keys())
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[evaluate] appended results to {csv_path}")


if __name__ == "__main__":
    main()