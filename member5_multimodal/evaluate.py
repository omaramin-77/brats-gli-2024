"""Member 5 — Multimodal test-set evaluation (Pipeline A — naive 4-channel baseline).

Loads the best checkpoint produced by ``member5_multimodal/train.py`` and reports
Dice/IoU/HD95 on the held-out test split. Appends one row to
``results/tables/test_metrics.csv``.

This is the FIRST and only place the test split is read. Do not call this
script during hyperparameter tuning or development — tune on val.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed  # noqa: E402
set_global_seed()

import csv  # noqa: E402

import torch  # noqa: E402

from shared.config import (
    CHECKPOINT_DIR,
    GLOBAL_SEED,
    SPLITS_DIR,
    TABLES_DIR,
    get_data_root,
)
from shared.dataset import BraTSDataset, get_dataloader
from shared.trainer import CheckpointManager, validate_one_epoch

from member5_multimodal.model import MultimodalUNet3D


MEMBER_NAME = "M5_Multimodal_4ch"
MODALITY = "multimodal"
ARCHITECTURE = "MultimodalUNet3D"

CSV_FIELDNAMES = [
    "member", "modality", "architecture", "seed",
    "dice_wt", "dice_tc", "dice_et",
    "iou_wt", "iou_tc", "iou_et",
    "hd95_wt", "hd95_tc", "hd95_et",
    "training_time_hrs", "gpu_memory_gb",
]


def main() -> None:
    DATA_ROOT = get_data_root()

    test_ds = BraTSDataset(
        data_root=DATA_ROOT,
        split_file=SPLITS_DIR / "test_ids.txt",
        modality=MODALITY,
        augment=False,
        full_volume=True,
    )
    test_loader = get_dataloader(test_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalUNet3D(in_channels=4, out_channels=3).to(device)

    ckpt = CheckpointManager(CHECKPOINT_DIR, MEMBER_NAME)
    model, _, _, _ = ckpt.load_best(model)
    model = model.to(device)

    metrics = validate_one_epoch(model, test_loader, device)

    # Pick up training_time_hrs / gpu_memory_gb from the sidecar JSON
    # written by train.py; fall back to NaN if absent.
    import json
    sidecar_path = CHECKPOINT_DIR / f"{MEMBER_NAME}_train_meta.json"
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    else:
        sidecar = {"training_time_hrs": float("nan"), "gpu_memory_gb": float("nan")}

    row = {
        "member": MEMBER_NAME,
        "modality": MODALITY,
        "architecture": ARCHITECTURE,
        "seed": GLOBAL_SEED,
        "dice_wt": metrics.get("dice_wt", float("nan")),
        "dice_tc": metrics.get("dice_tc", float("nan")),
        "dice_et": metrics.get("dice_et", float("nan")),
        "iou_wt":  metrics.get("iou_wt",  float("nan")),
        "iou_tc":  metrics.get("iou_tc",  float("nan")),
        "iou_et":  metrics.get("iou_et",  float("nan")),
        "hd95_wt": metrics.get("hd95_wt", float("nan")),
        "hd95_tc": metrics.get("hd95_tc", float("nan")),
        "hd95_et": metrics.get("hd95_et", float("nan")),
        "training_time_hrs": sidecar["training_time_hrs"],
        "gpu_memory_gb":     sidecar["gpu_memory_gb"],
    }

    csv_path = TABLES_DIR / "test_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Saved test metrics for {MEMBER_NAME} to test_metrics.csv")


if __name__ == "__main__":
    main()
