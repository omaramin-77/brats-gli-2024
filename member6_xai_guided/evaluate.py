"""Member 6 — XAI-guided multi-task M5 test-set evaluation.

Loads the best checkpoint produced by member6_xai_guided/train.py and reports
Dice/IoU/HD95 on the held-out test split. Appends one row to
``results/tables/test_metrics.csv`` with the canonical 12-metric schema, and
prints the post-training gate values so the report can compare them against
the XAI-derived initialisation.

Use --init {xai, random} to evaluate the matching checkpoint variant.
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
import json  # noqa: E402

import torch  # noqa: E402

from shared.config import (  # noqa: E402
    CHECKPOINT_DIR,
    GLOBAL_SEED,
    SPLITS_DIR,
    TABLES_DIR,
    get_data_root,
)
from shared.dataset import BraTSDataset, get_dataloader  # noqa: E402
from shared.trainer import CheckpointManager, validate_one_epoch  # noqa: E402

from member6_xai_guided.train import _name_for_init, build_model  # noqa: E402


MODALITY = "multimodal"
ARCHITECTURE = "XAIGuidedMultimodalUNet3D"

CSV_FIELDNAMES = [
    "member", "modality", "architecture", "seed",
    "dice_wt", "dice_tc", "dice_et",
    "iou_wt", "iou_tc", "iou_et",
    "hd95_wt", "hd95_tc", "hd95_et",
    "training_time_hrs", "gpu_memory_gb",
]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--init", choices=("xai", "random"), default="xai",
        help="Which trained variant to evaluate. Must match the train run.",
    )
    args = parser.parse_args()

    member_name, _ = _name_for_init(args.init)
    print(f"[eval] init={args.init}  member_name={member_name}")

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
    model = build_model(args.init).to(device)
    ckpt = CheckpointManager(CHECKPOINT_DIR, member_name)
    model, _, epoch, val_dice = ckpt.load_best(model)
    model = model.to(device)
    print(f"[eval] loaded best checkpoint @ epoch {epoch} (val_dice_wt={val_dice:.4f})")

    metrics = validate_one_epoch(model, test_loader, device)

    sidecar_path = CHECKPOINT_DIR / f"{member_name}_train_meta.json"
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    else:
        sidecar = {"training_time_hrs": float("nan"), "gpu_memory_gb": float("nan")}

    row = {
        "member": member_name,
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
        "training_time_hrs": sidecar.get("training_time_hrs", float("nan")),
        "gpu_memory_gb":     sidecar.get("gpu_memory_gb",     float("nan")),
    }

    csv_path = TABLES_DIR / "test_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nappended test metrics for {member_name} to {csv_path}")

    # Print the post-training gate values + a delta block vs naive M5.
    final_gates = model.gate_probabilities().detach().cpu().numpy()
    print()
    print("=" * 60)
    print(f" RESULT — {member_name}")
    print("=" * 60)
    print(f"  Dice_WT  : {row['dice_wt']:.4f}")
    print(f"  Dice_TC  : {row['dice_tc']:.4f}")
    print(f"  Dice_ET  : {row['dice_et']:.4f}")
    print(f"  HD95_WT  : {row['hd95_wt']:.2f}")
    print()
    print("  Final gate probabilities (cols: t1c, t1n, t2f, t2w):")
    for sr, vec in zip(("WT", "TC", "ET"), final_gates):
        print(f"    {sr}: {vec.round(4).tolist()}")
    if "final_gates" in sidecar:
        init_gates = sidecar.get("final_gates", {})
        print()
        print("  (See sidecar JSON for gates over the full training trajectory.)")

    # Delta vs naive M5 if present.
    baseline_row = None
    with csv_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("member") == "M5_Multimodal_4ch":
                baseline_row = r
    if baseline_row is not None:
        try:
            b_wt = float(baseline_row["dice_wt"])
            b_tc = float(baseline_row["dice_tc"])
            b_et = float(baseline_row["dice_et"])
            print()
            print(f"  Delta vs naive M5:")
            print(f"    WT: {row['dice_wt'] - b_wt:+.4f}  (baseline {b_wt:.4f})")
            print(f"    TC: {row['dice_tc'] - b_tc:+.4f}  (baseline {b_tc:.4f})")
            print(f"    ET: {row['dice_et'] - b_et:+.4f}  (baseline {b_et:.4f})")
        except (KeyError, ValueError):
            pass
    print("=" * 60)


if __name__ == "__main__":
    main()
