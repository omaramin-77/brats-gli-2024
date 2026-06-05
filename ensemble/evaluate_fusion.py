"""Evaluate the trained Latent Space Fusion Head on the test set.

Loads the best checkpoint produced by ensemble/train_fusion.py and reports
Dice / IoU / HD95 on the held-out test split. Appends one row to
results/tables/test_metrics.csv and prints deltas vs the M5_Multimodal_4ch
naive baseline so the team can immediately see whether the novel
contribution beats the floor.

Note on path resolution:
  shared.config has a Windows-specific bug — CHECKPOINT_DIR / TABLES_DIR
  resolve to D:\\... (empty) instead of <repo>/results/...  We import
  RESULTS_DIR (which IS correct) and derive correct paths from it.
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
from torch.utils.data import DataLoader  # noqa: E402

from shared.config import (  # noqa: E402
    GLOBAL_SEED,
    RESULTS_DIR,
    SPLITS_DIR,
)
from shared.trainer import CheckpointManager, dice_bce_loss  # noqa: E402

from ensemble.features_dataset import FeaturesDataset  # noqa: E402
from ensemble.fusion import ALL_MODALITIES, LatentFusionEnsemble, build_gate_bias  # noqa: E402
from ensemble.train_fusion import _member_name_for, validate_one_fusion_epoch  # noqa: E402


MODALITY = "multimodal-features"
ARCHITECTURE = "LatentFusionEnsemble"
BASELINE_MEMBER_NAME = "M5_Multimodal_4ch"

# Correct path overrides — see module docstring.
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
TABLES_DIR     = RESULTS_DIR / "tables"

CSV_FIELDNAMES = [
    "member", "modality", "architecture", "seed",
    "dice_wt", "dice_tc", "dice_et",
    "iou_wt", "iou_tc", "iou_et",
    "hd95_wt", "hd95_tc", "hd95_et",
    "training_time_hrs", "gpu_memory_gb",
]


def _read_baseline_row(csv_path: Path) -> dict | None:
    """Return the M5_Multimodal_4ch row from test_metrics.csv, or None."""
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("member") == BASELINE_MEMBER_NAME:
                return row
    return None


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate the trained Latent Space Fusion Head on the test split."
    )
    parser.add_argument(
        "--modalities", nargs="+", default=list(ALL_MODALITIES),
        choices=list(ALL_MODALITIES),
        help="Modality subset used at train time. Must match the train run "
             "you are evaluating (the checkpoint name is suffix-encoded by "
             "modality count).",
    )
    args = parser.parse_args()

    modalities = tuple(m for m in ALL_MODALITIES if m in args.modalities)
    member_name = _member_name_for(modalities)
    print(f"[eval-fusion] modalities: {modalities}")
    print(f"[eval-fusion] checkpoint name: {member_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval-fusion] device: {device}")

    test_ds = FeaturesDataset(str(SPLITS_DIR / "test_ids.txt"), modalities=modalities)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"[eval-fusion] test patients: {len(test_ds)}")

    # gate_bias is needed only for instantiation; the trained checkpoint
    # overwrites the conv bias when loaded.
    bias_path = TABLES_DIR / "modality_importance_scores.json"
    gate_bias = build_gate_bias(bias_path, modalities=modalities)

    # If the sidecar exists, replay the same normalisation flag the model
    # was trained with so loading the checkpoint succeeds.
    sidecar_path = CHECKPOINT_DIR / f"{member_name}_train_meta.json"
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    else:
        sidecar = {"training_time_hrs": float("nan"), "gpu_memory_gb": float("nan")}
    normalise_inputs = bool(sidecar.get("normalise_inputs", True))

    model = LatentFusionEnsemble(
        gate_bias_init=gate_bias,
        modalities=modalities,
        normalise_inputs=normalise_inputs,
    ).to(device)

    ckpt = CheckpointManager(CHECKPOINT_DIR, member_name)
    model, _, epoch, val_dice = ckpt.load_best(model)
    model = model.to(device)
    print(f"[eval-fusion] loaded best checkpoint @ epoch {epoch} (val_dice_wt={val_dice:.4f})")

    metrics = validate_one_fusion_epoch(model, test_loader, device, dice_bce_loss)

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
    print(f"[eval-fusion] appended row to {csv_path}")

    # Summary block — compare against the naive baseline.
    print()
    print("=" * 60)
    print("LATENT FUSION ENSEMBLE — TEST RESULTS")
    print("=" * 60)
    print(f"  member           : {member_name}")
    print(f"  architecture     : {ARCHITECTURE}")
    print(f"  Dice_WT          : {row['dice_wt']:.4f}")
    print(f"  Dice_TC          : {row['dice_tc']:.4f}")
    print(f"  Dice_ET          : {row['dice_et']:.4f}")
    print(f"  HD95_WT          : {row['hd95_wt']:.2f}")
    print()
    baseline = _read_baseline_row(csv_path)
    if baseline is None:
        print(f"  (no {BASELINE_MEMBER_NAME} row in test_metrics.csv — skip delta)")
    else:
        try:
            b_wt = float(baseline["dice_wt"])
            b_tc = float(baseline["dice_tc"])
            b_et = float(baseline["dice_et"])
            print(f"  Δ Dice_WT vs baseline : {row['dice_wt'] - b_wt:+.4f}  "
                  f"(baseline {b_wt:.4f})")
            print(f"  Δ Dice_TC vs baseline : {row['dice_tc'] - b_tc:+.4f}  "
                  f"(baseline {b_tc:.4f})")
            print(f"  Δ Dice_ET vs baseline : {row['dice_et'] - b_et:+.4f}  "
                  f"(baseline {b_et:.4f})")
        except (KeyError, ValueError) as e:
            print(f"  (could not parse baseline row: {e})")
    print("=" * 60)


if __name__ == "__main__":
    main()
