"""Member 5 — Modality ablation study (Pipeline B).

Loads the trained M5 naive 4-channel multimodal baseline and measures how
much each input modality contributes to each sub-region (WT/TC/ET) by
zeroing one channel at a time and recording the Dice drop.

Channel order (per shared/preprocessing.py:preprocess_patient_multimodal):
    channel 0 = t1c
    channel 1 = t1n
    channel 2 = t2f
    channel 3 = t2w

Outputs:
  - results/tables/ablation_scores.csv             (one row per modality)
  - results/tables/modality_importance_scores.json (nested {sr: {mod: drop}})
  - results/figures/ablation_bar_chart.png         (3-panel WT/TC/ET bar chart)

The JSON drop scores initialise the Latent Fusion Head's attention gate
bias in Pipeline C (M5.4). Pipeline C normalises these into logits.
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
from monai.inferers import sliding_window_inference  # noqa: E402

from shared.config import (  # noqa: E402
    CHECKPOINT_DIR,
    FIGURES_DIR,
    GLOBAL_SEED,
    PATCH_SIZE,
    SPLITS_DIR,
    TABLES_DIR,
    get_data_root,
)
from shared.dataset import BraTSDataset, get_dataloader  # noqa: E402
from shared.metrics import compute_all_metrics, MetricTracker  # noqa: E402
from shared.trainer import CheckpointManager  # noqa: E402
from shared.visualization import plot_modality_importance_bar  # noqa: E402

from member5_multimodal.model import MultimodalUNet3D  # noqa: E402


MEMBER_NAME = "M5_Multimodal_4ch"
ARCHITECTURE = "MultimodalUNet3D"
# Channel order MUST match preprocess_patient_multimodal.
MODALITY_CHANNEL_ORDER = ("t1c", "t1n", "t2f", "t2w")
SUBREGIONS = ("wt", "tc", "et")
CSV_COLUMNS = [
    "modality",
    "dice_wt_baseline", "dice_tc_baseline", "dice_et_baseline",
    "dice_wt_ablated",  "dice_tc_ablated",  "dice_et_ablated",
    "drop_wt", "drop_tc", "drop_et",
]


@torch.no_grad()
def infer_volume(model, image, device, patch_size):
    """Run sliding-window inference on a single full-volume input.

    image: (1, 4, 128, 128, 128) tensor on device
    Returns: (1, 3, 128, 128, 128) logits tensor on device.
    """
    roi = patch_size if isinstance(patch_size, tuple) else (patch_size,) * 3
    return sliding_window_inference(
        inputs=image,
        roi_size=roi,
        sw_batch_size=4,
        predictor=model,
        overlap=0.5,
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalUNet3D(in_channels=4, out_channels=3).to(device)
    ckpt = CheckpointManager(CHECKPOINT_DIR, MEMBER_NAME)
    model, _, epoch, val_dice = ckpt.load_best(model)
    model = model.to(device)
    model.eval()
    print(
        f"[ablation] loaded best checkpoint @ epoch {epoch} "
        f"(val_dice_wt={val_dice:.4f})"
    )

    test_ds = BraTSDataset(
        data_root=get_data_root(),
        split_file=str(SPLITS_DIR / "test_ids.txt"),
        modality="multimodal",
        augment=False,
        full_volume=True,
    )
    test_loader = get_dataloader(
        test_ds, batch_size=1, shuffle=False, num_workers=0
    )
    print(f"[ablation] test patients: {len(test_ds.patient_ids)}")

    trackers = {
        "baseline": MetricTracker(),
        "t1c":      MetricTracker(),
        "t1n":      MetricTracker(),
        "t2f":      MetricTracker(),
        "t2w":      MetricTracker(),
    }

    for i, batch in enumerate(test_loader):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        # Baseline pass (no ablation).
        logits = infer_volume(model, image, device, PATCH_SIZE)
        trackers["baseline"].update(compute_all_metrics(logits, label))

        # Per-modality ablation passes — zero one channel of the input
        # before the forward pass. Clone so the original image is not
        # mutated for subsequent iterations.
        for c in range(4):
            ablated = image.clone()
            ablated[:, c, :, :, :] = 0.0
            logits = infer_volume(model, ablated, device, PATCH_SIZE)
            metrics = compute_all_metrics(logits, label)
            trackers[MODALITY_CHANNEL_ORDER[c]].update(metrics)

        if (i + 1) % 5 == 0:
            print(f"[ablation] {i + 1}/{len(test_ds.patient_ids)} done")

    agg = {name: tracker.compute() for name, tracker in trackers.items()}
    baseline = agg["baseline"]
    print()
    print(
        f"[ablation] BASELINE:  WT={baseline['dice_wt']:.4f}  "
        f"TC={baseline['dice_tc']:.4f}  ET={baseline['dice_et']:.4f}"
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "ablation_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for mod in MODALITY_CHANNEL_ORDER:
            a = agg[mod]
            row = {
                "modality": mod,
                "dice_wt_baseline": baseline["dice_wt"],
                "dice_tc_baseline": baseline["dice_tc"],
                "dice_et_baseline": baseline["dice_et"],
                "dice_wt_ablated":  a["dice_wt"],
                "dice_tc_ablated":  a["dice_tc"],
                "dice_et_ablated":  a["dice_et"],
                "drop_wt": baseline["dice_wt"] - a["dice_wt"],
                "drop_tc": baseline["dice_tc"] - a["dice_tc"],
                "drop_et": baseline["dice_et"] - a["dice_et"],
            }
            writer.writerow(row)
            print(
                f"[ablation] {mod}: WT={a['dice_wt']:.4f} "
                f"(drop {row['drop_wt']:+.4f})  "
                f"TC={a['dice_tc']:.4f} (drop {row['drop_tc']:+.4f})  "
                f"ET={a['dice_et']:.4f} (drop {row['drop_et']:+.4f})"
            )
    print(f"[ablation] wrote {csv_path}")

    # JSON: nested {sub-region: {modality: drop}}. Pipeline C consumes
    # these absolute drops and turns them into attention-gate logits.
    # Negative drops are clamped to 0 — a negative drop means ablating
    # that modality actually helped, so the model didn't need it.
    importance = {sr: {} for sr in SUBREGIONS}
    for mod in MODALITY_CHANNEL_ORDER:
        a = agg[mod]
        for sr in SUBREGIONS:
            drop = baseline[f"dice_{sr}"] - a[f"dice_{sr}"]
            importance[sr][mod] = max(0.0, float(drop))
    json_path = TABLES_DIR / "modality_importance_scores.json"
    json_path.write_text(json.dumps(importance, indent=2), encoding="utf-8")
    print(f"[ablation] wrote {json_path}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "ablation_bar_chart.png"
    plot_modality_importance_bar(importance, save_path=str(fig_path))
    print(f"[ablation] wrote {fig_path}")

    print()
    print("=" * 60)
    print("MODALITY ABLATION COMPLETE")
    print("=" * 60)
    print(f"  baseline Dice_WT : {baseline['dice_wt']:.4f}")
    print(f"  baseline Dice_TC : {baseline['dice_tc']:.4f}")
    print(f"  baseline Dice_ET : {baseline['dice_et']:.4f}")
    print()
    print("Most-important modality per sub-region (largest Dice drop):")
    for sr in SUBREGIONS:
        ranked = sorted(importance[sr].items(), key=lambda kv: kv[1], reverse=True)
        print(f"  {sr.upper()}: {ranked[0][0]} (drop={ranked[0][1]:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
