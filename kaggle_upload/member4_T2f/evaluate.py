"""Member 4 — T2f (FLAIR) test-set evaluation.

Evaluates BOTH trained checkpoints on the test split, computes all 9 BraTS
metrics (Dice/IoU/HD95 × WT/TC/ET), and saves:
  results/T2F/tables/M4_test_full.csv   — both architectures
  results/tables/test_metrics.csv       — team-wide file, one row per model
                                          (appended, not overwritten)

Conventions (claude.md):
  • set_global_seed() first.
  • Full-volume inference only — never patch-based (§2.4b).
  • Best checkpoint via CheckpointManager.load_best.
  • Reports dice_wt, dice_tc, dice_et separately — BraTS protocol (§2.10).
  • Writes test_metrics.csv so xai_analysis.py contamination guard passes.

Usage:
  python evaluate.py                    # both archs
  python evaluate.py --arch resunet
  python evaluate.py --arch segresnet
"""
from __future__ import annotations

# ── Seed first ────────────────────────────────────────────────────────────────
from shared.seed import set_global_seed
set_global_seed()

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    for candidate in [
        '/kaggle/input/brats-code/shared',
        '/kaggle/input/brats-shared-infra',
    ]:
        parent = str(Path(candidate).parent)
        if Path(candidate).exists() and parent not in sys.path:
            sys.path.insert(0, parent)
            break
    REPO_ROOT    = Path('/kaggle/working')
    RESULTS_ROOT = Path('/kaggle/working/results')
else:
    REPO_ROOT    = Path(__file__).resolve().parent.parent
    RESULTS_ROOT = REPO_ROOT / 'results'
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from shared.config import (
    CHECKPOINT_DIR, NUM_WORKERS, PATCH_SIZE, SPLITS_DIR, TABLES_DIR,
    get_data_root,
)
from shared.dataset import BraTSDataset, get_dataloader
from shared.metrics import compute_all_metrics, MetricTracker
from shared.trainer import CheckpointManager

from model import build_model

MODALITY = "t2f"
LOCAL_TABLES = RESULTS_ROOT / "T2F" / "tables"
LOCAL_TABLES.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader, device: torch.device) -> dict:
    """Full-volume inference → mean 9-metric dict."""
    model.eval()
    tracker = MetricTracker()

    for batch in tqdm(loader, desc="  Test inference", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)   # (B,3,D,H,W)

        logits  = model(images)
        metrics = compute_all_metrics(logits, labels)
        tracker.update(metrics)

    return tracker.compute()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="M4 T2f evaluation")
    parser.add_argument("--arch", default="both",
                        choices=["resunet", "segresnet", "both"])
    args  = parser.parse_args()
    archs = ["resunet", "segresnet"] if args.arch == "both" else [args.arch]

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = get_data_root()
    patch_size = PATCH_SIZE[0] if isinstance(PATCH_SIZE, tuple) else PATCH_SIZE
    nw        = NUM_WORKERS

    # ── Test loader — full volumes (claude.md §2.4b) ──────────────────────────
    test_ds = BraTSDataset(
        data_root  = data_root,
        split_file = str(SPLITS_DIR / "test_ids.txt"),
        modality   = MODALITY,
        patch_size = patch_size,
        augment    = False,
        full_volume = True,
    )
    test_loader = get_dataloader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=nw, pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
    )
    print(f"[M4-eval] device={device} | test_volumes={len(test_ds)}")

    rows = []
    for arch in archs:
        ckpt_name = f"member4_T2f_{arch}"
        model     = build_model(arch, in_channels=1, out_channels=3)

        try:
            mgr                          = CheckpointManager(str(CHECKPOINT_DIR), ckpt_name)
            model, _, epoch, val_metrics = mgr.load_best(model)
            val_dice_wt = (
                val_metrics.get("dice_wt", val_metrics)
                if isinstance(val_metrics, dict)
                else float(val_metrics)
            )
            print(f"\n[{arch}] Loaded checkpoint epoch={epoch}  val_dice_wt={val_dice_wt:.4f}")
        except FileNotFoundError:
            print(f"\n[{arch}] No checkpoint at {CHECKPOINT_DIR}/{ckpt_name}_best.pt")
            print(f"         → Run: python train.py --arch {arch}")
            continue

        model = model.to(device)
        m     = evaluate_model(model, test_loader, device)

        print(f"[{arch}] TEST RESULTS:")
        print(f"  Dice  WT={m['dice_wt']:.4f}  TC={m['dice_tc']:.4f}  ET={m['dice_et']:.4f}")
        print(f"  IoU   WT={m['iou_wt']:.4f}  TC={m['iou_tc']:.4f}  ET={m['iou_et']:.4f}")
        print(f"  HD95  WT={m['hd95_wt']:.1f}  TC={m['hd95_tc']:.1f}  ET={m['hd95_et']:.1f} mm")

        row = {
            "member":        "M4",
            "modality":      "T2f (FLAIR)",
            "architecture":  model.arch_name,
            # 9 BraTS metrics
            "dice_wt":  round(m["dice_wt"],  4),
            "dice_tc":  round(m["dice_tc"],  4),
            "dice_et":  round(m["dice_et"],  4),
            "iou_wt":   round(m["iou_wt"],   4),
            "iou_tc":   round(m["iou_tc"],   4),
            "iou_et":   round(m["iou_et"],   4),
            "hd95_wt":  round(m["hd95_wt"],  1),
            "hd95_tc":  round(m["hd95_tc"],  1),
            "hd95_et":  round(m["hd95_et"],  1),
            # bookkeeping
            "val_dice_wt":  round(val_dice_wt, 4),
            "best_epoch":   epoch,
        }
        rows.append(row)

    if not rows:
        print("\n[M4-eval] No checkpoints found. Run train.py first.")
        return

    df = pd.DataFrame(rows)

    # ── Save local detail file ────────────────────────────────────────────────
    df.to_csv(LOCAL_TABLES / "M4_test_full.csv", index=False)
    print(f"\nSaved: {LOCAL_TABLES / 'M4_test_full.csv'}")

    # ── Append to team-wide test_metrics.csv (creates if absent) ─────────────
    team_csv = TABLES_DIR / "test_metrics.csv"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if team_csv.exists():
        existing = pd.read_csv(team_csv)
        # Drop any previous M4 rows so we don't accumulate duplicates
        existing = existing[
            ~((existing["member"] == "M4") & (existing["architecture"].isin(df["architecture"].tolist())))
        ]
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined.to_csv(team_csv, index=False)
    print(f"Appended: {team_csv}  ← xai_analysis.py is now unlocked")

    # ── Comparison printout ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  M4 TEST RESULTS — Architecture Comparison")
    print(f"{'='*60}")
    cols = ["architecture", "dice_wt", "dice_tc", "dice_et", "hd95_wt"]
    print(df[cols].to_string(index=False))

    if len(rows) > 1:
        best_arch = df.loc[df["dice_wt"].idxmax(), "architecture"]
        print(f"\n  ✓ Best for T2f FLAIR: {best_arch}")
        print(f"    Use this checkpoint for extract_features.py")


if __name__ == "__main__":
    main()
