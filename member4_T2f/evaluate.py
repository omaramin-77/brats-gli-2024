"""Member 4 — T2f (FLAIR) test-set evaluation.

Evaluates BOTH trained checkpoints (ResUNet3D + SegResNet) on the test split,
computes Dice_WT / IoU_WT / HD95_WT, prints a comparison table, and saves:
  results/T2F/tables/M4_row.csv        ← your row for M5's comparison table
  results/T2F/tables/M4_test_full.csv  ← per-architecture detail

Usage:
  python evaluate.py                        # evaluates both archs
  python evaluate.py --arch resunet         # only ResUNet3D
  python evaluate.py --arch segresnet       # only SegResNet
"""
from __future__ import annotations

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

from shared.seed import set_global_seed
set_global_seed()

from shared.config import (
    get_data_root,
    SPLITS_DIR,
    CHECKPOINT_DIR,
    PATCH_SIZE,
)
from shared.dataset import BraTSDataset, get_dataloader
from shared.metrics import compute_all_metrics
from shared.trainer import CheckpointManager

from model import build_model

MODALITY   = "t2f"
TABLES_DIR = RESULTS_ROOT / "T2F" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation loop for one model
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader, device) -> dict:
    """Run model over loader, return mean Dice/IoU/HD95."""
    model.eval()
    all_dice, all_iou, all_hd95 = [], [], []

    pbar = tqdm(loader, desc="  Evaluating", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["mask"].to(device, non_blocking=True)

        # Ensure labels have channel dim: (B,1,D,H,W)
        if labels.dim() == 4:
            labels = labels.unsqueeze(1)

        logits = model(images, return_features=False)
        m      = compute_all_metrics(logits, labels)

        all_dice.append(m["dice"])
        all_iou.append(m["iou"])
        all_hd95.append(m["hd95"])
        pbar.set_postfix(dice=f"{m['dice']:.4f}")

    return {
        "dice_wt": float(np.nanmean(all_dice)),
        "iou_wt":  float(np.nanmean(all_iou)),
        "hd95_wt": float(np.nanmean(all_hd95)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="M4 T2f evaluation")
    parser.add_argument("--arch", default="both",
                        choices=["resunet", "segresnet", "both"])
    args   = parser.parse_args()
    archs  = ["resunet", "segresnet"] if args.arch == "both" else [args.arch]

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root  = get_data_root()
    patch_size = PATCH_SIZE[0]

    # ── Test DataLoader (never seen during training) ──────────────────────────
    nw = 4 if IS_KAGGLE else 0
    test_ds = BraTSDataset(
        data_root          = data_root,
        split_file         = str(SPLITS_DIR / "test_ids.txt"),
        modality           = MODALITY,
        patch_size         = patch_size,
        augment            = False,
        patches_per_volume = 2,
    )
    test_loader = get_dataloader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=nw, pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
    )
    print(f"[M4-eval] device={device} | test_patches={len(test_ds)}")

    rows = []
    for arch in archs:
        ckpt_name = f"member4_T2f_{arch}"
        model     = build_model(arch, in_channels=1)

        # Load best checkpoint
        try:
            manager        = CheckpointManager(str(CHECKPOINT_DIR), ckpt_name)
            model, _, epoch, val_dice = manager.load_best(model)
            print(f"\n[{arch}] Loaded checkpoint (epoch={epoch}, val_dice={val_dice:.4f})")
        except FileNotFoundError:
            print(f"\n[{arch}] No checkpoint found at {CHECKPOINT_DIR}/{ckpt_name}_best.pt")
            print(f"         → Run train.py --arch {arch} first")
            continue

        model = model.to(device)
        metrics = evaluate_model(model, test_loader, device)

        print(f"[{arch}] TEST RESULTS:")
        print(f"  Dice_WT = {metrics['dice_wt']:.4f}")
        print(f"  IoU_WT  = {metrics['iou_wt']:.4f}")
        print(f"  HD95_WT = {metrics['hd95_wt']:.1f} mm")

        rows.append({
            "Member":         "M4",
            "Modality":       "T2f (FLAIR)",
            "Architecture":   model.arch_name,
            "Dice_WT":        round(metrics["dice_wt"], 4),
            "IoU_WT":         round(metrics["iou_wt"],  4),
            "HD95_WT":        round(metrics["hd95_wt"], 1),
            "Val_Dice":       round(val_dice, 4),
            "Best_Epoch":     epoch,
        })

    if not rows:
        print("\n[M4-eval] No checkpoints found. Train first.")
        return

    # ── Save results ──────────────────────────────────────────────────────────
    df_full = pd.DataFrame(rows)
    df_full.to_csv(TABLES_DIR / "M4_test_full.csv", index=False)
    print(f"\nSaved: {TABLES_DIR / 'M4_test_full.csv'}")

    # Best row for M5's comparison table (highest Dice)
    best = df_full.loc[df_full["Dice_WT"].idxmax()]
    best_df = best.to_frame().T
    best_df.to_csv(TABLES_DIR / "M4_row.csv", index=False)
    print(f"Saved: {TABLES_DIR / 'M4_row.csv'}  ← share this with M5")

    # ── Print comparison ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  M4 TEST RESULTS — Architecture Comparison")
    print(f"{'='*55}")
    print(df_full[["Architecture", "Dice_WT", "IoU_WT", "HD95_WT"]].to_string(index=False))

    if len(rows) > 1:
        winner = df_full.loc[df_full["Dice_WT"].idxmax(), "Architecture"]
        print(f"\n  ✓ Best for T2f: {winner}")

    # ── Create test_metrics.csv so xai_analysis.py can run ───────────────────
    df_full.to_csv(TABLES_DIR / "test_metrics.csv", index=False)
    print(f"\ntest_metrics.csv written → xai_analysis.py is now unlocked")


if __name__ == "__main__":
    main()
