"""Member 4 — T2f (FLAIR) training entry point.

Trains BOTH architectures and logs each as a separate MLflow run:
  Run A: M4-T2f-ResUNet3D-seed42
  Run B: M4-T2f-SegResNet-seed42

Conventions (claude.md §3):
  • set_global_seed() is the first executable line.
  • Models output 3 channels (WT, TC, ET) — no sigmoid inside the model.
  • Early stopping drives on dice_wt (not dice_et — too noisy early).
  • Validation condition: epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS
  • val/test DataLoader uses full_volume=True — never patch-based (claude.md §2.4b).
  • CosineAnnealingLR stepped after every epoch.
  • MLflow run name format: 'M{N}-{MODALITY}-{ArchName}-seed42'.

Usage (Kaggle or local):
  python train.py                      # trains both
  python train.py --arch resunet       # one arch only
  python train.py --arch segresnet
  python train.py --epochs 10          # quick smoke test
"""
from __future__ import annotations

# ── Seed first — before any other import ────────────────────────────────────
from shared.seed import set_global_seed
set_global_seed()

import argparse
import os
import sys
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch
import torch.optim as optim
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

# ── Shared infrastructure ─────────────────────────────────────────────────────
from shared.config import (
    AMP_ENABLED, BATCH_SIZE, CHECKPOINT_DIR, LR, MLRUNS_DIR,
    NUM_EPOCHS, NUM_WORKERS, PATCH_SIZE, PATIENCE, SPLITS_DIR,
    VAL_EVERY_N_EPOCHS, WEIGHT_DECAY, BEST_METRIC,
    get_data_root,
)
from shared.dataset import BraTSDataset, get_dataloader
from shared.trainer import (
    CheckpointManager, EarlyStopper,
    dice_bce_loss,
    train_one_epoch, validate_one_epoch,
)

from model import build_model

# ── Constants ─────────────────────────────────────────────────────────────────
MEMBER_NAME = "member4_T2f"
MODALITY    = "t2f"

TABLES_DIR  = RESULTS_ROOT / "T2F" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Single-arch training run
# ══════════════════════════════════════════════════════════════════════════════

def run_training(
    arch:         str,
    num_epochs:   int,
    train_loader,
    val_loader,
    device:       torch.device,
) -> dict:
    """Train one architecture; return a summary dict."""

    patch_size = PATCH_SIZE[0] if isinstance(PATCH_SIZE, tuple) else PATCH_SIZE
    model      = build_model(arch, in_channels=1, out_channels=3).to(device)
    params     = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # MLflow run name — format from claude.md §3.8
    run_name  = f"M4-{MODALITY.upper()}-{model.arch_name}-seed42"
    ckpt_name = f"member4_T2f_{arch}"

    print(f"\n{'='*60}")
    print(f"  Arch    : {model.arch_name}")
    print(f"  Params  : {params:,}")
    print(f"  Run     : {run_name}")
    print(f"  Epochs  : {num_epochs}  |  Device: {device}")
    print(f"{'='*60}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler    = (
        torch.cuda.amp.GradScaler()
        if AMP_ENABLED and device.type == "cuda" else None
    )

    ckpt_mgr  = CheckpointManager(save_dir=str(CHECKPOINT_DIR), member_name=ckpt_name)
    stopper   = EarlyStopper(patience=PATIENCE)
    best_dice = -1.0
    best_epoch = 0
    t0 = time.time()

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("brats-gli-2024")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "member":       4,
            "modality":     MODALITY,
            "architecture": model.arch_name,
            "params_M":     round(params / 1e6, 2),
            "patch_size":   patch_size,
            "batch_size":   BATCH_SIZE,
            "lr":           LR,
            "weight_decay": WEIGHT_DECAY,
            "num_epochs":   num_epochs,
            "patience":     PATIENCE,
            "amp":          AMP_ENABLED,
            "device":       str(device),
            "is_kaggle":    IS_KAGGLE,
            "best_metric":  BEST_METRIC,
        })

        for epoch in range(1, num_epochs + 1):
            # ── Training ──────────────────────────────────────────────────────
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scaler, device, dice_bce_loss
            )
            scheduler.step()

            mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)
            print(f"Epoch {epoch:03d}/{num_epochs} | train_loss={train_metrics['loss']:.4f}", end="")

            # ── Validation (per VAL_EVERY_N_EPOCHS + final epoch) ─────────────
            if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == num_epochs:
                val_metrics = validate_one_epoch(
                    model, val_loader, device, dice_bce_loss
                )

                # Log all 9 sub-region metrics
                for k, v in val_metrics.items():
                    if isinstance(v, float):
                        mlflow.log_metric(f"val_{k}", v, step=epoch)

                val_dice = val_metrics.get(BEST_METRIC, -1.0)
                is_best  = val_dice > best_dice
                if is_best:
                    best_dice  = val_dice
                    best_epoch = epoch

                print(
                    f" | dice_wt={val_metrics.get('dice_wt', 0):.4f}"
                    f" dice_tc={val_metrics.get('dice_tc', 0):.4f}"
                    f" dice_et={val_metrics.get('dice_et', 0):.4f}"
                    + (" ← best" if is_best else "")
                )

                # CheckpointManager.save accepts full metric dict (plan.md step 4)
                ckpt_mgr.save(
                    model, optimizer, epoch, val_metrics,
                    is_best=is_best, best_metric_key=BEST_METRIC,
                )

                if stopper.should_stop(val_metrics):
                    print(f"\n[M4-{arch}] Early stopping at epoch {epoch}")
                    break
            else:
                print()

        elapsed = time.time() - t0
        mlflow.log_metrics({
            "best_val_dice_wt": best_dice,
            "best_epoch":       best_epoch,
            "train_time_min":   round(elapsed / 60, 2),
        })
        run_id = run.info.run_id

    print(f"\n[M4-{arch}] Done. best_val_dice_wt={best_dice:.4f} @ epoch {best_epoch}"
          f"  ({elapsed/60:.1f} min)  mlflow_run={run_id}")

    return {
        "arch":              model.arch_name,
        "params_M":          round(params / 1e6, 2),
        "best_val_dice_wt":  round(best_dice, 4),
        "best_epoch":        best_epoch,
        "train_min":         round(elapsed / 60, 1),
        "mlflow_run_id":     run_id,
        "ckpt_name":         ckpt_name,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="M4 T2f training")
    parser.add_argument("--arch",   default="both",
                        choices=["resunet", "segresnet", "both"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override NUM_EPOCHS from config")
    args = parser.parse_args()

    num_epochs = args.epochs or NUM_EPOCHS
    archs      = ["resunet", "segresnet"] if args.arch == "both" else [args.arch]

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[M4] GPU: {torch.cuda.get_device_name(0)} | {vram:.1f} GB")
    else:
        print("[M4] CPU only — training will be slow")

    # ── Data ──────────────────────────────────────────────────────────────────
    data_root  = get_data_root()
    patch_size = PATCH_SIZE[0] if isinstance(PATCH_SIZE, tuple) else PATCH_SIZE
    nw         = NUM_WORKERS    # 4 on Kaggle (Linux), 0 on Windows

    print(f"[M4] data_root={data_root}  patch_size={patch_size}")

    # Training: patch-based with tumour bias (claude.md §2.4b)
    train_ds = BraTSDataset(
        data_root          = data_root,
        split_file         = str(SPLITS_DIR / "train_ids.txt"),
        modality           = MODALITY,
        patch_size         = patch_size,
        augment            = True,
        patches_per_volume = 4,
    )
    # Validation: full-volume (not patch-based) — required by claude.md §2.4b
    val_ds = BraTSDataset(
        data_root          = data_root,
        split_file         = str(SPLITS_DIR / "val_ids.txt"),
        modality           = MODALITY,
        patch_size         = patch_size,
        augment            = False,
        full_volume        = True,
    )

    train_loader = get_dataloader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=nw,
)
    val_loader = get_dataloader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=nw,
    )
    print(f"[M4] train_patches={len(train_ds)} | val_volumes={len(val_ds)}")

    # ── Train each architecture ───────────────────────────────────────────────
    results = []
    for arch in archs:
        result = run_training(arch, num_epochs, train_loader, val_loader, device)
        results.append(result)

    # ── Comparison summary ────────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  ARCHITECTURE COMPARISON — validation Dice_WT")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r['arch']:12s} | params={r['params_M']:.1f}M"
                  f" | val_dice_wt={r['best_val_dice_wt']:.4f}"
                  f" | best_epoch={r['best_epoch']}"
                  f" | time={r['train_min']:.0f}min")

        winner = max(results, key=lambda r: r["best_val_dice_wt"])
        print(f"\n  ✓ Winner: {winner['arch']}  (dice_wt={winner['best_val_dice_wt']:.4f})")
        print(f"    Use ckpt_name='{winner['ckpt_name']}' in evaluate.py / extract_features.py\n")

        df = pd.DataFrame(results)
        out_csv = TABLES_DIR / "M4_arch_comparison.csv"
        df.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")

    print(f"\n[M4] Training complete.")
    print(f"  MLflow: python -m mlflow ui --backend-store-uri {MLRUNS_DIR}")
    print(f"  Open:   http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
