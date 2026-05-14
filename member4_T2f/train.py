"""Member 4 — T2f (FLAIR) training entry point.

Trains BOTH architectures sequentially and logs each as a separate MLflow run:
  Run A: ResUNet3D      (original plan, ~9.9 M params)
  Run B: SegResNet      (MONAI, ~4.7 M params, faster)

After both runs, a comparison summary is printed and saved to
  results/T2F/tables/M4_arch_comparison.csv

Usage (Kaggle or local):
  python train.py                     # trains both
  python train.py --arch resunet      # trains only ResUNet3D
  python train.py --arch segresnet    # trains only SegResNet
  python train.py --arch resunet --epochs 30   # override epoch count
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
import mlflow
from tqdm.auto import tqdm

# ── Environment & path setup ──────────────────────────────────────────────────
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    # On Kaggle: shared code is uploaded as a dataset called 'brats-code'
    # Adjust the dataset name if yours differs
    for candidate in [
        '/kaggle/input/brats-code/shared',
        '/kaggle/input/brats-shared-infra',
    ]:
        parent = str(Path(candidate).parent)
        if Path(candidate).exists() and parent not in sys.path:
            sys.path.insert(0, parent)
            break
    REPO_ROOT = Path('/kaggle/working')
    RESULTS_ROOT = Path('/kaggle/working/results')
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    RESULTS_ROOT = REPO_ROOT / 'results'
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

# ── Shared infrastructure ─────────────────────────────────────────────────────
from shared.seed import set_global_seed
set_global_seed()  # always first

from shared.config import (  # noqa: E402
    AMP_ENABLED, BATCH_SIZE, LR, NUM_EPOCHS, NUM_WORKERS, PATCH_SIZE,
    PATIENCE, VAL_EVERY_N_EPOCHS, WEIGHT_DECAY,
    CHECKPOINT_DIR, FIGURES_DIR, MLRUNS_DIR, SPLITS_DIR, TABLES_DIR,
    TARGET_CHANNELS, TARGET_CHANNEL_NAMES, BEST_METRIC, GLOBAL_SEED,
    get_data_root,
)
from shared.dataset import BraTSDataset, get_dataloader, load_splits  # noqa: E402,F401


MEMBER_NAME = "member4_T2f"
MODALITY    = "t2f"

# ── Output dirs ───────────────────────────────────────────────────────────────
TABLES_DIR = RESULTS_ROOT / "T2F" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        use_amp = AMP_ENABLED and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images, return_features=False)
            loss    = dice_bce_loss(outputs, masks)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    pbar = tqdm(loader, desc="  Val  ", leave=False, unit="batch")

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device, non_blocking=True)

        outputs = model(images, return_features=False)
        loss    = dice_bce_loss(outputs, masks)

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        inter = (preds * masks).sum()
        union = preds.sum() + masks.sum()

        dice = (2.0 * inter) / (union + 1e-8)
        iou  = inter / (union - inter + 1e-8)

        total_loss += loss.item()
        total_dice += dice.item()
        total_iou  += iou.item()
        pbar.set_postfix(dice=f"{dice.item():.4f}")

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou":  total_iou  / n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Single-arch training pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_training(arch: str, num_epochs: int, train_loader, val_loader, device) -> dict:
    """Train one architecture, return best metrics dict."""

    patch_size  = PATCH_SIZE[0]
    model       = build_model(arch, in_channels=1).to(device)
    params      = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_name    = f"M4-{arch}-T2f-seed42"
    ckpt_name   = f"member4_T2f_{arch}"

    print(f"\n{'='*60}")
    print(f"  Architecture : {model.arch_name}")
    print(f"  Parameters   : {params:,}")
    print(f"  MLflow run   : {run_name}")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {num_epochs}")
    print(f"{'='*60}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler    = (
        torch.cuda.amp.GradScaler()
        if AMP_ENABLED and device.type == "cuda" else None
    )

    ckpt_manager = CheckpointManager(
        save_dir    = str(CHECKPOINT_DIR),
        member_name = ckpt_name,
    )
    stopper   = EarlyStopper(patience=PATIENCE)
    best_dice = -1.0
    best_epoch = 0
    t_start = time.time()

    mlflow.set_experiment("brats-gli-2024")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "member":         4,
            "modality":       MODALITY,
            "architecture":   model.arch_name,
            "params_M":       round(params / 1e6, 2),
            "patch_size":     patch_size,
            "batch_size":     BATCH_SIZE,
            "lr":             LR,
            "weight_decay":   WEIGHT_DECAY,
            "num_epochs":     num_epochs,
            "patience":       PATIENCE,
            "amp":            AMP_ENABLED,
            "device":         str(device),
            "is_kaggle":      IS_KAGGLE,
        })

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
            scheduler.step()

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            print(f"Epoch {epoch:03d}/{num_epochs} | train_loss={train_loss:.4f}", end="")

            if epoch % VAL_EVERY_N_EPOCHS == 0:
                val = validate_one_epoch(model, val_loader, device)
                val_dice = val["dice"]

                mlflow.log_metrics({
                    "val_loss": val["loss"],
                    "val_dice": val_dice,
                    "val_iou":  val["iou"],
                }, step=epoch)

                is_best = val_dice > best_dice
                if is_best:
                    best_dice  = val_dice
                    best_epoch = epoch

                print(f" | val_dice={val_dice:.4f} iou={val['iou']:.4f}"
                      + (" ← best" if is_best else ""))

                ckpt_manager.save(model, optimizer, epoch, val_dice, is_best=is_best)

                if stopper.should_stop(val_dice):
                    print(f"\n[M4-{arch}] Early stopping at epoch {epoch}")
                    break
            else:
                print()

        elapsed = time.time() - t_start
        mlflow.log_metrics({
            "best_val_dice":   best_dice,
            "best_epoch":      best_epoch,
            "train_time_min":  round(elapsed / 60, 2),
        })

        run_id = run.info.run_id

    print(f"\n[M4-{arch}] Done. best_val_dice={best_dice:.4f} @ epoch {best_epoch}"
          f" ({elapsed/60:.1f} min)  MLflow run_id={run_id}")

    return {
        "arch":          model.arch_name,
        "params_M":      round(params / 1e6, 2),
        "best_val_dice": round(best_dice, 4),
        "best_epoch":    best_epoch,
        "train_min":     round(elapsed / 60, 1),
        "mlflow_run_id": run_id,
        "ckpt_name":     ckpt_name,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="M4 T2f training")
    parser.add_argument("--arch",   default="both",
                        choices=["resunet", "segresnet", "both"],
                        help="Which architecture to train")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override NUM_EPOCHS from config")
    args = parser.parse_args()

    num_epochs = args.epochs or NUM_EPOCHS
    archs      = ["resunet", "segresnet"] if args.arch == "both" else [args.arch]

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[M4] GPU: {torch.cuda.get_device_name(0)} | {vram:.1f} GB VRAM")
    else:
        print("[M4] Running on CPU — training will be slow")

    # ── Data ──────────────────────────────────────────────────────────────────
    data_root  = get_data_root()
    patch_size = PATCH_SIZE[0]
    print(f"[M4] data_root  = {data_root}")
    print(f"[M4] patch_size = {patch_size}")

    # num_workers: 4 on Kaggle (Linux), 0 on Windows
    nw = 4 if IS_KAGGLE else 0

    train_ds = BraTSDataset(
        data_root          = data_root,
        split_file         = str(SPLITS_DIR / "train_ids.txt"),
        modality           = MODALITY,
        patch_size         = patch_size,
        augment            = True,
        patches_per_volume = 4,
    )
    val_ds = BraTSDataset(
        data_root          = data_root,
        split_file         = str(SPLITS_DIR / "val_ids.txt"),
        modality           = MODALITY,
        patch_size         = patch_size,
        augment            = False,
        patches_per_volume = 2,
    )

    train_loader = get_dataloader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=nw, pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
    )
    val_loader = get_dataloader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=nw, pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
    )
    print(f"[M4] train patches={len(train_ds)} | val patches={len(val_ds)}")

    # ── Train each architecture ───────────────────────────────────────────────
    results = []
    for arch in archs:
        result = run_training(arch, num_epochs, train_loader, val_loader, device)
        results.append(result)

    # ── Comparison summary ────────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  ARCHITECTURE COMPARISON (validation Dice)")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r['arch']:12s} | params={r['params_M']:.1f}M "
                  f"| val_dice={r['best_val_dice']:.4f} "
                  f"| best_epoch={r['best_epoch']} "
                  f"| time={r['train_min']:.0f}min")

        winner = max(results, key=lambda r: r["best_val_dice"])
        print(f"\n  ✓ Winner: {winner['arch']} (Dice={winner['best_val_dice']:.4f})")
        print(f"    Use ckpt_name='{winner['ckpt_name']}' in evaluate.py\n")

        df = pd.DataFrame(results)
        out_path = TABLES_DIR / "M4_arch_comparison.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    print("\n[M4] Training complete. Open MLflow to compare runs:")
    print("  python -m mlflow ui   →  http://127.0.0.1:5000")
    print("  (on Kaggle: install ngrok or use kaggle-specific port forwarding)")


if __name__ == "__main__":
    main()
