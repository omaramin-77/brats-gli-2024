"""Member 4 — T2f (FLAIR) training entry point.

Uses M5's shared infrastructure:
    shared.config   — paths, hyperparameters, device detection
    shared.seed     — global random seed (42, same for every member)
    shared.dataset  — BraTSDataset, get_dataloader, load_splits
    shared.trainer  — train_one_epoch, validate_one_epoch, EarlyStopper,
                      CheckpointManager

Run from the repo root in VS Code terminal:
    python member4_T2f/train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── Make sure repo root is on the path so shared.* imports work ──────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Shared infrastructure ────────────────────────────────────────────────────
from shared.seed import set_global_seed
set_global_seed()  # seed=42, must happen before anything else

from shared.config import (
    get_data_root,
    SPLITS_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    PATIENCE,
    VAL_EVERY_N_EPOCHS,
    LR,
    WEIGHT_DECAY,
    AMP_ENABLED,
    PATCH_SIZE,
)
from shared.dataset import BraTSDataset, get_dataloader
from shared.trainer import (
    train_one_epoch,
    validate_one_epoch,
    EarlyStopper,
    CheckpointManager,
    dice_bce_loss,
)

# ── Member-specific imports ──────────────────────────────────────────────────
import torch
import torch.optim as optim
import mlflow

from model import T2fSegModel   # our model.py in same folder

# ── Config ───────────────────────────────────────────────────────────────────
MEMBER_NAME = "member4_T2f"
MODALITY    = "t2f"


def main() -> None:
    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M4] device = {device}")

    # ── Data root (reads DATA_PATH.local.txt) ────────────────────────────────
    data_root = get_data_root()
    print(f"[M4] data root = {data_root}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    patch_size = PATCH_SIZE[0]   # e.g. 64 or 96 — set by config based on VRAM

    train_ds = BraTSDataset(
        data_root  = data_root,
        split_file = str(SPLITS_DIR / "train_ids.txt"),
        modality   = MODALITY,
        patch_size = patch_size,
        augment    = True,
        patches_per_volume = 4,
    )
    val_ds = BraTSDataset(
        data_root  = data_root,
        split_file = str(SPLITS_DIR / "val_ids.txt"),
        modality   = MODALITY,
        patch_size = patch_size,
        augment    = False,
        patches_per_volume = 2,
    )

    train_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = get_dataloader(val_ds,   batch_size=1,          shuffle=False)
    print(f"[M4] train patches={len(train_ds)}  val patches={len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = T2fSegModel(in_channels=1, base_ch=32).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[M4] model params = {params:,}")

    # ── Optimiser + scheduler ────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ── AMP scaler (GPU only) ────────────────────────────────────────────────
    scaler = (
        torch.cuda.amp.GradScaler()
        if AMP_ENABLED and device.type == "cuda"
        else None
    )

    # ── Checkpoint + early stopping ──────────────────────────────────────────
    ckpt_manager = CheckpointManager(
        save_dir    = str(CHECKPOINT_DIR),
        member_name = MEMBER_NAME,
    )
    stopper    = EarlyStopper(patience=PATIENCE)
    best_dice  = -1.0

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow.set_experiment("brats-gli-2024")
    with mlflow.start_run(run_name=f"{MEMBER_NAME}-seed42"):
        mlflow.log_params({
            "member":      4,
            "modality":    MODALITY,
            "arch":        "ResUNet3D",
            "patch_size":  patch_size,
            "batch_size":  BATCH_SIZE,
            "lr":          LR,
            "epochs":      NUM_EPOCHS,
            "patience":    PATIENCE,
        })

        # ── Training loop ────────────────────────────────────────────────────
        for epoch in range(1, NUM_EPOCHS + 1):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scaler, device, dice_bce_loss
            )
            scheduler.step()

            print(
                f"Epoch {epoch:03d}/{NUM_EPOCHS} "
                f"| train loss {train_metrics['loss']:.4f}"
            )
            mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)

            # Validate every N epochs
            if epoch % VAL_EVERY_N_EPOCHS == 0:
                val_metrics = validate_one_epoch(model, val_loader, device, dice_bce_loss)
                val_dice = val_metrics.get("dice", 0.0)

                print(
                    f"         val → dice={val_dice:.4f} "
                    f"iou={val_metrics.get('iou', 0):.4f} "
                    f"hd95={val_metrics.get('hd95', float('nan')):.1f}"
                )

                mlflow.log_metrics(
                    {
                        "val_dice": val_dice,
                        "val_iou":  val_metrics.get("iou",  0.0),
                        "val_hd95": val_metrics.get("hd95", float("nan")),
                        "val_loss": val_metrics.get("loss", 0.0),
                    },
                    step=epoch,
                )

                is_best = val_dice > best_dice
                if is_best:
                    best_dice = val_dice

                ckpt_manager.save(
                    model, optimizer, epoch, val_dice, is_best=is_best
                )

                if stopper.should_stop(val_dice):
                    print(f"[M4] early stopping at epoch {epoch}")
                    break

        mlflow.log_metric("best_val_dice", best_dice)
        print(f"\n[M4] training complete. best val Dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()
