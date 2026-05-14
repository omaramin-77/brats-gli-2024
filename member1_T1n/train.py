"""Member 1 — T1n training entry point.

REFERENCE TEMPLATE — copy this pattern into your own train.py.

This stub demonstrates the complete shared-infrastructure wiring: seeding,
config, splits, dataset, AMP scaler, training/validation loops, MLflow
logging, and checkpoint management. Members 2-5 should base their train.py
on this file but swap in their own model, loss tweaks, and hyperparameters.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so ``import shared.x`` works regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Seed must be set before any randomness is consumed.
from shared.seed import set_global_seed  # noqa: E402
set_global_seed()

import torch  # noqa: E402

from shared.config import (  # noqa: E402
    AMP_ENABLED,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    LR,
    MLRUNS_DIR,
    NUM_EPOCHS,
    NUM_WORKERS,
    PATCH_SIZE,
    PATIENCE,
    SPLITS_DIR,
    VAL_EVERY_N_EPOCHS,
    WEIGHT_DECAY,
    get_data_root,
)
from shared.dataset import BraTSDataset, get_dataloader, load_splits  # noqa: E402
from shared.trainer import (  # noqa: E402
    CheckpointManager,
    EarlyStopper,
    dice_bce_loss,
    train_one_epoch,
    validate_one_epoch,
)


MEMBER_NAME = "member1_T1n"
MODALITY = "t1n"


def build_model() -> torch.nn.Module:
    """TODO: instantiate your T1n model from member1_T1n/model.py."""
    raise NotImplementedError("Member 1: import T1nSegModel and return it here.")


def main() -> None:
    data_root = get_data_root()
    splits = load_splits(SPLITS_DIR)

    patch_side = PATCH_SIZE[0] if isinstance(PATCH_SIZE, tuple) else PATCH_SIZE

    train_ds = BraTSDataset(
        data_root=data_root,
        split_file=str(SPLITS_DIR / "train_ids.txt"),
        modality=MODALITY,
        patch_size=patch_side,
        augment=True,
        patches_per_volume=4,
    )
    val_ds = BraTSDataset(
        data_root=data_root,
        split_file=str(SPLITS_DIR / "val_ids.txt"),
        modality=MODALITY,
        patch_size=patch_side,
        augment=False,
        patches_per_volume=1,
    )

    train_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = get_dataloader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler() if (AMP_ENABLED and device.type == "cuda") else None

    stopper = EarlyStopper(patience=PATIENCE)
    ckpt = CheckpointManager(save_dir=str(CHECKPOINT_DIR), member_name=MEMBER_NAME)

    # TODO: configure MLflow tracking — uncomment after you `pip install mlflow`.
    # import mlflow
    # mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    # mlflow.set_experiment(MEMBER_NAME)
    # mlflow.start_run()

    best_dice = -1.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, dice_bce_loss)
        print(f"[epoch {epoch:03d}] train_loss={train_metrics['loss']:.4f}")
        # mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)

        # Members 2-5: copy this condition exactly — do not use epoch % N alone.
        # The `or epoch == NUM_EPOCHS` clause guarantees a final validation when
        # NUM_EPOCHS is not a multiple of VAL_EVERY_N_EPOCHS, otherwise the
        # last training step's state never reaches EarlyStopper / CheckpointManager.
        if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS:
            val_metrics = validate_one_epoch(model, val_loader, device, dice_bce_loss)
            print(
                f"[epoch {epoch:03d}] val_loss={val_metrics['loss']:.4f} "
                f"dice={val_metrics.get('dice', float('nan')):.4f} "
                f"iou={val_metrics.get('iou', float('nan')):.4f} "
                f"hd95={val_metrics.get('hd95', float('nan')):.2f}"
            )
            # for k, v in val_metrics.items():
            #     mlflow.log_metric(f"val_{k}", v, step=epoch)

            is_best = val_metrics.get("dice", -1.0) > best_dice
            if is_best:
                best_dice = val_metrics["dice"]
            ckpt.save(model, optimizer, epoch, val_metrics.get("dice", 0.0), is_best=is_best)

            if stopper.should_stop(val_metrics.get("dice", -1.0)):
                print(f"[epoch {epoch:03d}] early stopping triggered")
                break

    # mlflow.end_run()


if __name__ == "__main__":
    main()