"""Member 3 — T2w training entry point.

Based on the reference template in member1_T1n/train.py.
Only changes from the template:
  - MEMBER_NAME = "member3_T2w"
  - MODALITY    = "t2w"
  - build_model() imports T2wSegModel (Swin-UNETR)
  - LR shadowed to 1e-4 (AdamW + Transformer standard)
  - PATCH_SIZE / BATCH_SIZE / FEATURE_SIZE shadowed for 6 GB VRAM (RTX 3050)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make repo root importable so ``import shared.x`` works regardless of CWD.
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

# Member-3 specific imports
from model import build_model  # noqa: E402


# ---------------------------------------------------------------------------
# Member-3 identity
# ---------------------------------------------------------------------------
MEMBER_NAME = "member3_T2w"
MODALITY    = "t2w"

# ---------------------------------------------------------------------------
# RTX 3050 (6 GB VRAM) overrides — shadow shared defaults, do NOT edit config.py
# ---------------------------------------------------------------------------
PATCH_SIZE    = (64, 64, 64)   # 96^3 would OOM on 6 GB
BATCH_SIZE    = 1              # one patch at a time
FEATURE_SIZE  = 24             # smaller Swin encoder — fits in 6 GB
LR            = 1e-4           # AdamW + Transformer standard


def main() -> None:
    data_root = get_data_root()
    splits = load_splits(SPLITS_DIR)

    patch_side = PATCH_SIZE[0]   # always a tuple now

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

    train_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = get_dataloader(val_ds,   batch_size=1,          shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{MEMBER_NAME}] device={device}  patch={PATCH_SIZE}  feature_size={FEATURE_SIZE}")
    print(f"[{MEMBER_NAME}] train={len(train_ds)} samples  val={len(val_ds)} samples")

    model = build_model(img_size=PATCH_SIZE, feature_size=FEATURE_SIZE).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{MEMBER_NAME}] Swin-UNETR  {total_params:.1f} M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.cuda.amp.GradScaler() if (AMP_ENABLED and device.type == "cuda") else None

    stopper = EarlyStopper(patience=PATIENCE)
    ckpt    = CheckpointManager(save_dir=str(CHECKPOINT_DIR), member_name=MEMBER_NAME)

    # Optional MLflow — uncomment after `pip install mlflow`
    # import mlflow
    # mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    # mlflow.set_experiment(MEMBER_NAME)
    # mlflow.start_run()

    best_dice = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):

        # ── Train ──────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, dice_bce_loss
        )
        print(f"[epoch {epoch:03d}] train_loss={train_metrics['loss']:.4f}")
        # mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)

        # ── Validate every N epochs (copy this condition exactly per template)
        if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS:
            val_metrics = validate_one_epoch(
                model, val_loader, device, dice_bce_loss
            )
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

            ckpt.save(
                model, optimizer, epoch,
                val_metrics.get("dice", 0.0),
                is_best=is_best,
            )

            if stopper.should_stop(val_metrics.get("dice", -1.0)):
                print(f"[epoch {epoch:03d}] early stopping triggered")
                break

    # mlflow.end_run()

    print(f"\n[{MEMBER_NAME}] Training complete. Best val Dice = {best_dice:.4f}")
    print(f"[{MEMBER_NAME}] Phase 1 target: dice > 0.60  →  {'PASS ✓' if best_dice > 0.60 else 'not yet — keep tuning'}")


if __name__ == "__main__":
    main()
