"""Member 2 — T1c training entry point.

3D Attention-Gated U-Net trained on the T1c modality. Two experiments
share this file via the ``--loss`` flag:

    python member2_T1c/train.py                  # primary, dice_bce_loss
    python member2_T1c/train.py --loss focal     # secondary, Dice + focal

The focal variant is M2's design.md §4 second experiment. The two runs
produce independent checkpoints, sidecar JSONs, MLflow runs, and
training-curves PNGs so evaluate.py / xai_analysis.py / extract_features.py
can default to the dicebce (canonical) checkpoint without overwriting
the ablation.
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

import argparse  # noqa: E402

import torch  # noqa: E402

# tqdm gives batch-level progress visibility. With ~1100 batches/epoch on M2
# the per-epoch print otherwise leaves ~15-20 min of silence between
# "Epoch N starting..." and the train_loss line. Falls back to a no-op
# wrapper if tqdm is not installed so training never fails on a missing
# progress library.
try:
    from tqdm import tqdm  # noqa: E402
except ImportError:                                                # pragma: no cover
    def tqdm(iterable=None, **_kwargs):                            # type: ignore[no-redef]
        return iterable

from shared.config import (  # noqa: E402
    AMP_ENABLED,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    FIGURES_DIR,
    GLOBAL_SEED,
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
from shared.dataset import BraTSDataset, get_dataloader  # noqa: E402
from shared.trainer import (  # noqa: E402
    CheckpointManager,
    EarlyStopper,
    dice_bce_loss,
    focal_loss,
    train_one_epoch,
    validate_one_epoch,
)

from member2_T1c.model import AttentionUNet3D  # noqa: E402


MODALITY = "t1c"
ARCHITECTURE = "AttentionUNet3D"


def build_model() -> torch.nn.Module:
    """Instantiate the M2 T1c architecture (3D Attention-Gated U-Net)."""
    return AttentionUNet3D(in_channels=1, out_channels=3)


def dice_focal_loss(pred_logits, target):
    """Dice + focal — replaces the BCE component of dice_bce_loss with focal_loss.

    Per design.md §4 M2, M2's second experiment swaps focal in for BCE.
    The Dice term is preserved (it handles the per-channel class imbalance
    that BCE/focal alone cannot); focal is mixed in at 0.5 weight to match
    the BCE weighting in the primary loss.
    """
    dice_only = dice_bce_loss(pred_logits, target, dice_weight=1.0, bce_weight=0.0)
    focal = focal_loss(pred_logits, target, gamma=2.0, alpha=0.25)
    return dice_only + 0.5 * focal


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Training requires CUDA. No GPU detected — check your conda env / "
            "CUDA toolkit. Set CUDA_VISIBLE_DEVICES if you have multiple GPUs."
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", choices=["dicebce", "focal"], default="dicebce")
    args = parser.parse_args()

    if args.loss == "focal":
        member_name = "M2_T1c_AttUNet_focal"
        mlflow_run_name = "M2-T1c-AttUNet-focal"
        loss_fn = dice_focal_loss
        loss_label = "dice+focal"
    else:
        member_name = "M2_T1c_AttUNet"
        mlflow_run_name = "M2-T1c-AttUNet-dicebce"
        loss_fn = dice_bce_loss
        loss_label = "dice+bce"

    print(f"[train] loss={loss_label}  member={member_name}  run={mlflow_run_name}")

    import time
    train_start = time.time()

    data_root = get_data_root()

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
        augment=False,
        full_volume=True,                  # full-volume eval, SWI tiles internally
    )

    train_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = get_dataloader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler("cuda") if (AMP_ENABLED and device.type == "cuda") else None

    stopper = EarlyStopper(patience=PATIENCE)
    ckpt = CheckpointManager(save_dir=str(CHECKPOINT_DIR), member_name=member_name)

    # MLflow tracking. Lazy-imported so a plain ``import member2_T1c.train``
    # has no side-effects beyond seeding/config; if mlflow is not installed,
    # training continues without logging rather than failing.
    try:
        import mlflow  # type: ignore
        mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
        mlflow.set_experiment(member_name)
        mlflow_run = mlflow.start_run(run_name=mlflow_run_name)
    except ImportError:
        mlflow = None
        mlflow_run = None

    if mlflow is not None:
        mlflow.log_param("member",       member_name)
        mlflow.log_param("modality",     MODALITY)
        mlflow.log_param("architecture", ARCHITECTURE)
        mlflow.log_param("seed",         GLOBAL_SEED)
        mlflow.log_param("batch_size",   BATCH_SIZE)
        mlflow.log_param("lr",           LR)
        mlflow.log_param("weight_decay", WEIGHT_DECAY)
        mlflow.log_param("num_epochs",   NUM_EPOCHS)
        mlflow.log_param("patch_size",   PATCH_SIZE)
        mlflow.log_param("amp",          AMP_ENABLED)
        mlflow.log_param("loss",         loss_label)

    print("Dataloaders created")
    print("Starting epochs...")

    from collections import defaultdict
    history = defaultdict(list)

    best_dice = -1.0
    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"Epoch {epoch} starting...")
            train_bar = tqdm(
                train_loader,
                desc=f"epoch {epoch:03d}/{NUM_EPOCHS} train",
                leave=False,
                dynamic_ncols=True,
            )
            train_metrics = train_one_epoch(
                model, train_bar, optimizer, scaler, device, loss_fn
            )
            print(f"[epoch {epoch:03d}] train_loss={train_metrics['loss']:.4f}")
            history["train_loss"].append(train_metrics["loss"])
            if mlflow is not None:
                mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)

            scheduler.step()
            if mlflow is not None:
                mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

            # Final-epoch validation guard (claude.md §3.7 / design.md §10.5):
            # without ``or epoch == NUM_EPOCHS`` the final epoch never validates
            # when NUM_EPOCHS is not a multiple of VAL_EVERY_N_EPOCHS, so
            # EarlyStopper and CheckpointManager operate on stale numbers.
            if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS:
                val_bar = tqdm(
                    val_loader,
                    desc=f"epoch {epoch:03d}/{NUM_EPOCHS} val",
                    leave=False,
                    dynamic_ncols=True,
                )
                val_metrics = validate_one_epoch(
                    model, val_bar, device, loss_fn
                )
                print(
                    f"[epoch {epoch:03d}] val_loss={val_metrics.get('loss', 0.0):.4f} "
                    f"WT={val_metrics.get('dice_wt', float('nan')):.4f} "
                    f"TC={val_metrics.get('dice_tc', float('nan')):.4f} "
                    f"ET={val_metrics.get('dice_et', float('nan')):.4f} "
                    f"HD95_WT={val_metrics.get('hd95_wt', float('nan')):.2f}"
                )
                if mlflow is not None:
                    for k, v in val_metrics.items():
                        mlflow.log_metric(f"val_{k}", v, step=epoch)

                for key in ("dice_wt", "dice_tc", "dice_et",
                            "iou_wt", "iou_tc", "iou_et",
                            "hd95_wt", "hd95_tc", "hd95_et"):
                    history[f"val_{key}"].append(val_metrics.get(key, float("nan")))
                history["lr"].append(optimizer.param_groups[0]["lr"])

                current_wt = val_metrics.get("dice_wt", -1.0)
                is_best = current_wt > best_dice
                if is_best:
                    best_dice = current_wt
                ckpt.save(model, optimizer, epoch, val_metrics, is_best=is_best,
                          best_metric_key="dice_wt")

                if stopper.should_stop(val_metrics):
                    print(f"[epoch {epoch:03d}] early stopping triggered")
                    break
    finally:
        training_time_hrs = (time.time() - train_start) / 3600.0
        gpu_memory_gb = (
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda" else 0.0
        )
        print(f"[train] total time: {training_time_hrs:.2f} h, peak GPU mem: {gpu_memory_gb:.2f} GB")

        if mlflow is not None:
            mlflow.log_metric("training_time_hrs", training_time_hrs)
            mlflow.log_metric("gpu_memory_gb",     gpu_memory_gb)

        # Sidecar JSON so evaluate.py can read training metadata that MLflow
        # holds but is not pulled from MLflow at eval time.
        import json
        sidecar = CHECKPOINT_DIR / f"{member_name}_train_meta.json"
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps({
            "training_time_hrs": training_time_hrs,
            "gpu_memory_gb":     gpu_memory_gb,
        }, indent=2), encoding="utf-8")

        # Save training curves figure (Phase 1 acceptance criterion).
        from shared.visualization import plot_training_curves
        curves_path = FIGURES_DIR / f"training_curves_{member_name}.png"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_training_curves(dict(history), member_name, save_path=str(curves_path))
        print(f"[train] saved training curves to {curves_path}")
        if mlflow is not None:
            mlflow.log_artifact(str(curves_path))

        if mlflow is not None and mlflow_run is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
