"""Member 5 — Multimodal training entry point (Pipeline A — naive 4-channel baseline).

Mirrors member1_T1n/train.py exactly, with these differences:
- MODALITY = "multimodal"  → BraTSDataset returns (4, H, W, D) volumes
- model is MultimodalUNet3D(in_channels=4, out_channels=3) — no deep supervision
- loss is dice_bce_loss directly (no deep_supervision_loss wrapper)
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
from shared.dataset import BraTSDataset, get_dataloader, load_splits  # noqa: E402
from shared.trainer import (  # noqa: E402
    CheckpointManager,
    EarlyStopper,
    dice_bce_loss,
    train_one_epoch,
    validate_one_epoch,
)

from member5_multimodal.model import MultimodalUNet3D  # noqa: E402


MEMBER_NAME = "M5_Multimodal_4ch"
MODALITY = "multimodal"
# MLflow run name follows claude.md §3.8: 'M{N}-{MODALITY}-{ArchName}-seed42'.
MLFLOW_RUN_NAME = "M5-multimodal-naive-seed42"


def build_model() -> torch.nn.Module:
    """Instantiate the M5 Pipeline A architecture (4-channel ResUNet, no deep supervision)."""
    return MultimodalUNet3D(in_channels=4, out_channels=3)


def main() -> None:
    import time
    train_start = time.time()

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
        augment=False,
        full_volume=True,                  
    )

    train_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    val_loader = get_dataloader(val_ds, batch_size=1, shuffle=False, num_workers=2)

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
    ckpt = CheckpointManager(save_dir=str(CHECKPOINT_DIR), member_name=MEMBER_NAME)

    # MLflow tracking. Imported lazily so a plain ``import member5_multimodal.train``
    # has no side-effects beyond seeding/config; if mlflow is not installed,
    # training continues without logging rather than failing.
    try:
        import mlflow  # type: ignore
        mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
        mlflow.set_experiment(MEMBER_NAME)
        mlflow_run = mlflow.start_run(run_name=MLFLOW_RUN_NAME)
    except ImportError:
        mlflow = None
        mlflow_run = None

    if mlflow is not None:
        mlflow.log_param("member",       MEMBER_NAME)
        mlflow.log_param("modality",     MODALITY)
        mlflow.log_param("architecture", "MultimodalUNet3D")
        mlflow.log_param("seed",         GLOBAL_SEED)
        mlflow.log_param("batch_size",   BATCH_SIZE)
        mlflow.log_param("lr",           LR)
        mlflow.log_param("weight_decay", WEIGHT_DECAY)
        mlflow.log_param("num_epochs",   NUM_EPOCHS)
        mlflow.log_param("patch_size",   PATCH_SIZE)
        mlflow.log_param("amp",          AMP_ENABLED)

    print("Dataloaders created")
    print("Starting epochs...")

    from collections import defaultdict
    history = defaultdict(list)

    best_dice = -1.0
    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"Epoch {epoch} starting...")
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scaler, device, dice_bce_loss
            )
            print(f"[epoch {epoch:03d}] train_loss={train_metrics['loss']:.4f}")
            history["train_loss"].append(train_metrics["loss"])
            if mlflow is not None:
                mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)

            scheduler.step()
            if mlflow is not None:
                mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

            # Members 2-5: copy this condition exactly — do not use epoch % N alone.
            # The `or epoch == NUM_EPOCHS` clause guarantees a final validation when
            # NUM_EPOCHS is not a multiple of VAL_EVERY_N_EPOCHS, otherwise the
            # last training step's state never reaches EarlyStopper / CheckpointManager.
            if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS:
                val_metrics = validate_one_epoch(
                    model, val_loader, device, dice_bce_loss
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

        # Sidecar JSON so evaluate.py can read training metadata.
        import json
        sidecar = CHECKPOINT_DIR / f"{MEMBER_NAME}_train_meta.json"
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps({
            "training_time_hrs": training_time_hrs,
            "gpu_memory_gb":     gpu_memory_gb,
        }, indent=2), encoding="utf-8")

        # Save training curves figure (Phase 1 acceptance criterion).
        from shared.visualization import plot_training_curves
        curves_path = FIGURES_DIR / f"training_curves_{MEMBER_NAME}.png"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_training_curves(dict(history), MEMBER_NAME, save_path=str(curves_path))
        print(f"[train] saved training curves to {curves_path}")
        if mlflow is not None:
            mlflow.log_artifact(str(curves_path))

        if mlflow is not None and mlflow_run is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
