"""Train the Latent Space Fusion Head (Pipeline C).

Loads per-modality bottleneck features from disk (pre-computed by
M1-M4's extract_features.py) and trains a small fusion + decoder
network. The attention gates are initialised from the modality
importance scores produced by Pipeline B.

This is FAST training: no NIfTI loading, no encoder forward passes,
just the small fusion-and-decoder network. 30 epochs is sufficient.

Note on path resolution:
  shared.config has a Windows-specific bug — CHECKPOINT_DIR / TABLES_DIR /
  FIGURES_DIR resolve to D:\\checkpoints etc. (empty) instead of
  <repo>/results/... where the real artefacts live. We import RESULTS_DIR
  (which IS correct) and derive the per-output directories from it.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed  # noqa: E402
set_global_seed()

from collections import defaultdict  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.optim.lr_scheduler import CosineAnnealingLR  # noqa: E402

from shared.config import (  # noqa: E402
    AMP_ENABLED,
    GLOBAL_SEED,
    MLRUNS_DIR,
    NUM_WORKERS,
    RESULTS_DIR,
    SPLITS_DIR,
    VAL_EVERY_N_EPOCHS,
    WEIGHT_DECAY,
)
from shared.metrics import compute_all_metrics, MetricTracker  # noqa: E402
from shared.trainer import CheckpointManager, EarlyStopper, dice_bce_loss  # noqa: E402
from shared.visualization import plot_training_curves  # noqa: E402

from ensemble.features_dataset import FeaturesDataset  # noqa: E402
from ensemble.fusion import LatentFusionEnsemble, build_gate_bias  # noqa: E402


MEMBER_NAME = "M5_LatentFusion_XAI"
MLFLOW_RUN_NAME = "M5-fusion-XAI-initialised-seed42"
# Fusion-specific hyperparameters (override config defaults).
FUSION_NUM_EPOCHS = 30
FUSION_BATCH_SIZE = 4   # features are small; larger batch fits comfortably
FUSION_LR = 3e-4

# Correct path overrides — see module docstring.
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR    = RESULTS_DIR / "figures"
TABLES_DIR     = RESULTS_DIR / "tables"


def train_one_fusion_epoch(model, loader, optimizer, scaler, device, loss_fn):
    model.train()
    use_amp = scaler is not None and device.type == "cuda"
    total_loss, total_batches = 0.0, 0
    for batch in loader:
        features = {m: t.to(device, non_blocking=True)
                    for m, t in batch["features"].items()}
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits, _ = model(features)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
    return {"loss": total_loss / max(total_batches, 1)}


@torch.no_grad()
def validate_one_fusion_epoch(model, loader, device, loss_fn):
    model.eval()
    tracker = MetricTracker()
    total_loss, total_batches = 0.0, 0
    for batch in loader:
        features = {m: t.to(device, non_blocking=True)
                    for m, t in batch["features"].items()}
        labels = batch["label"].to(device, non_blocking=True)
        logits, _ = model(features)
        loss = loss_fn(logits, labels)
        metrics = compute_all_metrics(logits, labels)
        metrics["loss"] = float(loss.item())
        tracker.update(metrics)
        total_loss += float(loss.item())
        total_batches += 1
    agg = tracker.compute()
    agg.setdefault("loss", total_loss / max(total_batches, 1))
    return agg


def main() -> None:
    train_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"[fusion] device: {device}")

    train_ds = FeaturesDataset(str(SPLITS_DIR / "train_ids.txt"))
    val_ds = FeaturesDataset(str(SPLITS_DIR / "val_ids.txt"))
    print(f"[fusion] train patients: {len(train_ds)}  val patients: {len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=FUSION_BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=pin, drop_last=False, persistent_workers=True,
    prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=pin, drop_last=False, persistent_workers=True,
    prefetch_factor=2,
    )

    bias_path = TABLES_DIR / "modality_importance_scores.json"
    if not bias_path.exists():
        raise FileNotFoundError(
            f"modality_importance_scores.json not found at {bias_path}. "
            "Run python member5_multimodal/ablation.py first."
        )
    gate_bias = build_gate_bias(bias_path)
    print(f"[fusion] gate bias (init): {gate_bias.tolist()}")
    print(f"[fusion] attention prior:   {torch.softmax(gate_bias, dim=0).tolist()}")
    print("[fusion] modality order:    (t1c, t1n, t2f, t2w)")

    model = LatentFusionEnsemble(gate_bias_init=gate_bias).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[fusion] parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FUSION_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=FUSION_NUM_EPOCHS)
    scaler = (
        torch.amp.GradScaler("cuda")
        if (AMP_ENABLED and device.type == "cuda")
        else None
    )

    stopper = EarlyStopper(patience=15)
    ckpt = CheckpointManager(save_dir=str(CHECKPOINT_DIR), member_name=MEMBER_NAME)

    # MLflow lazy import — fusion runs even if mlflow isn't installed.
    try:
        import mlflow  # type: ignore
        mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
        mlflow.set_experiment(MEMBER_NAME)
        mlflow_run = mlflow.start_run(run_name=MLFLOW_RUN_NAME)
    except ImportError:
        mlflow = None
        mlflow_run = None

    if mlflow is not None:
        mlflow.log_param("member",        MEMBER_NAME)
        mlflow.log_param("architecture",  "LatentFusionEnsemble")
        mlflow.log_param("seed",          GLOBAL_SEED)
        mlflow.log_param("batch_size",    FUSION_BATCH_SIZE)
        mlflow.log_param("lr",            FUSION_LR)
        mlflow.log_param("weight_decay",  WEIGHT_DECAY)
        mlflow.log_param("num_epochs",    FUSION_NUM_EPOCHS)
        mlflow.log_param("amp",           AMP_ENABLED)
        for i, mod in enumerate(("t1c", "t1n", "t2f", "t2w")):
            mlflow.log_param(f"gate_bias_{mod}", float(gate_bias[i].item()))

    print("[fusion] starting training...")
    history = defaultdict(list)
    best_dice = -1.0

    try:
        for epoch in range(1, FUSION_NUM_EPOCHS + 1):
            print(f"[fusion] epoch {epoch} starting...")
            train_metrics = train_one_fusion_epoch(
                model, train_loader, optimizer, scaler, device, dice_bce_loss
            )
            print(f"[fusion] epoch {epoch:03d}  train_loss={train_metrics['loss']:.4f}")
            history["train_loss"].append(train_metrics["loss"])
            if mlflow is not None:
                mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)

            scheduler.step()
            if mlflow is not None:
                mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

            # The `or epoch == FUSION_NUM_EPOCHS` guarantees a final validation
            # when FUSION_NUM_EPOCHS is not a multiple of VAL_EVERY_N_EPOCHS.
            if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == FUSION_NUM_EPOCHS:
                val_metrics = validate_one_fusion_epoch(
                    model, val_loader, device, dice_bce_loss
                )
                print(
                    f"[fusion] epoch {epoch:03d}  val_loss={val_metrics.get('loss', 0.0):.4f}  "
                    f"WT={val_metrics.get('dice_wt', float('nan')):.4f}  "
                    f"TC={val_metrics.get('dice_tc', float('nan')):.4f}  "
                    f"ET={val_metrics.get('dice_et', float('nan')):.4f}  "
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
                    print(f"[fusion] epoch {epoch:03d}  early stopping triggered")
                    break
    finally:
        training_time_hrs = (time.time() - train_start) / 3600.0
        gpu_memory_gb = (
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda" else 0.0
        )
        print(f"[fusion] total time: {training_time_hrs:.2f} h  peak GPU mem: {gpu_memory_gb:.2f} GB")

        if mlflow is not None:
            mlflow.log_metric("training_time_hrs", training_time_hrs)
            mlflow.log_metric("gpu_memory_gb",     gpu_memory_gb)

        # Sidecar JSON so evaluate_fusion.py can read training metadata.
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        sidecar = CHECKPOINT_DIR / f"{MEMBER_NAME}_train_meta.json"
        sidecar.write_text(json.dumps({
            "training_time_hrs": training_time_hrs,
            "gpu_memory_gb":     gpu_memory_gb,
            "gate_bias_init":    gate_bias.tolist(),
        }, indent=2), encoding="utf-8")

        # Save training curves figure.
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        curves_path = FIGURES_DIR / f"training_curves_{MEMBER_NAME}.png"
        plot_training_curves(dict(history), MEMBER_NAME, save_path=str(curves_path))
        print(f"[fusion] saved training curves to {curves_path}")
        if mlflow is not None:
            mlflow.log_artifact(str(curves_path))
            if mlflow_run is not None:
                mlflow.end_run()


if __name__ == "__main__":
    main()
