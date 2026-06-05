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
from ensemble.fusion import ALL_MODALITIES, LatentFusionEnsemble, build_gate_bias  # noqa: E402


# Base member name; the actual on-disk checkpoint name is suffixed with the
# modality count so 3-modality and 4-modality runs do not overwrite each other.
BASE_MEMBER_NAME = "M5_LatentFusion_XAI"
# Fusion-specific hyperparameters (override config defaults).
FUSION_NUM_EPOCHS = 60     # bumped from 30 — small head with cold-start decoder needs more time
FUSION_BATCH_SIZE = 4
FUSION_LR = 3e-4
FUSION_WARMUP_EPOCHS = 5   # linear LR warmup 0 -> FUSION_LR over first 5 epochs

# Correct path overrides — see module docstring.
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR    = RESULTS_DIR / "figures"
TABLES_DIR     = RESULTS_DIR / "tables"


def _member_name_for(modalities: tuple[str, ...]) -> str:
    """Distinct checkpoint name per modality subset."""
    if len(modalities) == len(ALL_MODALITIES):
        return BASE_MEMBER_NAME
    short = "".join(m for m in modalities)
    return f"{BASE_MEMBER_NAME}_only_{short}"


def _mlflow_run_name_for(modalities: tuple[str, ...]) -> str:
    if len(modalities) == len(ALL_MODALITIES):
        return "M5-fusion-XAI-initialised-seed42"
    short = "-".join(modalities)
    return f"M5-fusion-XAI-{short}-seed42"


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
    import argparse
    parser = argparse.ArgumentParser(
        description="Train the Latent Space Fusion Head (Pipeline C)."
    )
    parser.add_argument(
        "--modalities", nargs="+", default=list(ALL_MODALITIES),
        choices=list(ALL_MODALITIES),
        help="Modality subset to fuse. Default = all 4. Pass e.g. "
             "'--modalities t1c t1n t2f' to drop t2w (or any 3-modality "
             "ablation). The model architecture, checkpoint name, and "
             "MLflow run name all adapt automatically.",
    )
    parser.add_argument(
        "--epochs", type=int, default=FUSION_NUM_EPOCHS,
        help=f"Total epochs (default: {FUSION_NUM_EPOCHS}).",
    )
    parser.add_argument(
        "--warmup", type=int, default=FUSION_WARMUP_EPOCHS,
        help=f"Linear LR warmup epochs from 0 to lr (default: {FUSION_WARMUP_EPOCHS}).",
    )
    parser.add_argument(
        "--lr", type=float, default=FUSION_LR,
        help=f"Peak LR after warmup (default: {FUSION_LR}).",
    )
    parser.add_argument(
        "--no-normalise", action="store_true",
        help="Ablation flag: disable per-modality input normalisation "
             "(forces the attention conv to learn across raw, statistically "
             "incompatible bottleneck distributions). Default is to normalise.",
    )
    args = parser.parse_args()

    # Preserve canonical modality order (matches ALL_MODALITIES) so weight
    # columns line up across runs even if the user passes modalities in
    # arbitrary order on the command line.
    modalities = tuple(m for m in ALL_MODALITIES if m in args.modalities)
    member_name = _member_name_for(modalities)
    mlflow_run_name = _mlflow_run_name_for(modalities)
    num_epochs = int(args.epochs)
    warmup_epochs = max(0, int(args.warmup))
    peak_lr = float(args.lr)
    normalise_inputs = not args.no_normalise

    train_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"[fusion] device: {device}")
    print(f"[fusion] modalities: {modalities}  ({len(modalities)}-modality run)")
    print(f"[fusion] checkpoint name: {member_name}")
    print(f"[fusion] normalise inputs: {normalise_inputs}")
    print(f"[fusion] epochs={num_epochs} warmup={warmup_epochs} peak_lr={peak_lr}")

    train_ds = FeaturesDataset(str(SPLITS_DIR / "train_ids.txt"), modalities=modalities)
    val_ds = FeaturesDataset(str(SPLITS_DIR / "val_ids.txt"), modalities=modalities)
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
    gate_bias = build_gate_bias(bias_path, modalities=modalities)
    print(f"[fusion] gate bias (init): {gate_bias.tolist()}")
    print(f"[fusion] attention prior:   {torch.softmax(gate_bias, dim=0).tolist()}")
    print(f"[fusion] modality order:    {modalities}")

    model = LatentFusionEnsemble(
        gate_bias_init=gate_bias,
        modalities=modalities,
        normalise_inputs=normalise_inputs,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[fusion] parameters: {total_params:,}")

    # AdamW is built at peak LR; we manually scale the LR per epoch during the
    # warmup window, then hand control to CosineAnnealingLR for the remaining
    # epochs. Cosine T_max is total_epochs - warmup so the cosine cycle ends
    # at the last epoch regardless of warmup length.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak_lr, weight_decay=WEIGHT_DECAY
    )
    cosine_t_max = max(1, num_epochs - warmup_epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=cosine_t_max)

    def apply_warmup_lr(epoch_one_indexed: int) -> float:
        """Linear warmup from 0 -> peak_lr over warmup_epochs.

        Returns the LR actually set so it can be logged. Called BEFORE the
        epoch runs; cosine.step() is called AFTER the epoch finishes once
        warmup is over.
        """
        if warmup_epochs == 0 or epoch_one_indexed > warmup_epochs:
            return optimizer.param_groups[0]["lr"]
        scaled = peak_lr * (epoch_one_indexed / warmup_epochs)
        for group in optimizer.param_groups:
            group["lr"] = scaled
        return scaled
    scaler = (
        torch.amp.GradScaler("cuda")
        if (AMP_ENABLED and device.type == "cuda")
        else None
    )

    stopper = EarlyStopper(patience=15)
    ckpt = CheckpointManager(save_dir=str(CHECKPOINT_DIR), member_name=member_name)

    # MLflow lazy import — fusion runs even if mlflow isn't installed.
    try:
        import mlflow  # type: ignore
        mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
        mlflow.set_experiment(member_name)
        mlflow_run = mlflow.start_run(run_name=mlflow_run_name)
    except ImportError:
        mlflow = None
        mlflow_run = None

    if mlflow is not None:
        mlflow.log_param("member",        member_name)
        mlflow.log_param("architecture",  "LatentFusionEnsemble")
        mlflow.log_param("seed",          GLOBAL_SEED)
        mlflow.log_param("batch_size",    FUSION_BATCH_SIZE)
        mlflow.log_param("peak_lr",       peak_lr)
        mlflow.log_param("warmup_epochs", warmup_epochs)
        mlflow.log_param("weight_decay",  WEIGHT_DECAY)
        mlflow.log_param("num_epochs",    num_epochs)
        mlflow.log_param("amp",           AMP_ENABLED)
        mlflow.log_param("modalities",    ",".join(modalities))
        mlflow.log_param("normalise_inputs", normalise_inputs)
        for i, mod in enumerate(modalities):
            mlflow.log_param(f"gate_bias_{mod}", float(gate_bias[i].item()))

    print("[fusion] starting training...")
    history = defaultdict(list)
    best_dice = -1.0

    try:
        for epoch in range(1, num_epochs + 1):
            # Set LR via linear warmup for the first `warmup_epochs`; afterwards
            # the cosine scheduler steps at the end of the epoch.
            current_lr = apply_warmup_lr(epoch)
            print(f"[fusion] epoch {epoch} starting (lr={current_lr:.6f})...")
            train_metrics = train_one_fusion_epoch(
                model, train_loader, optimizer, scaler, device, dice_bce_loss
            )
            print(f"[fusion] epoch {epoch:03d}  train_loss={train_metrics['loss']:.4f}")
            history["train_loss"].append(train_metrics["loss"])
            if mlflow is not None:
                mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)

            # Only step the cosine scheduler once warmup has completed,
            # otherwise the cosine cycle starts before warmup is over.
            if epoch > warmup_epochs:
                scheduler.step()
            if mlflow is not None:
                mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

            # The `or epoch == num_epochs` guarantees a final validation
            # when num_epochs is not a multiple of VAL_EVERY_N_EPOCHS.
            if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == num_epochs:
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
        sidecar = CHECKPOINT_DIR / f"{member_name}_train_meta.json"
        sidecar.write_text(json.dumps({
            "training_time_hrs": training_time_hrs,
            "gpu_memory_gb":     gpu_memory_gb,
            "gate_bias_init":    gate_bias.tolist(),
            "modalities":        list(modalities),
            "normalise_inputs":  normalise_inputs,
            "warmup_epochs":     warmup_epochs,
            "peak_lr":           peak_lr,
            "num_epochs":        num_epochs,
        }, indent=2), encoding="utf-8")

        # Save training curves figure.
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        curves_path = FIGURES_DIR / f"training_curves_{member_name}.png"
        plot_training_curves(dict(history), member_name, save_path=str(curves_path))
        print(f"[fusion] saved training curves to {curves_path}")
        if mlflow is not None:
            mlflow.log_artifact(str(curves_path))
            if mlflow_run is not None:
                mlflow.end_run()


if __name__ == "__main__":
    main()
