"""Member 6 — XAI-guided multi-task M5 training entry point.

Identical to member5_multimodal/train.py except:
- The model is XAIGuidedMultimodalUNet3D, NOT MultimodalUNet3D.
- A --init {xai, random} CLI flag swaps between XAI-prior gate init (the
  headline configuration) and uniform gate init (the ablation control).
- Per-validation epoch we log the current (3, 4) gate probabilities to
  MLflow so the report can show whether the XAI prior survived training.

Naive M5 baseline (MultimodalUNet3D) is NOT touched — its checkpoint stays
on disk as the comparison row in test_metrics.csv.
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
    FIGURES_DIR,
    GLOBAL_SEED,
    LR,
    MLRUNS_DIR,
    NUM_EPOCHS,
    NUM_WORKERS,
    PATCH_SIZE,
    PATIENCE,
    SPLITS_DIR,
    TABLES_DIR,
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

from member6_xai_guided.model import (  # noqa: E402
    XAIGuidedMultimodalUNet3D,
    build_xai_gate_init,
)


MODALITY = "multimodal"
BASE_MEMBER_NAME = "M6_XAIGuided_Multimodal"


def _name_for_init(init: str) -> tuple[str, str]:
    """Distinct checkpoint and MLflow run names for the xai vs random variants."""
    if init == "xai":
        return BASE_MEMBER_NAME, "M6-multimodal-xai-guided-seed42"
    if init == "random":
        return f"{BASE_MEMBER_NAME}_random_init", "M6-multimodal-xai-guided-random-seed42"
    raise ValueError(f"unknown init='{init}'; must be 'xai' or 'random'")


def build_model(init: str) -> XAIGuidedMultimodalUNet3D:
    """Instantiate the M6 model with the chosen gate initialisation."""
    if init == "xai":
        importance_path = TABLES_DIR / "modality_importance_scores.json"
        if not importance_path.exists():
            raise FileNotFoundError(
                f"modality_importance_scores.json not found at {importance_path}. "
                "Run python member5_multimodal/ablation.py first."
            )
        gate_init = build_xai_gate_init(importance_path)
    else:
        gate_init = None        # falls back to uniform softmax (zeros)
    return XAIGuidedMultimodalUNet3D(
        in_channels=4, out_channels=3, gate_init_logits=gate_init,
    )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Train Member 6 — XAI-guided multi-task M5."
    )
    parser.add_argument(
        "--init", choices=("xai", "random"), default="xai",
        help="Gate initialisation. 'xai' uses ablation-derived per-sub-region "
             "per-modality weights (the headline config). 'random' uses "
             "uniform softmax (= zeros logits) — the ablation control "
             "that isolates the value of the XAI prior.",
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help=f"Total epochs (default: {NUM_EPOCHS}; matches M1-M5).",
    )
    args = parser.parse_args()

    member_name, mlflow_run_name = _name_for_init(args.init)
    num_epochs = int(args.epochs)
    print(f"[M6] init={args.init}  member_name={member_name}  epochs={num_epochs}")

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

    train_loader = get_dataloader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=6,
    )
    val_loader = get_dataloader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(args.init).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Log the initial gate values before any training step.
    init_gates = model.gate_probabilities().detach().cpu().numpy().round(4)
    print(f"[M6] initial gate probabilities (rows wt/tc/et, cols t1c/t1n/t2f/t2w):")
    for row, sr in zip(init_gates, ("wt", "tc", "et")):
        print(f"        {sr}: {row.tolist()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler("cuda") if (AMP_ENABLED and device.type == "cuda") else None

    stopper = EarlyStopper(patience=PATIENCE)
    ckpt = CheckpointManager(save_dir=str(CHECKPOINT_DIR), member_name=member_name)

    # MLflow tracking (lazy).
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
        mlflow.log_param("architecture", "XAIGuidedMultimodalUNet3D")
        mlflow.log_param("init",         args.init)
        mlflow.log_param("seed",         GLOBAL_SEED)
        mlflow.log_param("batch_size",   BATCH_SIZE)
        mlflow.log_param("lr",           LR)
        mlflow.log_param("weight_decay", WEIGHT_DECAY)
        mlflow.log_param("num_epochs",   num_epochs)
        mlflow.log_param("patch_size",   PATCH_SIZE)
        mlflow.log_param("amp",          AMP_ENABLED)
        # Log initial gates as metrics so we can plot drift vs epoch later.
        for i, sr in enumerate(("wt", "tc", "et")):
            for j, mod in enumerate(("t1c", "t1n", "t2f", "t2w")):
                mlflow.log_metric(f"gate_init_{sr}_{mod}", float(init_gates[i, j]))

    print("Dataloaders created")
    print("Starting epochs...")

    from collections import defaultdict
    history = defaultdict(list)

    best_dice = -1.0
    try:
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch} starting...")
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scaler, device, dice_bce_loss,
            )
            print(f"[epoch {epoch:03d}] train_loss={train_metrics['loss']:.4f}")
            history["train_loss"].append(train_metrics["loss"])
            if mlflow is not None:
                mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)

            scheduler.step()
            if mlflow is not None:
                mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

            # Validation on the BEST_METRIC cadence; the `or last` clause
            # guarantees a final validation even when epochs % VAL_EVERY != 0.
            if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == num_epochs:
                val_metrics = validate_one_epoch(
                    model, val_loader, device, dice_bce_loss,
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

                # Track gate drift each validation cycle.
                cur_gates = model.gate_probabilities().detach().cpu().numpy()
                if mlflow is not None:
                    for i, sr in enumerate(("wt", "tc", "et")):
                        for j, mod in enumerate(("t1c", "t1n", "t2f", "t2w")):
                            mlflow.log_metric(
                                f"gate_{sr}_{mod}", float(cur_gates[i, j]), step=epoch,
                            )
                print(f"[epoch {epoch:03d}] gates wt={cur_gates[0].round(3).tolist()}")
                print(f"[epoch {epoch:03d}] gates tc={cur_gates[1].round(3).tolist()}")
                print(f"[epoch {epoch:03d}] gates et={cur_gates[2].round(3).tolist()}")

                current_wt = val_metrics.get("dice_wt", -1.0)
                is_best = current_wt > best_dice
                if is_best:
                    best_dice = current_wt
                ckpt.save(
                    model, optimizer, epoch, val_metrics,
                    is_best=is_best, best_metric_key="dice_wt",
                )

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

        import json
        sidecar = CHECKPOINT_DIR / f"{member_name}_train_meta.json"
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        final_gates = model.gate_probabilities().detach().cpu().numpy().tolist()
        sidecar.write_text(json.dumps({
            "training_time_hrs": training_time_hrs,
            "gpu_memory_gb":     gpu_memory_gb,
            "init":              args.init,
            "final_gates": {
                "wt": final_gates[0],
                "tc": final_gates[1],
                "et": final_gates[2],
            },
        }, indent=2), encoding="utf-8")

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
