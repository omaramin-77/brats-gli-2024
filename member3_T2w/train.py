# member3_T2w/train.py
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed
set_global_seed()

import torch

from shared.config import (
    AMP_ENABLED,
    BEST_METRIC,
    CHECKPOINT_DIR,
    FIGURES_DIR,
    GLOBAL_SEED,
    LR,
    NUM_EPOCHS,
    NUM_WORKERS,
    PATCH_SIZE,
    PATIENCE,
    SPLITS_DIR,
    VAL_EVERY_N_EPOCHS,
    WEIGHT_DECAY,
    get_data_root,
)
from shared.dataset import BraTSDataset, get_dataloader
from shared.trainer import CheckpointManager, EarlyStopper, dice_bce_loss, validate_one_epoch
from shared.visualization import plot_training_curves

from model import build_model


MEMBER_NAME = "M3_T2w_SwinUNETR"
MODALITY = "t2w"
ARCHITECTURE = "SwinUNETR"

LOCAL_PATCH_SIZE = PATCH_SIZE
LOCAL_BATCH_SIZE = 1
FEATURE_SIZE = 24
LOCAL_LR = LR


def progress_bar(percent: float, width: int = 30) -> str:
    done = int(width * percent / 100)
    left = width - done
    return "[" + "=" * done + ">" + "." * left + "]"


def format_time(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def train_one_epoch_with_progress(
    model,
    loader,
    optimizer,
    scaler,
    device,
    loss_fn,
    epoch: int,
    total_epochs: int,
):
    model.train()
    use_amp = scaler is not None and device.type == "cuda"

    total_loss = 0.0
    n_batches = len(loader)
    start = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.item())

        percent = 100.0 * batch_idx / max(n_batches, 1)
        elapsed = time.time() - start
        avg_time = elapsed / batch_idx
        eta = avg_time * (n_batches - batch_idx)

        print(
            f"\rEpoch {epoch:03d}/{total_epochs:03d} "
            f"{progress_bar(percent)} {percent:6.2f}% "
            f"| Batch {batch_idx}/{n_batches} "
            f"| Loss {loss.item():.4f} "
            f"| Elapsed {format_time(elapsed)} "
            f"| ETA {format_time(eta)}",
            end="",
            flush=True,
        )

    print()
    return {"loss": total_loss / max(n_batches, 1)}


def main() -> None:
    run_start = time.time()

    data_root = get_data_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"Member       : {MEMBER_NAME}")
    print(f"Modality     : {MODALITY}")
    print(f"Architecture : {ARCHITECTURE}")
    print(f"Device       : {device}")
    print(f"Patch size   : {LOCAL_PATCH_SIZE}")
    print(f"Batch size   : {LOCAL_BATCH_SIZE}")
    print(f"Feature size : {FEATURE_SIZE}")
    print(f"Learning rate: {LOCAL_LR}")
    print(f"Epochs       : {NUM_EPOCHS}")
    print("=" * 80)

    train_ds = BraTSDataset(
        data_root=data_root,
        split_file=str(SPLITS_DIR / "train_ids.txt"),
        modality=MODALITY,
        patch_size=LOCAL_PATCH_SIZE[0],
        augment=True,
        patches_per_volume=4,
        full_volume=False,
    )

    val_ds = BraTSDataset(
        data_root=data_root,
        split_file=str(SPLITS_DIR / "val_ids.txt"),
        modality=MODALITY,
        patch_size=LOCAL_PATCH_SIZE[0],
        augment=False,
        patches_per_volume=1,
        full_volume=True,
    )

    train_loader = get_dataloader(
        train_ds,
        batch_size=LOCAL_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = get_dataloader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Train dataset samples : {len(train_ds)}")
    print(f"Train batches/epoch   : {len(train_loader)}")
    print(f"Validation patients   : {len(val_ds)}")
    print("=" * 80)

    model = build_model(
        img_size=LOCAL_PATCH_SIZE,
        feature_size=FEATURE_SIZE,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters  : {trainable_params:,}")
    print("=" * 80)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LOCAL_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    scaler = torch.cuda.amp.GradScaler() if AMP_ENABLED and device.type == "cuda" else None

    stopper = EarlyStopper(patience=PATIENCE)
    ckpt = CheckpointManager(save_dir=str(CHECKPOINT_DIR), member_name=MEMBER_NAME)

    history = defaultdict(list)
    best_dice_wt = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_one_epoch_with_progress(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            loss_fn=dice_bce_loss,
            epoch=epoch,
            total_epochs=NUM_EPOCHS,
        )

        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])

        print(f"[epoch {epoch:03d}] train_loss={train_metrics['loss']:.4f}")

        if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS:
            print(f"[epoch {epoch:03d}] validation started...")

            val_metrics = validate_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                loss_fn=dice_bce_loss,
            )

            for key, value in val_metrics.items():
                history[f"val_{key}"].append(value)

            val_loss = val_metrics.get("loss", float("nan"))
            dice_wt = val_metrics.get("dice_wt", float("nan"))
            dice_tc = val_metrics.get("dice_tc", float("nan"))
            dice_et = val_metrics.get("dice_et", float("nan"))
            iou_wt = val_metrics.get("iou_wt", float("nan"))
            hd95_wt = val_metrics.get("hd95_wt", float("nan"))

            print(
                f"[epoch {epoch:03d}] "
                f"val_loss={val_loss:.4f} | "
                f"dice_wt={dice_wt:.4f} | "
                f"dice_tc={dice_tc:.4f} | "
                f"dice_et={dice_et:.4f} | "
                f"iou_wt={iou_wt:.4f} | "
                f"hd95_wt={hd95_wt:.2f}"
            )

            is_best = dice_wt > best_dice_wt
            if is_best:
                best_dice_wt = dice_wt

            ckpt.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_metrics=val_metrics,
                is_best=is_best,
                best_metric_key=BEST_METRIC,
            )

            if stopper.should_stop(val_metrics):
                print(f"[epoch {epoch:03d}] early stopping triggered")
                break

    training_time_hrs = (time.time() - run_start) / 3600.0

    gpu_memory_gb = 0.0
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = CHECKPOINT_DIR / f"{MEMBER_NAME}_train_meta.json"

    meta = {
        "member": MEMBER_NAME,
        "modality": MODALITY,
        "architecture": ARCHITECTURE,
        "seed": GLOBAL_SEED,
        "patch_size": list(LOCAL_PATCH_SIZE),
        "batch_size": LOCAL_BATCH_SIZE,
        "feature_size": FEATURE_SIZE,
        "learning_rate": LOCAL_LR,
        "num_epochs": NUM_EPOCHS,
        "best_metric": BEST_METRIC,
        "best_dice_wt": best_dice_wt,
        "training_time_hrs": training_time_hrs,
        "gpu_memory_gb": gpu_memory_gb,
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_curves(
        dict(history),
        MEMBER_NAME,
        save_path=str(FIGURES_DIR / f"training_curves_{MEMBER_NAME}.png"),
    )

    print("=" * 80)
    print("Training complete")
    print(f"Best val dice_wt : {best_dice_wt:.4f}")
    print(f"Training hours   : {training_time_hrs:.2f}")
    print(f"GPU memory GB    : {gpu_memory_gb:.2f}")
    print(f"Meta saved       : {meta_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()