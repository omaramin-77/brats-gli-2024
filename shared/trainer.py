"""Reusable training utilities: early stopping, checkpoint manager, loss, AMP loops.

Members copy ``train_one_epoch`` / ``validate_one_epoch`` if they need custom
behaviour, but the defaults here are what every pipeline starts from.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.inferers import sliding_window_inference
    _HAVE_SWI = True
except ImportError:  # pragma: no cover
    _HAVE_SWI = False

from shared.config import GRAD_CLIP_NORM, PATCH_SIZE
from shared.metrics import MetricTracker, compute_all_metrics


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopper:
    """Stop training once validation Dice has plateaued."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-3):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_dice: float = -float("inf")
        self.bad_checks: int = 0

    def should_stop(self, val_dice: float) -> bool:
        if val_dice > self.best_dice + self.min_delta:
            print(f"[early-stop] val Dice improved {self.best_dice:.4f} -> {val_dice:.4f}")
            self.best_dice = val_dice
            self.bad_checks = 0
            return False
        self.bad_checks += 1
        print(
            f"[early-stop] no improvement ({self.bad_checks}/{self.patience}) "
            f"best={self.best_dice:.4f} latest={val_dice:.4f}"
        )
        return self.bad_checks >= self.patience


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------
class CheckpointManager:
    """Save / restore checkpoints for a single member's experiment."""

    def __init__(self, save_dir: str, member_name: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.member_name = member_name

    def _path(self, suffix: str) -> Path:
        return self.save_dir / f"{self.member_name}_{suffix}.pt"

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_dice: float,
        is_best: bool = False,
    ) -> None:
        payload = {
            "epoch": int(epoch),
            "val_dice": float(val_dice),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        epoch_path = self._path(f"epoch{epoch:03d}")
        torch.save(payload, epoch_path)
        size_mb = os.path.getsize(epoch_path) / 1e6
        print(f"[ckpt] {epoch_path.name} saved ({size_mb:.1f} MB, dice={val_dice:.4f})")

        if is_best:
            best_path = self._path("best")
            torch.save(payload, best_path)
            print(f"[ckpt] new best -> {best_path.name}")

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> tuple[nn.Module, Optional[torch.optim.Optimizer], int, float]:
        path = self._path("best")
        if not path.exists():
            raise FileNotFoundError(
                f"No best checkpoint at {path}. "
                "Train for at least one validation step before evaluating."
            )
        payload = torch.load(path, map_location="cpu")
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        return model, optimizer, int(payload["epoch"]), float(payload["val_dice"])


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def dice_bce_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 1.0,
    bce_weight: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Combined soft-Dice + binary cross-entropy loss.

    Plain cross-entropy alone fails on BraTS because tumour voxels make up
    less than 1 % of the volume — the network can drive the BCE loss towards
    its minimum simply by predicting all-background. The soft-Dice term is
    invariant to that imbalance because it normalises by the union, so adding
    it puts pressure on the model to actually overlap the tumour mask.
    """
    target = target.float()
    if target.shape != pred_logits.shape:
        target = target.view_as(pred_logits)

    probs = torch.sigmoid(pred_logits)
    inter = (probs * target).sum()
    denom = probs.sum() + target.sum()
    dice_loss = 1.0 - (2.0 * inter + eps) / (denom + eps)

    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    return dice_weight * dice_loss + bce_weight * bce


# ---------------------------------------------------------------------------
# Train / validate epoch loops
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional["torch.cuda.amp.GradScaler"],
    device: torch.device,
    loss_fn: Callable = dice_bce_loss,
) -> dict:
    """Single training epoch with optional AMP and gradient clipping."""
    model.train()
    use_amp = scaler is not None and device.type == "cuda"

    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if labels.dim() == 4:
            labels = labels.unsqueeze(1)  # ensure (B, 1, H, W, D)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    mean_loss = total_loss / max(total_batches, 1)
    return {"loss": mean_loss}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    loss_fn: Callable = dice_bce_loss,
) -> dict:
    """Validation loop with sliding-window inference for full-volume metrics."""
    model.eval()
    tracker = MetricTracker()
    total_loss = 0.0
    total_batches = 0

    roi = PATCH_SIZE if isinstance(PATCH_SIZE, tuple) else (PATCH_SIZE,) * 3

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if labels.dim() == 4:
            labels = labels.unsqueeze(1)

        if _HAVE_SWI:
            logits = sliding_window_inference(
                inputs=images,
                roi_size=roi,
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
            )
        else:  # pragma: no cover
            logits = model(images)

        loss = loss_fn(logits, labels)
        metrics = compute_all_metrics(logits, labels)
        metrics["loss"] = float(loss.item())
        tracker.update(metrics)
        total_loss += float(loss.item())
        total_batches += 1

    agg = tracker.compute()
    agg.setdefault("loss", total_loss / max(total_batches, 1))
    return agg
