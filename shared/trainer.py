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

    def should_stop(self, val_metric) -> bool:
        # Accept dict (preferred — pulls BEST_METRIC) or float (legacy).
        if isinstance(val_metric, dict):
            from shared.config import BEST_METRIC
            val_dice = float(val_metric.get(BEST_METRIC, -1.0))
        else:
            val_dice = float(val_metric)
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
        val_metrics,                # float or dict
        is_best: bool = False,
        best_metric_key: str = "dice_wt",
    ) -> None:
        if isinstance(val_metrics, dict):
            val_dice = float(val_metrics.get(best_metric_key, 0.0))
            metric_payload = dict(val_metrics)
        else:
            val_dice = float(val_metrics)
            metric_payload = {"val_dice": val_dice}
        payload = {
            "epoch": int(epoch),
            "val_dice": val_dice,
            "val_metrics": metric_payload,
            "best_metric_key": best_metric_key,
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
    channel_weights = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-channel soft-Dice + BCE. Reduces over spatial dims only,
    then averages over batch and channel. Backward-compatible with
    single-channel input."""
    target = target.float()
    if target.shape != pred_logits.shape:
        target = target.view_as(pred_logits)

    probs = torch.sigmoid(pred_logits)
    # Spatial dims only — assume (B, C, ...spatial).
    dims = tuple(range(2, pred_logits.ndim))
    inter = (probs * target).sum(dim=dims)          # (B, C)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice_per_ch = 1.0 - (2.0 * inter + eps) / (denom + eps)   # (B, C)

    if channel_weights is not None:
        w = torch.as_tensor(channel_weights, device=pred_logits.device,
                            dtype=dice_per_ch.dtype)
        dice_loss = (dice_per_ch * w).sum(dim=1).mean() / w.sum()
    else:
        dice_loss = dice_per_ch.mean()

    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    return dice_weight * dice_loss + bce_weight * bce


def focal_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Per-channel focal loss for severely imbalanced binary tasks."""
    target = target.float()
    if target.shape != pred_logits.shape:
        target = target.view_as(pred_logits)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    probs = torch.sigmoid(pred_logits)
    p_t = probs * target + (1 - probs) * (1 - target)
    focal_w = (1 - p_t) ** gamma
    alpha_w = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_w * focal_w * bce).mean()


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
            with torch.amp.autocast(device_type="cuda"):
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
