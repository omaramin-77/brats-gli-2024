"""Segmentation metrics: Dice, IoU, Hausdorff-95, and an epoch tracker.

All metrics are computed on the binary whole-tumour task by default. Per-class
metrics are member-specific and live alongside their evaluate.py.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch

try:
    from monai.metrics import HausdorffDistanceMetric
    _HD_METRIC = HausdorffDistanceMetric(percentile=95, include_background=False, reduction="mean")
except Exception:  # pragma: no cover
    _HD_METRIC = None


def _to_binary_tensor(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).to(torch.bool)


def compute_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Soft-binary Dice coefficient. Operates on already-thresholded inputs."""
    p = _to_binary_tensor(pred)
    t = _to_binary_tensor(target)
    inter = (p & t).sum().item()
    denom = p.sum().item() + t.sum().item()
    return (2.0 * inter + eps) / (denom + eps)


def compute_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Intersection-over-Union (Jaccard index)."""
    p = _to_binary_tensor(pred)
    t = _to_binary_tensor(target)
    inter = (p & t).sum().item()
    union = (p | t).sum().item()
    return (inter + eps) / (union + eps)


def compute_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """95th-percentile Hausdorff distance via MONAI.

    Returns NaN (with a one-time warning) when either mask is empty — HD is
    geometrically undefined in that case and the standard practice is to
    exclude such cases from the mean.
    """
    if _HD_METRIC is None:
        warnings.warn("MONAI HausdorffDistanceMetric unavailable; returning NaN.")
        return float("nan")

    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)
    if not p.any() or not t.any():
        warnings.warn(
            "compute_hd95: empty prediction or target — returning NaN.",
            stacklevel=2,
        )
        return float("nan")

    # MONAI expects (B, C, H, W, D) with values in {0, 1}.
    p_t = torch.from_numpy(p.astype(np.uint8))[None, None].float()
    t_t = torch.from_numpy(t.astype(np.uint8))[None, None].float()
    _HD_METRIC.reset()
    _HD_METRIC(y_pred=p_t, y=t_t)
    val = _HD_METRIC.aggregate().item()
    _HD_METRIC.reset()
    return float(val)


def compute_all_metrics(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    channel_names: tuple = ("wt", "tc", "et"),
    threshold: float = 0.5,
) -> dict:
    """Sigmoid + threshold pred_logits, then compute per-channel
    Dice / IoU / HD95.

    Accepts (B, C, H, W, D) or (C, H, W, D). Returns a dict with
    keys dice_{name}, iou_{name}, hd95_{name} for each channel name.
    """
    if pred_logits.ndim == 4:
        pred_logits = pred_logits.unsqueeze(0)
    if target.ndim == 4:
        target = target.unsqueeze(0)

    probs = torch.sigmoid(pred_logits)
    pred_bin = (probs >= threshold).to(torch.bool)
    target_bin = (target > 0).to(torch.bool)

    out: dict = {}
    for c, name in enumerate(channel_names):
        p_c = pred_bin[:, c:c+1]
        t_c = target_bin[:, c:c+1]

        out[f"dice_{name}"] = compute_dice(p_c, t_c)
        out[f"iou_{name}"] = compute_iou(p_c, t_c)

        hd_vals = []
        for b in range(p_c.shape[0]):
            p_np = p_c[b, 0].detach().cpu().numpy()
            t_np = t_c[b, 0].detach().cpu().numpy()
            hd_vals.append(compute_hd95(p_np, t_np))
        out[f"hd95_{name}"] = float(np.nanmean(hd_vals)) if hd_vals else float("nan")

    return out


class MetricTracker:
    """Accumulate batch-level metrics and aggregate at the end of an epoch."""

    def __init__(self) -> None:
        self._buf: dict[str, list[float]] = defaultdict(list)

    def update(self, metrics: dict) -> None:
        for k, v in metrics.items():
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isnan(fv):
                # Skip NaNs (e.g. HD95 when masks are empty) so the mean
                # is not poisoned by undefined values.
                continue
            self._buf[k].append(fv)

    def compute(self) -> dict:
        return {k: float(np.mean(v)) if v else float("nan") for k, v in self._buf.items()}

    def reset(self) -> None:
        self._buf.clear()
