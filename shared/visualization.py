"""Matplotlib utilities for figures used in the report and slide deck.

All functions take a ``save_path`` argument; when provided the figure is
written to disk at 150 DPI and immediately closed so headless scripts do not
accumulate open figures.
"""
from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe; does not interfere with notebooks
import matplotlib.pyplot as plt
import numpy as np


def _best_axial_slice(seg: np.ndarray) -> int:
    """Return the axial index with the largest tumour area."""
    if seg.size == 0 or seg.sum() == 0:
        return seg.shape[2] // 2
    counts = (seg > 0).sum(axis=(0, 1))
    return int(np.argmax(counts))


def plot_patient_overview(
    vol_dict: dict,
    seg: np.ndarray,
    patient_id: str,
    save_path: Optional[str] = None,
) -> None:
    """2x3 figure: four MRI modalities + T1c-with-overlay + segmentation alone."""
    z = _best_axial_slice(seg)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    modality_order = ("t1c", "t1n", "t2f", "t2w")
    for ax, mod in zip(axes[:4], modality_order):
        vol = vol_dict.get(mod)
        if vol is None:
            ax.set_visible(False)
            continue
        ax.imshow(vol[:, :, z].T, cmap="gray", origin="lower")
        ax.set_title(f"{mod.upper()}")
        ax.axis("off")

    overlay_ax = axes[4]
    base = vol_dict.get("t1c")
    if base is not None:
        overlay_ax.imshow(base[:, :, z].T, cmap="gray", origin="lower")
        mask = np.ma.masked_where(seg[:, :, z].T == 0, seg[:, :, z].T)
        overlay_ax.imshow(mask, cmap="autumn", alpha=0.45, origin="lower")
    overlay_ax.set_title("T1c + segmentation")
    overlay_ax.axis("off")

    seg_ax = axes[5]
    seg_ax.imshow(seg[:, :, z].T, cmap="viridis", origin="lower")
    seg_ax.set_title("Segmentation labels")
    seg_ax.axis("off")

    fig.suptitle(f"{patient_id}  (axial slice {z})", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gradcam_overlay(
    vol_slice: np.ndarray,
    cam_slice: np.ndarray,
    seg_slice: np.ndarray,
    title: str,
    save_path: Optional[str] = None,
) -> None:
    """Three-panel comparison: raw MRI | Grad-CAM overlay | GT overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(vol_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("MRI")
    axes[0].axis("off")

    axes[1].imshow(vol_slice.T, cmap="gray", origin="lower")
    axes[1].imshow(cam_slice.T, cmap=plt.cm.hot, alpha=0.5, origin="lower")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(vol_slice.T, cmap="gray", origin="lower")
    gt = np.ma.masked_where(seg_slice.T == 0, seg_slice.T)
    axes[2].imshow(gt, cmap="autumn", alpha=0.5, origin="lower")
    axes[2].set_title("Ground truth")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    train_losses: list,
    val_dices: list,
    member_name: str,
    save_path: Optional[str] = None,
) -> None:
    """Two-panel figure: training loss + validation Dice across epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(range(1, len(train_losses) + 1), train_losses, label="train loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{member_name} — training loss")
    axes[0].grid(True, alpha=0.3)

    if val_dices:
        xs = range(1, len(val_dices) + 1)
        axes[1].plot(xs, val_dices, color="tab:green", label="val Dice")
        best_epoch = int(np.argmax(val_dices)) + 1
        axes[1].axvline(best_epoch, color="black", linestyle="--", alpha=0.6,
                        label=f"best @ {best_epoch}")
        axes[1].legend()
    axes[1].set_xlabel("validation step")
    axes[1].set_ylabel("Dice")
    axes[1].set_title(f"{member_name} — validation Dice")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
