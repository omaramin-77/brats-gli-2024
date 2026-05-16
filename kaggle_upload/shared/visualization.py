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
    history: dict,
    member_name: str,
    save_path: Optional[str] = None,
) -> None:
    """Two-panel figure: training loss + per-subregion validation Dice.

    ``history`` is a dict-of-lists keyed by metric name, e.g.
    ``{"train_loss": [...], "val_dice_wt": [...], "val_dice_tc": [...],
       "val_dice_et": [...]}``. Missing keys are skipped silently.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tl = history.get("train_loss", [])
    if tl:
        axes[0].plot(range(1, len(tl) + 1), tl, label="train_loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].set_title(f"{member_name} — training loss")
        axes[0].grid(True, alpha=0.3)

    for key, colour in (("val_dice_wt", "tab:blue"),
                        ("val_dice_tc", "tab:orange"),
                        ("val_dice_et", "tab:red")):
        vals = history.get(key, [])
        if vals:
            axes[1].plot(range(1, len(vals) + 1), vals, label=key, color=colour)
    best_wt = history.get("val_dice_wt", [])
    if best_wt:
        be = int(np.argmax(best_wt)) + 1
        axes[1].axvline(be, color="black", linestyle="--", alpha=0.5,
                        label=f"best_wt@{be}")
    axes[1].set_xlabel("validation step")
    axes[1].set_ylabel("Dice")
    axes[1].set_title(f"{member_name} — per-subregion Dice")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gradcam_three_view(
    volume: np.ndarray,
    heatmap: np.ndarray,
    seg: np.ndarray,
    patient_id: str,
    subregion: str,
    save_path_prefix: Optional[str] = None,
) -> None:
    """Save axial / coronal / sagittal Grad-CAM overlays for one
    (patient, sub-region) pair. Three PNGs total."""
    views = {"axial": 2, "coronal": 1, "sagittal": 0}
    for view_name, axis in views.items():
        if seg.sum() > 0:
            other = tuple(a for a in (0, 1, 2) if a != axis)
            idx = int(np.argmax((seg > 0).sum(axis=other)))
        else:
            idx = seg.shape[axis] // 2
        if axis == 0:
            vs, cs, ss = volume[idx], heatmap[idx], seg[idx]
        elif axis == 1:
            vs, cs, ss = volume[:, idx], heatmap[:, idx], seg[:, idx]
        else:
            vs, cs, ss = volume[:, :, idx], heatmap[:, :, idx], seg[:, :, idx]
        title = f"{patient_id} — {subregion} — {view_name} (slice {idx})"
        save = f"{save_path_prefix}_{view_name}.png" if save_path_prefix else None
        plot_gradcam_overlay(vs, cs, ss, title=title, save_path=save)


def plot_subregion_overlay(
    vol_slice: np.ndarray,
    target_3ch_slice: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Show MRI slice with WT (blue), TC (yellow), ET (red) overlays."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(vol_slice.T, cmap="gray", origin="lower")
    colors = [(0.2, 0.5, 1.0), (1.0, 0.85, 0.0), (1.0, 0.2, 0.2)]
    for ch_idx, col in enumerate(colors):
        mask = target_3ch_slice[ch_idx].T
        if mask.sum() == 0:
            continue
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask > 0] = (*col, 0.35)
        ax.imshow(overlay, origin="lower")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gradcam_per_subregion(
    volume: np.ndarray,
    cams: dict,
    seg_3ch: np.ndarray,
    patient_id: str,
    save_path: Optional[str] = None,
) -> None:
    """3-row x 3-col figure: rows = WT/TC/ET, cols = MRI/CAM/GT.

    ``cams`` is ``{"wt": cam_array, "tc": ..., "et": ...}``;
    ``seg_3ch`` is shape ``(3, H, W, D)``.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    subregions = ("wt", "tc", "et")
    for i, sr in enumerate(subregions):
        seg = seg_3ch[i]
        if seg.sum() > 0:
            z = int(np.argmax((seg > 0).sum(axis=(0, 1))))
        else:
            z = volume.shape[2] // 2
        axes[i, 0].imshow(volume[:, :, z].T, cmap="gray", origin="lower")
        axes[i, 0].set_title(f"MRI ({sr})")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(volume[:, :, z].T, cmap="gray", origin="lower")
        axes[i, 1].imshow(cams[sr][:, :, z].T, cmap="hot", alpha=0.5, origin="lower")
        axes[i, 1].set_title(f"Grad-CAM ({sr})")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(volume[:, :, z].T, cmap="gray", origin="lower")
        gt = np.ma.masked_where(seg[:, :, z].T == 0, seg[:, :, z].T)
        axes[i, 2].imshow(gt, cmap="autumn", alpha=0.5, origin="lower")
        axes[i, 2].set_title(f"GT ({sr})")
        axes[i, 2].axis("off")
    fig.suptitle(f"{patient_id} — per-subregion Grad-CAM", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_modality_importance_bar(
    importance_per_subregion: dict,
    save_path: Optional[str] = None,
) -> None:
    """3 panels (WT/TC/ET), each a 4-bar chart of modality importance.

    ``importance_per_subregion`` is
    ``{"wt": {"t1c": 0.4, "t1n": ..., "t2f": ..., "t2w": ...}, "tc": {...}, "et": {...}}``.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, sr in zip(axes, ("wt", "tc", "et")):
        imp = importance_per_subregion.get(sr, {})
        if not imp:
            ax.set_title(f"{sr.upper()} — no data")
            ax.axis("off")
            continue
        mods = list(imp.keys())
        vals = [imp[m] for m in mods]
        ax.bar(mods, vals)
        ax.set_title(f"Importance ({sr.upper()})")
        ax.set_ylabel("Dice drop when ablated")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
