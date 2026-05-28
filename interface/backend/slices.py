"""Generate PNG slices for the MRI viewer.

The frontend can either (a) load the NIfTI directly with NiiVue, or (b) ask
this backend for pre-rendered PNGs (simpler, no WebGL needed). Both paths are
supported; these helpers cover path (b).

Views
-----
- axial    — slicing along axis 2 (Z), shown as (H, W)
- coronal  — slicing along axis 1 (Y), shown as (H, D)
- sagittal — slicing along axis 0 (X), shown as (W, D)

Overlay
-------
Each WT/TC/ET region renders in a distinct colour with alpha, layered ET on
top of TC on top of WT so the smallest region stays visible.
"""
from __future__ import annotations

import io
from typing import Literal, Optional

import numpy as np
from PIL import Image

View = Literal["axial", "coronal", "sagittal"]
Region = Literal["wt", "tc", "et", "all"]

# Per-region overlay colours (R, G, B). Chosen to match the standard BraTS
# colour scheme so the UI is familiar to anyone who has looked at the dataset.
_REGION_COLOR = {
    "wt": (50, 180, 220),   # cyan-blue — broadest region
    "tc": (255, 165, 0),    # orange    — middle
    "et": (220, 50, 60),    # crimson   — smallest, most clinical
}
_OVERLAY_ALPHA = 100  # 0-255 over the underlying greyscale


def take_slice(volume: np.ndarray, view: View, index: int) -> np.ndarray:
    """Slice a 3D volume by view and integer index. Returns a 2D array."""
    if volume.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape {volume.shape}")
    if view == "axial":
        idx = _clip(index, volume.shape[2])
        return volume[:, :, idx]
    if view == "coronal":
        idx = _clip(index, volume.shape[1])
        return volume[:, idx, :]
    if view == "sagittal":
        idx = _clip(index, volume.shape[0])
        return volume[idx, :, :]
    raise ValueError(f"unknown view '{view}'")


def _clip(idx: int, axis_len: int) -> int:
    return max(0, min(axis_len - 1, int(idx)))


def normalise_for_display(s: np.ndarray) -> np.ndarray:
    """Map a slice to uint8 [0, 255]. Robust to constant slices."""
    s = np.asarray(s, dtype=np.float32)
    lo = float(np.percentile(s, 1.0))
    hi = float(np.percentile(s, 99.0))
    if hi - lo < 1e-6:
        return np.zeros(s.shape, dtype=np.uint8)
    s = np.clip((s - lo) / (hi - lo), 0.0, 1.0)
    return (s * 255.0).astype(np.uint8)


def slice_to_png_bytes(
    image_slice: np.ndarray,
    overlay_3ch: Optional[np.ndarray] = None,
    region: Region = "all",
    alpha: int = _OVERLAY_ALPHA,
) -> bytes:
    """Render a slice (optionally with a WT/TC/ET overlay) to PNG bytes.

    Parameters
    ----------
    image_slice : 2D ndarray
        The underlying greyscale MRI slice (any range; normalised internally).
    overlay_3ch : Optional[ndarray]
        Shape ``(3, H, W)`` binary masks for WT, TC, ET on the same slice.
    region : "wt" | "tc" | "et" | "all"
        Which region(s) to overlay. "all" stacks ET on TC on WT.
    """
    grey = normalise_for_display(image_slice)
    # Rotate 90deg so anterior/posterior reads naturally in the browser
    # (NIfTI convention is array-axis-first; humans expect anatomical-up).
    grey = np.rot90(grey)
    rgb = np.stack([grey, grey, grey], axis=-1)  # (H, W, 3)

    if overlay_3ch is not None and region != "none":
        if overlay_3ch.shape[0] != 3:
            raise ValueError(
                f"overlay_3ch must be (3, H, W); got {overlay_3ch.shape}"
            )
        # Same rotation as the underlying image so they line up pixel-perfect.
        rotated = np.stack([np.rot90(overlay_3ch[i]) for i in range(3)], axis=0)

        # Painter's order: WT first (broadest), then TC, then ET on top.
        order = ("wt", "tc", "et") if region == "all" else (region,)
        for name in order:
            ch_idx = {"wt": 0, "tc": 1, "et": 2}[name]
            color = np.array(_REGION_COLOR[name], dtype=np.uint8)
            mask = rotated[ch_idx].astype(bool)
            if not mask.any():
                continue
            # Alpha-blend: out = (1 - a/255) * grey + (a/255) * colour
            a = alpha / 255.0
            for c in range(3):
                rgb[..., c] = np.where(
                    mask,
                    (rgb[..., c].astype(np.float32) * (1 - a) + color[c] * a).astype(np.uint8),
                    rgb[..., c],
                )

    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
