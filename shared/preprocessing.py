"""Preprocessing primitives shared across all five member pipelines.

Design rules
------------
* Per-patient z-score normalisation over brain voxels only. Using global
  dataset statistics would leak test information into training and would
  fail when a scanner's intensity scale shifts.
* The brain bounding box crop removes air/skull background so the network
  spends parameters on tissue rather than zeros.
* Segmentation masks are resized with nearest-neighbour (order=0) — bilinear
  interpolation would invent fractional label values that do not exist.
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import nibabel as nib
import numpy as np
from monai import transforms as mt
from scipy.ndimage import zoom


# ---------------------------------------------------------------------------
# Volume loading and intensity normalisation
# ---------------------------------------------------------------------------
def load_and_normalise(path: str) -> np.ndarray:
    """Load a NIfTI volume and z-score normalise within the brain mask only.

    BraTS volumes are skull-stripped, so background voxels are exactly zero.
    Normalising over the whole volume would let the large background block
    dominate the mean/std and squash the brain intensity range. Using only
    ``vol > 0`` voxels keeps the contrast inside the brain meaningful.
    """
    vol = nib.load(path).get_fdata().astype(np.float32)

    background = vol == 0
    brain = vol[~background]

    if brain.size == 0:
        # Pathological case: empty volume. Return zeros to avoid NaNs.
        return np.zeros_like(vol, dtype=np.float32)

    mean = float(brain.mean())
    std = float(brain.std())

    # epsilon prevents divide-by-zero on flat volumes.
    vol = (vol - mean) / (std + 1e-8)

    # Restore the original background to exactly zero — otherwise the shift
    # by -mean/std turns it into a constant negative value and the network
    # would waste capacity learning to ignore it.
    vol[background] = 0.0
    return vol


# ---------------------------------------------------------------------------
# Brain bounding-box crop
# ---------------------------------------------------------------------------
def crop_to_brain(
    vol: np.ndarray,
    seg: Optional[np.ndarray],
    pad: int = 8,
) -> tuple[np.ndarray, Optional[np.ndarray], tuple]:
    """Crop ``vol`` (and ``seg`` if given) to a tight brain bounding box + pad.

    Returns the cropped volume, the cropped segmentation (or None), and the
    slice tuple used so the caller can reuse the same crop for sibling
    modalities (multimodal preprocessing relies on this).
    """
    coords = np.argwhere(vol > 0)
    if coords.size == 0:
        slc = tuple(slice(0, s) for s in vol.shape)
        return vol, seg, slc

    mins = coords.min(axis=0) - pad
    maxs = coords.max(axis=0) + pad + 1  # +1 so the inclusive max becomes a Python slice end

    # Bug-fix: clip per-axis against the per-axis size, not a single scalar.
    # np.clip(mins, 0, vol.shape[0]) would mis-bound on non-cubic volumes.
    shape = np.array(vol.shape)
    mins = np.clip(mins, 0, shape - 1)
    maxs = np.clip(maxs, 1, shape)

    slc = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))
    cropped_vol = vol[slc]
    cropped_seg = seg[slc] if seg is not None else None
    return cropped_vol, cropped_seg, slc


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------
def resize_volume(
    vol: np.ndarray,
    target: tuple = (128, 128, 128),
    order: int = 1,
) -> np.ndarray:
    """Resize a 3D volume to ``target`` shape via scipy.ndimage.zoom.

    ``order=1`` (trilinear) is correct for image volumes; ``order=0``
    (nearest neighbour) is mandatory for segmentation masks because integer
    label values must be preserved exactly.
    """
    factors = [t / s for t, s in zip(target, vol.shape)]
    out = zoom(vol, factors, order=order)
    # zoom can produce shapes off-by-one due to floating-point rounding; trim
    # or pad so the caller's downstream tensors line up.
    if out.shape != tuple(target):
        out = _force_shape(out, target)
    return out


def _force_shape(vol: np.ndarray, target: tuple) -> np.ndarray:
    """Crop or zero-pad ``vol`` so its shape exactly matches ``target``."""
    out = vol
    pad_widths = []
    crop_slices = []
    for axis_size, target_size in zip(out.shape, target):
        if axis_size >= target_size:
            start = (axis_size - target_size) // 2
            crop_slices.append(slice(start, start + target_size))
            pad_widths.append((0, 0))
        else:
            crop_slices.append(slice(0, axis_size))
            diff = target_size - axis_size
            pad_widths.append((diff // 2, diff - diff // 2))
    out = out[tuple(crop_slices)]
    if any(p != (0, 0) for p in pad_widths):
        out = np.pad(out, pad_widths, mode="constant", constant_values=0)
    return out


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------
def extract_patch(
    vol: np.ndarray,
    seg: np.ndarray,
    size: int = 64,
    tumour_bias: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a ``size``-cubed patch from ``vol`` (C, H, W, D) and ``seg`` (H, W, D).

    With probability ``tumour_bias`` the centre is drawn from a tumour voxel
    so the network sees positive examples — random sampling alone would yield
    almost entirely background patches because tumours occupy <1 % of the
    volume on average.
    """
    if vol.ndim != 4:
        raise ValueError(f"vol must be 4D (C,H,W,D); got shape {vol.shape}")
    if seg.shape != vol.shape[1:]:
        raise ValueError(
            f"seg shape {seg.shape} must match vol spatial shape {vol.shape[1:]}"
        )

    spatial = np.array(vol.shape[1:])
    half = size // 2

    use_tumour = (np.random.rand() < tumour_bias) and (seg.sum() > 0)
    if use_tumour:
        tumour_voxels = np.argwhere(seg > 0)
        centre = tumour_voxels[np.random.randint(len(tumour_voxels))]
    else:
        centre = np.array([
            np.random.randint(half, max(half + 1, s - half))
            for s in spatial
        ])

    # Clamp the centre so the patch stays inside the volume.
    centre = np.clip(centre, half, spatial - (size - half))

    starts = centre - half
    ends = starts + size

    s = tuple(slice(int(a), int(b)) for a, b in zip(starts, ends))
    patch_vol = vol[(slice(None),) + s]
    patch_seg = seg[s]
    return patch_vol, patch_seg


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def get_augmentation_transforms() -> mt.Compose:
    """Patch-level augmentations applied during training only.

    Spatial flips and 90° rotations are safe for medical volumes (they do not
    change tissue identity, unlike arbitrary affine warps which can introduce
    interpolation artefacts that confuse 3D segmentation models).
    """
    return mt.Compose([
        mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        mt.RandRotate90d(keys=["image", "label"], prob=0.5),
        mt.RandGaussianNoised(keys=["image"], prob=0.3, std=0.05),
        mt.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
    ])


# ---------------------------------------------------------------------------
# Per-patient pipelines
# ---------------------------------------------------------------------------
_MODALITY_GLOB = {
    "t1c": "*t1c*.nii*",
    "t1n": "*t1n*.nii*",
    "t2f": "*t2f*.nii*",
    "t2w": "*t2w*.nii*",
}


def _find_one(pattern: str, patient_dir: str) -> str:
    matches = sorted(glob.glob(os.path.join(patient_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {patient_dir}")
    return matches[0]


def preprocess_patient(
    patient_dir: str,
    modality: str,
    target_size: tuple = (128, 128, 128),
) -> tuple[np.ndarray, np.ndarray]:
    """Full single-modality preprocessing pipeline for one patient.

    Returns a (1, H, W, D) volume and a (H, W, D) segmentation. The leading
    channel axis lets the dataset stack patches with no further reshaping.
    """
    if modality not in _MODALITY_GLOB:
        raise ValueError(f"Unknown modality: {modality}")

    mod_path = _find_one(_MODALITY_GLOB[modality], patient_dir)
    seg_path = _find_one("*seg*.nii*", patient_dir)

    vol = load_and_normalise(mod_path)
    seg = nib.load(seg_path).get_fdata().astype(np.int16)

    vol_c, seg_c, _ = crop_to_brain(vol, seg)
    vol_r = resize_volume(vol_c, target_size, order=1)
    seg_r = resize_volume(seg_c, target_size, order=0)

    return vol_r[np.newaxis].astype(np.float32), seg_r.astype(np.int16)


def preprocess_patient_multimodal(
    patient_dir: str,
    target_size: tuple = (128, 128, 128),
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess all four modalities together with a shared brain bbox.

    The crop is computed from T1c because the contrast-enhanced sequence
    typically delineates the brain volume most reliably. Reusing the same
    slices across modalities keeps voxel correspondence intact.
    """
    paths = {m: _find_one(_MODALITY_GLOB[m], patient_dir) for m in ("t1c", "t1n", "t2f", "t2w")}
    seg_path = _find_one("*seg*.nii*", patient_dir)

    vols = {m: load_and_normalise(p) for m, p in paths.items()}
    seg = nib.load(seg_path).get_fdata().astype(np.int16)

    # Crop slices come from T1c; reapply to every other modality + seg.
    _, seg_c, slc = crop_to_brain(vols["t1c"], seg)
    cropped = {m: vols[m][slc] for m in vols}

    resized = {m: resize_volume(cropped[m], target_size, order=1) for m in cropped}
    seg_r = resize_volume(seg_c, target_size, order=0)

    stacked = np.stack(
        [resized["t1c"], resized["t1n"], resized["t2f"], resized["t2w"]],
        axis=0,
    ).astype(np.float32)
    return stacked, seg_r.astype(np.int16)
