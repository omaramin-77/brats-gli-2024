"""Inference pipeline — preprocess uploads, run sliding-window inference,
compute metrics. Wraps ``shared.preprocessing`` and ``shared.metrics`` so the
numerics here are byte-identical to ``shared.trainer.validate_one_epoch``.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from monai.inferers import sliding_window_inference

from shared.config import MODALITIES, PATCH_SIZE, TARGET_CHANNEL_NAMES
from shared.metrics import compute_all_metrics
from shared.preprocessing import (
    preprocess_patient,
    preprocess_patient_multimodal,
)

from interface.backend.config import DEVICE
from interface.backend.registry import ModelEntry, registry
from interface.backend.sessions import sessions


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_session(sid: str, entry: ModelEntry) -> tuple[np.ndarray, np.ndarray]:
    """Run the shared preprocessing pipeline for ``entry``'s input shape.

    Returns
    -------
    image : np.ndarray
        ``(C, 128, 128, 128)`` — C=1 for unimodal, C=4 for multimodal.
    seg : np.ndarray
        ``(3, 128, 128, 128)`` int8 WT/TC/ET ground-truth mask.

    Raises
    ------
    FileNotFoundError
        If a required modality (for the model) or the segmentation is missing.
    """
    patient_dir = sessions.upload_dir(sid)
    if entry.in_channels == 1:
        vol, seg = preprocess_patient(str(patient_dir), entry.modality)
        # vol: (1, 128, 128, 128), seg: (128, 128, 128) raw labels
    else:
        # Multimodal stack uses the shared (t1c, t1n, t2f, t2w) order.
        vol, seg = preprocess_patient_multimodal(str(patient_dir))

    # Build the 3-channel WT/TC/ET ground-truth target — same construction as
    # BraTSDataset._multichannel_label. Inlined here because the dataset class
    # only exposes it via __getitem__ which we don't want to pay for.
    wt = (seg == 1) | (seg == 2) | (seg == 3)
    tc = (seg == 1) | (seg == 3)
    et = (seg == 3)
    target_3ch = np.stack([wt, tc, et], axis=0).astype(np.int8)
    return vol.astype(np.float32), target_3ch


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference_for(model_key: str, sid: str) -> dict:
    """Run a single model on a session's uploads. Cache result on disk.

    Returns a dict with:
      - dice_wt / dice_tc / dice_et
      - iou_wt / iou_tc / iou_et
      - hd95_wt / hd95_tc / hd95_et
      - inference_time_s
      - model_key, model_display_name
      - volume_mm3_wt / volume_mm3_tc / volume_mm3_et (predicted tumour volumes)
    """
    entry = registry.get(model_key)
    if not entry.enabled:
        raise RuntimeError(
            f"Model '{model_key}' not available: {entry.load_error}"
        )

    image_np, target_3ch = preprocess_session(sid, entry)

    model = registry.load(model_key)

    image_t = torch.from_numpy(image_np).unsqueeze(0).to(DEVICE).float()      # (1, C, 128, 128, 128)
    target_t = torch.from_numpy(target_3ch).unsqueeze(0).to(DEVICE).float()    # (1, 3, 128, 128, 128)

    roi = PATCH_SIZE if isinstance(PATCH_SIZE, tuple) else (PATCH_SIZE,) * 3

    t0 = time.perf_counter()
    logits = sliding_window_inference(
        inputs=image_t,
        roi_size=roi,
        sw_batch_size=2,
        predictor=model,
        overlap=0.5,
    )
    # Some training-mode models return tuples (M1 deep supervision); eval-mode
    # is single-tensor, but be defensive in case any wrapper slips through.
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    elapsed = time.perf_counter() - t0

    # Per the design (shared/metrics.compute_all_metrics): sigmoid + threshold
    # 0.5 → independent binary masks per channel.
    metrics = compute_all_metrics(logits, target_t)
    metrics["inference_time_s"] = float(elapsed)

    probs = torch.sigmoid(logits)[0].cpu().numpy().astype(np.float32)      # (3, H, W, D)
    binary = (probs >= 0.5).astype(np.uint8)

    # Tumour volume: voxel count × voxel volume (mm^3). We approximate voxel
    # volume from the original NIfTI affine. After resize to 128**3 the voxels
    # are no longer 1mm isotropic, so we report volume in the *original* space
    # by scaling back; this gives a clinically interpretable number.
    affine = sessions.get_affine(sid)
    voxel_mm3 = _approx_voxel_volume_mm3(affine, image_np.shape[1:])
    for i, name in enumerate(TARGET_CHANNEL_NAMES):
        metrics[f"volume_mm3_{name}"] = float(int(binary[i].sum()) * voxel_mm3)

    metrics["model_key"] = entry.key
    metrics["model_display_name"] = entry.display_name
    metrics["model_short_name"] = entry.short_name
    metrics["architecture"] = entry.architecture
    metrics["modality"] = entry.modality

    sessions.store_prediction(
        sid=sid,
        model_key=model_key,
        binary_mask=binary,
        probs=probs,
        metrics=metrics,
        affine=affine,
    )
    return metrics


def _approx_voxel_volume_mm3(
    affine: Optional[np.ndarray],
    resized_shape: tuple[int, ...],
) -> float:
    """Approximate per-voxel volume in the resized 128**3 space.

    The original NIfTI affine encodes voxel spacing in the source volume. We
    take the geometric mean of the three column norms (which is the voxel
    volume in mm**3) and scale by the shape ratio. This is good enough for
    a tumour-volume readout in the UI — clinical reporting would use the
    original-resolution mask, not the resized one.
    """
    if affine is None:
        return 1.0
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    src_voxel_mm3 = float(np.prod(spacing))
    # If the original was 240x240x155 and we resize to 128x128x128, each new
    # voxel covers more of the source. We don't know the original shape here
    # without re-reading the NIfTI; assume the BraTS standard 240x240x155.
    src_shape = (240.0, 240.0, 155.0)
    factor = float(np.prod(src_shape) / np.prod(resized_shape))
    return src_voxel_mm3 * factor


# ---------------------------------------------------------------------------
# Helpers for the routes layer
# ---------------------------------------------------------------------------
def get_or_run_prediction(model_key: str, sid: str) -> dict:
    """Return cached metrics if present; otherwise run inference and cache."""
    if sessions.has_prediction(sid, model_key):
        return sessions.load_prediction_metrics(sid, model_key)
    return run_inference_for(model_key, sid)


def required_modalities_for(model_key: str) -> tuple[str, ...]:
    """Which raw upload kinds must be present before this model can run."""
    entry = registry.get(model_key)
    if entry.in_channels == 1:
        return (entry.modality, "seg")
    # Multimodal needs all four
    return tuple(list(MODALITIES) + ["seg"])


def missing_modalities(model_key: str, sid: str) -> list[str]:
    meta = sessions.load(sid)
    needed = required_modalities_for(model_key)
    return [k for k in needed if k not in meta.uploads]
