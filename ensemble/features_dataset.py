"""FeaturesDataset — loads pre-computed bottleneck features for Pipeline C.

For each patient, loads four per-modality bottleneck tensors saved by
M1-M4's extract_features.py scripts, plus the matching 3-channel
WT/TC/ET label rebuilt from the raw segmentation NIfTI.

Note on feature shape normalisation:
  The Pipeline C spec expects every per-modality tensor to be shape
  (256, 8, 8, 8) after batch-dim squeeze. M1 and M2 save (1, 256, 8, 8, 8).
  M4 saves (256, 8, 8, 8). M3 currently saves (1, 256, 4, 4, 4) because
  M3's extract_features.py ran inference with Swin-UNETR's img_size set
  to PATCH_SIZE=64³ instead of full_volume 128³. Re-extracting M3 is out
  of scope here, so this dataset trilinearly upsamples any feature whose
  spatial dims do not equal (8, 8, 8). A one-time warning prints the
  first time this happens so the bug stays visible.

Note on path resolution:
  shared.config has a Windows-specific bug — RESULTS_DIR / "/features"
  resolves to D:\\features (empty) instead of <repo>/results/features.
  We import RESULTS_DIR (which IS correct) and derive the features root
  manually as RESULTS_DIR / "features".
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from shared.config import RESULTS_DIR, get_data_root
from shared.preprocessing import resize_volume


_FEATURE_SPATIAL_TARGET = (8, 8, 8)
_FEATURE_CHANNELS = 256
_WARNED_RESHAPES: set[str] = set()


def _normalise_feature_shape(t: torch.Tensor, mod: str, pid: str) -> torch.Tensor:
    """Coerce a saved feature tensor into (256, 8, 8, 8).

    Handles:
      - leading batch dim of size 1 (M1, M2, M3)
      - missing batch dim (M4)
      - spatial dims that do not match (8, 8, 8) (M3 saves 4x4x4)
    """
    if t.ndim == 5 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim != 4:
        raise RuntimeError(
            f"feature {pid}/{mod}: expected 4D after squeeze, got shape "
            f"{tuple(t.shape)}"
        )
    if t.shape[0] != _FEATURE_CHANNELS:
        raise RuntimeError(
            f"feature {pid}/{mod}: expected {_FEATURE_CHANNELS} channels, "
            f"got {t.shape[0]} (shape {tuple(t.shape)})"
        )
    if t.shape[1:] != _FEATURE_SPATIAL_TARGET:
        if mod not in _WARNED_RESHAPES:
            _WARNED_RESHAPES.add(mod)
            print(
                f"[FeaturesDataset] WARNING: {mod} features have spatial shape "
                f"{tuple(t.shape[1:])} != {_FEATURE_SPATIAL_TARGET}. Trilinearly "
                f"upsampling to {_FEATURE_SPATIAL_TARGET}. Re-run "
                f"member{'3' if mod == 't2w' else '?'}_*/extract_features.py "
                "with full-volume 128^3 input to fix at source."
            )
        t = F.interpolate(
            t.unsqueeze(0).float(),
            size=_FEATURE_SPATIAL_TARGET,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
    return t.float()


class FeaturesDataset(Dataset):
    """Pre-computed bottleneck features for the Latent Fusion Head.

    Per patient, loads four (1, 256, 8, 8, 8) feature tensors from
    results/features/{M1_T1n,M2_T1c,M3_T2w,M4_T2f}/{patient_id}.pt
    and the matching 3-channel WT/TC/ET label built from the raw
    segmentation. The label is at full 128^3 resolution; the decoder
    upsamples the 8^3 fused features 16x back to that resolution
    before the loss is computed.

    Returns:
        {
            "features": dict with keys "t1c","t1n","t2f","t2w",
                        each shape (256, 8, 8, 8)  (channel-first, no batch dim)
            "label":    (3, 128, 128, 128) float tensor (WT, TC, ET)
            "patient_id": str
        }
    """

    FEATURE_DIRS = {
        "t1c": "M2_T1c",
        "t1n": "M1_T1n",
        "t2f": "M4_T2f",
        "t2w": "M3_T2w",
    }

    def __init__(self, split_file: str, data_root: str | None = None):
        path = Path(split_file)
        if not path.exists():
            raise FileNotFoundError(f"split file not found: {path}")
        self.patient_ids = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.data_root = Path(data_root) if data_root else Path(get_data_root())
        # NOTE: derive from RESULTS_DIR rather than shared.config.FEATURE_DIR
        # because config.FEATURE_DIR is broken on Windows (resolves to D:/features).
        self.features_root = RESULTS_DIR / "features"

        # Pre-flight check: every patient must have all 4 feature files.
        missing = []
        for pid in self.patient_ids:
            for mod, dirname in self.FEATURE_DIRS.items():
                fpath = self.features_root / dirname / f"{pid}.pt"
                if not fpath.exists():
                    missing.append(str(fpath))
        if missing:
            raise FileNotFoundError(
                f"FeaturesDataset: {len(missing)} feature files missing. "
                f"First missing: {missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _load_label(self, patient_id: str) -> torch.Tensor:
        """Load the (3, 128, 128, 128) WT/TC/ET target.

        Reads from the on-disk cache produced by ensemble/cache_labels.py
        if present. Falls back to live NIfTI loading + resize otherwise,
        so the dataset stays functional when the cache has not been
        built yet (e.g. in CI smoke tests).

        The cache stores uint8 for compactness; we cast to float here
        because BCE / Dice expect float targets.
        """
        cache_path = self.features_root / "labels" / f"{patient_id}.pt"
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu").float()

        # Fallback: rebuild from NIfTI. Identical to the original logic.
        pdir = self.data_root / patient_id
        seg_path = next(pdir.glob("*seg*.nii*"))
        seg_raw = nib.load(str(seg_path)).get_fdata().astype(np.int16)
        seg = resize_volume(seg_raw, (128, 128, 128), order=0).astype(np.int16)
        wt = ((seg == 1) | (seg == 2) | (seg == 3))
        tc = ((seg == 1) | (seg == 3))
        et = (seg == 3)
        return torch.from_numpy(
            np.stack([wt, tc, et], axis=0).astype(np.float32)
        )

    def __getitem__(self, idx: int) -> dict:
        pid = self.patient_ids[idx]
        features = {}
        for mod, dirname in self.FEATURE_DIRS.items():
            fpath = self.features_root / dirname / f"{pid}.pt"
            t = torch.load(fpath, map_location="cpu")
            features[mod] = _normalise_feature_shape(t, mod, pid)

        label = self._load_label(pid)
        return {"features": features, "label": label, "patient_id": pid}
