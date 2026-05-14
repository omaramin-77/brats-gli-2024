"""BraTSDataset and DataLoader factory used by every member pipeline.

Each member instantiates this class with the modality they own:
    - member1: 't1n'
    - member2: 't1c'
    - member3: 't2w'
    - member4: 't2f'
    - member5: 'multimodal'

The class returns a 3-channel WT/TC/ET target tensor (BraTS protocol). Each
channel is an independent binary mask; channels are overlapping by
construction (ET ⊂ TC ⊂ WT). Loss and metrics treat them as independent
binary tasks (sigmoid per channel — never softmax).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from shared.preprocessing import (
    extract_patch,
    get_augmentation_transforms,
    preprocess_patient,
    preprocess_patient_multimodal,
)


# Try MONAI CacheDataset; fall back gracefully if unavailable.
try:
    from monai.data import CacheDataset  # noqa: F401
    _HAVE_MONAI_CACHE = True
except ImportError:  # pragma: no cover
    _HAVE_MONAI_CACHE = False


def load_splits(splits_dir: str) -> dict:
    """Read train/val/test patient ID lists written by create_splits.py.

    Members call this once at the top of train.py:
        splits = load_splits(SPLITS_DIR)
        train_ids, val_ids = splits['train'], splits['val']
    """
    splits_dir = Path(splits_dir)
    out = {}
    for name in ("train", "val", "test"):
        f = splits_dir / f"{name}_ids.txt"
        if not f.exists():
            raise FileNotFoundError(
                f"Split file {f} missing. Run shared/create_splits.py once."
            )
        ids = [line.strip() for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        out[name] = ids
    return out


class BraTSDataset(Dataset):
    """Patch-based dataset over BraTS GLI-2024 volumes.

    __len__ multiplies the number of patients by ``patches_per_volume`` so a
    single epoch sees several patches per case. The patch index modulo
    ``patches_per_volume`` is unused — sampling is independent each call,
    which is what we want for stochastic optimisation.
    """

    def __init__(
        self,
        data_root: str,
        split_file: str,
        modality: str = "t1n",
        patch_size: int = 64,
        tumour_bias: float = 0.8,
        augment: bool = False,
        patches_per_volume: int = 4,
    ):
        VALID_MODALITIES = {"t1n", "t1c", "t2w", "t2f", "multimodal"}
        if modality not in VALID_MODALITIES:
            raise ValueError(
                f"Unknown modality '{modality}'. "
                f"Must be one of {sorted(VALID_MODALITIES)}."
            )

        self.data_root = data_root
        self.modality = modality
        self.patch_size = int(patch_size)
        self.tumour_bias = float(tumour_bias)
        self.augment = bool(augment)
        self.patches_per_volume = int(patches_per_volume)

        self.patient_ids = self._load_ids(split_file)
        self._aug = get_augmentation_transforms() if self.augment else None

    @staticmethod
    def _load_ids(split_file: str) -> list[str]:
        path = Path(split_file)
        if not path.exists():
            raise FileNotFoundError(f"split_file {path} not found")
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.patient_ids) * self.patches_per_volume

    def _patient_dir(self, patient_id: str) -> str:
        return os.path.join(self.data_root, patient_id)

    def _load_volume(self, patient_id: str) -> tuple[np.ndarray, np.ndarray]:
        pdir = self._patient_dir(patient_id)
        if self.modality == "multimodal":
            return preprocess_patient_multimodal(pdir)
        return preprocess_patient(pdir, self.modality)

    def _multichannel_label(self, seg: np.ndarray) -> np.ndarray:
        """Build a (3, H, W, D) binary target with channels (WT, TC, ET).

        From the raw integer segmentation (labels 0-4):
            WT = whole tumour    = (seg == 1) | (seg == 2) | (seg == 3)
            TC = tumour core     = (seg == 1) | (seg == 3)
            ET = enhancing       = (seg == 3)
        Label 4 (resection cavity) is excluded per BraTS 2024 protocol.

        Channels are overlapping by construction: ET ⊂ TC ⊂ WT. Each
        channel is treated as an independent binary task by the loss.
        """
        wt = (seg == 1) | (seg == 2) | (seg == 3)
        tc = (seg == 1) | (seg == 3)
        et = (seg == 3)
        return np.stack([wt, tc, et], axis=0).astype(np.int64)

    def __getitem__(self, idx: int) -> dict:
        patient_idx = idx // self.patches_per_volume
        patient_id = self.patient_ids[patient_idx]

        vol, seg = self._load_volume(patient_id)

        # Build the 3-channel WT/TC/ET target from the raw integer seg first,
        # then crop it with the same window extract_patch picked for the image.
        target_3ch = self._multichannel_label(seg)             # (3, H, W, D) int64

        # extract_patch needs a 3D mask for tumour-aware sampling.
        seg_for_sampling = (seg > 0).astype(np.int64)

        patch_vol, _, (starts, ends) = extract_patch(
            vol,
            seg_for_sampling,
            size=self.patch_size,
            tumour_bias=self.tumour_bias,
        )
        s = tuple(slice(int(a), int(b)) for a, b in zip(starts, ends))
        patch_target = target_3ch[(slice(None),) + s]          # (3, P, P, P)

        if self._aug is not None:
            data = {
                "image": torch.from_numpy(patch_vol).float(),
                "label": torch.from_numpy(patch_target).float(),
            }
            data = self._aug(data)
            image = data["image"].float()
            label = data["label"].float()
        else:
            image = torch.from_numpy(patch_vol).float()
            label = torch.from_numpy(patch_target).float()

        assert label.shape == (3, self.patch_size, self.patch_size, self.patch_size), \
            f"Expected label shape (3,P,P,P); got {label.shape}"

        return {"image": image, "label": label, "patient_id": patient_id}


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Return a DataLoader. Multi-worker loading is disabled by default.

    Note: keep num_workers=0 on Windows (DataLoader workers re-import the
    parent script and frequently deadlock). Linux/WSL2 can safely use 4.
    The MONAI CacheDataset path is left as a hook for member5's heavy
    multimodal pipeline; it can wrap the existing dataset's __getitem__
    output if memory permits.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
