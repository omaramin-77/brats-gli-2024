"""Member 2 — T1c bottleneck-feature extraction for Latent Space Fusion.

After M2 has trained, this script loads the best checkpoint, iterates
over every patient in the train + val + test splits at full volume
(128**3 after preprocessing), and saves the M2 bottleneck feature map
``(1, 256, 8, 8, 8)`` per patient to ``results/features/M2_T1c/{pid}.pt``.

M5's Latent Fusion Head consumes one .pt file per modality per
patient. The patient IDs match the splits exactly so M5 can pair
``features/M2_T1c/{pid}.pt`` with ``features/M1_T1n/{pid}.pt`` etc.
without any cross-reference table.

Total output: 350 .pt files in ``results/features/M2_T1c/`` — one per
patient in train + val + test. Each file loads back to a tensor of
shape ``(1, 256, 8, 8, 8)`` (256 bottleneck channels at H/16 of the
128**3 preprocessed volume).

Acceptance:
- 350 .pt files exist after running.
- Each file's tensor shape is (1, 256, 8, 8, 8).
- File names exactly match patient IDs in the split files.
"""
from shared.seed import set_global_seed
set_global_seed()

from pathlib import Path

import torch

from shared.config import (
    CHECKPOINT_DIR,
    RESULTS_DIR,
    SPLITS_DIR,
    get_data_root,
)
from shared.dataset import BraTSDataset, get_dataloader
from shared.trainer import CheckpointManager

from member2_T1c.model import AttentionUNet3D


MEMBER_NAME = "M2_T1c_AttUNet"
MODALITY = "t1c"
SPLITS = ("train", "val", "test")
FEATURES_DIR = RESULTS_DIR / "features" / "M2_T1c"


def main() -> None:
    data_root = get_data_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Concatenate train + val + test patient IDs into one ordered list. We
    # build one BraTSDataset per split (full_volume=True, no augmentation,
    # no patches) and chain them via ConcatDataset so a single DataLoader
    # walks all 350 patients in deterministic order.
    datasets = []
    expected_total = 0
    for name in SPLITS:
        ds = BraTSDataset(
            data_root=data_root,
            split_file=str(SPLITS_DIR / f"{name}_ids.txt"),
            modality=MODALITY,
            augment=False,
            full_volume=True,
        )
        datasets.append(ds)
        expected_total += len(ds.patient_ids)
        print(f"[features] split={name:5s} n_patients={len(ds.patient_ids)}")

    combined = torch.utils.data.ConcatDataset(datasets)
    loader = get_dataloader(combined, batch_size=1, shuffle=False)

    model = AttentionUNet3D(in_channels=1, out_channels=3).to(device)
    ckpt = CheckpointManager(CHECKPOINT_DIR, MEMBER_NAME)
    model, _, _, _ = ckpt.load_best(model)
    model = model.to(device)
    model.eval()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[features] writing to {FEATURES_DIR}")
    print(f"[features] expecting {expected_total} patients")

    written = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            patient_id = batch["patient_id"][0]
            features = model.forward_features(image)              # (1, 256, 8, 8, 8) for 128**3 input
            out_path = FEATURES_DIR / f"{patient_id}.pt"
            torch.save(features.cpu(), out_path)
            written += 1
            if written % 25 == 0:
                print(f"[features] {written}/{expected_total}")

    print(f"[features] wrote {written} tensors")

    # ── Acceptance checks ────────────────────────────────────────────────
    saved = sorted(p.name for p in FEATURES_DIR.glob("*.pt"))
    assert len(saved) == expected_total, (
        f"Expected {expected_total} .pt files in {FEATURES_DIR}, "
        f"found {len(saved)}"
    )
    sample_tensor = torch.load(FEATURES_DIR / saved[0], map_location="cpu")
    assert tuple(sample_tensor.shape) == (1, 256, 8, 8, 8), (
        f"Sample feature shape {tuple(sample_tensor.shape)} != (1, 256, 8, 8, 8); "
        "did you run with full_volume=True at 128**3 preprocessing?"
    )
    print(f"[features] acceptance OK — {len(saved)} files, sample shape {tuple(sample_tensor.shape)}")


if __name__ == "__main__":
    main()
