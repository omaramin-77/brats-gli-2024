"""Member 1 — extract bottleneck features for M5's Latent Space Fusion.

Loads the trained M1 checkpoint, runs forward_features() on every patient
in train + val + test splits, and saves the resulting (1, 256, 8, 8, 8)
bottleneck tensors to disk. These are the inputs M5's fusion head consumes.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed  # noqa: E402
set_global_seed()

import torch  # noqa: E402

from shared.config import (  # noqa: E402
    CHECKPOINT_DIR,
    RESULTS_DIR,
    SPLITS_DIR,
    get_data_root,
)
from shared.dataset import BraTSDataset, get_dataloader, load_splits  # noqa: E402
from shared.trainer import CheckpointManager  # noqa: E402

from member1_T1n.model import ResidualUNet3D  # noqa: E402


MEMBER_NAME = "M1_T1n_ResUNet"
MODALITY = "t1n"
FEATURES_DIR = RESULTS_DIR / "features" / "M1_T1n"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet3D(in_channels=1, out_channels=3).to(device)

    ckpt = CheckpointManager(CHECKPOINT_DIR, MEMBER_NAME)
    model, _, epoch, val_dice = ckpt.load_best(model)
    model = model.to(device)
    model.eval()
    print(f"[features] loaded best checkpoint @ epoch {epoch} (dice_wt={val_dice:.4f})")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    data_root = get_data_root()
    splits = load_splits(str(SPLITS_DIR))
    all_ids = splits["train"] + splits["val"] + splits["test"]
    print(
        f"[features] processing {len(all_ids)} patients "
        f"(train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])})"
    )

    # Build one dataset, then override its patient list so we iterate every
    # split in a single pass. full_volume=True makes __getitem__ return the
    # whole (1, 128, 128, 128) preprocessed volume — no patches.
    ds = BraTSDataset(
        data_root=data_root,
        split_file=str(SPLITS_DIR / "train_ids.txt"),  # placeholder; overridden below
        modality=MODALITY,
        augment=False,
        full_volume=True,
    )
    ds.patient_ids = all_ids
    loader = get_dataloader(ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            pid = batch["patient_id"]
            if isinstance(pid, (list, tuple)):
                pid = pid[0]
            image = batch["image"].to(device, non_blocking=True)
            features = model.forward_features(image)
            # features shape: (1, 256, 8, 8, 8) for 128³ input.
            save_path = FEATURES_DIR / f"{pid}.pt"
            torch.save(features.cpu(), save_path)
            if (i + 1) % 25 == 0:
                print(
                    f"[features] {i + 1}/{len(all_ids)} patients done; "
                    f"latest shape: {tuple(features.shape)}"
                )

    print()
    print(f"[features] saved {len(all_ids)} feature tensors to {FEATURES_DIR}")
    print("[features] expected shape per file: (1, 256, 8, 8, 8)")


if __name__ == "__main__":
    main()
