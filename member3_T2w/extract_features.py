# member3_T2w/extract_features.py
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed
set_global_seed()

import torch

from shared.config import NUM_WORKERS, RESULTS_DIR, SPLITS_DIR, get_data_root
from shared.dataset import BraTSDataset, get_dataloader, load_splits
from shared.trainer import CheckpointManager

from member3_T2w.model import build_model


MEMBER_NAME = "M3_T2w_SwinUNETR"
# The feature-cache directory uses the short, modality-based name that
# the Latent Fusion Head (ensemble/features_dataset.py) expects, matching
# the convention used by M1_T1n / M2_T1c / M4_T2f. The full MEMBER_NAME
# is still used for the checkpoint via CheckpointManager.
FEATURE_DIR_NAME = "M3_T2w"
MODALITY = "t2w"
FEATURE_SIZE = 24
# Full-volume preprocessing target inside BraTSDataset is always 128^3,
# regardless of PATCH_SIZE — see shared/dataset.py:BraTSDataset.__init__.
FULL_VOLUME_SIZE = (128, 128, 128)

# Path override — shared.config.CHECKPOINT_DIR resolves to D:\checkpoints on
# Windows (Path("...") / "/checkpoints" → absolute), not <repo>/results/...
# Derive from RESULTS_DIR (which is correctly set) instead.
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = get_data_root()

    # Build Swin-UNETR sized for the actual full-volume input (128^3), not for
    # the training patch size. forward_features now upsamples the projected
    # bottleneck to H/16 to honor the shared Latent Fusion contract.
    model = build_model(img_size=FULL_VOLUME_SIZE, feature_size=FEATURE_SIZE).to(device)
    ckpt = CheckpointManager(save_dir=str(CHECKPOINT_DIR), member_name=MEMBER_NAME)
    model, _, epoch, best_val_dice = ckpt.load_best(model, optimizer=None)
    model.eval()

    print(f"[features] loaded checkpoint epoch={epoch}, val_dice_wt={best_val_dice:.4f}")

    out_dir = RESULTS_DIR / "features" / FEATURE_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(str(SPLITS_DIR))
    all_ids = splits["train"] + splits["val"] + splits["test"]

    start = time.time()

    for i, pid in enumerate(all_ids, start=1):
        temp_file = out_dir / f"_single_{pid}.txt"
        temp_file.write_text(pid + "\n", encoding="utf-8")

        # full_volume=True ignores patch_size and returns the entire 128^3
        # preprocessed volume + label — exactly what the trained Swin needs
        # for its bottleneck features to be /16 of the actual scan extent.
        ds = BraTSDataset(
            data_root=data_root,
            split_file=str(temp_file),
            modality=MODALITY,
            augment=False,
            patches_per_volume=1,
            full_volume=True,
        )

        loader = get_dataloader(ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

        batch = next(iter(loader))
        image = batch["image"].to(device)

        with torch.no_grad():
            features = model.forward_features(image).cpu()

        save_path = out_dir / f"{pid}.pt"
        torch.save(features, save_path)

        temp_file.unlink(missing_ok=True)

        percent = 100.0 * i / len(all_ids)
        elapsed = time.time() - start
        print(
            f"[features] {percent:6.2f}% | {i}/{len(all_ids)} | {pid} | saved {tuple(features.shape)}",
            flush=True,
        )

    print(f"[features] saved all features to {out_dir}")


if __name__ == "__main__":
    main()