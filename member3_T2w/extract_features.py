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

from shared.config import CHECKPOINT_DIR, NUM_WORKERS, PATCH_SIZE, RESULTS_DIR, SPLITS_DIR, get_data_root
from shared.dataset import BraTSDataset, get_dataloader, load_splits
from shared.trainer import CheckpointManager

from model import build_model


MEMBER_NAME = "M3_T2w_SwinUNETR"
MODALITY = "t2w"
FEATURE_SIZE = 24


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = get_data_root()

    model = build_model(img_size=PATCH_SIZE, feature_size=FEATURE_SIZE).to(device)
    ckpt = CheckpointManager(save_dir=str(CHECKPOINT_DIR), member_name=MEMBER_NAME)
    model, _, epoch, best_val_dice = ckpt.load_best(model, optimizer=None)
    model.eval()

    print(f"[features] loaded checkpoint epoch={epoch}, val_dice_wt={best_val_dice:.4f}")

    out_dir = RESULTS_DIR / "features" / MEMBER_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(str(SPLITS_DIR))
    all_ids = splits["train"] + splits["val"] + splits["test"]

    start = time.time()

    for i, pid in enumerate(all_ids, start=1):
        temp_file = out_dir / f"_single_{pid}.txt"
        temp_file.write_text(pid + "\n", encoding="utf-8")

        ds = BraTSDataset(
            data_root=data_root,
            split_file=str(temp_file),
            modality=MODALITY,
            patch_size=PATCH_SIZE[0],
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