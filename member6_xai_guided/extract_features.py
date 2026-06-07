"""Member 6 — extract bottleneck features for ensemble compatibility.

The XAI-guided M5 shares the same encoder backbone as M5, so its
``forward_features`` returns the standard ``(B, 256, H/16, W/16, D/16)``
contract. Saving these lets the LateEnsemble or LatentFusion paths optionally
include M6 as a fusion source.

CLI mirrors evaluate.py: --init {xai, random} selects which trained variant
to use.
"""
from __future__ import annotations

import sys
import time
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

from member6_xai_guided.train import _name_for_init, build_model  # noqa: E402


MODALITY = "multimodal"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init", choices=("xai", "random"), default="xai",
    )
    args = parser.parse_args()

    member_name, _ = _name_for_init(args.init)
    feature_dir_name = "M6_XAIGuided" if args.init == "xai" else "M6_XAIGuided_random"
    features_dir = RESULTS_DIR / "features" / feature_dir_name
    features_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.init).to(device)
    ckpt = CheckpointManager(CHECKPOINT_DIR, member_name)
    model, _, epoch, val_dice = ckpt.load_best(model)
    model = model.to(device)
    model.eval()
    print(f"[features] loaded best ckpt epoch={epoch} val_dice_wt={val_dice:.4f}")
    print(f"[features] writing to {features_dir}")

    data_root = get_data_root()
    splits = load_splits(str(SPLITS_DIR))
    all_ids = splits["train"] + splits["val"] + splits["test"]
    print(f"[features] processing {len(all_ids)} patients")

    ds = BraTSDataset(
        data_root=data_root,
        split_file=str(SPLITS_DIR / "train_ids.txt"),   # placeholder; overridden
        modality=MODALITY,
        augment=False,
        full_volume=True,
    )
    ds.patient_ids = all_ids
    loader = get_dataloader(ds, batch_size=1, shuffle=False)

    start = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            pid = batch["patient_id"]
            if isinstance(pid, (list, tuple)):
                pid = pid[0]
            image = batch["image"].to(device, non_blocking=True)
            features = model.forward_features(image)        # (1, 256, 8, 8, 8)
            torch.save(features.cpu(), features_dir / f"{pid}.pt")
            if i % 25 == 0 or i == len(all_ids):
                elapsed = time.time() - start
                print(f"[features] {i}/{len(all_ids)}  ({elapsed:.1f}s elapsed)")

    print(f"\n[features] saved {len(all_ids)} feature tensors to {features_dir}")


if __name__ == "__main__":
    main()
