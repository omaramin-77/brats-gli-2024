# member3_T2w/xai_analysis.py
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed
set_global_seed()

import torch

from shared.config import CHECKPOINT_DIR, FIGURES_DIR, NUM_WORKERS, PATCH_SIZE, SPLITS_DIR, TABLES_DIR, get_data_root
from shared.dataset import BraTSDataset, get_dataloader, load_splits
from shared.grad_cam_3d import GradCAM3D
from shared.trainer import CheckpointManager
from shared.visualization import plot_gradcam_three_view, plot_gradcam_per_subregion

from model import build_model


MEMBER_NAME = "M3_T2w_SwinUNETR"
MODALITY = "t2w"
FEATURE_SIZE = 24
SUBREGIONS = ("wt", "tc", "et")


def main() -> None:
    test_metrics_path = TABLES_DIR / "test_metrics.csv"
    if not test_metrics_path.exists():
        raise RuntimeError(
            "Run member3_T2w/evaluate.py first. "
            "XAI must only run after final test evaluation."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = get_data_root()

    model = build_model(img_size=PATCH_SIZE, feature_size=FEATURE_SIZE).to(device)
    ckpt = CheckpointManager(save_dir=str(CHECKPOINT_DIR), member_name=MEMBER_NAME)
    model, _, epoch, best_val_dice = ckpt.load_best(model, optimizer=None)
    model.eval()

    print(f"[xai] loaded checkpoint epoch={epoch}, val_dice_wt={best_val_dice:.4f}")

    splits = load_splits(str(SPLITS_DIR))
    test_ids = splits["test"]
    selected_ids = [
        test_ids[0],
        test_ids[len(test_ids) // 2],
        test_ids[-1],
    ]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    target_layer = model.get_target_layer()

    for pid in selected_ids:
        temp_file = FIGURES_DIR / f"_xai_{pid}.txt"
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
        label = batch["label"]

        volume_np = image[0, 0].detach().cpu().numpy()
        label_np = label[0].detach().cpu().numpy()

        print(f"[xai] patient={pid}")

        cams = {}

        for sr in SUBREGIONS:
            print(f"[xai] Grad-CAM subregion={sr}")

            with GradCAM3D(model, target_layer) as cam:
                heatmap = cam.generate(image, target_channel=sr)

            cams[sr] = heatmap

            sr_index = SUBREGIONS.index(sr)
            sr_seg = label_np[sr_index]

            prefix = FIGURES_DIR / f"gradcam_M3_{pid}_{sr}"
            plot_gradcam_three_view(
                volume=volume_np,
                heatmap=heatmap,
                seg=sr_seg,
                patient_id=pid,
                subregion=sr,
                save_path_prefix=str(prefix),
            )

        plot_gradcam_per_subregion(
            volume=volume_np,
            cams=cams,
            seg_3ch=label_np,
            patient_id=pid,
            save_path=str(FIGURES_DIR / f"gradcam_M3_{pid}_all_subregions.png"),
        )

        temp_file.unlink(missing_ok=True)

    print(f"[xai] saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()