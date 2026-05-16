"""Member 4 — T2f bottleneck feature extraction for Latent Fusion.

Loads the best trained checkpoint and runs forward_features() on every
patient in train + val + test splits, saving one .pt file per patient:
  results/features/M4_T2f/{patient_id}.pt  — shape (256, D/16, H/16, W/16)

M5 loads these .pt files instead of running the T2f encoder at fusion time.
This makes fusion training fast (no NIfTI loading, no encoder forward passes).

Latent Fusion contract (design.md §9.2):
  forward_features(x) → (B, 256, H/16, W/16, D/16)

Usage:
  python extract_features.py                    # best checkpoint (auto-detected)
  python extract_features.py --arch resunet
  python extract_features.py --arch segresnet
  python extract_features.py --split train      # one split only
"""
from __future__ import annotations

# ── Seed first ────────────────────────────────────────────────────────────────
from shared.seed import set_global_seed
set_global_seed()

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_KAGGLE:
    for candidate in [
        '/kaggle/input/brats-code/shared',
        '/kaggle/input/brats-shared-infra',
    ]:
        parent = str(Path(candidate).parent)
        if Path(candidate).exists() and parent not in sys.path:
            sys.path.insert(0, parent)
            break
    REPO_ROOT    = Path('/kaggle/working')
    RESULTS_ROOT = Path('/kaggle/working/results')
else:
    REPO_ROOT    = Path(__file__).resolve().parent.parent
    RESULTS_ROOT = REPO_ROOT / 'results'
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from shared.config import (
    CHECKPOINT_DIR, NUM_WORKERS, PATCH_SIZE, SPLITS_DIR,
    get_data_root,
)
from shared.dataset import BraTSDataset, get_dataloader
from shared.trainer import CheckpointManager

from model import build_model, BOTTLENECK_CHANNELS

MODALITY     = "t2f"
FEATURES_DIR = RESULTS_ROOT / "features" / "M4_T2f"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILES = {
    "train": "train_ids.txt",
    "val":   "val_ids.txt",
    "test":  "test_ids.txt",
}


@torch.no_grad()
def extract_and_save(
    model:     torch.nn.Module,
    loader,
    device:    torch.device,
    save_dir:  Path,
) -> int:
    """Run forward_features on every batch; save one .pt per patient."""
    model.eval()
    saved = 0

    for batch in tqdm(loader, desc="  Extracting", leave=False):
        images      = batch["image"].to(device, non_blocking=True)
        patient_ids = batch["patient_id"]   # list[str], len=batch_size

        feats = model.forward_features(images)   # (B, 256, D/16, H/16, W/16)

        for i, pid in enumerate(patient_ids):
            out_path = save_dir / f"{pid}.pt"
            torch.save(feats[i].cpu(), out_path)
            saved += 1

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="M4 feature extraction")
    parser.add_argument("--arch",  default="best",
                        choices=["resunet", "segresnet", "best"],
                        help="'best' picks whichever arch has the checkpoint")
    parser.add_argument("--split", default="all",
                        choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root  = get_data_root()
    patch_size = PATCH_SIZE[0] if isinstance(PATCH_SIZE, tuple) else PATCH_SIZE
    nw         = NUM_WORKERS

    # ── Load model ────────────────────────────────────────────────────────────
    arch_to_try = ["resunet", "segresnet"] if args.arch == "best" else [args.arch]
    model = None

    for arch in arch_to_try:
        ckpt_name = f"member4_T2f_{arch}"
        _model    = build_model(arch, in_channels=1, out_channels=3)
        try:
            mgr            = CheckpointManager(str(CHECKPOINT_DIR), ckpt_name)
            _model, _, epoch, _ = mgr.load_best(_model)
            model = _model.to(device)
            print(f"[extract] Loaded {arch} checkpoint (epoch={epoch})")
            break
        except FileNotFoundError:
            print(f"[extract] No checkpoint for '{arch}' — skipping")

    if model is None:
        print("[extract] No checkpoint found. Run train.py first.")
        return

    model.eval()

    # ── Verify forward_features output shape ──────────────────────────────────
    dummy = torch.zeros(1, 1, patch_size, patch_size, patch_size, device=device)
    feat  = model.forward_features(dummy)
    exp_ch = BOTTLENECK_CHANNELS
    assert feat.shape[1] == exp_ch, (
        f"forward_features returned {feat.shape[1]} channels, expected {exp_ch}. "
        "Check the 1×1×1 projection in model.py."
    )
    exp_spatial = patch_size // 16
    assert feat.shape[2] == exp_spatial, (
        f"forward_features spatial dim = {feat.shape[2]}, expected {exp_spatial} "
        f"(patch_size={patch_size} / 16)."
    )
    print(f"[extract] forward_features shape check OK: {tuple(feat.shape)}")

    # ── Extract per split ─────────────────────────────────────────────────────
    splits = list(SPLIT_FILES.keys()) if args.split == "all" else [args.split]
    total  = 0

    for split in splits:
        ds = BraTSDataset(
            data_root   = data_root,
            split_file  = str(SPLITS_DIR / SPLIT_FILES[split]),
            modality    = MODALITY,
            patch_size  = patch_size,
            augment     = False,
            full_volume = True,
        )
        loader = get_dataloader(
            ds, batch_size=1, shuffle=False,
            num_workers=nw, pin_memory=(device.type == "cuda"),
            persistent_workers=(nw > 0),
        )
        print(f"[extract] Split={split}  volumes={len(ds)}")
        n     = extract_and_save(model, loader, device, FEATURES_DIR)
        total += n
        print(f"           Saved {n} feature files → {FEATURES_DIR}")

    print(f"\n[extract] Done. {total} files at {FEATURES_DIR}")
    print(f"  Shape per file: ({BOTTLENECK_CHANNELS}, {exp_spatial}, {exp_spatial}, {exp_spatial})")
    print(f"  M5 loads these with: torch.load('{FEATURES_DIR}/{{patient_id}}.pt')")


if __name__ == "__main__":
    main()
