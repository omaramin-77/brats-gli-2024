"""Pre-compute and cache 3-channel WT/TC/ET labels for Pipeline C.

Runs once. For every patient in train+val+test splits, builds the
(3, 128, 128, 128) WT/TC/ET target the same way
FeaturesDataset._load_label does (load raw seg, resize to 128^3
with order=0, stack WT/TC/ET binary channels) and saves the result
as torch.uint8 at:
    results/features/labels/{patient_id}.pt

Re-running is idempotent — files that already exist on disk are
skipped. To force re-cache, delete results/features/labels/ first.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed
set_global_seed()

import nibabel as nib
import numpy as np
import torch

from shared.config import RESULTS_DIR, SPLITS_DIR, get_data_root
from shared.dataset import load_splits
from shared.preprocessing import resize_volume


LABELS_DIR = RESULTS_DIR / "features" / "labels"


def build_label(patient_dir: Path) -> torch.Tensor:
    """Reproduce FeaturesDataset._load_label exactly.

    Loads the raw segmentation NIfTI, resizes to 128^3 with order=0
    (nearest neighbour — never bilinear on integer labels), constructs
    the 3-channel WT/TC/ET stack, returns torch.uint8 of shape
    (3, 128, 128, 128) with values in {0, 1}.
    """
    seg_path = next(patient_dir.glob("*seg*.nii*"))
    seg_raw = nib.load(str(seg_path)).get_fdata().astype(np.int16)
    seg = resize_volume(seg_raw, (128, 128, 128), order=0).astype(np.int16)
    wt = ((seg == 1) | (seg == 2) | (seg == 3))
    tc = ((seg == 1) | (seg == 3))
    et = (seg == 3)
    stack = np.stack([wt, tc, et], axis=0).astype(np.uint8)
    return torch.from_numpy(stack)


def main() -> None:
    data_root = Path(get_data_root())
    splits = load_splits(str(SPLITS_DIR))
    all_ids = splits["train"] + splits["val"] + splits["test"]
    print(f"[cache] data root: {data_root}")
    print(f"[cache] caching labels for {len(all_ids)} patients "
          f"(train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])})")

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[cache] output:    {LABELS_DIR}")

    already = 0
    built = 0
    failed: list[tuple[str, str]] = []
    start = time.time()

    for i, pid in enumerate(all_ids):
        out_path = LABELS_DIR / f"{pid}.pt"
        if out_path.exists():
            already += 1
            continue
        patient_dir = data_root / pid
        try:
            label = build_label(patient_dir)
            if tuple(label.shape) != (3, 128, 128, 128):
                raise RuntimeError(
                    f"label shape {tuple(label.shape)} != (3,128,128,128)"
                )
            if label.dtype != torch.uint8:
                raise RuntimeError(f"label dtype {label.dtype} != torch.uint8")
            if label.max() > 1:
                raise RuntimeError(
                    f"label has values > 1 (max={int(label.max())}); "
                    "binary masks expected"
                )
            torch.save(label, out_path)
            built += 1
        except Exception as e:
            failed.append((pid, str(e)))
            print(f"[cache] FAILED {pid}: {e}")

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            print(f"[cache] {i + 1}/{len(all_ids)} done "
                  f"({built} built, {already} cached) "
                  f"elapsed {elapsed:.1f}s")

    elapsed = time.time() - start
    print()
    print("=" * 60)
    print("LABEL CACHE COMPLETE")
    print("=" * 60)
    print(f"  total patients   : {len(all_ids)}")
    print(f"  already cached   : {already}")
    print(f"  built this run   : {built}")
    print(f"  failed           : {len(failed)}")
    if failed:
        print("  failures:")
        for pid, err in failed[:10]:
            print(f"    {pid}: {err}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")
    print(f"  output directory : {LABELS_DIR}")
    print(f"  elapsed time     : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 60)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
