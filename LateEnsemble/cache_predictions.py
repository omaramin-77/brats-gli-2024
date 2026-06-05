"""Cache full-volume logit predictions from each trained member.

For each member in MEMBER_SPECS and each patient in val + test, runs
sliding-window inference at PATCH_SIZE and saves the resulting
(3, 128, 128, 128) logit tensor (float16, on disk) to:

    results/predictions/{member.short}/{patient_id}.pt

It ALSO caches the matching ground-truth label, taken from BraTSDataset's
brain-cropped-and-resized output (same coordinate frame as the
predictions), to:

    results/predictions/labels/{patient_id}.pt   # uint8 (3, 128, 128, 128)

Why label caching lives here, not in ensemble/cache_labels.py: that older
script rebuilds labels from the raw NIfTI via resize-without-crop, which
puts labels in a different coordinate frame from the predictions
(predictions go through preprocess_patient which crops to brain first).
Using mis-aligned labels gives label/pred IoU ~0.06 on TC/ET and
catastrophically low Dice that the stacker cannot recover from. We keep
LateEnsemble's labels self-contained to avoid that trap.

This is the slow step. Run once. The downstream stacker training and
evaluation operate entirely on these cached logits — no NIfTI loading,
no encoder forward passes, just a 15-parameter softmax-weighted sum.

Run-time budget
---------------
350 patients x 5 members ~= 1750 SWI inferences. On an RTX 3090 each one
is ~1-2 seconds at PATCH_SIZE=96 with overlap=0.5, so ~1 hour total.
On a smaller GPU expect 2-3 hours.

Storage budget
--------------
Each .pt is (3, 128, 128, 128) float16 ~= 12 MB. 5 x 70 = 350 files for
val + test = ~4.2 GB. We do not cache train predictions (the stacker
trains on val).

CLI
---
    python LateEnsemble/cache_predictions.py
    python LateEnsemble/cache_predictions.py --members m1 m2
    python LateEnsemble/cache_predictions.py --splits val
    python LateEnsemble/cache_predictions.py --max-patients 3   # smoke test
    python LateEnsemble/cache_predictions.py --labels-only      # only fill in
                                                                # missing labels
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed  # noqa: E402
set_global_seed()

import argparse  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

try:
    from monai.inferers import sliding_window_inference  # noqa: E402
    _HAVE_SWI = True
except ImportError:  # pragma: no cover
    _HAVE_SWI = False

from shared.config import (  # noqa: E402
    PATCH_SIZE,
    RESULTS_DIR,
    SPLITS_DIR,
    get_data_root,
)
from shared.dataset import BraTSDataset, get_dataloader, load_splits  # noqa: E402
from shared.trainer import CheckpointManager  # noqa: E402

from LateEnsemble.members import MEMBER_SPECS  # noqa: E402


# Note: shared.config.RESULTS_DIR is correctly set on every machine
# (the previous Windows path bug only affects the derived subdirs).
PREDICTIONS_ROOT = RESULTS_DIR / "predictions"
LABELS_DIR = PREDICTIONS_ROOT / "labels"
CHECKPOINT_DIR_FIXED = RESULTS_DIR / "checkpoints"


@torch.no_grad()
def _run_inference(model, image, device) -> torch.Tensor:
    """Sliding-window inference at PATCH_SIZE, overlap=0.5."""
    roi = PATCH_SIZE if isinstance(PATCH_SIZE, tuple) else (PATCH_SIZE,) * 3
    if _HAVE_SWI:
        return sliding_window_inference(
            inputs=image,
            roi_size=roi,
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
        )
    # Fallback (should not happen in this repo — MONAI is in requirements).
    return model(image)  # pragma: no cover


def _save_label_if_missing(batch_label: torch.Tensor, pid: str) -> bool:
    """Save the BraTSDataset label for one patient as uint8.

    The label is whatever BraTSDataset's full_volume mode returned — i.e.
    in the brain-cropped-and-resized frame, identical to the coordinate
    system the predictions live in. Returns True if we wrote a new file.
    """
    out_path = LABELS_DIR / f"{pid}.pt"
    if out_path.exists():
        return False
    label = batch_label[0] if batch_label.ndim == 5 else batch_label
    # BraTSDataset returns float in {0, 1}; uint8 cuts disk by 4x.
    tensor = (label > 0.5).to(torch.uint8).contiguous()
    if tuple(tensor.shape) != (3, 128, 128, 128):
        raise RuntimeError(
            f"label tensor for {pid} has shape {tuple(tensor.shape)}, "
            "expected (3, 128, 128, 128)"
        )
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, out_path)
    return True


def _cache_labels_only(splits_to_cache, patient_limit) -> None:
    """Populate LABELS_DIR without loading any model.

    Uses a cheap unimodal dataset (modality='t1n') — the segmentation
    label is identical regardless of which modality the dataset was
    instantiated with, and t1n's preprocessing is the fastest.
    """
    print(f"\n[cache] === LABELS ONLY (no model inference) ===")
    data_root = get_data_root()
    total_written = 0
    total_skipped = 0
    label_start = time.time()

    for split_name in splits_to_cache:
        split_file = SPLITS_DIR / f"{split_name}_ids.txt"
        ds = BraTSDataset(
            data_root=data_root,
            split_file=str(split_file),
            modality="t1n",
            augment=False,
            full_volume=True,
        )
        if patient_limit is not None:
            ds.patient_ids = ds.patient_ids[:patient_limit]
        loader = get_dataloader(ds, batch_size=1, shuffle=False, num_workers=0)
        print(
            f"[cache] split={split_name:5s} n_patients={len(ds.patient_ids)} "
            f"out={LABELS_DIR}"
        )

        for i, batch in enumerate(loader, start=1):
            pid = batch["patient_id"]
            if isinstance(pid, (list, tuple)):
                pid = pid[0]
            if _save_label_if_missing(batch["label"], pid):
                total_written += 1
            else:
                total_skipped += 1
            if i % 10 == 0 or i == len(ds.patient_ids):
                print(f"[cache]   labels {i}/{len(ds.patient_ids)} (last={pid})")

    elapsed = time.time() - label_start
    print(
        f"[cache] labels done: written={total_written} "
        f"skipped(existing)={total_skipped} elapsed={elapsed/60:.2f} min"
    )


def _cache_one_member(spec, splits_to_cache, patient_limit, device):
    """Run inference for one member across the requested splits."""
    out_dir = PREDICTIONS_ROOT / spec.short
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[cache] === {spec.short.upper()} ({spec.member_name}) ===")
    model = spec.builder().to(device)
    ckpt = CheckpointManager(str(CHECKPOINT_DIR_FIXED), spec.member_name)
    model, _, epoch, val_dice = ckpt.load_best(model)
    model = model.to(device)
    model.eval()
    print(f"[cache] loaded best ckpt @ epoch {epoch} (val_dice_wt={val_dice:.4f})")

    data_root = get_data_root()
    total_saved = 0
    total_skipped = 0
    member_start = time.time()

    for split_name in splits_to_cache:
        split_file = SPLITS_DIR / f"{split_name}_ids.txt"
        ds = BraTSDataset(
            data_root=data_root,
            split_file=str(split_file),
            modality=spec.modality,
            augment=False,
            full_volume=True,
        )
        if patient_limit is not None:
            ds.patient_ids = ds.patient_ids[:patient_limit]
        loader = get_dataloader(ds, batch_size=1, shuffle=False, num_workers=0)
        print(
            f"[cache] split={split_name:5s} n_patients={len(ds.patient_ids)} "
            f"out={out_dir}"
        )

        split_start = time.time()
        for i, batch in enumerate(loader, start=1):
            pid = batch["patient_id"]
            if isinstance(pid, (list, tuple)):
                pid = pid[0]

            # Always opportunistically save the (BraTSDataset, brain-cropped)
            # label — idempotent, costs nothing on subsequent member passes.
            # This is what keeps labels in the same coordinate frame as the
            # predictions and avoids the IoU=0.06 misalignment trap.
            _save_label_if_missing(batch["label"], pid)

            save_path = out_dir / f"{pid}.pt"
            if save_path.exists():
                total_skipped += 1
                continue

            image = batch["image"].to(device, non_blocking=True)
            logits = _run_inference(model, image, device)   # (1, 3, 128, 128, 128)

            # Squeeze batch dim, downcast to fp16, move to CPU before saving.
            tensor = logits[0].to(torch.float16).cpu().contiguous()
            torch.save(tensor, save_path)
            total_saved += 1

            if i % 10 == 0 or i == len(ds.patient_ids):
                rate = i / max(time.time() - split_start, 1e-6)
                print(
                    f"[cache]   {i}/{len(ds.patient_ids)} "
                    f"({rate:.2f} patient/s)  last={pid} shape={tuple(tensor.shape)}"
                )

    elapsed = time.time() - member_start
    print(
        f"[cache] {spec.short} done: saved={total_saved} "
        f"skipped(existing)={total_skipped} elapsed={elapsed/60:.2f} min"
    )

    # Free GPU memory before the next member.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--members", nargs="+", default=None,
        choices=[s.short for s in MEMBER_SPECS],
        help="Subset of members to cache (default: all).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=("val", "test"),
        choices=("train", "val", "test"),
        help="Splits to cache. Train predictions are not used by the "
             "stacker but the option is here for completeness.",
    )
    parser.add_argument(
        "--max-patients", type=int, default=None,
        help="Limit per-split patient count (for smoke testing).",
    )
    parser.add_argument(
        "--labels-only", action="store_true",
        help="Populate results/predictions/labels/ from BraTSDataset and skip "
             "all model inference. Cheap (no GPU, no checkpoints). Use this "
             "when predictions already exist but the labels cache is missing "
             "or was built by the buggy ensemble/cache_labels.py.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cache] device: {device}  PATCH_SIZE={PATCH_SIZE}")
    print(f"[cache] predictions root: {PREDICTIONS_ROOT}")
    print(f"[cache] labels root:      {LABELS_DIR}")

    specs = MEMBER_SPECS
    if args.members:
        wanted = set(args.members)
        specs = [s for s in MEMBER_SPECS if s.short in wanted]

    splits = load_splits(str(SPLITS_DIR))
    print(
        f"[cache] splits: "
        + ", ".join(f"{name}={len(splits[name])}" for name in args.splits)
    )

    total_start = time.time()

    if args.labels_only:
        _cache_labels_only(args.splits, args.max_patients)
        print(
            f"\n[cache] labels-only run finished in "
            f"{(time.time() - total_start) / 60:.2f} min"
        )
        return

    for spec in specs:
        _cache_one_member(spec, args.splits, args.max_patients, device)

    print(
        f"\n[cache] all members done in "
        f"{(time.time() - total_start) / 60:.2f} min"
    )


if __name__ == "__main__":
    main()
