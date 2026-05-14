"""Copy the cases listed in data/splits/ from the full BraTS dataset to a local folder.

Examples (run from the repo root):
    python download_split.py --dest data/raw/subset
    python download_split.py --dest data/raw/subset --modality t1n
    python download_split.py --dest data/raw/subset --split train --modality t2f

Defaults: all 350 cases (train+val+test), every modality + segmentation mask.
A modality filter still copies the seg mask (you need labels for training).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SPLITS_DIR = REPO_ROOT / "data" / "splits"
DATA_PATH_FILE = REPO_ROOT / "data" / "raw" / "DATA_PATH.txt"
DATA_PATH_LOCAL_FILE = REPO_ROOT / "data" / "raw" / "DATA_PATH.local.txt"

MODALITIES = ("t1c", "t1n", "t2f", "t2w")
SPLITS = ("train", "val", "test")


def resolve_source() -> Path:
    for f in (DATA_PATH_LOCAL_FILE, DATA_PATH_FILE):
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    return Path(line)
    raise FileNotFoundError("No DATA_PATH.txt or DATA_PATH.local.txt found in data/raw/")


def load_ids(split: str) -> list[str]:
    return [
        line.strip()
        for line in (SPLITS_DIR / f"{split}_ids.txt").read_text().splitlines()
        if line.strip()
    ]


def files_for_case(case_id: str, modality: str) -> list[str]:
    mods = MODALITIES if modality == "all" else (modality,)
    return [f"{case_id}-{m}.nii.gz" for m in mods] + [f"{case_id}-seg.nii.gz"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", required=True, type=Path, help="Destination folder.")
    parser.add_argument("--modality", default="all", choices=("all", *MODALITIES))
    parser.add_argument("--split", default="all", choices=("all", *SPLITS))
    args = parser.parse_args()

    source = resolve_source()
    if not source.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")

    splits = SPLITS if args.split == "all" else (args.split,)
    case_ids = [cid for s in splits for cid in load_ids(s)]

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"src: {source}")
    print(f"dst: {args.dest.resolve()}")
    print(f"cases: {len(case_ids)} | split={args.split} | modality={args.modality}")

    copied = skipped = missing = 0
    for i, cid in enumerate(case_ids, 1):
        src_dir = source / cid
        dst_dir = args.dest / cid
        dst_dir.mkdir(exist_ok=True)
        for fname in files_for_case(cid, args.modality):
            src, dst = src_dir / fname, dst_dir / fname
            if dst.exists():
                skipped += 1
            elif not src.exists():
                print(f"  [missing] {src}")
                missing += 1
            else:
                shutil.copy2(src, dst)
                copied += 1
        if i % 25 == 0 or i == len(case_ids):
            print(f"  [{i}/{len(case_ids)}]  copied={copied} skipped={skipped} missing={missing}")


if __name__ == "__main__":
    main()
