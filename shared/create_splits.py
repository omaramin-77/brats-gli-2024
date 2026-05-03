"""Build the stratified 350-case subset and 80/10/10 train/val/test split.

Run ONCE. The committed split files under data/splits/ are the canonical
partition for the whole project; re-running this script would silently
change what each member trains on and invalidate prior results.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nibabel as nib  # noqa: E402

from shared.config import GLOBAL_SEED, SPLITS_DIR, get_data_root  # noqa: E402


SUBSET_SIZE = 350
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10  # informational; remainder of train+val
N_STRATA = 5


def _is_valid_patient(folder: Path) -> bool:
    return any(folder.glob("*seg*.nii*"))


def discover_patients(data_root: str) -> list[Path]:
    root = Path(data_root)
    patients = sorted(p for p in root.iterdir() if p.is_dir() and _is_valid_patient(p))
    return patients


def tumour_fraction(patient_dir: Path) -> float:
    seg_path = next(patient_dir.glob("*seg*.nii*"))
    seg = nib.load(str(seg_path)).get_fdata()
    total = seg.size
    if total == 0:
        return 0.0
    return float((seg > 0).sum() / total)


def stratify(values: np.ndarray, n_strata: int) -> np.ndarray:
    """Bin values by ``n_strata`` quantile edges.

    Returns an array of integer stratum labels in [0, n_strata-1].
    """
    quantiles = np.linspace(0.0, 1.0, n_strata + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.digitize(values, edges)


def select_subset(
    patients: list[Path],
    fractions: np.ndarray,
    target: int,
    rng: np.random.Generator,
) -> list[int]:
    """Stratified sampling of ``target`` indices balanced across strata."""
    strata = stratify(fractions, N_STRATA)
    per_stratum = target // N_STRATA

    selected: list[int] = []
    leftover_pool: list[int] = []

    for s in range(N_STRATA):
        idxs = np.where(strata == s)[0]
        if len(idxs) <= per_stratum:
            selected.extend(idxs.tolist())
            print(f"[splits] stratum {s}: took all {len(idxs)} (target {per_stratum})")
        else:
            chosen = rng.choice(idxs, size=per_stratum, replace=False)
            selected.extend(chosen.tolist())
            leftover_pool.extend([i for i in idxs if i not in set(chosen.tolist())])
            print(f"[splits] stratum {s}: sampled {per_stratum} of {len(idxs)}")

    # Top up from the largest leftover pool if any stratum was undersized.
    needed = target - len(selected)
    if needed > 0 and leftover_pool:
        topup = rng.choice(leftover_pool, size=min(needed, len(leftover_pool)), replace=False)
        selected.extend(topup.tolist())
        print(f"[splits] topped up {len(topup)} cases from leftover pool")

    return sorted(set(selected))


def write_split_file(path: Path, ids: Iterable[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(GLOBAL_SEED)

    data_root = get_data_root()
    print(f"[splits] DATA_ROOT = {data_root}")

    patients = discover_patients(data_root)
    print(f"[splits] found {len(patients)} valid patient folders")
    if not patients:
        raise SystemExit("No valid patients found — check DATA_PATH.txt.")

    print("[splits] computing tumour_fraction per case (this scans each seg) ...")
    fractions = np.array([tumour_fraction(p) for p in patients], dtype=np.float64)

    target = min(SUBSET_SIZE, len(patients))
    selected_idx = select_subset(patients, fractions, target=target, rng=rng)
    selected_ids = [patients[i].name for i in selected_idx]
    selected_fracs = fractions[selected_idx]
    selected_strata = stratify(selected_fracs, N_STRATA)

    # 80/10/10 split, stratified on the same tumour-fraction bins.
    train_ids, holdout_ids, train_strata, holdout_strata = train_test_split(
        selected_ids,
        selected_strata,
        test_size=(VAL_FRAC + TEST_FRAC),
        stratify=selected_strata,
        random_state=GLOBAL_SEED,
    )
    val_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=holdout_strata,
        random_state=GLOBAL_SEED,
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    write_split_file(SPLITS_DIR / "train_ids.txt", train_ids)
    write_split_file(SPLITS_DIR / "val_ids.txt", val_ids)
    write_split_file(SPLITS_DIR / "test_ids.txt", test_ids)

    # Reproducibility check: no patient may appear in more than one split.
    overlap_tv = set(train_ids) & set(val_ids)
    overlap_tt = set(train_ids) & set(test_ids)
    overlap_vt = set(val_ids) & set(test_ids)
    assert not overlap_tv, f"train/val overlap: {overlap_tv}"
    assert not overlap_tt, f"train/test overlap: {overlap_tt}"
    assert not overlap_vt, f"val/test overlap: {overlap_vt}"

    id_to_frac = {patients[i].name: float(fractions[i]) for i in selected_idx}
    stats = {
        "total_available": int(len(patients)),
        "subset_size": int(len(selected_ids)),
        "train": int(len(train_ids)),
        "val": int(len(val_ids)),
        "test": int(len(test_ids)),
        "seed": GLOBAL_SEED,
        "tumor_fraction_mean_train": float(np.mean([id_to_frac[i] for i in train_ids])),
        "tumor_fraction_mean_val": float(np.mean([id_to_frac[i] for i in val_ids])),
        "tumor_fraction_mean_test": float(np.mean([id_to_frac[i] for i in test_ids])),
        "created_by": "create_splits.py",
        "note": "Do not re-run this script. All members load from these files.",
    }
    (SPLITS_DIR / "split_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 60)
    print(" stratified 350-case subset & 80/10/10 split")
    print("=" * 60)
    print(f" total available     : {stats['total_available']}")
    print(f" subset size         : {stats['subset_size']}")
    print(f" train               : {stats['train']}")
    print(f" val                 : {stats['val']}")
    print(f" test                : {stats['test']}")
    print(f" seed                : {stats['seed']}")
    print(f" mean frac (train)   : {stats['tumor_fraction_mean_train']:.6f}")
    print(f" mean frac (val)     : {stats['tumor_fraction_mean_val']:.6f}")
    print(f" mean frac (test)    : {stats['tumor_fraction_mean_test']:.6f}")
    print(" overlap check       : OK (no patient shared across splits)")
    print("=" * 60)


if __name__ == "__main__":
    main()
