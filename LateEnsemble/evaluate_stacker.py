"""Evaluate the LateEnsemble on the held-out test split.

Reads cached test logits from results/predictions/{m1..m5}/ and applies
three weighting variants, writing one row per variant to test_metrics.csv:

  1. LateEnsemble_Uniform     — equal weights (1/5) for every member.
  2. LateEnsemble_XAI_prior   — XAI-derived weights, no training.
  3. LateEnsemble_XAI_stacked — XAI-init + val-fit weights (the headline).

Prints the learned weight table and the per-variant deltas against the
M5_Multimodal_4ch baseline so the comparison fits in one screen.

The variants are evaluated under a single CLI call because they share data
and the per-variant cost is negligible (just 15 scalars × 35 patients).

CLI
---
    python LateEnsemble/evaluate_stacker.py
    python LateEnsemble/evaluate_stacker.py --m5-prior 0.6      # for "XAI_prior"
    python LateEnsemble/evaluate_stacker.py --variants stacked  # only run one
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
import csv  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from shared.config import GLOBAL_SEED, RESULTS_DIR, SPLITS_DIR  # noqa: E402
from shared.metrics import compute_all_metrics, MetricTracker  # noqa: E402

from LateEnsemble.members import (  # noqa: E402
    MEMBER_SPECS,
    NUM_MEMBERS,
    SHORT_TO_ABLATION_KEY,
)
from LateEnsemble.stacker import (  # noqa: E402
    PerSubregionStacker,
    build_uniform_logits,
    build_xai_init_logits,
    format_weight_table,
)
from LateEnsemble.train_stacker import CachedLogitsDataset  # noqa: E402


CHECKPOINT_DIR_FIXED = RESULTS_DIR / "checkpoints"
TABLES_DIR_FIXED = RESULTS_DIR / "tables"
BASELINE_MEMBER_NAME = "M5_Multimodal_4ch"
STACKER_CKPT_NAME = "LateEnsemble_stacker"

CSV_FIELDNAMES = [
    "member", "modality", "architecture", "seed",
    "dice_wt", "dice_tc", "dice_et",
    "iou_wt", "iou_tc", "iou_et",
    "hd95_wt", "hd95_tc", "hd95_et",
    "training_time_hrs", "gpu_memory_gb",
]

VARIANT_LABELS = {
    "uniform": "LateEnsemble_Uniform",
    "prior":   "LateEnsemble_XAI_prior",
    "stacked": "LateEnsemble_XAI_stacked",
}


# ---------------------------------------------------------------------------
# Variant construction
# ---------------------------------------------------------------------------
def _build_stacker_for_variant(variant: str, members_short: list[str],
                               m5_prior: float) -> PerSubregionStacker:
    if variant == "uniform":
        init = build_uniform_logits(NUM_MEMBERS, 3)
        return PerSubregionStacker(NUM_MEMBERS, 3, init_logits=init)

    if variant == "prior":
        importance_path = TABLES_DIR_FIXED / "modality_importance_scores.json"
        if not importance_path.exists():
            raise FileNotFoundError(
                f"modality_importance_scores.json not found at {importance_path}."
            )
        init = build_xai_init_logits(
            importance_json=importance_path,
            members_short=members_short,
            ablation_keys=SHORT_TO_ABLATION_KEY,
            m5_prior=m5_prior,
        )
        return PerSubregionStacker(NUM_MEMBERS, 3, init_logits=init)

    if variant == "stacked":
        ckpt_path = CHECKPOINT_DIR_FIXED / f"{STACKER_CKPT_NAME}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Stacker checkpoint not found at {ckpt_path}. "
                "Run LateEnsemble/train_stacker.py first."
            )
        payload = torch.load(ckpt_path, map_location="cpu")
        # Sanity: the checkpoint must be over the same members in the same order.
        ckpt_members = payload.get("members_short")
        if ckpt_members is not None and ckpt_members != members_short:
            raise RuntimeError(
                f"Stacker checkpoint was saved with members {ckpt_members}, "
                f"but the current MEMBER_SPECS order is {members_short}. "
                "Re-train the stacker or restore the original ordering."
            )
        stacker = PerSubregionStacker(NUM_MEMBERS, 3)
        stacker.load_state_dict(payload["stacker_state"])
        return stacker

    raise ValueError(f"Unknown variant: {variant!r}")


# ---------------------------------------------------------------------------
# Test-set evaluation for one variant
# ---------------------------------------------------------------------------
@torch.no_grad()
def _evaluate_variant(stacker, loader, device) -> dict:
    stacker.eval().to(device)
    tracker = MetricTracker()
    for batch in loader:
        logits_in = batch["logits"].to(device, non_blocking=True)   # (B, M, S, H, W, D)
        labels = batch["label"].to(device, non_blocking=True)       # (B, S, H, W, D)
        out = stacker(logits_in)                                    # (B, S, H, W, D)
        metrics = compute_all_metrics(out, labels)
        tracker.update(metrics)
    return tracker.compute()


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _read_baseline_row(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("member") == BASELINE_MEMBER_NAME:
                return row
    return None


def _append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _format_delta_block(metrics: dict, baseline_row: dict | None) -> str:
    lines = [
        f"  Dice_WT          : {metrics['dice_wt']:.4f}",
        f"  Dice_TC          : {metrics['dice_tc']:.4f}",
        f"  Dice_ET          : {metrics['dice_et']:.4f}",
        f"  HD95_WT          : {metrics['hd95_wt']:.2f}",
    ]
    if baseline_row is None:
        lines.append(f"  (no {BASELINE_MEMBER_NAME} row in test_metrics.csv — skip delta)")
        return "\n".join(lines)
    try:
        b_wt = float(baseline_row["dice_wt"])
        b_tc = float(baseline_row["dice_tc"])
        b_et = float(baseline_row["dice_et"])
        lines += [
            f"  Δ Dice_WT vs naive M5 : {metrics['dice_wt'] - b_wt:+.4f}  (baseline {b_wt:.4f})",
            f"  Δ Dice_TC vs naive M5 : {metrics['dice_tc'] - b_tc:+.4f}  (baseline {b_tc:.4f})",
            f"  Δ Dice_ET vs naive M5 : {metrics['dice_et'] - b_et:+.4f}  (baseline {b_et:.4f})",
        ]
    except (KeyError, ValueError) as exc:
        lines.append(f"  (could not parse baseline row: {exc})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--variants", nargs="+",
        default=("uniform", "prior", "stacked"),
        choices=("uniform", "prior", "stacked"),
    )
    parser.add_argument("--m5-prior", type=float, default=0.5,
                        help="Only used for the 'prior' variant.")
    parser.add_argument("--no-csv", action="store_true",
                        help="Skip appending rows to test_metrics.csv "
                             "(useful for dry runs).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval-stacker] device: {device}")

    # ---- Data ---------------------------------------------------------------
    test_ds = CachedLogitsDataset(SPLITS_DIR / "test_ids.txt")
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print(f"[eval-stacker] test patients: {len(test_ds)}")

    # ---- Read training metadata for time/memory columns (only for 'stacked') -
    sidecar_path = CHECKPOINT_DIR_FIXED / f"{STACKER_CKPT_NAME}_train_meta.json"
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    else:
        sidecar = {}
    training_time_hrs = sidecar.get("training_time_hrs", float("nan"))
    # Stacker is a 15-scalar model — peak memory is dominated by the cached
    # logits in CPU RAM, not GPU. Report 0.0 GB to be honest.
    gpu_memory_gb = 0.0

    members_short = [s.short for s in MEMBER_SPECS]
    csv_path = TABLES_DIR_FIXED / "test_metrics.csv"
    baseline_row = _read_baseline_row(csv_path)

    # ---- Run every requested variant ---------------------------------------
    all_results: dict[str, dict] = {}
    for variant in args.variants:
        print(f"\n[eval-stacker] === variant={variant} ===")
        stacker = _build_stacker_for_variant(variant, members_short, args.m5_prior)

        weights_now = stacker.weights.detach().cpu()
        print(format_weight_table(weights_now, members_short))

        metrics = _evaluate_variant(stacker, test_loader, device)
        all_results[variant] = metrics

        row = {
            "member":       VARIANT_LABELS[variant],
            "modality":     "ensemble-late",
            "architecture": "PerSubregionStacker",
            "seed":         GLOBAL_SEED,
            "dice_wt": metrics.get("dice_wt", float("nan")),
            "dice_tc": metrics.get("dice_tc", float("nan")),
            "dice_et": metrics.get("dice_et", float("nan")),
            "iou_wt":  metrics.get("iou_wt",  float("nan")),
            "iou_tc":  metrics.get("iou_tc",  float("nan")),
            "iou_et":  metrics.get("iou_et",  float("nan")),
            "hd95_wt": metrics.get("hd95_wt", float("nan")),
            "hd95_tc": metrics.get("hd95_tc", float("nan")),
            "hd95_et": metrics.get("hd95_et", float("nan")),
            "training_time_hrs": training_time_hrs if variant == "stacked" else 0.0,
            "gpu_memory_gb":     gpu_memory_gb,
        }

        if not args.no_csv:
            _append_row(csv_path, row)
            print(f"[eval-stacker] appended row '{VARIANT_LABELS[variant]}' to {csv_path}")

        print()
        print("-" * 60)
        print(f" RESULT — {VARIANT_LABELS[variant]}")
        print("-" * 60)
        print(_format_delta_block(metrics, baseline_row))
        print("-" * 60)

    # ---- Summary across variants -------------------------------------------
    if len(all_results) > 1:
        print()
        print("=" * 70)
        print(" SUMMARY — all variants on test")
        print("=" * 70)
        print(f" {'variant':<28} {'WT':>8} {'TC':>8} {'ET':>8} {'mean':>8}")
        print("-" * 70)
        if baseline_row is not None:
            b = baseline_row
            try:
                b_wt, b_tc, b_et = float(b["dice_wt"]), float(b["dice_tc"]), float(b["dice_et"])
                b_mean = (b_wt + b_tc + b_et) / 3.0
                print(f" {'(baseline) M5_Multimodal_4ch':<28} "
                      f"{b_wt:>8.4f} {b_tc:>8.4f} {b_et:>8.4f} {b_mean:>8.4f}")
            except (KeyError, ValueError):
                pass
        for variant, metrics in all_results.items():
            wt = metrics.get("dice_wt", float("nan"))
            tc = metrics.get("dice_tc", float("nan"))
            et = metrics.get("dice_et", float("nan"))
            mean = (wt + tc + et) / 3.0
            print(f" {VARIANT_LABELS[variant]:<28} "
                  f"{wt:>8.4f} {tc:>8.4f} {et:>8.4f} {mean:>8.4f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
