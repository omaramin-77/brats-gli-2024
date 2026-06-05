"""Fit the PerSubregionStacker on val from cached logits.

Consumes:
  results/predictions/{m1..m5}/{pid}.pt   — (3, 128, 128, 128) fp16 logits
  results/predictions/labels/{pid}.pt     — (3, 128, 128, 128) uint8 target,
                                            in the BraTSDataset brain-cropped
                                            frame (same coords as predictions)

Produces:
  results/checkpoints/LateEnsemble_stacker_best.pt
  results/checkpoints/LateEnsemble_stacker_train_meta.json
  results/figures/training_curves_LateEnsemble.png

Why val and not train:
  M1–M5 were trained on the train split, so their *train-split* predictions
  are over-confident and not representative of generalisation. Fitting the
  15 stacking weights on val gives an unbiased estimate of how to combine
  their *test-time* predictions. Val is small (35 patients) but the
  stacker has so few parameters (15 logits) that overfitting is unlikely.

CLI
---
    python LateEnsemble/train_stacker.py
    python LateEnsemble/train_stacker.py --epochs 100 --lr 0.1
    python LateEnsemble/train_stacker.py --m5-prior 0.6
    python LateEnsemble/train_stacker.py --init uniform   # control
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
import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.optim.lr_scheduler import CosineAnnealingLR  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from shared.config import (  # noqa: E402
    GLOBAL_SEED,
    MLRUNS_DIR,
    RESULTS_DIR,
    SPLITS_DIR,
)
from shared.metrics import compute_all_metrics, MetricTracker  # noqa: E402
from shared.trainer import dice_bce_loss  # noqa: E402
from shared.visualization import plot_training_curves  # noqa: E402

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


MEMBER_NAME = "LateEnsemble_stacker"
MLFLOW_RUN_NAME = "LateEnsemble-XAI-stacked-seed42"

PREDICTIONS_ROOT = RESULTS_DIR / "predictions"
# Aligned label cache produced by LateEnsemble/cache_predictions.py.
# Do NOT point this at results/features/labels/ — that older cache was
# built by ensemble/cache_labels.py from raw-resized NIfTI without the
# brain-bbox crop, putting labels in a different coordinate system from
# the predictions. label/pred IoU on TC/ET drops to ~0.06 under that
# mismatch and the stacker plateaus at WT~0.35.
LABELS_DIR = PREDICTIONS_ROOT / "labels"
CHECKPOINT_DIR_FIXED = RESULTS_DIR / "checkpoints"
FIGURES_DIR_FIXED = RESULTS_DIR / "figures"
TABLES_DIR_FIXED = RESULTS_DIR / "tables"


# ---------------------------------------------------------------------------
# Dataset over cached logits + cached labels
# ---------------------------------------------------------------------------
class CachedLogitsDataset(Dataset):
    """Per patient: stack M1–M5 logits along a new axis; load the matching label.

    Returns:
        {
            "logits": (M, 3, 128, 128, 128) float32 tensor,
            "label":  (3, 128, 128, 128) float32 binary tensor,
            "patient_id": str,
        }

    Logits are cast back to float32 here because the loss/metrics on a
    few patients of fp16 produced small but observable HD95 instabilities
    in earlier tests.
    """

    def __init__(self, split_file: Path):
        path = Path(split_file)
        if not path.exists():
            raise FileNotFoundError(f"split file missing: {path}")
        self.patient_ids = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Pre-flight: confirm every (member, patient) prediction AND label exist.
        missing: list[str] = []
        for pid in self.patient_ids:
            for spec in MEMBER_SPECS:
                fpath = PREDICTIONS_ROOT / spec.short / f"{pid}.pt"
                if not fpath.exists():
                    missing.append(str(fpath))
            lab = LABELS_DIR / f"{pid}.pt"
            if not lab.exists():
                missing.append(str(lab))
        if missing:
            raise FileNotFoundError(
                f"CachedLogitsDataset: {len(missing)} files missing. "
                f"First missing: {missing[0]}.\n"
                "Run LateEnsemble/cache_predictions.py first. If predictions "
                "already exist but labels do not, use "
                "`python LateEnsemble/cache_predictions.py --labels-only`."
            )

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _load_label(self, pid: str) -> torch.Tensor:
        """Load the aligned WT/TC/ET label for ``pid``.

        Only reads from LABELS_DIR (the BraTSDataset-aligned cache produced
        by LateEnsemble/cache_predictions.py). The older
        results/features/labels/ cache from ensemble/cache_labels.py is
        deliberately NOT used as a fallback — its labels are in a different
        coordinate frame (raw-resize, no brain crop) and produce IoU ~0.06
        on TC/ET against the predictions. If the right file isn't here, we
        raise loudly rather than silently train on broken data.
        """
        cache_path = LABELS_DIR / f"{pid}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Aligned label for {pid} not found at {cache_path}. "
                "Run `python LateEnsemble/cache_predictions.py --labels-only` "
                "to populate it."
            )
        return torch.load(cache_path, map_location="cpu").float()

    def __getitem__(self, idx: int) -> dict:
        pid = self.patient_ids[idx]
        per_member = []
        for spec in MEMBER_SPECS:
            t = torch.load(PREDICTIONS_ROOT / spec.short / f"{pid}.pt",
                           map_location="cpu")
            per_member.append(t.float())                  # (3, 128, 128, 128)
        logits = torch.stack(per_member, dim=0)           # (M, 3, 128, 128, 128)
        label = self._load_label(pid)
        return {"logits": logits, "label": label, "patient_id": pid}


# ---------------------------------------------------------------------------
# Train / validate loops
# ---------------------------------------------------------------------------
def _train_one_epoch(stacker, loader, optimizer, device, loss_fn):
    stacker.train()
    total, n = 0.0, 0
    for batch in loader:
        logits_in = batch["logits"].to(device, non_blocking=True)   # (B, M, S, H, W, D)
        labels = batch["label"].to(device, non_blocking=True)       # (B, S, H, W, D)
        optimizer.zero_grad(set_to_none=True)
        out = stacker(logits_in)                                    # (B, S, H, W, D)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n += 1
    return {"loss": total / max(n, 1)}


@torch.no_grad()
def _validate(stacker, loader, device, loss_fn):
    """Same data is used as 'val' here (we fit on val); this is just bookkeeping
    so the training curve is interpretable. Honest test numbers come from
    evaluate_stacker.py.
    """
    stacker.eval()
    tracker = MetricTracker()
    total, n = 0.0, 0
    for batch in loader:
        logits_in = batch["logits"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        out = stacker(logits_in)
        loss = loss_fn(out, labels)
        metrics = compute_all_metrics(out, labels)
        metrics["loss"] = float(loss.item())
        tracker.update(metrics)
        total += float(loss.item())
        n += 1
    agg = tracker.compute()
    agg.setdefault("loss", total / max(n, 1))
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05,
                        help="LR for the 15 stacking logits. Higher than the "
                             "unimodal training LRs because we have only 15 "
                             "params and few data points.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-volume batches; volumes are 128³×3×5 fp32 "
                             "which is ~150 MB each.")
    parser.add_argument("--m5-prior", type=float, default=0.5,
                        help="Probability mass reserved for M5 at init.")
    parser.add_argument(
        "--init", choices=("xai", "uniform"), default="xai",
        help="Initial weights. 'xai' uses ablation scores + m5-prior; "
             "'uniform' is the control variant (all weights 1/M).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stacker] device: {device}")

    # ---- Data ---------------------------------------------------------------
    val_ds = CachedLogitsDataset(SPLITS_DIR / "val_ids.txt")
    print(f"[stacker] val patients: {len(val_ds)}")
    pin = device.type == "cuda"
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=pin, drop_last=False,
    )
    # No separate "validation" loader — we fit on val. We re-use the same
    # loader (without shuffle) for in-training metric reporting.
    val_loader_metric = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=pin, drop_last=False,
    )

    # ---- Model --------------------------------------------------------------
    members_short = [s.short for s in MEMBER_SPECS]
    if args.init == "xai":
        importance_path = TABLES_DIR_FIXED / "modality_importance_scores.json"
        if not importance_path.exists():
            raise FileNotFoundError(
                f"modality_importance_scores.json not found at {importance_path}. "
                "Run member5_multimodal/ablation.py first."
            )
        init_logits = build_xai_init_logits(
            importance_json=importance_path,
            members_short=members_short,
            ablation_keys=SHORT_TO_ABLATION_KEY,
            m5_prior=args.m5_prior,
        )
    else:
        init_logits = build_uniform_logits(NUM_MEMBERS, 3)

    stacker = PerSubregionStacker(
        num_members=NUM_MEMBERS,
        num_subregions=3,
        init_logits=init_logits,
    ).to(device)

    n_params = sum(p.numel() for p in stacker.parameters())
    print(f"[stacker] parameters: {n_params}  init={args.init}")
    print(f"[stacker] initial weights:\n{format_weight_table(stacker.weights.detach().cpu(), members_short)}")

    optimizer = torch.optim.Adam(stacker.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- MLflow (best-effort) ----------------------------------------------
    try:
        import mlflow  # type: ignore
        mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
        mlflow.set_experiment(MEMBER_NAME)
        mlflow_run = mlflow.start_run(run_name=MLFLOW_RUN_NAME)
    except ImportError:
        mlflow = None
        mlflow_run = None

    if mlflow is not None:
        mlflow.log_param("member",      MEMBER_NAME)
        mlflow.log_param("seed",        GLOBAL_SEED)
        mlflow.log_param("epochs",      args.epochs)
        mlflow.log_param("lr",          args.lr)
        mlflow.log_param("batch_size",  args.batch_size)
        mlflow.log_param("init",        args.init)
        mlflow.log_param("m5_prior",    args.m5_prior)

    # ---- Training loop ------------------------------------------------------
    history = defaultdict(list)
    best_mean_dice = -1.0
    train_start = time.time()

    try:
        for epoch in range(1, args.epochs + 1):
            tr = _train_one_epoch(stacker, val_loader, optimizer, device, dice_bce_loss)
            history["train_loss"].append(tr["loss"])
            scheduler.step()

            # Cheap metric pass on the same set every 5 epochs (and the last).
            if epoch % 5 == 0 or epoch == args.epochs:
                m = _validate(stacker, val_loader_metric, device, dice_bce_loss)
                mean_dice = float(np.mean([
                    m.get("dice_wt", float("nan")),
                    m.get("dice_tc", float("nan")),
                    m.get("dice_et", float("nan")),
                ]))
                print(
                    f"[stacker] epoch {epoch:3d}  "
                    f"train_loss={tr['loss']:.4f}  "
                    f"val WT={m['dice_wt']:.4f} TC={m['dice_tc']:.4f} "
                    f"ET={m['dice_et']:.4f}  mean={mean_dice:.4f}"
                )
                for key in ("dice_wt", "dice_tc", "dice_et",
                            "iou_wt", "iou_tc", "iou_et",
                            "hd95_wt", "hd95_tc", "hd95_et"):
                    history[f"val_{key}"].append(m.get(key, float("nan")))
                history["val_mean_dice"].append(mean_dice)
                history["lr"].append(optimizer.param_groups[0]["lr"])

                if mlflow is not None:
                    mlflow.log_metric("train_loss", tr["loss"], step=epoch)
                    for k, v in m.items():
                        mlflow.log_metric(f"val_{k}", v, step=epoch)
                    mlflow.log_metric("val_mean_dice", mean_dice, step=epoch)

                # Best-checkpoint selection on mean(WT, TC, ET) — see the
                # write-up's "Day 0" note: the previous bug was picking the
                # WT-only peak, which on this kind of fast-saturating model
                # captured the worst TC/ET. Mean is the right selector here.
                if mean_dice > best_mean_dice:
                    best_mean_dice = mean_dice
                    CHECKPOINT_DIR_FIXED.mkdir(parents=True, exist_ok=True)
                    best_path = CHECKPOINT_DIR_FIXED / f"{MEMBER_NAME}_best.pt"
                    torch.save({
                        "epoch": epoch,
                        "val_mean_dice": mean_dice,
                        "val_metrics": dict(m),
                        "init": args.init,
                        "m5_prior": args.m5_prior,
                        "members_short": members_short,
                        "stacker_state": stacker.state_dict(),
                    }, best_path)
                    print(
                        f"[stacker] new best -> {best_path.name} "
                        f"(mean_dice={mean_dice:.4f})"
                    )
    finally:
        elapsed_hrs = (time.time() - train_start) / 3600.0
        print(f"[stacker] training time: {elapsed_hrs:.4f} h")

        # Sidecar for evaluate_stacker.py.
        CHECKPOINT_DIR_FIXED.mkdir(parents=True, exist_ok=True)
        sidecar = CHECKPOINT_DIR_FIXED / f"{MEMBER_NAME}_train_meta.json"
        sidecar.write_text(json.dumps({
            "training_time_hrs": elapsed_hrs,
            "best_val_mean_dice": best_mean_dice,
            "init": args.init,
            "m5_prior": args.m5_prior,
            "members_short": members_short,
        }, indent=2), encoding="utf-8")

        # Training curves.
        FIGURES_DIR_FIXED.mkdir(parents=True, exist_ok=True)
        curves_path = FIGURES_DIR_FIXED / f"training_curves_{MEMBER_NAME}.png"
        plot_training_curves(dict(history), MEMBER_NAME, save_path=str(curves_path))
        print(f"[stacker] saved training curves to {curves_path}")
        if mlflow is not None:
            mlflow.log_artifact(str(curves_path))
            if mlflow_run is not None:
                mlflow.end_run()

        # Print final weight table.
        final_weights = stacker.weights.detach().cpu()
        print()
        print("=" * 60)
        print(" FINAL STACKER WEIGHTS")
        print("=" * 60)
        print(format_weight_table(final_weights, members_short))
        print("=" * 60)
        print(f" best val mean(WT,TC,ET) = {best_mean_dice:.4f}")
        print(f" checkpoint: {CHECKPOINT_DIR_FIXED / (MEMBER_NAME + '_best.pt')}")
        print("=" * 60)


if __name__ == "__main__":
    main()
