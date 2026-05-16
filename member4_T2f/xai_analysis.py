"""Member 4 — T2f explainability: per-sub-region Grad-CAM + SHAP.

Produces per patient × per sub-region × per view:
  • 9 Grad-CAM PNGs per patient (3 sub-regions × 3 views) — 27 PNGs for 3 patients
  • Architecture comparison PNGs (ResUNet3D vs SegResNet, axial only)
  • SHAP importance maps per sub-region (run with --shap_bg N)
  • Pearson r agreement score between Grad-CAM and SHAP

Filename format (claude.md §3.9):
  gradcam_M4_{pid}_{subregion}_{view}.png
  shap_M4_{pid}_{subregion}_axial.png

Conventions (claude.md):
  • set_global_seed() first.
  • GradCAM3D always used as context manager (§2.5).
  • target_channel specified per sub-region — never full-channel average (§2.8).
  • test_metrics.csv guard enforces evaluate → xai order (§2.4).
  • Full-volume inputs for GradCAM (not patches).

Usage:
  python xai_analysis.py                          # Grad-CAM only, 3 patients
  python xai_analysis.py --n_patients 3 --shap_bg 20   # + SHAP (slow)
  python xai_analysis.py --arch resunet           # single arch
"""
from __future__ import annotations

# ── Seed first ────────────────────────────────────────────────────────────────
from shared.seed import set_global_seed
set_global_seed()

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    CHECKPOINT_DIR, FIGURES_DIR, PATCH_SIZE, SPLITS_DIR, TABLES_DIR,
    TARGET_CHANNEL_NAMES,
    get_data_root,
)
from shared.grad_cam_3d import GradCAM3D
from shared.preprocessing import preprocess_patient
from shared.trainer import CheckpointManager
from shared.visualization import plot_gradcam_three_view

from model import build_model

MODALITY    = "t2f"
LOCAL_FIGS  = RESULTS_ROOT / "T2F" / "figures"
LOCAL_FIGS.mkdir(parents=True, exist_ok=True)
LOCAL_TABS  = RESULTS_ROOT / "T2F" / "tables"
LOCAL_TABS.mkdir(parents=True, exist_ok=True)

# ── Sub-regions (frozen order — claude.md §3.4) ───────────────────────────────
SUBREGIONS = TARGET_CHANNEL_NAMES   # ("wt", "tc", "et")
VIEWS      = ("axial", "coronal", "sagittal")

BG_COLOR = "#0D1117"
TEAL     = "#5DCAA5"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _peak_slice(seg_3ch: np.ndarray, subregion: str) -> dict[str, int]:
    """Return {axial, coronal, sagittal} peak slice indices for a sub-region."""
    ch = {"wt": 0, "tc": 1, "et": 2}[subregion]
    mask = seg_3ch[ch]   # (D, H, W)
    if mask.sum() == 0:
        D, H, W = mask.shape
        return {"axial": D // 2, "coronal": H // 2, "sagittal": W // 2}
    return {
        "axial":    int(mask.sum(axis=(1, 2)).argmax()),
        "coronal":  int(mask.sum(axis=(0, 2)).argmax()),
        "sagittal": int(mask.sum(axis=(0, 1)).argmax()),
    }


def _extract_slice(arr: np.ndarray, view: str, idx: int) -> np.ndarray:
    if view == "axial":    return arr[idx]
    if view == "coronal":  return arr[:, idx, :]
    return arr[:, :, idx]


def _save_gradcam_figure(
    vol_np:     np.ndarray,   # (D, H, W)
    cam_np:     np.ndarray,   # (D, H, W)
    seg_3ch:    np.ndarray,   # (3, D, H, W)
    subregion:  str,
    view:       str,
    idx:        int,
    arch_name:  str,
    pid:        str,
    save_path:  Path,
) -> None:
    """3-panel: FLAIR | GT overlay | Grad-CAM overlay for one view/sub-region."""
    mri_s = _extract_slice(vol_np, view, idx)
    cam_s = _extract_slice(cam_np, view, idx)
    ch    = {"wt": 0, "tc": 1, "et": 2}[subregion]
    gt_s  = _extract_slice(seg_3ch[ch], view, idx)

    sr_colours = {"wt": "#5DCAA5", "tc": "#EF9F27", "et": "#FF4B4B"}
    sr_colour  = sr_colours[subregion]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=BG_COLOR)
    labels = [f"FLAIR — {view} z={idx}", f"GT ({subregion.upper()})", f"Grad-CAM ({subregion.upper()})"]

    for ax in axes:
        ax.set_facecolor(BG_COLOR)
        ax.imshow(mri_s.T, cmap="gray", origin="lower", interpolation="bilinear")
        ax.axis("off")

    if gt_s.sum() > 0:
        axes[1].contour(gt_s.T, levels=[0.5], colors=[sr_colour], linewidths=1.8)
    axes[2].imshow(cam_s.T, cmap="inferno", alpha=0.60, origin="lower", vmin=0, vmax=1)
    if gt_s.sum() > 0:
        axes[2].contour(gt_s.T, levels=[0.5], colors=[sr_colour], linewidths=1.4)

    for ax, lbl in zip(axes, labels):
        ax.set_title(lbl, color="white", fontsize=9, pad=4)

    fig.suptitle(f"M4 {arch_name} | {pid} | {subregion.upper()} | {view}",
                 color="white", fontsize=10, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


def _arch_comparison_figure(
    vol_np:    np.ndarray,
    seg_3ch:   np.ndarray,
    cam_dict:  dict[str, np.ndarray],   # arch_name → cam_np
    subregion: str,
    pid:       str,
    save_path: Path,
) -> None:
    """Side-by-side Grad-CAM per arch for one sub-region (axial only)."""
    ch  = {"wt": 0, "tc": 1, "et": 2}[subregion]
    z   = _peak_slice(seg_3ch, subregion)["axial"]
    mri = vol_np[z]
    gt  = seg_3ch[ch, z]

    n   = 1 + len(cam_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), facecolor=BG_COLOR)

    axes[0].imshow(mri.T, cmap="gray", origin="lower")
    if gt.sum() > 0:
        axes[0].contour(gt.T, levels=[0.5], colors=[TEAL], linewidths=1.8)
    axes[0].set_title(f"FLAIR z={z} | {subregion.upper()}", color="white", fontsize=10)
    axes[0].axis("off")

    cmaps = ["inferno", "plasma"]
    for idx, (arch_name, cam_np) in enumerate(cam_dict.items()):
        ax = axes[idx + 1]
        ax.imshow(mri.T, cmap="gray", origin="lower")
        ax.imshow(cam_np[z].T, cmap=cmaps[idx % 2], alpha=0.55, origin="lower", vmin=0, vmax=1)
        if gt.sum() > 0:
            ax.contour(gt.T, levels=[0.5], colors=[TEAL], linewidths=1.4)
        ax.set_title(f"Grad-CAM | {arch_name}", color="white", fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Arch comparison — {pid} | {subregion.upper()}", color="white", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_per_subregion(
    model:     torch.nn.Module,
    vol_t:     torch.Tensor,
    n_bg:      int,
    subregion: str,
) -> np.ndarray | None:
    """SHAP GradientExplainer for one sub-region. Returns (D,H,W) map or None."""
    try:
        import shap
    except ImportError:
        print("  [SHAP] pip install shap")
        return None

    ch = {"wt": 0, "tc": 1, "et": 2}[subregion]
    model.eval()

    # Background: near-zero random patches
    bg = torch.randn(n_bg, *vol_t.shape[1:]) * 0.05

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = model(x)
            return torch.sigmoid(logits[:, ch:ch+1]).squeeze(1).reshape(x.shape[0], -1)

    try:
        explainer   = shap.GradientExplainer(model, bg)
        shap_vals   = explainer.shap_values(vol_t)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_map = np.abs(shap_vals).squeeze()
        if shap_map.max() > 1e-9:
            shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min())
        return shap_map
    except Exception as e:
        print(f"  [SHAP] Error: {e}")
        return None


def _save_shap_figure(
    vol_np:    np.ndarray,
    cam_np:    np.ndarray,
    shap_map:  np.ndarray,
    seg_3ch:   np.ndarray,
    subregion: str,
    pearson_r: float,
    arch_name: str,
    pid:       str,
    save_path: Path,
) -> None:
    ch  = {"wt": 0, "tc": 1, "et": 2}[subregion]
    z   = _peak_slice(seg_3ch, subregion)["axial"]
    mri = vol_np[z]
    gt  = seg_3ch[ch, z]
    sr  = subregion.upper()
    lvl = "HIGH" if pearson_r > 0.7 else "MODERATE" if pearson_r > 0.4 else "LOW"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG_COLOR)
    titles = [f"FLAIR + GT ({sr})", f"Grad-CAM ({sr})", f"SHAP ({sr})"]
    maps   = [None, cam_np[z], shap_map[z] if shap_map.ndim == 3 else shap_map]
    cmaps  = ["gray", "inferno", "viridis"]
    alphas = [1.0, 0.55, 0.55]

    for ax, title, hmap, cmap, alpha in zip(axes, titles, maps, cmaps, alphas):
        ax.set_facecolor(BG_COLOR)
        ax.imshow(mri.T, cmap="gray", origin="lower")
        if hmap is not None:
            ax.imshow(hmap.T, cmap=cmap, alpha=alpha, origin="lower", vmin=0, vmax=1)
        if gt.sum() > 0:
            ax.contour(gt.T, levels=[0.5], colors=[TEAL], linewidths=1.5)
        ax.set_title(title, color="white", fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"{pid} | {arch_name} | {sr} | Pearson r={pearson_r:.3f} ({lvl} agreement)",
        color="white", fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Contamination guard (claude.md §2.4) ─────────────────────────────────
    _guard = TABLES_DIR / "test_metrics.csv"
    if not _guard.exists():
        raise RuntimeError(
            f"\n\ntest_metrics.csv not found at {_guard}\n"
            "Run evaluate.py BEFORE xai_analysis.py to prevent test-set peeking."
        )

    parser = argparse.ArgumentParser(description="M4 XAI analysis")
    parser.add_argument("--arch",       default="both",
                        choices=["resunet", "segresnet", "both"])
    parser.add_argument("--n_patients", type=int, default=3,
                        help="Number of test patients (first/middle/last from test_ids)")
    parser.add_argument("--shap_bg",   type=int, default=0,
                        help="SHAP background samples (0=skip SHAP, slow otherwise)")
    args = parser.parse_args()

    archs     = ["resunet", "segresnet"] if args.arch == "both" else [args.arch]
    data_root = get_data_root()

    # ── Test patient IDs — first / middle / last (deterministic, diverse) ────
    with open(SPLITS_DIR / "test_ids.txt") as f:
        all_ids = [l.strip() for l in f if l.strip()]
    n     = min(args.n_patients, len(all_ids))
    idxs  = [0, len(all_ids) // 2, len(all_ids) - 1][:n]
    pids  = [all_ids[i] for i in idxs]
    print(f"[XAI] Patients: {pids}")

    # ── Load checkpoints ──────────────────────────────────────────────────────
    models: dict[str, torch.nn.Module] = {}
    for arch in archs:
        ckpt_name = f"member4_T2f_{arch}"
        model     = build_model(arch, in_channels=1, out_channels=3)
        try:
            mgr              = CheckpointManager(str(CHECKPOINT_DIR), ckpt_name)
            model, _, ep, _  = mgr.load_best(model)
            model.eval()
            models[model.arch_name] = model
            print(f"[XAI] Loaded {arch} epoch={ep}")
        except FileNotFoundError:
            print(f"[XAI] No checkpoint for '{arch}' — skipping")

    if not models:
        print("[XAI] No models loaded. Run train.py first.")
        return

    pearson_records = []

    # ── Per-patient analysis ──────────────────────────────────────────────────
    for pid in tqdm(pids, desc="Patients"):
        patient_dir = str(Path(data_root) / pid)
        print(f"\n[XAI] Patient: {pid}")

        try:
            vol, seg = preprocess_patient(patient_dir, MODALITY)
        except Exception as e:
            print(f"  [skip] preprocess failed: {e}")
            continue

        vol_np  = vol.squeeze()       # (D, H, W)
        seg_3ch = seg                 # (3, D, H, W) — WT/TC/ET channels
        vol_t   = torch.tensor(vol[np.newaxis]).float()   # (1, 1, D, H, W)

        # cam_dict_per_sr: arch_name → {sr: cam_np}
        cam_store: dict[str, dict[str, np.ndarray]] = {}

        for arch_name, model in models.items():
            print(f"  → {arch_name}")
            layers = model.get_encoder_layers()
            if not layers:
                print("    [Grad-CAM] No encoder layers — skip")
                continue
            target_layer = layers[-1]   # enc4 — deepest, most semantic
            cam_store[arch_name] = {}

            for sr in SUBREGIONS:
                # ── Grad-CAM per sub-region (claude.md §2.8) ─────────────────
                with GradCAM3D(model, target_layer) as cam_gen:
                    cam_np = cam_gen.generate(vol_t, target_channel=sr)

                cam_store[arch_name][sr] = cam_np

                # 9 PNGs per patient: 3 sub-regions × 3 views
                for view in VIEWS:
                    idx  = _peak_slice(seg_3ch, sr)[view]
                    name = f"gradcam_M4_{pid}_{sr}_{view}.png"
                    _save_gradcam_figure(
                        vol_np, cam_np, seg_3ch,
                        subregion=sr, view=view, idx=idx,
                        arch_name=arch_name, pid=pid,
                        save_path=LOCAL_FIGS / name,
                    )

                # ── SHAP per sub-region ───────────────────────────────────────
                if args.shap_bg > 0:
                    print(f"    [SHAP] {sr.upper()} with {args.shap_bg} bg samples …")
                    shap_map = compute_shap_per_subregion(
                        model, vol_t, args.shap_bg, sr
                    )
                    if shap_map is not None:
                        from scipy.stats import pearsonr
                        r, p = pearsonr(cam_np.flatten(), shap_map.flatten())
                        lvl  = "HIGH" if r > 0.7 else "MODERATE" if r > 0.4 else "LOW"
                        print(f"    Grad-CAM vs SHAP {sr.upper()}: r={r:.4f} ({lvl})")
                        pearson_records.append({
                            "patient":    pid,
                            "arch":       arch_name,
                            "subregion":  sr,
                            "pearson_r":  round(r, 4),
                            "p_value":    round(float(p), 6),
                            "agreement":  lvl,
                        })
                        shap_save = LOCAL_FIGS / f"shap_M4_{pid}_{sr}_axial.png"
                        _save_shap_figure(
                            vol_np, cam_np, shap_map, seg_3ch,
                            subregion=sr, pearson_r=r,
                            arch_name=arch_name, pid=pid,
                            save_path=shap_save,
                        )

        # ── Architecture comparison (one per sub-region) ──────────────────────
        if len(cam_store) >= 2:
            for sr in SUBREGIONS:
                cam_dict_for_sr = {
                    arch: cams[sr]
                    for arch, cams in cam_store.items()
                    if sr in cams
                }
                if len(cam_dict_for_sr) >= 2:
                    comp_save = LOCAL_FIGS / f"M4_arch_comparison_{pid}_{sr}.png"
                    _arch_comparison_figure(
                        vol_np, seg_3ch, cam_dict_for_sr, sr, pid, comp_save
                    )

    # ── SHAP agreement summary ────────────────────────────────────────────────
    if pearson_records:
        import pandas as pd
        df = pd.DataFrame(pearson_records)
        df.to_csv(LOCAL_TABS / "M4_shap_agreement.csv", index=False)
        print(f"\nSHAP agreement: {LOCAL_TABS}/M4_shap_agreement.csv")
        print(df.to_string(index=False))

        # Paragraph-length summary for the report (as a comment in the script)
        # -------------------------------------------------------------------
        # "For the FLAIR model, Grad-CAM and SHAP showed [HIGH/MODERATE/LOW]
        # spatial agreement (mean Pearson r = X.XX ± Y.YY). Agreement was
        # strongest for WT (r=...) and weakest for ET (r=...), consistent with
        # the class-imbalance challenge for the smallest sub-region. Where
        # agreement was high, Grad-CAM is validated as a reliable proxy for
        # FLAIR-based tumour localisation. Where agreement was low (ET), SHAP
        # is the more trustworthy explainability method."
        # -------------------------------------------------------------------
        mean_r = df["pearson_r"].mean()
        print(f"\n  Mean Pearson r (all sub-regions): {mean_r:.4f}")
        level = "HIGH" if mean_r > 0.7 else "MODERATE" if mean_r > 0.4 else "LOW"
        print(f"  Overall XAI agreement: {level}")

    print(f"\n[XAI] Figures saved to: {LOCAL_FIGS}")
    print(f"[XAI] Expected files: {len(pids)} patients × {len(SUBREGIONS)} sub-regions"
          f" × {len(VIEWS)} views = {len(pids)*len(SUBREGIONS)*len(VIEWS)} Grad-CAM PNGs")
    print("[XAI] Done ✓")


if __name__ == "__main__":
    main()
