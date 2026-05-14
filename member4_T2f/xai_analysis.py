"""Member 4 — T2f explainability: Grad-CAM + SHAP on both architectures.

Produces, for each architecture × each test patient:
  1. Grad-CAM overlay (axial / coronal / sagittal) — fast
  2. SHAP importance map (run overnight with --shap_bg 20)
  3. Side-by-side Grad-CAM comparison: ResUNet3D vs SegResNet

Also computes Pearson r agreement between Grad-CAM and SHAP maps.

Usage:
  python xai_analysis.py                    # Grad-CAM only, 3 patients
  python xai_analysis.py --shap_bg 20       # adds SHAP (slow)
  python xai_analysis.py --n_patients 5     # more patients
  python xai_analysis.py --arch resunet     # single arch
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless on Kaggle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
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

from shared.seed import set_global_seed
set_global_seed()

from shared.config import (
    get_data_root,
    SPLITS_DIR,
    CHECKPOINT_DIR,
    PATCH_SIZE,
)
from shared.preprocessing import preprocess_patient
from shared.trainer import CheckpointManager

from model import build_model

MODALITY    = "t2f"
FIGURES_DIR = RESULTS_ROOT / "T2F" / "figures"
TABLES_DIR  = RESULTS_ROOT / "T2F" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ── Test-set contamination guard ──────────────────────────────────────────────
_metrics_csv = TABLES_DIR / "test_metrics.csv"
if not _metrics_csv.exists():
    raise RuntimeError(
        "\n\ntest_metrics.csv not found.\n"
        "Run evaluate.py BEFORE xai_analysis.py.\n"
        f"Expected: {_metrics_csv}\n"
    )

# Colour palette
TEAL   = "#5DCAA5"
AMBER  = "#EF9F27"
BG     = "#0D1117"
PANEL  = "#161B22"


# ══════════════════════════════════════════════════════════════════════════════
# Grad-CAM
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM3D:
    """Lightweight Grad-CAM context manager for 3D models."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model  = model
        self._feats: dict = {}
        self._grads: dict = {}
        self._h1    = target_layer.register_forward_hook(
            lambda m, i, o: self._feats.update({"f": o})
        )
        self._h2    = target_layer.register_full_backward_hook(
            lambda m, gi, go: self._grads.update({"g": go[0]})
        )

    def generate(self, vol_tensor: torch.Tensor) -> np.ndarray:
        """Return (D,H,W) heatmap in [0,1]."""
        self.model.zero_grad()
        pred  = self.model(vol_tensor)
        prob  = torch.sigmoid(pred)
        score = (prob * (prob > 0.5).float()).sum()
        score.backward()

        weights = self._grads["g"].mean(dim=(2, 3, 4), keepdim=True)
        cam     = F.relu((weights * self._feats["f"]).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=vol_tensor.shape[2:],
                                mode="trilinear", align_corners=False)
        cam_np  = cam.squeeze().detach().cpu().numpy()

        if cam_np.max() > 1e-6:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        return cam_np

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._h1.remove()
        self._h2.remove()


def get_gradcam_layer(model: torch.nn.Module):
    """Return the last encoder layer to hook for Grad-CAM."""
    layers = model.get_encoder_layers()
    if layers:
        return layers[-1]
    # Fallback for SegResNet MONAI internals
    try:
        return list(model._net.encoder)[-1]
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap(model, vol_tensor: torch.Tensor, n_bg: int = 20) -> np.ndarray | None:
    """Compute SHAP GradientExplainer importance map. Returns (D,H,W) or None."""
    try:
        import shap
    except ImportError:
        print("  [SHAP] shap not installed — pip install shap")
        return None

    model.eval()
    background = [
        torch.randn_like(vol_tensor) * 0.1  # near-zero background samples
        for _ in range(n_bg)
    ]
    background_t = torch.cat(background, dim=0)  # (n_bg, 1, D, H, W)

    def model_fn(x):
        with torch.no_grad():
            return torch.sigmoid(model(x)).squeeze(1).reshape(x.shape[0], -1)

    try:
        explainer   = shap.GradientExplainer(model, background_t)
        shap_values = explainer.shap_values(vol_tensor)  # list or array
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_map = np.abs(shap_values).squeeze()
        if shap_map.max() > 1e-9:
            shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min())
        return shap_map
    except Exception as e:
        print(f"  [SHAP] Failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _best_slices(seg_np: np.ndarray) -> tuple[int, int, int]:
    tumour = (seg_np > 0)
    z = int(tumour.sum(axis=(1, 2)).argmax()) if tumour.any() else seg_np.shape[0] // 2
    y = int(tumour.sum(axis=(0, 2)).argmax()) if tumour.any() else seg_np.shape[1] // 2
    x = int(tumour.sum(axis=(0, 1)).argmax()) if tumour.any() else seg_np.shape[2] // 2
    return z, y, x


def plot_gradcam_overlay(
    vol_np: np.ndarray,
    seg_np: np.ndarray,
    cam_np: np.ndarray,
    title:  str,
    save_path: Path,
) -> None:
    """3×2 grid: (row0=MRI+GT, row1=Grad-CAM) × (axial, coronal, sagittal)."""
    z, y, x = _best_slices(seg_np)
    tumour   = (seg_np > 0).astype(np.float32)

    views = [
        ("Axial",    vol_np[z],        tumour[z],        cam_np[z]),
        ("Coronal",  vol_np[:, y, :],  tumour[:, y, :],  cam_np[:, y, :]),
        ("Sagittal", vol_np[:, :, x],  tumour[:, :, x],  cam_np[:, :, x]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=BG)
    fig.suptitle(title, color="white", fontsize=13, y=0.98)

    for col, (name, mri, gt, heat) in enumerate(views):
        for row in range(2):
            ax = axes[row, col]
            ax.set_facecolor(PANEL)
            ax.imshow(mri.T, cmap="gray", origin="lower", interpolation="bilinear")

            if row == 1:           # Grad-CAM row
                ax.imshow(heat.T, cmap="inferno", alpha=0.55,
                          origin="lower", vmin=0, vmax=1)

            if gt.sum() > 0:
                ax.contour(gt.T, levels=[0.5], colors=[TEAL], linewidths=1.4)

            label = name if row == 0 else f"Grad-CAM | {name}"
            ax.set_title(label, color="white", fontsize=9, pad=3)
            ax.axis("off")

    # Row labels
    for row, label in enumerate(["Raw FLAIR + GT contour", "Grad-CAM overlay"]):
        axes[row, 0].set_ylabel(label, color="#aaa", fontsize=8, rotation=90, labelpad=5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_arch_comparison(
    vol_np:    np.ndarray,
    seg_np:    np.ndarray,
    cam_dict:  dict,          # {arch_name: cam_np}
    patient_id: str,
    save_path:  Path,
) -> None:
    """Side-by-side Grad-CAM: ResUNet3D vs SegResNet (axial only)."""
    z = _best_slices(seg_np)[0]
    tumour = (seg_np > 0).astype(np.float32)
    mri_z  = vol_np[z]
    gt_z   = tumour[z]

    n_cols = 1 + len(cam_dict)        # MRI | arch1 | arch2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), facecolor=BG)

    # MRI panel
    axes[0].imshow(mri_z.T, cmap="gray", origin="lower")
    if gt_z.sum() > 0:
        axes[0].contour(gt_z.T, levels=[0.5], colors=[TEAL], linewidths=1.5)
    axes[0].set_title(f"FLAIR — axial z={z}", color="white", fontsize=10)
    axes[0].axis("off")

    colours = ["inferno", "plasma"]
    for idx, (arch_name, cam_np) in enumerate(cam_dict.items()):
        ax = axes[idx + 1]
        ax.imshow(mri_z.T, cmap="gray", origin="lower")
        ax.imshow(cam_np[z].T, cmap=colours[idx % 2], alpha=0.55,
                  origin="lower", vmin=0, vmax=1)
        if gt_z.sum() > 0:
            ax.contour(gt_z.T, levels=[0.5], colors=[TEAL], linewidths=1.5)
        ax.set_title(f"Grad-CAM | {arch_name}", color="white", fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Arch Comparison — {patient_id}", color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_shap_vs_gradcam(
    vol_np:     np.ndarray,
    seg_np:     np.ndarray,
    cam_np:     np.ndarray,
    shap_map:   np.ndarray,
    pearson_r:  float,
    arch_name:  str,
    patient_id: str,
    save_path:  Path,
) -> None:
    """3-panel: MRI | Grad-CAM | SHAP with agreement score."""
    z      = _best_slices(seg_np)[0]
    tumour = (seg_np > 0).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    titles = ["FLAIR + GT", f"Grad-CAM ({arch_name})", f"SHAP ({arch_name})"]
    maps   = [None, cam_np[z], shap_map[z] if shap_map.ndim == 3 else shap_map]
    cmaps  = ["gray", "inferno", "viridis"]
    alphas = [1.0, 0.55, 0.55]

    for ax, title, hmap, cmap, alpha in zip(axes, titles, maps, cmaps, alphas):
        ax.imshow(vol_np[z].T, cmap="gray", origin="lower")
        if hmap is not None:
            ax.imshow(hmap.T, cmap=cmap, alpha=alpha, origin="lower", vmin=0, vmax=1)
        if tumour[z].sum() > 0:
            ax.contour(tumour[z].T, levels=[0.5], colors=[TEAL], linewidths=1.5)
        ax.set_title(title, color="white", fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"{patient_id} | Pearson r = {pearson_r:.3f} "
        f"({'HIGH' if pearson_r > 0.7 else 'MODERATE' if pearson_r > 0.4 else 'LOW'} agreement)",
        color="white", fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="M4 XAI analysis")
    parser.add_argument("--arch",       default="both",
                        choices=["resunet", "segresnet", "both"])
    parser.add_argument("--n_patients", type=int, default=3,
                        help="Number of test patients to analyse")
    parser.add_argument("--shap_bg",    type=int, default=0,
                        help="SHAP background samples (0 = skip SHAP)")
    args = parser.parse_args()

    archs      = ["resunet", "segresnet"] if args.arch == "both" else [args.arch]
    device     = torch.device("cpu")   # XAI always on CPU
    data_root  = get_data_root()
    patch_size = PATCH_SIZE[0]

    # ── Load test patient IDs ─────────────────────────────────────────────────
    test_ids_file = SPLITS_DIR / "test_ids.txt"
    with open(test_ids_file) as f:
        test_ids = [l.strip() for l in f if l.strip()]
    test_ids = test_ids[: args.n_patients]
    print(f"[XAI] Analysing {len(test_ids)} test patient(s)")

    # ── Load checkpoints ──────────────────────────────────────────────────────
    models = {}
    for arch in archs:
        ckpt_name = f"member4_T2f_{arch}"
        model     = build_model(arch, in_channels=1)
        try:
            manager = CheckpointManager(str(CHECKPOINT_DIR), ckpt_name)
            model, _, epoch, val_dice = manager.load_best(model)
            model.eval().to(device)
            models[arch] = model
            print(f"[XAI] Loaded {arch} (epoch={epoch}, val_dice={val_dice:.4f})")
        except FileNotFoundError:
            print(f"[XAI] No checkpoint for '{arch}' — skipping")

    if not models:
        print("[XAI] No models loaded. Run train.py first.")
        return

    pearson_records = []

    # ── Per-patient analysis ──────────────────────────────────────────────────
    for pid in tqdm(test_ids, desc="Patients"):
        patient_dir = str(Path(data_root) / pid)
        print(f"\n[XAI] Patient: {pid}")

        # Preprocess FLAIR volume
        try:
            vol, seg = preprocess_patient(patient_dir, MODALITY)
        except Exception as e:
            print(f"  [skip] preprocess failed: {e}")
            continue

        vol_np  = vol.squeeze()
        seg_np  = seg
        vol_t   = torch.tensor(vol[np.newaxis]).float()  # (1,1,D,H,W)

        cam_dict = {}   # arch_name → cam_np for comparison plot

        for arch_name, model in models.items():
            print(f"  → {model.arch_name}")

            # ── Grad-CAM ─────────────────────────────────────────────────────
            target_layer = get_gradcam_layer(model)
            if target_layer is None:
                print("    [Grad-CAM] No target layer found — skipping")
                continue

            with GradCAM3D(model, target_layer) as cam_gen:
                cam_np = cam_gen.generate(vol_t)

            cam_dict[model.arch_name] = cam_np

            # Save Grad-CAM overlay
            save_name = f"M4_gradcam_{arch_name}_{pid}_combined.png"
            plot_gradcam_overlay(
                vol_np, seg_np, cam_np,
                title     = f"M4 Grad-CAM | {model.arch_name} | {pid}",
                save_path = FIGURES_DIR / save_name,
            )

            # ── SHAP ─────────────────────────────────────────────────────────
            if args.shap_bg > 0:
                print(f"    [SHAP] running with {args.shap_bg} background samples …")
                shap_map = compute_shap(model, vol_t, n_bg=args.shap_bg)

                if shap_map is not None:
                    from scipy.stats import pearsonr
                    r, p = pearsonr(cam_np.flatten(), shap_map.flatten())
                    level = "HIGH" if r > 0.7 else "MODERATE" if r > 0.4 else "LOW"
                    print(f"    Grad-CAM vs SHAP: Pearson r={r:.4f} (p={p:.2e}) → {level}")
                    pearson_records.append({
                        "patient":    pid,
                        "arch":       model.arch_name,
                        "pearson_r":  round(r, 4),
                        "p_value":    round(p, 6),
                        "agreement":  level,
                    })

                    shap_save = FIGURES_DIR / f"M4_shap_{arch_name}_{pid}.png"
                    plot_shap_vs_gradcam(
                        vol_np, seg_np, cam_np, shap_map,
                        pearson_r  = r,
                        arch_name  = model.arch_name,
                        patient_id = pid,
                        save_path  = shap_save,
                    )

        # ── Arch comparison plot (only when both models ran) ──────────────────
        if len(cam_dict) >= 2:
            comp_save = FIGURES_DIR / f"M4_arch_comparison_{pid}.png"
            plot_arch_comparison(vol_np, seg_np, cam_dict, pid, comp_save)

    # ── SHAP agreement summary ────────────────────────────────────────────────
    if pearson_records:
        import pandas as pd
        df = pd.DataFrame(pearson_records)
        df.to_csv(TABLES_DIR / "M4_shap_agreement.csv", index=False)
        print(f"\nSHAP agreement table saved to {TABLES_DIR}/M4_shap_agreement.csv")
        print(df.to_string(index=False))

    print(f"\n[XAI] All figures saved to: {FIGURES_DIR}")
    print("[XAI] Done ✓")


if __name__ == "__main__":
    main()
