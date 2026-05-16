"""Member 4 — T2f FLAIR Segmentation Demo
Gradio app — runs in a Kaggle notebook cell or locally.
Shares a public link (72 h) that anyone can open.

Usage (in notebook):
    %run app.py

Usage (terminal):
    python app.py

How it works:
    1. Loads both trained checkpoints (ResUNet3D + SegResNet)
    2. User picks a test patient and adjusts slice slider
    3. Shows FLAIR | Prediction Overlay | Grad-CAM side-by-side
    4. Shows architecture comparison (both models' predictions + CAMs)
    5. Shows real-time metric table
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
from shared.metrics import compute_all_metrics

from model import build_model

# ── Colours ───────────────────────────────────────────────────────────────────
TEAL  = "#5DCAA5"
AMBER = "#EF9F27"
RED   = "#FF4B4B"
BG    = "#0D1117"

# ══════════════════════════════════════════════════════════════════════════════
# Load models & data at startup (cached)
# ══════════════════════════════════════════════════════════════════════════════

DATA_ROOT  = get_data_root()
PATCH_SIZE = PATCH_SIZE[0]
DEVICE     = torch.device("cpu")   # Gradio inference on CPU

def _load_model(arch: str) -> torch.nn.Module | None:
    ckpt_name = f"member4_T2f_{arch}"
    model     = build_model(arch, in_channels=1)
    try:
        mgr             = CheckpointManager(str(CHECKPOINT_DIR), ckpt_name)
        model, _, _, _  = mgr.load_best(model)
        model.eval().to(DEVICE)
        return model
    except FileNotFoundError:
        return None

print("Loading models …")
MODELS = {}
for arch in ("resunet", "segresnet"):
    m = _load_model(arch)
    if m is not None:
        MODELS[m.arch_name] = m
        print(f"  ✓ {m.arch_name}")
    else:
        print(f"  ✗ {arch} — no checkpoint found")

# Load test patient IDs
with open(SPLITS_DIR / "test_ids.txt") as f:
    TEST_IDS = [l.strip() for l in f if l.strip()]

# Pre-process all test patients into RAM (volumes are ~8 MB each)
print(f"Pre-loading {len(TEST_IDS)} test patients …")
PATIENT_DATA: dict[str, dict] = {}
for pid in TEST_IDS:
    try:
        vol, seg = preprocess_patient(str(Path(DATA_ROOT) / pid), "t2f")
        PATIENT_DATA[pid] = {"vol": vol, "seg": seg}
    except Exception as e:
        print(f"  Skip {pid}: {e}")

PATIENT_IDS = sorted(PATIENT_DATA.keys())
print(f"Ready — {len(PATIENT_IDS)} patients available\n")


# ══════════════════════════════════════════════════════════════════════════════
# Grad-CAM helper (stateless)
# ══════════════════════════════════════════════════════════════════════════════

def compute_gradcam(model: torch.nn.Module, vol_t: torch.Tensor) -> np.ndarray:
    """Return (D,H,W) Grad-CAM heatmap in [0,1]."""
    layers = model.get_encoder_layers()
    if not layers:
        return np.zeros(vol_t.shape[2:])

    target_layer = layers[-1]
    feats, grads = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: feats.update({"f": o}))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.update({"g": go[0]}))

    model.zero_grad()
    pred  = model(vol_t)
    prob  = torch.sigmoid(pred)
    score = (prob * (prob > 0.5).float()).sum()
    score.backward()

    h1.remove(); h2.remove()

    weights = grads["g"].mean(dim=(2, 3, 4), keepdim=True)
    cam     = F.relu((weights * feats["f"]).sum(dim=1, keepdim=True))
    cam     = F.interpolate(cam, size=vol_t.shape[2:], mode="trilinear", align_corners=False)
    cam_np  = cam.squeeze().detach().cpu().numpy()

    if cam_np.max() > 1e-6:
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
    return cam_np


# ══════════════════════════════════════════════════════════════════════════════
# Figure builders
# ══════════════════════════════════════════════════════════════════════════════

def _apply_colormap(heatmap: np.ndarray, cmap_name: str = "inferno") -> np.ndarray:
    """Convert (H,W) float [0,1] to (H,W,4) RGBA."""
    cmap = plt.get_cmap(cmap_name)
    return cmap(heatmap)


def make_main_figure(
    pid:        str,
    slice_idx:  int,
    view:       str,           # "Axial" | "Coronal" | "Sagittal"
    arch_name:  str,
    show_gt:    bool,
    show_cam:   bool,
) -> np.ndarray:
    """3-panel figure: FLAIR | Prediction | Grad-CAM (one view/slice)."""

    data = PATIENT_DATA[pid]
    vol_np = data["vol"].squeeze()    # (D,H,W)
    seg_np = data["seg"]              # (D,H,W)
    vol_t  = torch.tensor(data["vol"][np.newaxis]).float()

    model = MODELS.get(arch_name)
    if model is None:
        fig, ax = plt.subplots(facecolor=BG)
        ax.text(0.5, 0.5, f"Model '{arch_name}' not loaded",
                color="white", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img

    # Inference
    with torch.no_grad():
        logits = model(vol_t, return_features=False)
        pred   = (torch.sigmoid(logits) > 0.5).float().squeeze().numpy()

    # Grad-CAM
    cam_np = compute_gradcam(model, vol_t) if show_cam else np.zeros_like(vol_np)

    # Slice extraction
    D, H, W = vol_np.shape
    slice_idx = max(0, min(slice_idx, {"Axial": D, "Coronal": H, "Sagittal": W}[view] - 1))

    def extract(arr):
        if view == "Axial":    return arr[slice_idx]
        if view == "Coronal":  return arr[:, slice_idx, :]
        return arr[:, :, slice_idx]

    mri_s  = extract(vol_np)
    gt_s   = extract((seg_np > 0).astype(np.float32))
    pred_s = extract(pred)
    cam_s  = extract(cam_np)
    tumour_s = extract((seg_np > 0).astype(np.float32))

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    titles = ["FLAIR", f"Prediction ({arch_name})", "Grad-CAM"]

    for ax in axes:
        ax.set_facecolor(BG)
        ax.axis("off")

    # Panel 1 — FLAIR + optional GT contour
    axes[0].imshow(mri_s.T, cmap="gray", origin="lower", interpolation="bilinear")
    if show_gt and tumour_s.sum() > 0:
        axes[0].contour(tumour_s.T, levels=[0.5], colors=[TEAL], linewidths=1.8)

    # Panel 2 — Prediction overlay
    axes[1].imshow(mri_s.T, cmap="gray", origin="lower")
    if pred_s.sum() > 0:
        axes[1].contour(pred_s.T, levels=[0.5], colors=[AMBER], linewidths=1.8,
                        linestyles="--")
    if show_gt and tumour_s.sum() > 0:
        axes[1].contour(tumour_s.T, levels=[0.5], colors=[TEAL], linewidths=1.4)

    # Panel 3 — Grad-CAM
    axes[2].imshow(mri_s.T, cmap="gray", origin="lower")
    if show_cam:
        axes[2].imshow(cam_s.T, cmap="inferno", alpha=0.6, origin="lower", vmin=0, vmax=1)
    if show_gt and tumour_s.sum() > 0:
        axes[2].contour(tumour_s.T, levels=[0.5], colors=[TEAL], linewidths=1.4)

    for ax, t in zip(axes, titles):
        ax.set_title(t, color="white", fontsize=11, pad=6)

    # Legend
    from matplotlib.patches import Patch
    legend_items = [Patch(color=TEAL, label="Ground truth")]
    if pred_s.sum() > 0:
        legend_items.append(Patch(color=AMBER, label="Prediction"))
    axes[2].legend(handles=legend_items, loc="lower right",
                   facecolor="#222", labelcolor="white", fontsize=8, framealpha=0.8)

    fig.suptitle(
        f"{pid} | {view} z={slice_idx} | {arch_name}",
        color="white", fontsize=11, y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def make_comparison_figure(pid: str, slice_idx: int) -> np.ndarray:
    """Side-by-side arch comparison — all loaded models."""
    if not MODELS:
        return np.zeros((300, 600, 3), dtype=np.uint8)

    data   = PATIENT_DATA[pid]
    vol_np = data["vol"].squeeze()
    seg_np = data["seg"]
    vol_t  = torch.tensor(data["vol"][np.newaxis]).float()

    D     = vol_np.shape[0]
    z     = max(0, min(slice_idx, D - 1))
    mri_z = vol_np[z]
    gt_z  = (seg_np[z] > 0).astype(np.float32)

    n_models = len(MODELS)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5), facecolor=BG)

    # Base MRI panel
    axes[0].imshow(mri_z.T, cmap="gray", origin="lower")
    if gt_z.sum() > 0:
        axes[0].contour(gt_z.T, levels=[0.5], colors=[TEAL], linewidths=1.8)
    axes[0].set_title(f"FLAIR — axial z={z}", color="white", fontsize=10)
    axes[0].axis("off")

    cam_colours = ["inferno", "plasma"]
    pred_colours = [AMBER, "#FF8C69"]

    for idx, (arch_name, model) in enumerate(MODELS.items()):
        ax = axes[idx + 1]
        with torch.no_grad():
            logits = model(vol_t, return_features=False)
            pred   = (torch.sigmoid(logits) > 0.5).float().squeeze().numpy()
        cam_np = compute_gradcam(model, vol_t)

        ax.imshow(mri_z.T, cmap="gray", origin="lower")
        ax.imshow(cam_np[z].T, cmap=cam_colours[idx], alpha=0.55, origin="lower",
                  vmin=0, vmax=1)
        if pred[z].sum() > 0:
            ax.contour(pred[z].T, levels=[0.5], colors=[pred_colours[idx]],
                       linewidths=1.8, linestyles="--")
        if gt_z.sum() > 0:
            ax.contour(gt_z.T, levels=[0.5], colors=[TEAL], linewidths=1.4)
        ax.set_title(f"Grad-CAM | {arch_name}", color="white", fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Architecture Comparison — {pid}", color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def compute_metrics_table(pid: str, arch_name: str) -> str:
    """Return markdown metrics table for the given patient & arch."""
    data  = PATIENT_DATA[pid]
    vol_t = torch.tensor(data["vol"][np.newaxis]).float()
    seg_t = torch.tensor(data["seg"][np.newaxis, np.newaxis] > 0).float()

    model = MODELS.get(arch_name)
    if model is None:
        return f"Model '{arch_name}' not loaded."

    with torch.no_grad():
        logits = model(vol_t, return_features=False)
        m      = compute_all_metrics(logits, seg_t)

    lines = [
        f"| Metric | {arch_name} |",
        "|--------|------------|",
        f"| Dice WT | **{m['dice']:.4f}** |",
        f"| IoU WT  | {m['iou']:.4f} |",
        f"| HD95 WT | {m['hd95']:.1f} mm |",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════════════

def build_app():
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Install gradio: pip install gradio")

    arch_choices = list(MODELS.keys()) if MODELS else ["(no models loaded)"]
    vol_shape    = PATIENT_DATA[PATIENT_IDS[0]]["vol"].shape if PATIENT_IDS else (1, 128, 1, 1)
    max_slice    = vol_shape[1] - 1  # D dimension after squeeze

    with gr.Blocks(
        title="M4 — FLAIR Tumour Segmentation & XAI",
        css="""
        body, .gradio-container { background: #0D1117 !important; color: #e6edf3; }
        .gr-button { border-radius: 6px; }
        h1, h2, h3 { color: #5DCAA5; }
        .gr-panel { background: #161B22; border-radius: 8px; }
        """,
    ) as demo:

        gr.Markdown(
            """
# 🧠 Member 4 — T2f FLAIR Segmentation + XAI
**BraTS GLI-2024 | ANN Course Project**

Compares **3D Residual U-Net** vs **SegResNet (MONAI)** on FLAIR-only brain tumour segmentation.
Grad-CAM heatmaps show *what* each model focuses on.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                pid_dd     = gr.Dropdown(PATIENT_IDS, label="Patient", value=PATIENT_IDS[0])
                arch_dd    = gr.Dropdown(arch_choices, label="Architecture", value=arch_choices[0])
                view_dd    = gr.Dropdown(["Axial", "Coronal", "Sagittal"],
                                         label="View plane", value="Axial")
                slice_sl   = gr.Slider(0, max_slice, step=1, value=max_slice // 2,
                                       label="Slice index")
                show_gt    = gr.Checkbox(True,  label="Show ground truth contour (teal)")
                show_cam   = gr.Checkbox(True,  label="Show Grad-CAM heatmap")
                run_btn    = gr.Button("🔍 Analyse", variant="primary")

            with gr.Column(scale=3):
                main_img   = gr.Image(label="FLAIR | Prediction | Grad-CAM",
                                      type="numpy", show_label=True)
                metrics_md = gr.Markdown("*(metrics will appear here)*")

        gr.Markdown("---\n### 🔬 Architecture Comparison (Axial)")
        gr.Markdown(
            "_Both models' Grad-CAMs side-by-side. "
            "FLAIR usually shows broader activation than T1c._"
        )
        with gr.Row():
            comp_btn = gr.Button("⚡ Compare Architectures", variant="secondary")
            comp_img = gr.Image(label="Arch Comparison", type="numpy")

        gr.Markdown(
            """
---
### How to read the visualisations
| Colour | Meaning |
|--------|---------|
| **Teal contour** | Ground truth tumour boundary |
| **Amber dashed** | Model prediction boundary |
| **Inferno/Plasma heatmap** | Grad-CAM — hot = model focuses here |

> **FLAIR finding**: expect a *broader* heatmap than T1c (which gives sharp enhancing-tumour focus).
> FLAIR highlights the full infiltrative margin — this is correct behaviour.
            """
        )

        # ── Callbacks ─────────────────────────────────────────────────────────

        def on_run(pid, arch, view, sl, gt, cam):
            img = make_main_figure(pid, int(sl), view, arch, gt, cam)
            metrics = compute_metrics_table(pid, arch)
            return img, metrics

        def on_compare(pid, sl):
            return make_comparison_figure(pid, int(sl))

        run_btn.click(
            fn=on_run,
            inputs=[pid_dd, arch_dd, view_dd, slice_sl, show_gt, show_cam],
            outputs=[main_img, metrics_md],
        )
        comp_btn.click(
            fn=on_compare,
            inputs=[pid_dd, slice_sl],
            outputs=[comp_img],
        )

        # Auto-run on load
        demo.load(
            fn=on_run,
            inputs=[pid_dd, arch_dd, view_dd, slice_sl, show_gt, show_cam],
            outputs=[main_img, metrics_md],
        )

    return demo


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not PATIENT_IDS:
        print("ERROR: No patients loaded. Check DATA_PATH.local.txt / Kaggle dataset config.")
        sys.exit(1)

    demo = build_app()

    # share=True → public URL valid for 72 h (works in Kaggle notebooks too)
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
