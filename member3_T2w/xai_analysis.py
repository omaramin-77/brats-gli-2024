"""Member 3 — T2w XAI analysis.

Two complementary methods — this is the novel M3 contribution:

  A) Native Swin Transformer attention maps
       Since MONAI's WindowAttention does not return attention weights by
       default, we monkey-patch its forward method to intercept and store
       the softmax attention matrix before it is used. This gives us one
       (B*nW, nH, N, N) tensor per block, averaged into a 3-D saliency volume.
       Early blocks: local texture.  Later blocks: full oedema extent.

  B) 3D Grad-CAM on the CNN decoder
       Uses shared/grad_cam_3d.py on the last decoder block.
       Standard backprop saliency — directly comparable to M1-M4.

Cross-validation: do A and B agree?
  Agreement  → model attends and predicts from the same region.
  Disagreement → encoder context differs from decoder decision → discuss.

Usage:
    python member3_T2w/xai_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed
set_global_seed()

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.config import (
    CHECKPOINT_DIR, SPLITS_DIR, FIGURES_DIR, NUM_WORKERS,
    get_data_root,
)
from shared.dataset import BraTSDataset, get_dataloader
from shared.trainer import CheckpointManager
from shared.grad_cam_3d import GradCAM3D
from shared.visualization import plot_gradcam_overlay

from model import build_model

MEMBER_NAME     = "member3_T2w"
MODALITY        = "t2w"
PATCH_SIZE      = (64, 64, 64)
FEATURE_SIZE    = 24
NUM_XAI_SAMPLES = 5


# ---------------------------------------------------------------------------
# Attention capture — monkey-patch WindowAttention to intercept attn weights
# ---------------------------------------------------------------------------

def _patch_window_attention(model, storage: list) -> list:
    """
    MONAI's WindowAttention computes attention internally but never returns it.
    We wrap its forward method to store the softmax attention matrix each call.
    Returns list of (module, original_forward) pairs for unpatching.
    """
    originals = []
    try:
        from monai.networks.nets.swin_unetr import WindowAttention  # type: ignore
    except ImportError:
        print("  WARNING: could not import WindowAttention — attention maps skipped")
        return originals

    for module in model.model.modules():
        if isinstance(module, WindowAttention):
            orig_forward = module.forward

            def make_patched(orig, mod):
                def patched_forward(x, mask=None):
                    B_, N, C = x.shape
                    try:
                        qkv = mod.qkv(x)
                        qkv = qkv.reshape(B_, N, 3,
                                          mod.num_heads,
                                          C // mod.num_heads
                                          ).permute(2, 0, 3, 1, 4)
                        q, k, _ = qkv.unbind(0)
                        q = q * mod.scale
                        attn = q @ k.transpose(-2, -1)
                        attn = attn.softmax(dim=-1)
                        storage.append(attn.detach().cpu())
                    except Exception:
                        pass
                    return orig(x, mask)
                return patched_forward

            module.forward = make_patched(orig_forward, module)
            originals.append((module, orig_forward))

    return originals


def _unpatch_window_attention(originals: list) -> None:
    for module, orig in originals:
        module.forward = orig


# ---------------------------------------------------------------------------
# Helper: reshape raw attention weights to normalised 3-D volume
# ---------------------------------------------------------------------------

def attn_to_volume(attn: torch.Tensor, target_shape: tuple) -> np.ndarray:
    try:
        if attn.dim() == 4:
            avg = attn.mean(dim=1).mean(dim=2).mean(dim=0)  # (N,)
        elif attn.dim() == 3:
            avg = attn.mean(dim=0).mean(dim=0)
        else:
            return np.zeros(target_shape, dtype=np.float32)

        N  = avg.numel()
        ws = round(N ** (1 / 3))
        if ws ** 3 != N:
            return np.zeros(target_shape, dtype=np.float32)

        cube = avg.reshape(ws, ws, ws).numpy()
        t    = torch.from_numpy(cube).unsqueeze(0).unsqueeze(0).float()
        t    = F.interpolate(t, size=target_shape, mode="trilinear", align_corners=False)
        vol  = t.squeeze().numpy()
        mn, mx = vol.min(), vol.max()
        return ((vol - mn) / (mx - mn + 1e-8)).astype(np.float32)
    except Exception:
        return np.zeros(target_shape, dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-patient XAI
# ---------------------------------------------------------------------------

def run_xai(
    patient_id: str,
    image: torch.Tensor,
    label: torch.Tensor,
    model,
    device: torch.device,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    image    = image.to(device)
    spatial  = tuple(image.shape[2:])
    image_np = image.squeeze().cpu().numpy()
    label_np = label.squeeze().cpu().numpy()

    # Best axial slice = most tumour voxels
    if label_np.sum() > 0:
        z = int(np.argmax((label_np > 0).sum(axis=(0, 1))))
    else:
        z = spatial[2] // 2

    # ── A: Transformer attention maps ─────────────────────────────────────
    attn_storage: list = []
    originals = _patch_window_attention(model, attn_storage)

    model.eval()
    with torch.no_grad():
        _ = model(image)

    _unpatch_window_attention(originals)

    block_vols = []
    for b_idx, attn in enumerate(attn_storage):
        vol = attn_to_volume(attn, spatial)
        block_vols.append(vol)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image_np[:, :, z].T, cmap="gray", origin="lower")
        ax.imshow(vol[:, :, z].T, cmap="hot", alpha=0.55, vmin=0, vmax=1, origin="lower")
        ax.set_title(f"Swin Block {b_idx:02d} — {patient_id}")
        ax.axis("off")
        path = FIGURES_DIR / f"{MEMBER_NAME}_attn_block_{b_idx:02d}_{patient_id}.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    if block_vols:
        global_attn = np.stack(block_vols).mean(axis=0)
        mn, mx = global_attn.min(), global_attn.max()
        if mx > mn:
            global_attn = (global_attn - mn) / (mx - mn)
        print(f"  Attention maps captured: {len(block_vols)} blocks")
    else:
        global_attn = np.zeros(spatial, dtype=np.float32)
        print(f"  WARNING: no attention maps captured for {patient_id}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_np[:, :, z].T, cmap="gray", origin="lower")
    axes[0].set_title(f"T2w — {patient_id}")
    axes[0].axis("off")
    axes[1].imshow(image_np[:, :, z].T, cmap="gray", origin="lower")
    axes[1].imshow(global_attn[:, :, z].T, cmap="hot", alpha=0.6, origin="lower")
    axes[1].set_title(f"Global attention ({len(block_vols)} blocks averaged)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{MEMBER_NAME}_attn_global_{patient_id}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved global attention → {MEMBER_NAME}_attn_global_{patient_id}.png")

    # ── B: 3D Grad-CAM on CNN decoder ─────────────────────────────────────
    # plot_gradcam_overlay signature: (vol_slice, cam_slice, seg_slice, title, save_path)
    target_layer = model.get_target_layer()
    with GradCAM3D(model=model, target_layer=target_layer) as cam:
        image_grad = image.detach().requires_grad_(True)
        cam_vol    = cam.generate(image_grad)   # numpy (D, H, W) in [0,1]

    gradcam_path = str(FIGURES_DIR / f"{MEMBER_NAME}_gradcam_{patient_id}.png")
    plot_gradcam_overlay(
        vol_slice  = image_np[:, :, z],
        cam_slice  = cam_vol[:, :, z],
        seg_slice  = label_np[:, :, z],
        title      = f"M3 Grad-CAM — {patient_id}  (slice {z})",
        save_path  = gradcam_path,
    )
    print(f"  Saved Grad-CAM        → {MEMBER_NAME}_gradcam_{patient_id}.png")

    # ── Comparison: 3-panel side-by-side ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"M3 XAI — {patient_id}  (axial slice {z})", fontsize=11)

    axes[0].imshow(image_np[:, :, z].T, cmap="gray", origin="lower")
    axes[0].set_title("T2w image")
    axes[0].axis("off")

    axes[1].imshow(image_np[:, :, z].T, cmap="gray", origin="lower")
    axes[1].imshow(global_attn[:, :, z].T, cmap="hot", alpha=0.55, origin="lower")
    axes[1].set_title("Swin attention (global)")
    axes[1].axis("off")

    axes[2].imshow(image_np[:, :, z].T, cmap="gray", origin="lower")
    axes[2].imshow(cam_vol[:, :, z].T, cmap="jet", alpha=0.55, origin="lower")
    axes[2].set_title("Grad-CAM (decoder)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{MEMBER_NAME}_comparison_{patient_id}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison      → {MEMBER_NAME}_comparison_{patient_id}.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{MEMBER_NAME}/xai] device={device}")

    ckpt  = CheckpointManager(save_dir=str(CHECKPOINT_DIR), member_name=MEMBER_NAME)
    model = build_model(img_size=PATCH_SIZE, feature_size=FEATURE_SIZE).to(device)
    model, _, epoch, val_dice = ckpt.load_best(model, optimizer=None)
    model.eval()
    print(f"[{MEMBER_NAME}/xai] checkpoint epoch={epoch}  val_dice={val_dice:.4f}\n")

    data_root = get_data_root()
    test_ds = BraTSDataset(
        data_root=data_root,
        split_file=str(SPLITS_DIR / "test_ids.txt"),
        modality=MODALITY,
        patch_size=PATCH_SIZE[0],
        augment=False,
        patches_per_volume=1,
    )
    from torch.utils.data import Subset
    xai_ds     = Subset(test_ds, list(range(min(NUM_XAI_SAMPLES, len(test_ds)))))
    xai_loader = get_dataloader(xai_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f"[{MEMBER_NAME}/xai] generating figures for {len(xai_ds)} patients ...\n")

    for batch in xai_loader:
        pid   = batch["patient_id"][0]
        image = batch["image"]
        label = batch["label"]
        if label.dim() == 4:
            label = label.unsqueeze(1)
        print(f"  Patient: {pid}")
        run_xai(patient_id=pid, image=image, label=label,
                model=model, device=device)
        print()

    print(f"[{MEMBER_NAME}/xai] All figures saved to: {FIGURES_DIR}")
    print("""
── Report questions to answer ────────────────────────────────────────────
  1. Do Swin attention maps and Grad-CAM highlight the same tumour region?
  2. Do early blocks (0-1) attend locally, later blocks (6-7) globally?
  3. Does T2w Dice = 0.8549 beat M1 T1n? Expected yes — T2w shows oedema.
──────────────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()