"""Per-modality prediction visualization.

For one test patient, run each member's unimodal checkpoint on its own
modality and render a panel that shows, per modality:

    column 0 : raw modality slice
    column 1 : raw modality + predicted WT/TC/ET overlay
    column 2 : raw modality + ground-truth WT/TC/ET overlay

Usage:
    python visualize_per_modality.py                       # first test patient
    python visualize_per_modality.py --patient BraTS-GLI-02193-104
    python visualize_per_modality.py --members t1c,t2f     # subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.seed import set_global_seed
set_global_seed()

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from monai.inferers import sliding_window_inference
    _HAVE_SWI = True
except ImportError:
    _HAVE_SWI = False

from shared.config import (
    CHECKPOINT_DIR,
    FIGURES_DIR,
    PATCH_SIZE,
    SPLITS_DIR,
    get_data_root,
)
from shared.preprocessing import preprocess_patient


# ---------------------------------------------------------------------------
# Member registry — modality -> (member_name on disk, build_model fn)
# ---------------------------------------------------------------------------
def _build_m1():
    from member1_T1n.model import ResidualUNet3D
    return ResidualUNet3D(in_channels=1, out_channels=3)


def _build_m2():
    from member2_T1c.model import AttentionUNet3D
    return AttentionUNet3D(in_channels=1, out_channels=3)


def _build_m3():
    from member3_T2w.model import M3T2wSwinUNETR
    roi = PATCH_SIZE if isinstance(PATCH_SIZE, tuple) else (PATCH_SIZE,) * 3
    return M3T2wSwinUNETR(img_size=roi, in_channels=1, out_channels=3)


def _build_m4():
    from member4_T2f.model import build_model
    return build_model("resunet", in_channels=1, out_channels=3)


MEMBERS = {
    "t1n": {"member_name": "M1_T1n_ResUNet",   "build": _build_m1, "title": "M1 / T1n / ResUNet3D"},
    "t1c": {"member_name": "M2_T1c_AttUNet",   "build": _build_m2, "title": "M2 / T1c / AttentionUNet3D"},
    "t2w": {"member_name": "M3_T2w_SwinUNETR", "build": _build_m3, "title": "M3 / T2w / SwinUNETR"},
    "t2f": {"member_name": "M4_T2f_resunet",   "build": _build_m4, "title": "M4 / T2f / ResUNet3D"},
}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _load_checkpoint_into(model: torch.nn.Module, member_name: str) -> torch.nn.Module:
    path = Path(CHECKPOINT_DIR) / f"{member_name}_best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state, strict=True)
    return model


@torch.no_grad()
def _predict(model: torch.nn.Module, vol_1c: np.ndarray, device: torch.device) -> np.ndarray:
    """Return sigmoid probs of shape (3, H, W, D) for one (1, H, W, D) volume."""
    model.eval().to(device)
    image = torch.from_numpy(vol_1c).float().unsqueeze(0).to(device)  # (1,1,H,W,D)
    roi = PATCH_SIZE if isinstance(PATCH_SIZE, tuple) else (PATCH_SIZE,) * 3
    if _HAVE_SWI:
        logits = sliding_window_inference(
            inputs=image, roi_size=roi, sw_batch_size=1,
            predictor=model, overlap=0.5,
        )
    else:
        logits = model(image)
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    return probs


def _seg_to_3ch(seg: np.ndarray) -> np.ndarray:
    """Raw integer BraTS labels → (3, H, W, D) WT/TC/ET binary masks."""
    wt = (seg == 1) | (seg == 2) | (seg == 3)
    tc = (seg == 1) | (seg == 3)
    et = (seg == 3)
    return np.stack([wt, tc, et], axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
SUBREGION_COLOURS = (
    (0.20, 0.55, 1.00),   # WT — blue
    (1.00, 0.85, 0.00),   # TC — yellow
    (1.00, 0.20, 0.20),   # ET — red
)


def _best_axial_slice(gt_3ch: np.ndarray) -> int:
    wt = gt_3ch[0]
    if wt.sum() == 0:
        return wt.shape[2] // 2
    return int(np.argmax(wt.sum(axis=(0, 1))))


def _overlay(ax, vol_slice: np.ndarray, mask_3ch_slice: np.ndarray, alpha: float = 0.40) -> None:
    ax.imshow(vol_slice.T, cmap="gray", origin="lower")
    for ch, colour in enumerate(SUBREGION_COLOURS):
        m = mask_3ch_slice[ch].T
        if m.sum() == 0:
            continue
        rgba = np.zeros((*m.shape, 4))
        rgba[m > 0] = (*colour, alpha)
        ax.imshow(rgba, origin="lower")


def _normalise_for_display(vol: np.ndarray) -> np.ndarray:
    """Clip to 1–99 percentile and rescale to [0,1] for nicer contrast."""
    brain = vol[vol != 0]
    if brain.size == 0:
        return vol
    lo, hi = np.percentile(brain, (1.0, 99.0))
    out = np.clip(vol, lo, hi)
    out = (out - lo) / max(hi - lo, 1e-8)
    return out


def render_panel(
    modality_to_result: dict,
    gt_3ch: np.ndarray,
    patient_id: str,
    save_path: str,
) -> None:
    """One row per modality, three columns (raw / pred overlay / GT overlay)."""
    z = _best_axial_slice(gt_3ch)
    mods = list(modality_to_result.keys())
    n = len(mods)

    fig, axes = plt.subplots(n, 3, figsize=(11, 3.4 * n), squeeze=False)

    for r, mod in enumerate(mods):
        title = MEMBERS[mod]["title"]
        vol = modality_to_result[mod]["volume"]    # (1, H, W, D)
        pred_3ch = modality_to_result[mod]["pred"]  # (3, H, W, D)
        vol_disp = _normalise_for_display(vol[0])

        # column 0 — raw
        axes[r, 0].imshow(vol_disp[:, :, z].T, cmap="gray", origin="lower")
        axes[r, 0].set_ylabel(title, fontsize=10)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("Modality (z = %d)" % z, fontsize=11)

        # column 1 — prediction
        _overlay(axes[r, 1], vol_disp[:, :, z], pred_3ch[:, :, :, z])
        axes[r, 1].axis("off")
        if r == 0:
            axes[r, 1].set_title("Prediction overlay", fontsize=11)

        # column 2 — ground truth
        _overlay(axes[r, 2], vol_disp[:, :, z], gt_3ch[:, :, :, z])
        axes[r, 2].axis("off")
        if r == 0:
            axes[r, 2].set_title("Ground truth overlay", fontsize=11)

    handles = [
        mpatches.Patch(color=SUBREGION_COLOURS[0], label="WT — whole tumour"),
        mpatches.Patch(color=SUBREGION_COLOURS[1], label="TC — tumour core"),
        mpatches.Patch(color=SUBREGION_COLOURS[2], label="ET — enhancing tumour"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Per-modality model output — {patient_id}", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default=None,
                        help="Patient ID (e.g. BraTS-GLI-02193-104). "
                             "Defaults to the first entry in test_ids.txt.")
    parser.add_argument("--members", default="t1c,t1n,t2w,t2f",
                        help="Comma-separated modality keys to include.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid probability threshold for the overlay mask.")
    parser.add_argument("--out", default=None,
                        help="Output PNG path. Default: results/figures/per_modality_overlay_<patient>.png")
    args = parser.parse_args()

    # Resolve the patient ID.
    if args.patient is None:
        test_ids = (SPLITS_DIR / "test_ids.txt").read_text(encoding="utf-8").splitlines()
        test_ids = [t.strip() for t in test_ids if t.strip()]
        if not test_ids:
            raise RuntimeError("test_ids.txt is empty.")
        patient_id = test_ids[0]
    else:
        patient_id = args.patient.strip()

    mods = [m.strip().lower() for m in args.members.split(",") if m.strip()]
    unknown = [m for m in mods if m not in MEMBERS]
    if unknown:
        raise ValueError(f"Unknown modalities: {unknown}. Choices: {sorted(MEMBERS)}")

    data_root = get_data_root()
    patient_dir = Path(data_root) / patient_id
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Patient folder not found: {patient_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[viz] patient={patient_id}  device={device}  members={mods}")

    results: dict[str, dict] = {}
    seg_int = None

    for mod in mods:
        info = MEMBERS[mod]
        ckpt_path = Path(CHECKPOINT_DIR) / f"{info['member_name']}_best.pt"
        if not ckpt_path.exists():
            print(f"[viz] SKIP {mod}: checkpoint missing -> {ckpt_path}")
            continue

        print(f"[viz] {mod}: preprocess + load checkpoint")
        vol, seg_this = preprocess_patient(str(patient_dir), modality=mod)   # vol (1,H,W,D)
        if seg_int is None:
            seg_int = seg_this

        model = info["build"]()
        _load_checkpoint_into(model, info["member_name"])

        print(f"[viz] {mod}: inference on {vol.shape[1:]} volume…")
        probs = _predict(model, vol, device)
        pred_mask = (probs >= args.threshold).astype(np.uint8)

        results[mod] = {"volume": vol, "pred": pred_mask}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not results:
        raise RuntimeError("No models ran — every checkpoint was missing.")

    gt_3ch = _seg_to_3ch(seg_int)

    out = Path(args.out) if args.out else Path(FIGURES_DIR) / f"per_modality_overlay_{patient_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render_panel(results, gt_3ch, patient_id, str(out))
    print(f"[viz] saved -> {out}")


if __name__ == "__main__":
    main()
