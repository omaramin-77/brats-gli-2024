# XAI-Guided Multimodal Brain Tumour Segmentation (BraTS GLI-2024)

This repository hosts a five-person research project on the BraTS GLI-2024
adult-glioma dataset. We compare four single-modality 3D segmentation
baselines (T1n, T1c, T2w, T2-FLAIR) against a multimodal model that ingests
all four sequences, then use 3D Grad-CAM to interpret what each network
attends to. The shared pipeline below guarantees that every member trains
on the same patient subset, splits, preprocessing, and evaluation metrics —
the only thing that varies between members is the model architecture and
the modality.

## Repository structure

```
brats-gli-2024/
├── shared/                # preprocessing, dataset, metrics, Grad-CAM, viz, config
├── member1_T1n/           # Member 1 — T1n native pipeline (3D Residual U-Net + deep supervision)
├── member2_T1c/           # Member 2 — T1c contrast-enhanced pipeline (Attention-Gated U-Net)
├── member3_T2w/           # Member 3 — T2w pipeline (MONAI Swin-UNETR)
├── member4_T2f/           # Member 4 — T2-FLAIR pipeline (ResUNet + SegResNet) + Gradio demo
├── member5_multimodal/    # Member 5 — naive 4-channel multimodal baseline + ablation
├── ensemble/              # Pipeline C — Latent Space Fusion Head (built; underperformed naive M5)
├── LateEnsemble/          # Pipeline D — XAI-weighted late fusion stacker (current direction)
├── interface/             # Streamlit demo placeholder (member4 ships a working Gradio app)
├── data/
│   ├── raw/               # absolute path to BraTS root in DATA_PATH.txt
│   ├── processed/         # cached preprocessed volumes (gitignored)
│   └── splits/            # train_ids.txt, val_ids.txt, test_ids.txt (committed)
├── experiments/mlruns/    # MLflow tracking (gitignored)
└── results/
    ├── figures/           # PNGs for the report and slides
    ├── tables/            # CSV metric exports (test_metrics.csv is the headline)
    ├── checkpoints/       # .pt files (gitignored)
    ├── features/          # per-modality bottleneck tensors for Pipeline C (gitignored)
    └── predictions/       # per-member full-volume logits + aligned labels for Pipeline D (gitignored)
```

Project status: see `.claude/STATUS.md` for the current state of each phase
and what's left for the write-up.

## Setup

```bash
git clone <repo-url> brats-gli-2024
cd brats-gli-2024

# Conda is recommended; any Python 3.10+ environment works.
pip install -r requirements.txt
# PyTorch with CUDA must be installed separately for your driver:
# https://pytorch.org

# Point the repo at your local copy of the BraTS GLI-2024 root.
# Open data/raw/DATA_PATH.txt and replace the path with the one valid on
# your machine. The committed value is correct for the project's primary
# workstation; do not push your local edit upstream.
```

## How to start your pipeline

* **Member 1 (T1n)** — work in `member1_T1n/`. Implement `model.py`, then
  fill in `train.py` using the reference template already there.
* **Member 2 (T1c)** — work in `member2_T1c/`. Set `MODALITY = "t1c"` and
  copy the loop from `member1_T1n/train.py`.
* **Member 3 (T2w)** — work in `member3_T2w/`. Set `MODALITY = "t2w"`.
* **Member 4 (T2-FLAIR)** — work in `member4_T2f/`. Set `MODALITY = "t2f"`.
* **Member 5 (multimodal)** — work in `member5_multimodal/`. Set
  `MODALITY = "multimodal"`, and ensure your model's first convolution has
  `in_channels=4`.

Every `train.py` starts with:

```python
from shared.seed import set_global_seed
set_global_seed()
from shared.config import *
from shared.dataset import BraTSDataset, get_dataloader, load_splits
```

## Git workflow

* `main` is protected — never push directly.
* Branch naming: `member{N}/{short-topic}`, e.g. `member1/unet-baseline`.
* Open a PR for every change. At least one teammate reviews before merge.
* Rebase your branch on top of `main` before requesting review.
* Commits should be small, descriptive, and mention the affected module.

## Shared code contract

* **Never duplicate preprocessing.** Always import from `shared/`. If you
  need a new transform, add it to `shared/preprocessing.py` and tell the
  team — your teammates' results must remain reproducible from the same
  function.
* `shared/config.py` is the single source of truth for hyperparameters.
  You may shadow values inside your own train.py, but do not edit the
  defaults on `main` without consensus.
* `shared/seed.py` must be the first import in every entry-point script.

## Data policy

* Do **not** commit any `.nii.gz` or `.nii` file. The `.gitignore` will
  block this; if you ever see one in `git status`, abort and check.
* Do **not** re-run `shared/create_splits.py`. It has already been executed
  and its outputs are committed under `data/splits/`. Re-running it would
  silently change the train/val/test partitioning and invalidate every
  trained checkpoint.
* `data/raw/DATA_PATH.txt` points to the dataset on disk — keep its target
  read-only.

## Weekly milestones (original schedule, kept for reference)

| Week | Phase                          | Owner       | Deliverable                                            |
|------|--------------------------------|-------------|--------------------------------------------------------|
| 1    | Phase 0 — shared infrastructure| Lead        | this repository skeleton + sanity check passing        |
| 2    | Phase 1 — unimodal baselines   | M1, M2, M3, M4 | first training runs, baseline Dice on validation set  |
| 3    | Phase 1 wrap + Phase 2 multimodal | M5       | multimodal model trained, all baselines tuned          |
| 4    | Phase 3 — XAI                  | All         | Grad-CAM overlays for every member, qualitative figures|
| 5    | Phase 4 — ensemble + UI        | M5 + lead   | late-fusion ensemble + Streamlit demo                  |
| 6    | Report + slides                | All         | written report, final figures and tables               |

Phases 0–3 are complete. Phase 4 produced Pipeline C (latent fusion, in
`ensemble/`) which empirically underperformed; Pipeline D (XAI-weighted
late fusion, in `LateEnsemble/`) is the current attempt. See
`.claude/STATUS.md` for current numbers and outstanding work.
