# ONBOARDING — Read this first

You just opened the repo. This document is the single document you need so
that you (and your teammates) are not lost. It explains:

1. What the project is doing.
2. Every file that has been created so far and what it does.
3. The exact path to take, in order, the first time you sit down to work.
4. How splits and preprocessing reach your training loop.
5. How the day-to-day workflow looks once you are set up.
6. Where to look when something breaks.

Treat this as a map. Skim section 1 and 2 once, then follow section 3
literally on your first day. After that, sections 4-8 are reference.

---

## 1. What is this project?

We are building a 3D MRI brain-tumour segmentation system on the **BraTS
GLI-2024** adult-glioma dataset.

There are **five people** on the team. Four of us each train a 3D
segmentation model on a *single* MRI modality:

| Member | Folder                  | Modality | What that modality is                |
|--------|-------------------------|----------|--------------------------------------|
| 1      | `member1_T1n/`          | T1n      | Native T1 (no contrast agent)        |
| 2      | `member2_T1c/`          | T1c      | Contrast-enhanced T1                 |
| 3      | `member3_T2w/`          | T2w      | T2-weighted                          |
| 4      | `member4_T2f/`          | T2-FLAIR | T2 with fluid attenuation            |
| 5      | `member5_multimodal/`   | all 4    | Stacked 4-channel multimodal model   |

The point of doing it this way:
- The four single-modality baselines tell us **how much each modality alone
  contributes** to segmentation quality.
- Member 5's multimodal model tells us **how much we gain by fusing them**.
- We then run **3D Grad-CAM** on every model to see *what each network
  attends to* — that is the "XAI-guided" part of the project title.

Everyone trains on the **exact same patients**, with the **exact same
preprocessing**, splits, metrics, and evaluation code. The only thing that
varies between members is the model architecture and which modality it
sees. That is enforced by sharing all that code in the [shared/](shared/)
folder. **Do not duplicate preprocessing or metrics in your member folder.**

---

## 2. Every file in the repo and what it does

### 2.1 Top-level files

| Path | Purpose |
|---|---|
| [README.md](README.md) | High-level project pitch, repo layout, weekly milestones. Skim once. |
| [requirements.txt](requirements.txt) | Python dependencies. Install with `pip install -r requirements.txt`. PyTorch+CUDA must be installed separately from pytorch.org for your GPU driver. |
| [.gitignore](.gitignore) | Blocks NIfTI volumes, `data/processed/`, `experiments/mlruns/`, checkpoints, virtualenvs, and Python cruft from being committed. Do not edit casually. |
| ONBOARDING.md | **This file.** |

### 2.2 The [shared/](shared/) folder — code every member imports

This is the heart of the repo. **You will import from here, you will not
modify these files casually.** If you genuinely need a new transform or
metric, add it here and tell the team — your teammates' results must remain
reproducible from the same function.

| File | What it does | When you touch it |
|---|---|---|
| [shared/config.py](shared/config.py) | Single source of truth for hyperparameters and paths. Auto-detects GPU, picks `PATCH_SIZE = (96,96,96)` if you have ≥14 GB VRAM else `(64,64,64)`. Defines `LR`, `BATCH_SIZE`, `NUM_EPOCHS`, `PATIENCE`, label IDs, and resolves the data root from `data/raw/DATA_PATH.txt`. | Read-only for members. You may shadow values inside your own train.py (`LR = 5e-5`), but do not edit defaults on `main` without team consensus. |
| [shared/seed.py](shared/seed.py) | `set_global_seed(seed=42)` — seeds Python `random`, NumPy, PyTorch CPU and CUDA, disables cuDNN benchmark for determinism. **First import in every entry-point script.** | Never edit. |
| [shared/preprocessing.py](shared/preprocessing.py) | Per-patient z-score normalisation (over brain voxels only, not background), brain bounding-box crop, resize to `(128,128,128)`, tumour-biased patch extraction, MONAI augmentations, full per-patient pipelines (`preprocess_patient`, `preprocess_patient_multimodal`). | Add new transforms here, never inside member folders. |
| [shared/dataset.py](shared/dataset.py) | `BraTSDataset` (PyTorch `Dataset`) and `get_dataloader`, plus `load_splits(splits_dir)`. Accepts `modality ∈ {"t1n","t1c","t2w","t2f","multimodal"}`. Returns `{"image", "label", "patient_id"}` dicts. Default task is binary whole-tumour. | Read-only. |
| [shared/metrics.py](shared/metrics.py) | `compute_dice`, `compute_iou`, `compute_hd95` (95th-percentile Hausdorff via MONAI), `compute_all_metrics`, and a `MetricTracker` class that accumulates batch metrics and skips NaNs. | Read-only. |
| [shared/trainer.py](shared/trainer.py) | `EarlyStopper`, `CheckpointManager`, `dice_bce_loss`, `train_one_epoch`, `validate_one_epoch` (with sliding-window inference for full-volume metrics). The pieces every member's training loop is built from. | Read-only. |
| [shared/grad_cam_3d.py](shared/grad_cam_3d.py) | `GradCAM3D` class — forward + full-backward hooks on a target layer, generates a normalised heatmap interpolated back to the input volume's spatial shape. **Always call `.remove_hooks()` after use** or you leak GPU memory. | Read-only. |
| [shared/visualization.py](shared/visualization.py) | Matplotlib helpers: `plot_patient_overview`, `plot_gradcam_overlay`, `plot_training_curves`. Headless-safe (`Agg` backend). Saves to `results/figures/`. | Add new plots here. |
| [shared/create_splits.py](shared/create_splits.py) | Builds the stratified 350-case subset and 80/10/10 train/val/test split. **Run only once, ever.** Outputs are committed under `data/splits/`. Re-running silently re-partitions the data and invalidates every trained checkpoint. | **Do not run it.** |
| [shared/sanity_check.py](shared/sanity_check.py) | End-to-end Phase 0 verification. Checks splits exist, preprocesses one patient through all four modalities + multimodal, exercises `BraTSDataset`, computes the loss on random tensors, saves a sample figure. Exits non-zero on the first failure. | Run it once after install (see section 3). |

### 2.3 Member folders — `member1_T1n/`, `member2_T1c/`, `member3_T2w/`, `member4_T2f/`, `member5_multimodal/`

Each member folder has exactly **four files**, and they all follow the same
contract:

| File | What it does |
|---|---|
| `model.py` | Defines the `nn.Module` architecture for that modality. **Currently a stub** — every member must implement their own encoder-decoder. Member 5's first conv must be `in_channels=4`. |
| `train.py` | Training entry point. Wires up the shared dataset, optimizer, AMP scaler, train+validate loops, early stopping, MLflow logging, and checkpointing. **Member 1's `train.py` is the reference template** — members 2-5 copy that loop and only swap `MODALITY` and `build_model()`. |
| `evaluate.py` | Test-set evaluation. Loads the best checkpoint, runs `validate_one_epoch` on the held-out test split, writes a CSV row to `results/tables/`. Stub right now. |
| `xai_analysis.py` | 3D Grad-CAM. Restores the best checkpoint, attaches `GradCAM3D` to the chosen layer (typically just before the final 1×1×1 classifier conv), produces overlays via `plot_gradcam_overlay`. Stub right now. |

The fact that every member folder has the same four-file shape is the
point — when a teammate opens your folder they should know exactly where to
look. Do not invent extra files unless you absolutely have to.

### 2.4 [data/](data/) — datasets and splits

| Path | What it is |
|---|---|
| [data/raw/DATA_PATH.txt](data/raw/DATA_PATH.txt) | A single-line text file holding the **absolute** path to the BraTS root on your local machine. The committed value is `D:\ANN_project\data\raw\BraTS2024-BraTS-GLI-TrainingData\training_data1_v2`. **Edit this on your own machine** to point at where the dataset actually lives, and **do not push your local edit** — the committed path is correct for the project's primary workstation. |
| `data/raw/.gitkeep` | Empty placeholder so the folder exists in git even though `.nii.gz` files are gitignored. |
| `data/raw/<patient folders>/` | The actual NIfTI volumes. Never committed — `.gitignore` blocks `*.nii` and `*.nii.gz`. |
| `data/processed/` | Cache for preprocessed volumes. Auto-generated, gitignored, regenerable from raw + `shared/preprocessing.py`. |
| [data/splits/train_ids.txt](data/splits/train_ids.txt) | 280 patient IDs, one per line. **Committed.** |
| [data/splits/val_ids.txt](data/splits/val_ids.txt) | 35 patient IDs. **Committed.** |
| [data/splits/test_ids.txt](data/splits/test_ids.txt) | 35 patient IDs. **Committed. Never look at these during development.** |
| [data/splits/split_stats.json](data/splits/split_stats.json) | 350 cases sampled from 1350 available, stratified by tumour fraction across 5 strata, `seed=42`. Mean tumour fraction: train 1.02 %, val 0.98 %, test 1.01 % — well-balanced. |

### 2.5 [results/](results/) — what training produces

| Path | Contains |
|---|---|
| `results/checkpoints/` | `member<N>_<name>_best.pt` and `member<N>_<name>_epoch###.pt`. Gitignored. Each member keeps their own on disk. |
| `results/figures/` | PNGs for the report and slides — patient overviews, Grad-CAM overlays, training curves. |
| `results/tables/` | CSV exports of test metrics. |

### 2.6 The other folders

| Path | Status |
|---|---|
| `experiments/mlruns/` | MLflow tracking artefacts. Each member runs MLflow locally; the directory is gitignored. |
| `ensemble/` | Phase 4. Late-fusion / averaging across the five members. Empty placeholder for now. |
| `interface/` | Phase 4. Streamlit demo. Empty placeholder for now. |

---

## 3. Your first day — exact path to take

Follow this in order. Do not skip steps.

### Step 1 — Clone and install

```powershell
git clone <repo-url> brats-gli-2024
cd brats-gli-2024

# Create a virtualenv (python -m venv .venv) or conda env. Python 3.10+.
pip install -r requirements.txt

# PyTorch with CUDA must be installed separately for your GPU driver.
# Go to https://pytorch.org and copy the install command that matches your CUDA.
```

### Step 2 — Tell the repo where the data lives on **your** machine

Open [data/raw/DATA_PATH.txt](data/raw/DATA_PATH.txt) and replace the
single line with the absolute path to your local copy of the BraTS GLI-2024
training root. Each patient folder underneath that root must contain four
NIfTI modalities + a `*seg*.nii.gz` segmentation.

**Do not commit your local edit** — the value already in there is correct
for the project's primary workstation, and pushing yours will break
everyone else.

### Step 3 — Verify the install with the sanity check

```powershell
python shared/sanity_check.py
```

Expected outcome: it prints `PHASE 0 SANITY CHECK - ALL TESTS PASSED`,
saves [results/figures/sanity_check_patient.png](results/figures/), and
exits 0. If it fails it tells you exactly which assertion broke.

If this step passes, your environment is correctly wired to the shared
pipeline and you can start coding.

### Step 4 — **Do not run** `shared/create_splits.py`

The splits are already committed under [data/splits/](data/splits/). Running
the script again would re-shuffle who is in train vs. val vs. test, and
every checkpoint trained against the old splits would become meaningless.
The script exists for traceability of how the splits were built, not for
re-execution.

### Step 5 — Open your member folder and start with `model.py`

Pick the folder that matches your role and implement your architecture in
`model.py`. A 3D U-Net is the obvious starting point; MONAI ships one
(`monai.networks.nets.UNet`). For member 5 the only architectural
difference is `in_channels=4`.

### Step 6 — Implement `train.py`

**Member 1's [train.py](member1_T1n/train.py) is the reference template.**
It is the only file in the member folders that already wires everything
together. Members 2-5: read it top to bottom, copy the loop into your own
train.py, swap `MODALITY` to your modality, swap `build_model()` to import
your model, done.

The key things every train.py must do, in this order:

1. `from shared.seed import set_global_seed; set_global_seed()` — first executable line.
2. Import config + dataset utilities from `shared/`.
3. Build train + val `BraTSDataset` instances with `augment=True` for train, `augment=False` for val.
4. Build `DataLoader`s via `get_dataloader`.
5. Construct your model, move it to CUDA, build an `AdamW` optimizer.
6. Build an AMP `GradScaler` (only if CUDA is available and `AMP_ENABLED`).
7. Build an `EarlyStopper(patience=PATIENCE)` and a `CheckpointManager(CHECKPOINT_DIR, MEMBER_NAME)`.
8. Loop epochs 1..`NUM_EPOCHS`. Every epoch: `train_one_epoch`. Every `VAL_EVERY_N_EPOCHS` epochs: `validate_one_epoch`, then `ckpt.save(..., is_best=...)`, then `stopper.should_stop(val_dice)`.
9. (Optional but encouraged) wrap the whole thing in an `mlflow.start_run()` context and log `train_loss` / `val_dice` / `val_iou` / `val_hd95` per epoch.

### Step 7 — Train and watch the numbers

```powershell
python member1_T1n/train.py    # or whichever member folder you own
```

The console prints config info, per-epoch loss, and per-validation Dice.
Checkpoints land in [results/checkpoints/](results/checkpoints/) under
`member<N>_<name>_*.pt`. The `_best.pt` is the one your evaluate.py and
xai_analysis.py will consume.

### Step 8 — Implement `evaluate.py` and `xai_analysis.py`

Once training has produced a `best.pt`:

- **evaluate.py** loads `test_ids.txt` (only at the very end, never during
  hyperparameter tuning), restores the best checkpoint via
  `CheckpointManager.load_best`, runs `validate_one_epoch` on the test
  loader, writes a row to `results/tables/test_metrics.csv`.
- **xai_analysis.py** restores the best checkpoint, picks a target layer
  (the bottleneck or the conv just before the final 1×1×1 classifier),
  wraps it in `GradCAM3D`, generates heatmaps for a handful of test cases,
  and saves overlays via `plot_gradcam_overlay`. Always call
  `cam.remove_hooks()` at the end.

---

## 4. How splits and preprocessing reach your training loop

This section answers the two questions every member asks on day one:

- *"Where do the splits come from? Do I have to generate them?"*
- *"When and where does preprocessing actually run?"*

### 4.1 The splits are already on disk — you just read them

The 80/10/10 train/val/test split was generated **once**, by the lead, by
running [shared/create_splits.py](shared/create_splits.py). It sampled 350
cases from the 1350 available, stratified into 5 strata by tumour fraction,
with `seed=42`, and wrote three plain-text files:

- [data/splits/train_ids.txt](data/splits/train_ids.txt) — 280 IDs
- [data/splits/val_ids.txt](data/splits/val_ids.txt) — 35 IDs
- [data/splits/test_ids.txt](data/splits/test_ids.txt) — 35 IDs

These files are **committed to git**. After `git pull` you already have
them. There is no command for you to run to "reach" the split — it is
sitting in `data/splits/` next to the rest of the code, and
[data/splits/split_stats.json](data/splits/split_stats.json) records the
exact stratification statistics for the report.

**Never run [shared/create_splits.py](shared/create_splits.py) again.**
Re-running it re-shuffles who is in train vs val vs test, and every
checkpoint trained against the old split becomes meaningless — member 5's
multimodal model would no longer be comparable to members 1-4's unimodal
baselines, because they would have trained on different patients. The
script is committed for traceability of *how* the splits were built, not
for re-execution.

### 4.2 How your `train.py` picks the splits up

Inside your member's `train.py`, three lines do the wiring:

```python
from shared.config  import SPLITS_DIR, get_data_root
from shared.dataset import BraTSDataset, get_dataloader, load_splits

DATA_ROOT = get_data_root()           # reads data/raw/DATA_PATH.txt
splits    = load_splits(SPLITS_DIR)   # sanity: len(splits["train"]) == 280

train_ds = BraTSDataset(
    data_root  = DATA_ROOT,
    split_file = SPLITS_DIR / "train_ids.txt",
    modality   = MODALITY,            # "t1n" / "t1c" / "t2w" / "t2f" / "multimodal"
    augment    = True,
)
val_ds = BraTSDataset(
    data_root  = DATA_ROOT,
    split_file = SPLITS_DIR / "val_ids.txt",
    modality   = MODALITY,
    augment    = False,
)
```

Notes on the contract:

- [`BraTSDataset`](shared/dataset.py) reads patient IDs from a file path
  (`split_file`), **not** from the in-memory list `load_splits` returns.
  The committed text files are the canonical input; `load_splits` is
  mostly for sanity printing at the top of `train.py`.
- `tumour_bias=0.8` and `patches_per_volume=4` are identical for every
  member. The only legitimate per-member differences are `modality` and
  `augment`.
- **Do not pass `test_ids.txt` to anything until you reach
  [evaluate.py](member1_T1n/evaluate.py).** Tune on val.

### 4.3 When does preprocessing run?

**There is no preprocessing script. You do not run preprocessing as a
separate step before training.**

Preprocessing is invoked **lazily** by `BraTSDataset.__getitem__` every
time the DataLoader asks for a sample. The full per-patient pipeline
defined in [shared/preprocessing.py](shared/preprocessing.py) runs inside
that single call:

1. Load the NIfTI volume(s) for your modality (`nibabel.load`).
2. Z-score normalise within the brain mask only.
3. Crop to the brain bounding box.
4. Resize to `(128, 128, 128)`.
5. Extract a tumour-biased patch of `PATCH_SIZE`.
6. *(Train only)* Apply MONAI augmentations — flips, intensity jitter, etc.

So the very first forward pass of training already eats fully-preprocessed
tensors. Every patient is re-processed every epoch — there is no on-disk
cache today. That is intentional: it keeps the workflow stateless, the
augmentations stochastic, and the working tree small.

The trade-off is that the first iterations of each epoch are I/O- and
CPU-bound. If your GPU is sitting at <40 % utilisation during training,
that is the bottleneck. The fix is `num_workers > 0` on Linux/WSL2; keep
it at `0` on Windows (see section 7).

### 4.4 What is `data/processed/` then?

A placeholder for an optional cache that **nobody currently writes to**.
The directory is gitignored so it can later host pre-computed NPY tensors
— member 5 will probably want this for the 4-channel multimodal pipeline,
where re-loading four NIfTI files per `__getitem__` is expensive. For now,
treat it as empty and ignore it.

### 4.5 So when do you actually "call" the split?

Once per training run, from your `train.py`, in the lines shown in 4.2.
Per file:

| File              | Touches splits?                                                                       |
|-------------------|---------------------------------------------------------------------------------------|
| `model.py`        | No — architecture only, never sees data.                                              |
| `train.py`        | Yes — `train_ids.txt` and `val_ids.txt`.                                              |
| `evaluate.py`     | Yes — **`test_ids.txt` only**, at the very end of the project, never during tuning.  |
| `xai_analysis.py` | Indirectly — picks a handful of test patient IDs *after* evaluate.py has confirmed the model. |

---

## 5. Daily workflow once you are set up

### Branching and PRs

- `main` is protected — never push directly.
- Branch naming: `member{N}/{short-topic}`, e.g. `member1/unet-baseline`.
- Open a PR for every change. At least one teammate reviews before merge.
- Rebase your branch on top of `main` before requesting review.
- Commits should be small and descriptive — mention the affected module.

### Shared-code contract

- **Never duplicate preprocessing.** Always import from `shared/`.
- If you need a new transform or metric, add it to `shared/` and post in the team channel — your teammates' results must remain reproducible from the same function.
- `shared/config.py` is the single source of truth for hyperparameters. Shadow values inside your own train.py if you need to (`LR = 5e-5`), but do not edit defaults on `main` without team consensus.
- `shared/seed.py` must be the first import in every entry-point script.

### Data hygiene

- **Never commit a `.nii` or `.nii.gz`.** The `.gitignore` blocks it; if you ever see one in `git status`, abort and check.
- **Never re-run `shared/create_splits.py`.**
- **Do not look at the test split during development.** Tune on val. Test is for the final evaluate.py run.

---

## 6. The shared infrastructure cheat-sheet

When you forget which import does what, this table is enough:

```python
# always first
from shared.seed import set_global_seed
set_global_seed()

# config — paths, hyperparameters, GPU-aware defaults
from shared.config import (
    BATCH_SIZE, LR, NUM_EPOCHS, PATIENCE, VAL_EVERY_N_EPOCHS, WEIGHT_DECAY,
    PATCH_SIZE, AMP_ENABLED, NUM_WORKERS,
    CHECKPOINT_DIR, FIGURES_DIR, MLRUNS_DIR, SPLITS_DIR, TABLES_DIR,
    get_data_root,
)

# data
from shared.dataset import BraTSDataset, get_dataloader, load_splits

# training utilities
from shared.trainer import (
    CheckpointManager, EarlyStopper,
    dice_bce_loss, train_one_epoch, validate_one_epoch,
)

# metrics (validate_one_epoch already calls these for you)
from shared.metrics import compute_all_metrics, compute_dice, compute_iou, compute_hd95, MetricTracker

# plotting
from shared.visualization import plot_patient_overview, plot_gradcam_overlay, plot_training_curves

# explainability
from shared.grad_cam_3d import GradCAM3D
```

---

## 7. When something breaks — a triage checklist

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: DATA_PATH.txt missing` | Cloned the repo but did not run from inside it | Run from the repo root, or check `data/raw/DATA_PATH.txt` exists. |
| `FileNotFoundError: DATA_ROOT does not exist on this machine` | The path inside `DATA_PATH.txt` is not valid on your machine | Edit `data/raw/DATA_PATH.txt` to the correct local path. Do not commit. |
| `FileNotFoundError: Split file ... missing` | Splits were not committed or were lost | They should be in `data/splits/`. Pull from `main`. **Do not** run `create_splits.py`. |
| Sanity check fails on shape `(1, 128, 128, 128)` | A patient folder is missing a modality or seg | Inspect that patient folder; missing files are a data issue, not a code bug. |
| OOM during training | `PATCH_SIZE=(96,96,96)` on <14 GB VRAM, or batch too large | Drop `BATCH_SIZE` to 1 in your train.py, or temporarily set `PATCH_SIZE=(64,64,64)` locally. |
| DataLoader hangs forever on Windows | `num_workers > 0` on Windows | Keep `NUM_WORKERS=0` on Windows; bump to 4 on Linux/WSL2 only. |
| HD95 = NaN in metrics | The prediction or ground-truth mask was empty for that case | Expected. `MetricTracker` already skips NaNs in the mean. |
| `RuntimeError: Grad-CAM hooks did not fire` | The `target_layer` you passed is never executed in `forward()` | Pick a layer that is actually in the forward path — typically the last encoder block or the conv just before the classifier. |
| GPU memory grows after Grad-CAM | Forgot to call `cam.remove_hooks()` | Always release hooks after use. |
| `cudnn.benchmark = True` after training | Determinism was disabled somewhere | `set_global_seed()` re-asserts `deterministic=True`, `benchmark=False`. Make sure it is the first import. |

---

## 8. Weekly milestones

| Week | Phase                           | Owner          | Deliverable                                            |
|------|---------------------------------|----------------|--------------------------------------------------------|
| 1    | Phase 0 — shared infrastructure | Lead           | repo skeleton + sanity check passing                   |
| 2    | Phase 1 — unimodal baselines    | M1, M2, M3, M4 | first training runs, baseline Dice on validation set   |
| 3    | Phase 1 wrap + Phase 2 multimodal | M5           | multimodal model trained, all baselines tuned          |
| 4    | Phase 3 — XAI                   | All            | Grad-CAM overlays for every member, qualitative figures|
| 5    | Phase 4 — ensemble + UI         | M5 + lead      | late-fusion ensemble + Streamlit demo                  |
| 6    | Report + slides                 | All            | written report, final figures and tables               |

---

## TL;DR — first 30 minutes

1. `pip install -r requirements.txt` and install PyTorch+CUDA from pytorch.org.
2. Edit [data/raw/DATA_PATH.txt](data/raw/DATA_PATH.txt) to your local BraTS path. Don't commit.
3. `python shared/sanity_check.py` → must print `ALL TESTS PASSED`.
4. **Do not** run `shared/create_splits.py`. Splits are already committed.
5. Open your member folder. Implement `model.py`. Copy [member1_T1n/train.py](member1_T1n/train.py) into your `train.py`, swap `MODALITY`. Train.
6. Then implement `evaluate.py` and `xai_analysis.py`.
7. Branch as `member{N}/{topic}`. PR. Get a review. Merge.

When in doubt, re-read this file or ask in the team channel.
