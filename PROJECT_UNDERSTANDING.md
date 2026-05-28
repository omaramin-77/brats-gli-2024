# BraTS GLI-2024 — XAI-Guided Multimodal Brain Tumour Segmentation
## Master Technical Project Understanding (Reverse-Engineered Reference)

> Purpose: This document is a self-contained, AI-readable, deeply technical
> reverse-engineering of the entire `brats-gli-2024` repository. A reader (human
> or AI) should be able to fully reason about the project's research thesis,
> code architecture, training/evaluation pipelines, model designs, experiment
> management, and obtained numerical results without ever opening a source
> file. All quoted hyperparameters, shapes, metrics, and numbers are extracted
> directly from the code, training-meta sidecar JSONs, and CSV results.

---

## 1. Project Identity and Research Thesis

### 1.1 What the project is
A 5-person collaborative research project that trains five independent 3D
brain-tumour segmentation pipelines on the **BraTS GLI-2024 adult-glioma**
dataset, then uses 3D Grad-CAM and per-modality input ablation as an
**XAI-driven architecture-design signal** for a final multimodal **latent-space
fusion ensemble**.

The novel claim is not that another fusion ensemble works — it is that *XAI
attribution (Grad-CAM + ablation drops) can be used to initialise the
attention-gate prior of a learned fusion head*, replacing random initialisation
with a data-driven prior derived from how much each modality contributes to
each tumour sub-region.

### 1.2 Five members, five pipelines, one shared infrastructure

| Member | Folder | Modality | Architecture | Role |
|--------|--------|----------|-----------------|------|
| M1 | `member1_T1n/` | T1n (native T1) | **3D Residual U-Net + Deep Supervision** | Sparse-positive modality baseline |
| M2 | `member2_T1c/` | T1c (contrast-enhanced T1) | **3D Attention-Gated U-Net** | Bright-blob enhancing-tumour expert |
| M3 | `member3_T2w/` | T2w | **Swin-UNETR (MONAI)** | Long-range oedema context, transformer |
| M4 | `member4_T2f/` | T2-FLAIR | **3D Residual U-Net + (SegResNet alt.)** | Standardised FLAIR baseline |
| M5 | `member5_multimodal/` | Stacked 4-channel (T1c,T1n,T2f,T2w) | **MultimodalUNet3D** (M1 skeleton, in_channels=4) | Naive multimodal baseline + ablation source |
| M5.2 | `ensemble/` (Pipeline C) | Bottleneck features from M1–M4 | **LatentFusionEnsemble** (gated attention) | XAI-initialised late-fusion model |

### 1.3 Phases (executed in order)
- **Phase 0** – shared infrastructure (data, splits, preprocessing, metrics, GradCAM, sanity check).
- **Phase 1** – four unimodal baselines trained on T1n/T1c/T2w/T2f.
- **Phase 2** – naive 4-channel multimodal baseline (M5 Pipeline A).
- **Phase 3** – Grad-CAM per sub-region per view (27 PNG overlays per member) + modality ablation (Pipeline B, M5).
- **Phase 4** – Latent Space Fusion Head (Pipeline C, M5.2): bottlenecks of M1–M4 are pre-computed, then a small gated-attention + decoder is trained with attention bias initialised from ablation scores.
- **Phase 5** – FastAPI inference & comparison backend (`interface/backend/`) with model registry, sessions, sliding-window inference, comparison endpoints, PDF reports.

---

## 2. Repository Layout (Exhaustive)

```
brats-gli-2024/
├── README.md                  high-level pitch, milestones
├── ONBOARDING.md              full new-developer guide (kept in sync with shared/)
├── requirements.txt           dependencies (PyTorch + MONAI + MLflow + etc.)
├── .claude/                   internal AI-coding instructions (claude.md/design.md/plan.md)
├── download_split.py          dataset bootstrap helper
├── visualize_per_modality.py  manual visualisation helper
│
├── shared/                    SINGLE SOURCE OF TRUTH for everything reusable
│   ├── config.py              paths, seeds, GPU-aware hyperparameters, env detection
│   ├── seed.py                set_global_seed(seed=42) — first line in every entry point
│   ├── preprocessing.py       per-patient z-score, brain-bbox crop, resize, patches, augs
│   ├── dataset.py             BraTSDataset, get_dataloader, load_splits
│   ├── trainer.py             dice_bce_loss, focal_loss, train/val loops, EarlyStopper, CheckpointManager
│   ├── metrics.py             Dice, IoU, HD95, MetricTracker, compute_all_metrics (9 keys)
│   ├── grad_cam_3d.py         GradCAM3D context manager (3D extension of Grad-CAM)
│   ├── visualization.py       matplotlib helpers (training curves, Grad-CAM overlays, ablation bars)
│   ├── create_splits.py       stratified 350-patient subset builder (RUN ONCE, do not re-run)
│   └── sanity_check.py        Phase-0 end-to-end smoke test
│
├── data/
│   ├── raw/DATA_PATH.txt      absolute path to BraTS root on disk
│   ├── raw/DATA_PATH.local.txt (gitignored) machine-local override
│   ├── processed/             cache (gitignored; currently unused on Windows)
│   └── splits/                train_ids.txt (280), val_ids.txt (35), test_ids.txt (35), split_stats.json
│
├── member1_T1n/   model.py train.py evaluate.py xai_analysis.py extract_features.py
├── member2_T1c/   model.py train.py evaluate.py xai_analysis.py extract_features.py
├── member3_T2w/   model.py train.py evaluate.py xai_analysis.py extract_features.py
├── member4_T2f/   model.py train.py evaluate.py xai_analysis.py extract_features.py notebook_M4.ipynb app.py
├── member5_multimodal/   model.py train.py evaluate.py xai_analysis.py ablation.py extract_features.py
│
├── ensemble/
│   ├── fusion.py              ModalityGatedFusion, LatentFusionDecoder, LatentFusionEnsemble, build_gate_bias
│   ├── features_dataset.py    FeaturesDataset (loads pre-saved bottleneck tensors)
│   ├── cache_labels.py        pre-computes 3-channel WT/TC/ET labels to .pt for fast fusion training
│   ├── train_fusion.py        trains LatentFusionEnsemble (Pipeline C)
│   └── evaluate_fusion.py     test-set evaluation of fusion ensemble
│
├── interface/
│   ├── backend/   FastAPI service (main.py, registry.py, inference.py, sessions.py, routes/*)
│   ├── frontend/  Loveable React UI (placeholder)
│   └── LOVEABLE_PROMPT.md     prompt used to scaffold the UI
│
├── experiments/mlruns/        MLflow tracking artefacts (gitignored)
└── results/
    ├── checkpoints/           .pt model weights (gitignored except in repo here)
    ├── features/              .pt bottleneck tensors per (member, patient)
    │   ├── M1_T1n/   M2_T1c/   M3_T2w/   M4_T2f/   M5_Multimodal_4ch/   labels/
    ├── figures/               training_curves_*, gradcam_*, ablation_bar_chart, sanity_check_patient, per-modality subfolders T1n/T2f/T2w
    └── tables/                test_metrics.csv (master), ablation_scores.csv, modality_importance_scores.json, T2f/
```

---

## 3. Data Pipeline (Raw → Tensor)

### 3.1 Dataset facts (frozen, verified)
- 4 MRI modalities per patient: **T1c, T1n, T2f (FLAIR), T2w**, all 1mm
  isotropic, co-registered, skull-stripped by the BraTS organisers.
- Native shape: **182 × 218 × 182** (the spatial size before preprocessing).
- Raw seg labels: `0` background, `1` necrotic core (NCR), `2` peritumoural
  oedema (ED), `3` enhancing tumour (ET), `4` resection cavity (excluded).
- BraTS sub-region targets (constructed in `BraTSDataset._multichannel_label`):
  - **WT** (whole tumour) = `{1, 2, 3}` — channel 0
  - **TC** (tumour core) = `{1, 3}` — channel 1
  - **ET** (enhancing tumour) = `{3}` — channel 2
- **Channel order is frozen forever**: `("wt", "tc", "et")` for the 3-channel
  target, `("t1c", "t1n", "t2f", "t2w")` for multimodal input.
- Sub-regions OVERLAP by construction: ET ⊂ TC ⊂ WT. Therefore the loss uses
  **per-channel sigmoid** (never softmax) and the metric is computed
  per-channel.

### 3.2 Splits (committed, never re-generated)
- Sampled in `shared/create_splits.py` from 1350 available patients →
  **350-patient stratified subset**, 5 strata by tumour fraction, `seed=42`.
- Sizes: **train 280 / val 35 / test 35** (80/10/10).
- Mean tumour fractions per split (split_stats.json):
  train 1.0194%, val 0.9773%, test 1.0117% — extremely balanced.
- ET-positive cases: train 204, val 24, test 26 (used to ensure ET isn't
  entirely missing from any split).
- The script is documented as **never to be re-run** — splits are committed
  to `data/splits/`.

### 3.3 Preprocessing (`shared/preprocessing.py`)
All preprocessing is **lazy inside `BraTSDataset.__getitem__`** — there is no
offline preprocessing script. The full per-patient pipeline:

1. **Load NIfTI** with `nibabel`, cast to float32 (`load_and_normalise`).
2. **Z-score normalisation over brain voxels only**
   `vol[vol>0]`'s mean and std are used; background voxels are restored to
   exactly 0 after normalisation. Background-inclusive normalisation would
   shift the background to a constant negative value and waste capacity.
3. **Brain bounding-box crop** with 8-voxel padding (`crop_to_brain`). Per-axis
   clipping against `vol.shape` (not against a scalar) — this matters for the
   non-cubic native 182×218×182 input.
4. **Resize to (128, 128, 128)** (`resize_volume` → `scipy.ndimage.zoom`).
   - `order=1` (trilinear) for image volumes.
   - `order=0` (nearest neighbour) for segmentation masks — mandatory to keep
     integer labels exact. `_force_shape` then trims/pads off-by-ones.
5. **Multichannel label construction** (in `BraTSDataset`):
   `wt = (seg==1)|(seg==2)|(seg==3)`, `tc = (seg==1)|(seg==3)`, `et = (seg==3)`,
   stacked into `(3, 128, 128, 128) int64`.
6. **Tumour-biased patch extraction** (`extract_patch`, train only):
   `PATCH_SIZE = (96,96,96)` if VRAM ≥ 14 GB else `(64,64,64)`.
   With `tumour_bias = 0.8` the patch centre is drawn from a tumour voxel;
   otherwise random. The chosen `(starts, ends)` slice tuple is REUSED to crop
   the 3-channel target so image and label are byte-aligned.
   `patches_per_volume = 4` — dataset `__len__` is `num_patients * 4`.
7. **Augmentation** (`get_augmentation_transforms` — MONAI Compose, train only):
   - `RandFlipd(prob=0.5, spatial_axis=[0,1,2])`
   - `RandRotate90d(prob=0.5)`
   - `RandGaussianNoised(prob=0.3, std=0.05)`
   - `RandScaleIntensityd(factors=0.1, prob=0.3)`

For **multimodal** (`preprocess_patient_multimodal`):
- Single brain-bbox crop derived from T1c (most reliable delineation).
- Same crop slice reused for T1n / T2f / T2w / seg → guarantees voxel
  correspondence across modalities.
- Resize each modality independently with `order=1`, stack into `(4, 128³)`.

### 3.4 BraTSDataset contract (`shared/dataset.py`)
```python
BraTSDataset(
    data_root, split_file, modality,        # modality ∈ {"t1n","t1c","t2w","t2f","multimodal"}
    patch_size=64, tumour_bias=0.8,
    augment=False, patches_per_volume=4,
    full_volume=False                      # True → return entire 128³ volume; used by val/test
)
```
- `__getitem__` returns `{"image": (C,H,W,D) float, "label": (3,H,W,D) float, "patient_id": str}`.
- `full_volume=True` forces `patches_per_volume=1` and disallows augmentation
  — validation and test loaders **MUST** use this mode. Patch-based eval
  numbers are not comparable to BraTS leaderboards.
- `get_dataloader` returns a PyTorch `DataLoader` with `pin_memory=cuda?`,
  `num_workers=0` on Windows (fork is unreliable), `4` on Linux/WSL2.

### 3.5 FeaturesDataset (`ensemble/features_dataset.py`)
Used by Pipeline C only. For each patient:
- Loads four `.pt` files: `results/features/{M1_T1n,M2_T1c,M3_T2w,M4_T2f}/{pid}.pt`.
- Normalises shape to `(256, 8, 8, 8)` (M3's Swin saves `(1,256,4,4,4)` so a
  trilinear upsample to `(8,8,8)` is applied with a one-time warning).
- Loads cached 3-channel label `(3, 128, 128, 128)` from
  `results/features/labels/{pid}.pt` (built by `ensemble/cache_labels.py`).
- Returns `{"features": {"t1c","t1n","t2f","t2w"}, "label": ..., "patient_id": ...}`.

---

## 4. Shared Training Infrastructure (`shared/trainer.py`)

### 4.1 Losses

#### `dice_bce_loss(pred_logits, target, dice_weight=1.0, bce_weight=0.5, channel_weights=None, eps=1e-8)`
- Per-channel **soft-Dice + BCE**.
- Reduces over spatial dims only, then mean over `(batch, channel)`.
- Spatial dims taken dynamically (`range(2, pred.ndim)`), so it works for
  patch or full-volume tensors.
- BCE uses `binary_cross_entropy_with_logits` per voxel, reduced per-channel
  matching the Dice term, so rare ET is not dwarfed by WT in the gradient.
- The model outputs are RAW LOGITS — sigmoid happens inside the loss and at
  metric time, NEVER inside any model.

#### `focal_loss(pred_logits, target, gamma=2.0, alpha=0.25)`
- Per-channel focal loss; M2 ships a `--loss focal` ablation variant
  (`dice_focal_loss = dice_only + 0.5*focal`).

### 4.2 Train/validate loops

#### `train_one_epoch(model, loader, optimizer, scaler, device, loss_fn=dice_bce_loss)`
- `model.train()`.
- For each batch: forward in `torch.amp.autocast(device_type="cuda")` if AMP
  scaler is provided, otherwise plain forward.
- Backward → `scaler.unscale_` → `clip_grad_norm_(GRAD_CLIP_NORM=1.0)` →
  `scaler.step` → `scaler.update`.
- Returns `{"loss": mean batch loss}`.

#### `validate_one_epoch(model, loader, device, loss_fn=dice_bce_loss)`
- `model.eval()`, `torch.no_grad()`.
- For each batch (which is a full-volume tensor in val/test): runs
  `monai.inferers.sliding_window_inference(roi_size=PATCH_SIZE, sw_batch_size=4, overlap=0.5)`.
- Computes `compute_all_metrics(logits, labels)` per batch — 9 keys.
- Aggregates via `MetricTracker` (skips NaNs in mean).

### 4.3 Metrics (`shared/metrics.py`)
- `compute_dice(p, t, eps=1e-8)` — `2|p∩t| / (|p|+|t|)` on binary tensors.
- `compute_iou(p, t)` — `|p∩t| / |p∪t|`.
- `compute_hd95(p, t)` — uses MONAI `HausdorffDistanceMetric(percentile=95)`;
  returns NaN if either mask is empty (geometrically undefined).
- `compute_all_metrics(pred_logits, target, channel_names=("wt","tc","et"), threshold=0.5)`:
  applies sigmoid + 0.5 threshold per channel → 9 keys
  (`dice_wt`,`dice_tc`,`dice_et`,`iou_wt`,`iou_tc`,`iou_et`,`hd95_wt`,`hd95_tc`,`hd95_et`).
- `MetricTracker` accumulates batch dicts and computes NaN-skipping means.

### 4.4 Early stopping & checkpointing

#### `EarlyStopper(patience=15, min_delta=1e-3)`
- Tracks `best_dice` initialised to `-inf`.
- Accepts dict (preferred) or float; pulls `BEST_METRIC="dice_wt"` from config.
- Returns `True` after `patience` consecutive non-improvements.

#### `CheckpointManager(save_dir, member_name)`
- `.save(model, optimizer, epoch, val_metrics, is_best=False, best_metric_key="dice_wt")` →
  writes `{name}_epoch{NNN}.pt` always, `{name}_best.pt` when `is_best=True`.
  Payload: `{"epoch","val_dice","val_metrics","best_metric_key","model_state","optimizer_state"}`.
- `.load_best(model, optimizer=None)` → restores state-dicts; returns
  `(model, optimizer, epoch, val_dice_wt)`.

### 4.5 Configuration (`shared/config.py`)

| Constant | Value | Notes |
|---|---|---|
| `GLOBAL_SEED` | 42 | Frozen |
| `BATCH_SIZE` | 1 | Conservative for 3D volumes |
| `NUM_WORKERS` | 0 (Windows) / 4 (Linux) | Avoids DataLoader deadlock on Win32 |
| `PATCH_SIZE` | (96,96,96) if VRAM ≥14 GB else (64,64,64) | Auto-detected |
| `LR` | 1e-4 | AdamW |
| `WEIGHT_DECAY` | 1e-5 | |
| `NUM_EPOCHS` | 50 | Cap; EarlyStopper(patience=15) shortens |
| `PATIENCE` | 15 | |
| `VAL_EVERY_N_EPOCHS` | 5 | + always validate at last epoch |
| `GRAD_CLIP_NORM` | 1.0 | |
| `AMP_ENABLED` | True | `torch.amp.autocast` + `GradScaler` |
| `MODALITIES` | `("t1c","t1n","t2f","t2w")` | Frozen order |
| `TARGET_CHANNELS` | 3 | Always |
| `TARGET_CHANNEL_NAMES` | `("wt","tc","et")` | Frozen order |
| `BEST_METRIC` | `"dice_wt"` | WT is the stablest early-training signal |
| `IS_KAGGLE` | True/False | Triggers Kaggle-specific path setup |

### 4.6 Reproducibility — `set_global_seed(42)`
- Seeds Python `random`, NumPy, `torch`, `torch.cuda` (all devices).
- `torch.backends.cudnn.deterministic = True`, `cudnn.benchmark = False`.
- `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Sets `PYTHONHASHSEED=42`.
- **Must be the first executable line** of every entry-point script. This is
  enforced as a rule and verified by code review.

---

## 5. Model Architectures (Tensor-Level Specification)

All models share these contracts:
- **Input**: `(B, C, H, W, D)` float, `C=1` for unimodal, `C=4` for multimodal.
- **Output**: `(B, 3, H, W, D)` raw logits (channels = WT, TC, ET).
- Every model implements `forward_features(x) -> (B, 256, H/16, W/16, D/16)`.
  This is the binding contract that lets M5's Latent Fusion Head consume any
  member's bottleneck. Models with a native bottleneck ≠ 256 channels add a
  1×1×1 projection conv.

### 5.1 M1 — `ResidualUNet3D` (T1n) — `member1_T1n/model.py`
4-level 3D Residual U-Net with **deep supervision**.

```
in (B,1,64,64,64) or (B,1,128,128,128)
  enc1 ResidualBlock(1→32)     skip s1   --pool2→ (32, H/2,...)
  enc2 ResidualBlock(32→64)    skip s2   --pool2→ (64, H/4,...)
  enc3 ResidualBlock(64→128)   skip s3   --pool2→ (128, H/8,...)
  enc4 ResidualBlock(128→256)  skip s4   --pool2→ (256, H/16,...)
  bottleneck ResidualBlock(256→256)               ← forward_features
  dec1 ConvT3d(256→128, str2) ⨁ s4 → ResBlock(128+256→128)
  dec2 ConvT3d(128→64,  str2) ⨁ s3 → ResBlock(64+128→64)
  dec3 ConvT3d(64→32,   str2) ⨁ s2 → ResBlock(32+64→32)
  dec4 ConvT3d(32→32,   str2) ⨁ s1 → ResBlock(32+32→32)
  out_head Conv3d(32→3, 1×1×1)               → main_pred (B,3,H,W,D)
  aux2_head Conv3d(64→3, 1×1×1) on dec2 + trilinear up to H
  aux3_head Conv3d(32→3, 1×1×1) on dec3 + trilinear up to H
  aux4_head Conv3d(32→3, 1×1×1) on dec4 + (already full-res)
```

**ResidualBlock3D**: two 3×3×3 conv-BN-ReLU layers + identity skip (1×1×1
projection when in≠out channels).

**Forward contract**:
- **training mode** → returns 4-tuple `(main_pred, aux2, aux3, aux4)` all of
  shape `(B,3,H,W,D)`.
- **eval mode** → returns `main_pred` only.

**Loss (M1 train.py)**: `deep_supervision_loss(outputs, target)`:
`L = L_main + 0.5*L_aux2 + 0.25*L_aux3 + 0.125*L_aux4`, each summand is
`dice_bce_loss`.

**Grad-CAM target**: `enc4` (last encoder block; bottleneck-resolution feature
just before the bottleneck pool).

### 5.2 M2 — `AttentionUNet3D` (T1c) — `member2_T1c/model.py`
4-level 3D Attention-Gated U-Net (Oktay et al. 2018, 3D variant).

```
Identical 32→64→128→256 encoder/decoder spine, ConvBlock3D
(two conv-BN-ReLU, no residual skip).
Per skip connection:
    g_i = up_i(d_(i-1))                  # gating signal
    s_i_att = AttentionGate3D(g_i, s_i)  # multiplicative attention
    d_i = ConvBlock(cat([g_i, s_i_att]))
```
`AttentionGate3D` computes:
`alpha = sigmoid(psi(ReLU(W_g(g) + W_x(x))))`, then `out = x * alpha`.

**No deep supervision** (attention already regularises decoder levels).
**No deep-supervision tuple**: `forward` always returns single tensor in both
modes.

Two training variants are supported via `--loss` flag:
- `M2_T1c_AttUNet` (canonical) → `dice_bce_loss`
- `M2_T1c_AttUNet_focal` (experiment 2) → `dice_only + 0.5*focal_loss(γ=2,α=0.5)`

**XAI**: Grad-CAM (target layer `enc4`) **+ Occlusion Sensitivity** (8³ window,
stride 4 or 8) — M2 is the only member that runs occlusion in addition to
Grad-CAM; cost is ~30 minutes for one patient at 128³.

### 5.3 M3 — `M3T2wSwinUNETR` (T2w) — `member3_T2w/model.py`
Wraps `monai.networks.nets.SwinUNETR(img_size=(64,64,64), feature_size=24, use_checkpoint=True)`.

**forward_features specifics**:
- Native Swin bottleneck spatial = `H/32` (one patch_embed + 4 PatchMerging
  stages, each ×2) → for 64³ input, `(B, 384, 2, 2, 2)`.
- A trained `feature_projector = Conv3d(feature_size*16=384 → 256, 1×1×1)`
  brings channels to 256.
- Then `F.interpolate(mode="trilinear")` upsamples spatial back to `H/16` to
  match the cross-member contract (`(B, 256, 8, 8, 8)` for 128³ input).

**Grad-CAM target**: `model.decoder5` if present, else `model.encoder10`, else
the second-to-last child module.

**Patch size**: trained at 64³ (Swin requires `img_size` known); features were
extracted with the same 64³ window — which is why M3's saved features are
`(1, 256, 4, 4, 4)` and `FeaturesDataset` trilinearly upsamples them to
`(256, 8, 8, 8)` with a one-time warning.

### 5.4 M4 — `T2fSegModel` / `T2fSegResNet` (T2-FLAIR) — `member4_T2f/model.py`
Two architectures are trained side by side and compared:

#### (a) `T2fSegModel` (ResUNet3D) — ~9.9 M params
Identical 4-level residual U-Net with `base_ch=32` and a 1×1×1 projection
conv on the bottleneck output that guarantees 256 channels.
**This is the canonical M4 model** used downstream (checkpoint
`M4_T2f_resunet_best.pt`).

#### (b) `T2fSegResNet` — MONAI SegResNet wrapper, ~4.7 M params
- Wraps `monai.networks.nets.SegResNet(spatial_dims=3, init_filters=32, dropout_prob=0.1)`.
- At init time, runs a 32³ probe to discover the native bottleneck spatial
  and channel count by hooking `down_layers.*` modules.
- Computes `extra_pool = MaxPool3d(2)^k` to reach `H/16`.
- Adds a 1×1×1 projection to 256 channels.
- `forward_features` re-attaches a hook on the last encoder group and returns
  `proj(extra_pool(cached_bottleneck))`.

**Build factory**: `build_model(arch="resunet"|"segresnet")`.

**Train script** uses argparse `--arch {resunet,segresnet,both}` and saves
`M4_arch_comparison.csv` summarising the two runs. The ResUNet is shipped as
the canonical model; the SegResNet variant is documented as an alternative.

### 5.5 M5 (Pipeline A) — `MultimodalUNet3D` — `member5_multimodal/model.py`
M1's ResidualUNet3D skeleton with `in_channels=4` and **no deep supervision**
(returns single tensor in both modes).
- Encoder: `4 → 32 → 64 → 128 → 256`.
- Bottleneck (256 ch) is also exposed via `forward_features`.
- Decoder identical to M1.
- This model is the **floor** the Pipeline-C latent fusion must beat to
  validate the research thesis.

### 5.6 M5.2 (Pipeline C) — `LatentFusionEnsemble` — `ensemble/fusion.py`
The project's novel contribution. Operates on pre-computed bottlenecks.

#### `ModalityGatedFusion(channels=256, num_modalities=4, gate_bias_init: (4,))`
- `attention_conv = Conv3d(256*4=1024 → 4, 1×1×1, bias=True)`.
- When `gate_bias_init` is provided, the conv's weights are **zeroed** and its
  bias is set to `gate_bias_init`. At training step 0, attention(any voxel)
  = `softmax(bias)` → exactly the XAI-derived prior.
- Forward: concat 4 bottlenecks along channels → `attention_conv` → `softmax`
  across modality dim → weighted sum of the 4 bottlenecks.
- Returns `(fused (B,256,D,H,W), attention (B,4,D,H,W))`.

#### `LatentFusionDecoder(in_channels=256, out_channels=3)`
- Three `_UpBlock(ConvTranspose3d(×2) + 2× conv-BN-ReLU)`: 8→16→32→64.
- Final `F.interpolate(size=128, mode="trilinear", align_corners=False)`.
- Head `Conv3d(32→3, 1×1×1)`.

#### `LatentFusionEnsemble(gate_bias_init)`
- Wraps both modules.
- `forward(features: {"t1c","t1n","t2f","t2w"} of (B,256,8,8,8))` → `(logits (B,3,128,128,128), attention)`.

#### `build_gate_bias(json_path)`
Reads `modality_importance_scores.json`, averages drops across
{WT,TC,ET} per modality (ε=1e-3 to avoid log(0)), normalises into a softmax
prior, returns `log(avg / avg.sum())` as the bias vector. This is the
**XAI-as-design-signal** mechanism.

---

## 6. Per-Member Training Pipelines (Side-by-Side)

| | M1 | M2 | M3 | M4 | M5 | M5.2 (Fusion) |
|---|---|---|---|---|---|---|
| Arch | ResUNet3D | AttUNet3D | SwinUNETR | ResUNet3D | MultimodalUNet3D | LatentFusionEnsemble |
| Modality | t1n | t1c | t2w | t2f | multimodal (4ch) | features dict |
| Loss | DiceBCE + DS (0.5/0.25/0.125) | DiceBCE *or* Dice+0.5·focal | DiceBCE | DiceBCE | DiceBCE | DiceBCE |
| Optimizer | AdamW(lr=1e-4, wd=1e-5) | same | same | same | same | AdamW(lr=3e-4, wd=1e-5) |
| Scheduler | CosineAnnealingLR(T_max=50) | same | same | same | same | CosineAnnealingLR(T_max=30) |
| Patch size | 64³ or 96³ (auto) | same | (64,64,64) (Swin fixed) | same | same | n/a — fixed (256,8,8,8) features |
| Batch size | 1 | 1 | 1 | 1 | 1 | 4 |
| AMP | yes | yes | yes | yes | yes | yes |
| Augmentation | flips, rot90, gauss noise, intensity scale | same | same | same | same | none |
| Tumour bias | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | n/a |
| `patches_per_volume` | 4 | 4 | 4 | 4 | 4 | 1 |
| Val mode | `full_volume=True` + SWI | same | same | same | same | feature-level (no SWI) |
| Epochs | 50 | 50 | 50 | 50 | 50 | 30 |
| `VAL_EVERY_N_EPOCHS` | 5 (+ last) | 5 | 5 | 5 | 5 | 5 |
| EarlyStopper | patience=15 on dice_wt | same | same | same | same | same |
| MLflow run name | `M1-T1n-ResUNet-seed42` | `M2-T1c-AttUNet-dicebce` (or `-focal`) | `M3-T2w-SwinUNETR-seed42` | `M4-T2F-ResUNet3D-seed42` / `M4-T2F-SegResNet-seed42` | `M5-multimodal-naive-seed42` | `M5-fusion-XAI-initialised-seed42` |
| Checkpoint name | `M1_T1n_ResUNet` | `M2_T1c_AttUNet` (or `…_focal`) | `M3_T2w_SwinUNETR` | `member4_T2f_resunet` | `M5_Multimodal_4ch` | `M5_LatentFusion_XAI` |

### 6.1 Training loop skeleton (all members)
```python
set_global_seed()                                  # FIRST line
train_ds = BraTSDataset(..., patches_per_volume=4, augment=True)
val_ds   = BraTSDataset(..., full_volume=True, augment=False)
train_loader = get_dataloader(train_ds, batch_size=1, shuffle=True, num_workers=6)
val_loader   = get_dataloader(val_ds, batch_size=1, shuffle=False, num_workers=2)

model = build_model().to(device)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
scaler = torch.amp.GradScaler("cuda") if AMP_ENABLED else None
stopper = EarlyStopper(patience=15)
ckpt = CheckpointManager(CHECKPOINT_DIR, MEMBER_NAME)

for epoch in 1..NUM_EPOCHS:
    train_metrics = train_one_epoch(model, train_loader, opt, scaler, device, loss_fn)
    scheduler.step()
    if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS:
        val_metrics = validate_one_epoch(model, val_loader, device, loss_fn)
        is_best = val_metrics["dice_wt"] > best_dice
        ckpt.save(model, opt, epoch, val_metrics, is_best=is_best)
        if stopper.should_stop(val_metrics): break

# sidecar JSON: training_time_hrs, gpu_memory_gb
# training curves PNG saved via plot_training_curves
# MLflow run is wrapped around the whole loop with params + per-epoch metrics
```

### 6.2 Evaluation (`member*/evaluate.py`)
- Loads `test_ids.txt` (the only place the test split is read).
- Loads best checkpoint via `CheckpointManager.load_best`.
- Builds `BraTSDataset(..., full_volume=True)` and a `DataLoader(batch_size=1)`.
- Calls `validate_one_epoch` (which internally runs MONAI sliding-window
  inference with `roi_size=PATCH_SIZE, sw_batch_size=4, overlap=0.5`).
- Appends one row to `results/tables/test_metrics.csv`:
  `member,modality,architecture,seed,dice_wt,dice_tc,dice_et,iou_wt,iou_tc,iou_et,hd95_wt,hd95_tc,hd95_et,training_time_hrs,gpu_memory_gb`.

### 6.3 XAI analysis (`member*/xai_analysis.py`)
- **Test-set contamination guard**: refuses to run unless
  `test_metrics.csv` exists (i.e. evaluate.py has been executed already).
- Picks three test patients deterministically: `[ids[0], ids[mid], ids[-1]]`.
- For each patient:
  - Preprocess via `preprocess_patient`.
  - Build per-sub-region GT masks `wt_gt, tc_gt, et_gt`.
  - Open `with GradCAM3D(model, target_layer) as cam:` and call
    `cam.generate(x, target_channel=sr)` for `sr ∈ {wt, tc, et}`.
  - For each sub-region × view ∈ {axial(2), coronal(1), sagittal(0)}, find the
    slice with the largest sub-region area (`argmax(sum over other axes)`),
    extract `(vol_slice, cam_slice, sr_seg_slice)`, save via
    `plot_gradcam_overlay`.
  - Output filename: `gradcam_M{N}_{pid}_{sr}_{view}.png` (9 PNGs/patient,
    27 per member).
- M2 additionally runs **Occlusion Sensitivity** for the middle patient.

### 6.4 Feature extraction (`member*/extract_features.py`)
- Loads the best checkpoint.
- Iterates over **train + val + test patients in one pass** via the
  `full_volume=True` `BraTSDataset` (overriding `patient_ids`).
- For each patient: `features = model.forward_features(image)` → save to
  `results/features/{member}/{pid}.pt` (shape `(1, 256, 8, 8, 8)`; M3 saves
  `(1, 256, 4, 4, 4)` because of the Swin fixed-size constraint).

### 6.5 Ablation study (`member5_multimodal/ablation.py`)
- Loads `M5_Multimodal_4ch_best.pt`.
- For each test patient (sliding-window inference):
  - Baseline forward (no ablation).
  - 4 ablation passes — zero one of the 4 channels, re-run inference.
- Aggregates per-modality Dice drops: `drop = baseline - ablated`, clamped to
  `max(0, drop)` (negative drops mean the model didn't need that modality;
  a clip prevents the bias from rewarding *removing* a modality).
- Writes:
  - `results/tables/ablation_scores.csv` (one row per modality, all 9 cols).
  - `results/tables/modality_importance_scores.json` (nested
    `{sub-region: {modality: positive drop}}` — consumed by `build_gate_bias`).
  - `results/figures/ablation_bar_chart.png` (3-panel WT/TC/ET).

### 6.6 Fusion training (`ensemble/train_fusion.py`)
- Loads `FeaturesDataset(train)` and `FeaturesDataset(val)` — fast because no
  encoder forward passes are needed.
- `LatentFusionEnsemble(gate_bias_init=build_gate_bias(modality_importance_scores.json))`.
- `AdamW(lr=3e-4)`, `CosineAnnealingLR(T_max=30)`, AMP, `clip_grad_norm_(1.0)`.
- Same EarlyStopper + CheckpointManager pattern, on dice_wt.
- 30 epochs (no need for 50 since the fusion head is small and features are
  already learned features).
- Logged in MLflow with run name `M5-fusion-XAI-initialised-seed42` and per-modality
  `gate_bias_*` params.

### 6.7 Fusion evaluation (`ensemble/evaluate_fusion.py`)
- `FeaturesDataset(test_ids)` → `LatentFusionEnsemble` → `validate_one_fusion_epoch`.
- Appends one row to `test_metrics.csv` and prints `Δ Dice` vs the
  `M5_Multimodal_4ch` baseline row.

---

## 7. 3D Grad-CAM (`shared/grad_cam_3d.py`)

```python
with GradCAM3D(model, target_layer) as cam:        # MANDATORY context manager
    heatmap_np = cam.generate(input_tensor,
                              target_channel="wt"|"tc"|"et"|int,
                              target_mask=None)
    # shape: input_tensor.shape[2:]   range [0, 1]
```

Internals:
- Registers `register_forward_hook` (captures activation at `target_layer`)
  and `register_full_backward_hook` (captures gradient w.r.t. layer output).
- `__enter__` saves and forces `requires_grad_(True)` on all params; `__exit__`
  releases hooks and restores `requires_grad`.
- `generate`:
  1. Re-forwards the (cloned, requires-grad) input.
  2. Selects `channel_logits = output[:, ch:ch+1]`.
  3. Builds `target_mask` = `sigmoid(channel_logits) > 0.5` (predicted-positive
     voxels) unless caller provides one.
  4. Loss = `(channel_logits * target_mask).mean()` — scalar.
  5. `model.zero_grad(set_to_none=True); loss.backward()`.
  6. Weights = `mean(grad, dim=(2,3,4), keepdim=True)`.
  7. CAM = `ReLU((weights * activations).sum(dim=1, keepdim=True))`.
  8. Trilinear interpolation to input spatial shape.
  9. Min-max normalise to `[0, 1]` (with `+1e-8`).
- **Per-channel** is mandatory — averaging Grad-CAM across the 3 output
  channels is explicitly disallowed because it answers no useful question.

---

## 8. FastAPI Backend (`interface/backend/`)

Production-style inference & comparison service.

### 8.1 Architecture
- `main.py` — FastAPI app with `lifespan` startup; periodic session sweeper
  every 3600s; CORS middleware; exception handlers for `KeyError`/
  `FileNotFoundError`/`ValueError`.
- `registry.py` — Declarative `ModelEntry` catalogue with **lazy** model
  builders. `_LRUWeights(cap=MODEL_CACHE_SIZE=3)` caches loaded `nn.Module`s
  and `torch.cuda.empty_cache()` on eviction. Six entries:
  `m1_t1n, m2_t1c, m3_t2w, m4_t2f, m5_baseline, m6_late_fusion`. M6 is a
  reserved slot enabled when `ensemble/late_fusion.py` and
  `M6_LateFusion_best.pt` are dropped in — no code change.
- `sessions.py` — file-backed sessions under `interface/backend/sessions/{sid}/`
  with per-prediction caching (`prediction_{model_key}.json`, masks, slices).
  TTL (default 6 hours) is swept once per hour.
- `inference.py` — single-model inference path:
  1. `preprocess_session` calls `shared.preprocessing.preprocess_patient` (or
     `preprocess_patient_multimodal`) → `(C, 128³)` image + `(3, 128³)` GT.
  2. `registry.load(model_key)` returns model on `DEVICE`.
  3. MONAI `sliding_window_inference(roi=PATCH_SIZE, sw_batch_size=2, overlap=0.5)`.
  4. `compute_all_metrics(logits, target_t)` → 9 keys + `inference_time_s`.
  5. Tumour volume in mm³ computed by scaling resized-voxel count by the
     original NIfTI affine voxel size and the 240×240×155 → 128³ ratio.
  6. Stores predicted binary mask, sigmoid probabilities, and metrics under
     the session.
- `routes/` — `health`, `models` (catalogue), `sessions`, `predictions`,
  `comparisons`, `reports`.

### 8.2 Typical UI flow
```
POST /sessions  → {sid}
POST /sessions/{sid}/upload  kind=t1c|t1n|t2w|t2f|seg, file=*.nii.gz
POST /sessions/{sid}/predict   {"model_key":"m2_t1c"}
POST /sessions/{sid}/compare   {"model_keys":["m2_t1c","m5_baseline"]}
GET  /sessions/{sid}/predictions/m2_t1c/mask.nii.gz
GET  /sessions/{sid}/predictions/m2_t1c/slice?view=axial&index=64&region=all
GET  /sessions/{sid}/report.pdf
```

### 8.3 Frontend
Loveable-scaffolded React UI (see `interface/LOVEABLE_PROMPT.md`). Only a
README placeholder exists in `interface/frontend/`. Backend exposes
documented endpoints at `/docs` (FastAPI auto-generated).

---

## 9. Experiment Management

### 9.1 MLflow
- Tracking URI: `experiments/mlruns/` (gitignored).
- One experiment per `MEMBER_NAME`. One run per `MLFLOW_RUN_NAME`.
- Per-run params: `member, modality, architecture, seed, batch_size, lr,
  weight_decay, num_epochs, patch_size, amp, loss` (+ `gate_bias_{mod}` for
  fusion).
- Per-epoch metrics: `train_loss`, `lr`. Per-validation: all 9 sub-region
  keys plus `val_loss`. Per-run summary: `training_time_hrs`, `gpu_memory_gb`,
  `best_val_dice_wt`, `best_epoch`.
- MLflow is **lazily imported**; absent install never blocks training.

### 9.2 Sidecar JSON
Every training run writes `results/checkpoints/{MEMBER_NAME}_train_meta.json`
with at least `training_time_hrs` and `gpu_memory_gb`, picked up by
`evaluate.py` to populate the same fields in `test_metrics.csv`. M3's
sidecar additionally stores all hyperparams + `best_dice_wt`. M5 fusion's
sidecar additionally stores `gate_bias_init`.

### 9.3 Checkpoints layout
```
results/checkpoints/
  M1_T1n_ResUNet_best.pt
  M1_T1n_ResUNet_train_meta.json
  M2_T1c_AttUNet_best.pt
  M2_T1c_AttUNet_train_meta.json
  M3_T2w_SwinUNETR_best.pt
  M3_T2w_SwinUNETR_train_meta.json
  M4_T2f_resunet_best.pt
  M5_Multimodal_4ch_best.pt
  M5_Multimodal_4ch_train_meta.json
  M5_LatentFusion_XAI_best.pt
  M5_LatentFusion_XAI_epoch005…030.pt
  M5_LatentFusion_XAI_train_meta.json
```
Checkpoint payload (set by `CheckpointManager.save`):
```python
{ "epoch": int, "val_dice": float,
  "val_metrics": {... full metric dict ...},
  "best_metric_key": "dice_wt",
  "model_state": state_dict,
  "optimizer_state": state_dict }
```

### 9.4 Test-metrics CSV (`results/tables/test_metrics.csv`)
Master table — one row per (member, modality, architecture, seed). Columns:
```
member, modality, architecture, seed,
dice_wt, dice_tc, dice_et,
iou_wt, iou_tc, iou_et,
hd95_wt, hd95_tc, hd95_et,
training_time_hrs, gpu_memory_gb
```

---

## 10. RESULTS — Final Test-Set Metrics (Authoritative)

Source: `results/tables/test_metrics.csv` (n=35 held-out test patients,
seed=42). All Dice/IoU are higher-is-better; HD95 lower-is-better.

| Member | Modality | Architecture | Dice WT | Dice TC | Dice ET | IoU WT | IoU TC | IoU ET | HD95 WT | HD95 TC | HD95 ET | Train hrs | Peak GPU GB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | t1n | ResidualUNet3D | 0.6207 | 0.3365 | 0.2928 | 0.4958 | 0.2775 | 0.2356 | 14.85 | 18.35 | 18.18 | 1.92 | 2.35 |
| M2 | t1c | AttentionUNet3D | 0.5737 | **0.5935** | **0.5735** | 0.4457 | **0.5059** | **0.4897** | 18.10 | 10.35 | 9.56 | 2.38 | 2.91 |
| M3 | t2w | SwinUNETR | 0.7051 | 0.2993 | 0.2672 | 0.5720 | 0.2284 | 0.1998 | 19.07 | 22.88 | 24.36 | 13.88 | 2.11 |
| M4 | t2f | ResUNet3D | **0.7933** | 0.3227 | 0.3076 | **0.6864** | 0.2789 | 0.2658 | **7.87** | 22.06 | 21.58 | n/a | n/a |
| M5 | multimodal (4ch) | MultimodalUNet3D | **0.7662** | **0.6821** | **0.6881** | **0.6623** | **0.5989** | **0.6081** | 11.90 | **10.32** | **9.60** | 4.10 | 2.17 |
| M5.2 | multimodal-features | LatentFusionEnsemble | 0.4828 | 0.2152 | 0.1912 | 0.3418 | 0.1633 | 0.1404 | 14.15 | 19.22 | 19.51 | 2.41 | 15.72 |

Best per column shown in **bold**.

### 10.1 Unimodal observations
- **M4 (T2-FLAIR ResUNet3D)** is the strongest unimodal model on
  **whole-tumour Dice (0.7933)** and best HD95_WT (7.87 mm). This matches the
  clinical expectation that FLAIR delineates oedema, hence WT, most reliably.
- **M2 (T1c AttUNet)** dominates **tumour-core (TC, 0.5935) and
  enhancing-tumour (ET, 0.5735)** — by a large margin (M1/M3/M4 all
  produce 0.27–0.34 on TC/ET). Attention gates over a contrast-enhanced
  modality match the inductive bias: bright enhancing tumour against a dark
  background.
- **M3 (T2w SwinUNETR)** is second-best on WT (0.7051) but weakest on TC/ET
  (0.30/0.27) — T2w does not provide enhancement, and Swin's training is
  starved at patch size 64³ (cf. M3 training time 13.88h vs M1 1.92h).
- **M1 (T1n ResUNet + deep supervision)** is the weakest WT performer (0.62)
  — native T1 lacks contrast for many tumour regions; deep supervision helps
  ET slightly but ET Dice is still only 0.29.

### 10.2 Naive multimodal (M5 Pipeline A)
- WT 0.7662, TC **0.6821**, ET **0.6881** — best aggregate across all
  sub-regions. It loses to M4 on WT alone (-0.027) but **wins TC by +0.089
  and ET by +0.115 over M2**, and crushes M1/M3/M4 on TC/ET.
- This is the **floor** the XAI-guided fusion must beat. (Spoiler: it does
  not, in this run — see §10.4.)

### 10.3 Modality ablation results (Pipeline B, source of XAI prior)
Source: `results/tables/ablation_scores.csv` and
`modality_importance_scores.json`. Baseline = M5 multimodal Dice.
Drop = `baseline − ablated`. Negative drops clamped to 0 in the JSON.

Ablation Dice (zeroing one input channel of M5 at inference time):

| Ablated mod | WT_ablated | TC_ablated | ET_ablated | Δ WT | Δ TC | Δ ET |
|---|---:|---:|---:|---:|---:|---:|
| t1c | 0.6948 | 0.2061 | 0.2023 | **0.0714** | **0.4760** | **0.4859** |
| t1n | 0.6895 | 0.3751 | 0.4030 | 0.0768 | 0.3070 | 0.2852 |
| t2f | 0.1140 | 0.4234 | 0.4338 | **0.6523** | 0.2587 | 0.2543 |
| t2w | 0.7500 | 0.5876 | 0.5890 | 0.0162 | 0.0945 | 0.0992 |

Interpretation:
- **WT depends overwhelmingly on T2-FLAIR (ΔWT=0.652)** — zeroing T2f
  collapses WT Dice from 0.766 to 0.114. T2w can be removed with almost no
  WT penalty (ΔWT=0.016).
- **TC and ET depend most on T1c** (ΔTC=0.476, ΔET=0.486), confirming
  contrast-enhanced T1 is the channel that drives core/enhancing accuracy.
- **T2w is the least-important modality** across all three sub-regions
  (drops 0.02 / 0.09 / 0.10).
- **T1n's drops (0.08/0.31/0.29)** show it adds modest contribution to TC/ET
  by providing tissue contrast even without a contrast agent.

### 10.4 XAI-initialised gate bias used by Pipeline C
Source: `M5_LatentFusion_XAI_train_meta.json`. `gate_bias_init` of the
attention conv (order `(t1c, t1n, t2f, t2w)` after `build_gate_bias`):
```
t1c : -1.0923    softmax≈0.214
t1n : -1.5256    softmax≈0.135
t2f : -0.9725    softmax≈0.245
t2w : -2.6748    softmax≈0.043
```
i.e. T2f and T1c get the largest initial attention weights (matching the
ablation finding); T2w is heavily down-weighted, T1n receives moderate
weight. These are the conv biases at training step 0; the conv weights are
zeroed at init so attention is bias-driven until learning kicks in.

### 10.5 Pipeline C — Latent Fusion Ensemble
- **WT 0.4828, TC 0.2152, ET 0.1912** — well **below** all unimodal models
  except M1/M3 on TC/ET, and far below the naive M5 baseline
  (Δ vs M5: WT -0.283, TC -0.467, ET -0.497).
- Training time 2.41 h, peak GPU 15.72 GB (much higher than any unimodal
  because four 256-channel bottlenecks are concatenated to a 1024-channel
  attention input). 30 epochs.
- **Conclusion**: in this run the XAI-initialised latent fusion did **not**
  beat the naive multimodal baseline. Plausible causes:
  - M3's saved features were extracted at `(1,256,4,4,4)` instead of
    `(1,256,8,8,8)` (Swin patch-size constraint), then trilinearly upsampled
    in `FeaturesDataset`. That's a loss of spatial information for one of
    the four modalities.
  - Feature compatibility: M1/M2/M5 bottlenecks share the same residual
    skeleton but M4 is a different residual U-Net, and M3 is a transformer.
    The 256-channel projection is not a shared learnt embedding, so the
    fused features may not be semantically aligned across modalities.
  - The decoder is small (8³→128³ via three 2× upsamples + final trilinear
    interpolation to 128) versus the 5-level U-Net decoder of M5.
  - 30 epochs may be too few given the cold start of attention and decoder.

### 10.6 Resource usage
| Member | Training time (hrs) | Peak GPU mem (GB) |
|---|---:|---:|
| M1 | 1.92 | 2.35 |
| M2 | 2.38 | 2.91 |
| M3 (SwinUNETR) | **13.88** | 2.11 |
| M4 | n/a in CSV | n/a |
| M5 (naive multimodal) | 4.10 | 2.17 |
| M5.2 (Latent fusion) | 2.41 | **15.72** |

Insights:
- Swin is by far the slowest to train (≈7× the next slowest), reflecting
  the per-batch cost of windowed self-attention in 3D.
- Latent fusion is fast per epoch (no encoder forwards) but expensive in
  memory: concatenating four 256-ch bottlenecks gives a 1024-ch attention
  tensor.

### 10.7 Saved figures
- `results/figures/training_curves_{M1_T1n_ResUNet,M3_T2w_SwinUNETR,M5_Multimodal_4ch,M5_LatentFusion_XAI}.png`
  — two-panel loss + per-sub-region Dice curves with best-epoch vertical line.
- `results/figures/ablation_bar_chart.png` — three-panel WT/TC/ET bar chart of
  Dice drops per modality.
- `results/figures/sanity_check_patient.png` — Phase-0 patient overview.
- `results/figures/T1n/` — 27 PNGs (3 patients × 3 sub-regions × 3 views) per
  member. Same for `T2f/`, `T2w/`. Test patients
  `BraTS-GLI-02193-104, BraTS-GLI-02225-100, BraTS-GLI-03021-101` (the
  first / middle / last of `test_ids.txt`).
- `results/figures/T2f/` also stores `.npy` heatmap tensors for both
  `gradcam_M4_*` and `gradcam_resunet_*`, plus SHAP arrays
  (`shap_resunet_{pid}_{sr}.npy`) — extra XAI methods produced by M4.

### 10.8 Failure taxonomy documented in repo
`/.claude/CLAUDE.md` (the AI-coding instructions) explicitly enumerates the
expected failure modes used in qualitative XAI analysis:

| Failure | Detection | Likely cause | Sub-regions most affected |
|---|---|---|---|
| Small tumour missed (especially ET) | GT has ET, pred has none | Patch sampling bias too low; ET voxels rare | ET, TC |
| Boundary over-segmentation | Pred mask larger than GT at edges | Oedema mistaken for tumour core | TC vs WT |
| Modality confusion | CAM activates on wrong tissue | Similar intensity to tumour in modality | WT, TC for M3/M4 |
| False positives in healthy hemisphere | Activation contralateral | Symmetry heuristic | All sub-regions for M2 |
| ET predicted as TC | Channel mismatch | Channel-ordering bug | ET specifically |

---

## 11. Engineering & Reproducibility Patterns

### 11.1 Hard rules enforced by `.claude/CLAUDE.md` (project conventions)
- `set_global_seed()` is the **first executable line** of every entry-point.
- **Never softmax** the 3-channel output — sub-regions overlap; sigmoid per
  channel is correct.
- **Never re-run** `shared/create_splits.py` — splits are committed.
- **Never use `test_ids.txt`** until `evaluate.py`. `xai_analysis.py` enforces
  this via a `test_metrics.csv` existence guard.
- **Never use patch-based validation** — `full_volume=True` + sliding window
  is mandatory for val/test (claude.md §2.4b).
- **Never use bilinear interpolation on a seg mask** — always `order=0`.
- **Never compute Grad-CAM on the full 3-channel output** — always pass
  `target_channel`.
- **Never report only overall Dice** — always per-sub-region (WT/TC/ET).
- **Never drive early stopping on dice_et** — too noisy early. Use dice_wt.
- **Every model must implement `forward_features(x) → (B, 256, H/16, W/16, D/16)`**
  — binding contract for M5's fusion head.
- The `or epoch == NUM_EPOCHS` clause in the validation condition is
  mandatory so the final epoch validates even if `NUM_EPOCHS % VAL_EVERY_N_EPOCHS ≠ 0`.

### 11.2 Phase gates (acceptance criteria)
- **Phase 1 minimums** for unimodal members: `val_dice_wt > 0.60`,
  `val_dice_tc > 0.50`, `val_dice_et > 0.45`. (Realised test-set Dice came
  in lower in some cases due to held-out variance.)
- **Phase 3 minimums**: 27 Grad-CAM PNGs per member, populated
  `modality_importance_matrix.csv`, written per-member analyses.
- **Final**: sanity check passes on a clean clone, no absolute paths, no
  tracked `.nii*` or `.pt`, `requirements.txt` pinned.

### 11.3 Hyperparameter cheat-sheet (frozen)
```
GLOBAL_SEED        = 42
total patients     = 350 (sampled from 1350)
train / val / test = 280 / 35 / 35
TARGET_CHANNELS    = 3
TARGET_CHANNEL_NAMES = ("wt","tc","et")
MODALITIES         = ("t1c","t1n","t2f","t2w")
PATCH_SIZE         = 64³ or 96³ (auto by VRAM ≥14 GB)
volume after preproc = 128³
batch_size         = 1
patches_per_volume = 4
tumour_bias        = 0.8
lr                 = 1e-4   (3e-4 for fusion head)
weight_decay       = 1e-5
NUM_EPOCHS         = 50     (30 for fusion)
PATIENCE           = 15
VAL_EVERY_N_EPOCHS = 5
GRAD_CLIP_NORM     = 1.0
AMP_ENABLED        = True
deep_supervision_weights (M1) = (0.5, 0.25, 0.125)
focal_loss (M2 alt)            = γ=2.0, α=0.25 (default), α=0.5 in dice_focal_loss combo
overlap (SWI)      = 0.5
sw_batch_size      = 4 (validate) / 2 (interface inference)
HD95 reduction     = mean, percentile=95, include_background=False
empty-mask HD95    = NaN (skipped by MetricTracker)
```

### 11.4 Bottlenecks and known limitations
- **DataLoader on Windows**: `NUM_WORKERS=0` to avoid the fork deadlock — GPU
  utilisation often <40% early in each epoch. Linux/WSL2 with 4 workers
  removes this.
- **No on-disk preprocessing cache**: every `__getitem__` re-loads NIfTI and
  re-normalises. Intentional (keeps augmentation stochastic) but I/O-heavy.
- **M3 feature shape mismatch**: Swin trained at 64³ → features `(256,4,4,4)`
  upsampled in `FeaturesDataset` → information loss for the fusion model.
- **Path resolution bug**: `shared/config.py` resolves `CHECKPOINT_DIR` etc.
  via `RESULTS_DIR / "checkpoints"`. The ensemble scripts re-derive the same
  paths because earlier mis-resolution on Windows was discovered (documented
  in their docstrings).
- **Fusion head underperformance**: see §10.5. The XAI-initialised attention
  prior is a research idea — the negative result is interesting on its own
  and the prior bias is preserved in `M5_LatentFusion_XAI_train_meta.json`.

---

## 12. End-to-End Data & Control Flow (Runbook)

```
[1] One-time: shared/create_splits.py   →   data/splits/{train,val,test}_ids.txt
[2] One-time: shared/sanity_check.py    →   results/figures/sanity_check_patient.png

For each unimodal member (M1, M2, M3, M4):
  [3] memberN_*/train.py
       reads DATA_PATH.txt → DATA_ROOT
       loads splits/train_ids.txt + val_ids.txt
       BraTSDataset (patches, augmented) + BraTSDataset (full_volume)
       50 epochs, AMP, AdamW + Cosine, EarlyStopper(patience=15)
       writes results/checkpoints/{member}_best.pt + train_meta.json
       writes results/figures/training_curves_{member}.png
       logs to MLflow ({member} experiment)
  [4] memberN_*/evaluate.py
       loads test_ids.txt + best checkpoint
       SWI inference (roi=PATCH_SIZE, overlap=0.5)
       appends row to results/tables/test_metrics.csv
  [5] memberN_*/xai_analysis.py
       guards on test_metrics.csv existence
       3 patients × 3 sub-regions × 3 views = 27 Grad-CAM PNGs
  [6] memberN_*/extract_features.py
       saves (1, 256, 8, 8, 8) bottleneck per patient to
       results/features/{Mk_Modality}/{pid}.pt for train+val+test

For M5 (Pipeline A — naive multimodal):
  [7]  member5_multimodal/train.py            → M5_Multimodal_4ch_best.pt
  [8]  member5_multimodal/evaluate.py         → adds row to test_metrics.csv
  [9]  member5_multimodal/ablation.py         → ablation_scores.csv,
                                                modality_importance_scores.json,
                                                ablation_bar_chart.png
  [10] member5_multimodal/extract_features.py → M5 features (used only for completeness)

For Pipeline C — Latent Space Fusion Head:
  [11] ensemble/cache_labels.py    → results/features/labels/{pid}.pt (3,128³) uint8
  [12] ensemble/train_fusion.py    → builds gate bias from importance JSON,
                                     trains LatentFusionEnsemble(30 epochs, lr=3e-4)
                                     → M5_LatentFusion_XAI_best.pt
  [13] ensemble/evaluate_fusion.py → row in test_metrics.csv + Δ vs M5 baseline

For interactive use:
  [14] uvicorn interface.backend.main:app --reload --port 8000
       /sessions POST → upload .nii.gz files per modality →
       /predict {model_key} runs inference via registry-cached model →
       /compare runs multiple → /report.pdf → /predictions/.../mask.nii.gz
```

---

## 13. Dependencies (`requirements.txt`)

Pinned (approximately) — PyTorch + CUDA is installed separately:
- `torch~=2.2`, `torchvision~=0.17`
- `monai~=1.3` (SwinUNETR, SegResNet, sliding-window inference, HD95 metric, transforms)
- `nibabel~=5.2`, `numpy~=1.26`, `scipy~=1.13`, `scikit-learn~=1.4`
- `matplotlib~=3.8`, `nilearn~=0.10`
- `mlflow~=2.12`
- `streamlit~=1.33` (originally planned UI; superseded by FastAPI + Loveable)
- `shap~=0.45` (additional XAI on M4)
- `SimpleITK~=2.3`, `nnunetv2~=2.4` (reference)
- `tqdm~=4.66`, `pandas~=2.2`, `jupyterlab~=4.1`, `ipywidgets~=8.1`
- Backend extras (in `interface/backend/requirements.txt`): `fastapi`,
  `uvicorn[standard]`, etc.

---

## 14. Key Takeaways (For a Future AI Reading This)

1. **Five 3-channel WT/TC/ET segmentation models on BraTS GLI-2024**,
   plus a sixth XAI-initialised late-fusion model trained on cached
   bottleneck features. All on the **same** stratified 280/35/35 split,
   `seed=42`.
2. **Output is always 3-channel sigmoid** (WT, TC, ET), never softmax.
   Every model exposes `forward_features → (B, 256, H/16, ...)`.
3. **M4 (T2-FLAIR ResUNet3D) is the unimodal WT champion (Dice 0.79);
   M2 (T1c AttentionUNet) is the TC/ET champion (Dice 0.59/0.57).**
   M5's naive 4-channel multimodal model beats every unimodal model on
   TC/ET (0.68/0.69) and is second only to M4 on WT (0.77 vs 0.79).
4. **The XAI-initialised latent fusion (M5.2) did NOT beat the naive
   multimodal baseline in this run** (WT 0.48 / TC 0.22 / ET 0.19). The
   research idea — that attention priors derived from ablation drops
   help — is documented but not validated by the current numerical
   outcome. Plausible causes (feature-shape mismatch for M3, cross-arch
   bottleneck incompatibility, decoder capacity, training duration) are
   enumerated in §10.5.
5. **Ablation results are the most interesting scientific finding**:
   - WT collapses (Dice 0.77 → 0.11) when T2-FLAIR is removed.
   - TC and ET collapse (~0.69 → ~0.20) when T1c is removed.
   - T2w can be removed with almost no penalty across all sub-regions.
   - T1n contributes moderately to TC/ET (drops ~0.30) but little to WT.
6. **Reproducibility is the project's highest priority**: deterministic
   seeding, single source of truth in `shared/`, committed splits,
   `test_metrics.csv` guard for XAI, MLflow tracking with explicit
   run-naming convention `M{N}-{MODALITY}-{Arch}-seed42`.
7. **The repo also ships a production-style FastAPI inference backend**
   (`interface/backend/`) with declarative model registry, LRU model
   cache, lazy weight loading, session storage with TTL, sliding-window
   inference, and PDF report generation — engineered for an external
   React/Loveable UI.
