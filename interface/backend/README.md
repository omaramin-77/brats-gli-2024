# BraTS Backend — FastAPI

Inference + model-comparison service for the team's 5 trained segmentation models, plus a 6th slot reserved for a separately trained late-fusion model that can be plugged in later.

## Run

```powershell
# 1. From the repo root, install the backend extras on top of the repo deps.
pip install -r interface/backend/requirements.txt

# 2. Make sure shared/config.py can resolve the data root for any patient
#    you upload (DATA_PATH.local.txt). The backend itself does NOT depend on
#    DATA_PATH — uploads are stored under interface/backend/sessions/ — but
#    importing shared/config triggers the path check.

# 3. Launch.
uvicorn interface.backend.main:app --reload --port 8000
```

Interactive docs: <http://localhost:8000/docs>

## Configuration via env vars

| Variable | Default | Purpose |
|---|---|---|
| `BRATS_SESSION_DIR` | `interface/backend/sessions/` | Where uploads + cached predictions live |
| `BRATS_SESSION_TTL_HOURS` | `6` | How long a session lives without activity |
| `BRATS_MODEL_CACHE` | `3` | Max model weights resident in memory (LRU) |
| `BRATS_MAX_UPLOAD_MB` | `200` | Per-file upload limit |
| `BRATS_CORS_ORIGINS` | `http://localhost:5173,...` | Comma-separated frontend origins |

## Typical flow

```http
POST /sessions                                          → { sid }
POST /sessions/{sid}/upload  (kind=t1c, file=t1c.nii.gz)
POST /sessions/{sid}/upload  (kind=t1n, file=t1n.nii.gz)
POST /sessions/{sid}/upload  (kind=t2w, file=t2w.nii.gz)
POST /sessions/{sid}/upload  (kind=t2f, file=t2f.nii.gz)
POST /sessions/{sid}/upload  (kind=seg, file=seg.nii.gz)
POST /sessions/{sid}/predict      body { "model_key": "m2_t1c" }
POST /sessions/{sid}/compare      body { "model_keys": ["m2_t1c", "m5_baseline"] }
POST /sessions/{sid}/compare-all
GET  /sessions/{sid}/predictions/m2_t1c/mask.nii.gz
GET  /sessions/{sid}/predictions/m2_t1c/slice?view=axial&index=64&region=all
GET  /sessions/{sid}/report.pdf
DELETE /sessions/{sid}
```

## Plugging in the 6th model (late fusion)

When the separately trained late-fusion model is ready:

1. Save its checkpoint to `results/checkpoints/M6_LateFusion_best.pt` using the same payload format the rest of the team uses (`{"model_state": state_dict, ...}` — `shared.trainer.CheckpointManager` writes this shape).
2. Add `ensemble/late_fusion.py`:
   ```python
   def build_model():
       from ensemble.my_late_fusion_arch import LateFusionNet
       return LateFusionNet(in_channels=4, out_channels=3)
   ```
3. The `m6_late_fusion` entry in `GET /models` flips to `enabled=true` automatically — no server restart needed; the registry re-checks the filesystem on every `list_models` call.

## Model contract assumptions

- Every model takes input shape `(B, in_channels, 128, 128, 128)` after the shared preprocessing pipeline (z-score over brain voxels, bbox crop, resize).
- Every model returns logits at `(B, 3, H, W, D)`. Channel order: WT, TC, ET.
- Sigmoid + threshold 0.5 happens in the backend (inside `compute_all_metrics`), **never** inside the model.
- Multimodal models receive 4 channels in the order `(t1c, t1n, t2f, t2w)` — same as `shared.config.MODALITIES`.

## Limitations

- Single GPU only; concurrent inference requests are serialised via a global `asyncio.Lock`.
- Sliding-window inference uses `PATCH_SIZE` from `shared.config`, which is GPU-aware.
- Reports are deliberately simple; for a clinical-style report use a NIfTI viewer on the downloaded `mask.nii.gz`.
