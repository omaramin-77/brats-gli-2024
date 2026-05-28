"""Session storage — uploaded NIfTI volumes + cached prediction outputs.

Layout on disk
--------------
    SESSION_ROOT/<sid>/
        meta.json
        upload/
            <sid>-t1n.nii.gz       (BraTS-compatible naming so the
            <sid>-t1c.nii.gz        shared preprocessing globs pick
            <sid>-t2w.nii.gz        them up without modification)
            <sid>-t2f.nii.gz
            <sid>-seg.nii.gz       (optional; required for metrics)
        predictions/<model_key>/
            mask.nii.gz            3-channel binary WT/TC/ET prediction
            probs.npz              raw sigmoid probabilities (for slider overlay)
            metrics.json           dice/iou/hd95 + inference_time_s
        affine.npy                 affine from the first uploaded volume

The BraTS filename convention matters: ``shared.preprocessing.preprocess_patient``
uses globs like ``*t1n*.nii*`` to find each modality, so as long as the
filename contains the right substring the shared pipeline accepts the file
as-is.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from interface.backend.config import SESSION_ROOT, SESSION_TTL_HOURS, UPLOAD_KINDS


# ---------------------------------------------------------------------------
# Per-session metadata
# ---------------------------------------------------------------------------
@dataclass
class SessionMeta:
    sid: str
    created_at: float                       # unix seconds
    updated_at: float
    uploads: dict[str, str] = field(default_factory=dict)   # kind -> filename
    # affine kept as a list[list[float]] for JSON; the numpy version lives
    # alongside in affine.npy because round-tripping through JSON loses precision.
    has_affine: bool = False

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "SessionMeta":
        return cls(
            sid=data["sid"],
            created_at=float(data["created_at"]),
            updated_at=float(data["updated_at"]),
            uploads=dict(data.get("uploads", {})),
            has_affine=bool(data.get("has_affine", False)),
        )


# ---------------------------------------------------------------------------
# Session API
# ---------------------------------------------------------------------------
class SessionStore:
    """Thin disk-backed CRUD over SESSION_ROOT."""

    def __init__(self, root: Path = SESSION_ROOT) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # ----- paths -----
    def _dir(self, sid: str) -> Path:
        return self.root / sid

    def upload_dir(self, sid: str) -> Path:
        return self._dir(sid) / "upload"

    def prediction_dir(self, sid: str, model_key: str) -> Path:
        return self._dir(sid) / "predictions" / model_key

    def meta_path(self, sid: str) -> Path:
        return self._dir(sid) / "meta.json"

    def affine_path(self, sid: str) -> Path:
        return self._dir(sid) / "affine.npy"

    # ----- lifecycle -----
    def create(self) -> SessionMeta:
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        meta = SessionMeta(sid=sid, created_at=now, updated_at=now)
        (self._dir(sid) / "upload").mkdir(parents=True, exist_ok=True)
        (self._dir(sid) / "predictions").mkdir(parents=True, exist_ok=True)
        self._write_meta(meta)
        return meta

    def exists(self, sid: str) -> bool:
        return self.meta_path(sid).exists()

    def load(self, sid: str) -> SessionMeta:
        if not self.exists(sid):
            raise KeyError(f"Session {sid} not found")
        data = json.loads(self.meta_path(sid).read_text(encoding="utf-8"))
        return SessionMeta.from_json(data)

    def delete(self, sid: str) -> None:
        d = self._dir(sid)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    def _write_meta(self, meta: SessionMeta) -> None:
        meta.updated_at = time.time()
        self.meta_path(meta.sid).write_text(
            json.dumps(meta.to_json(), indent=2),
            encoding="utf-8",
        )

    # ----- uploads -----
    def store_upload(self, sid: str, kind: str, content: bytes, original_name: str) -> str:
        """Save an uploaded NIfTI volume under BraTS-compatible naming.

        The filename embeds the kind so ``shared.preprocessing``'s globs
        (``*t1n*.nii*`` etc.) find it. The original filename is recorded in
        meta.json for traceability.
        """
        if kind not in UPLOAD_KINDS:
            raise ValueError(
                f"Unknown upload kind '{kind}'. Must be one of {UPLOAD_KINDS}."
            )

        # Decide the on-disk filename. ``.nii`` and ``.nii.gz`` are both valid;
        # we keep the original extension so the user can verify by hash.
        ext = ".nii.gz" if original_name.lower().endswith(".nii.gz") else ".nii"
        target_name = f"{sid}-{kind}{ext}"
        target = self.upload_dir(sid) / target_name
        target.write_bytes(content)

        # Cache the affine from the first uploaded volume — every modality is
        # co-registered to the same space in BraTS so any of them is correct.
        if not self.affine_path(sid).exists():
            try:
                affine = nib.load(str(target)).affine
                np.save(self.affine_path(sid), affine)
            except Exception:  # pragma: no cover
                # Don't fail the upload on an unreadable file; fall back later.
                pass

        meta = self.load(sid)
        meta.uploads[kind] = target_name
        meta.has_affine = self.affine_path(sid).exists()
        self._write_meta(meta)
        return target_name

    def upload_path(self, sid: str, kind: str) -> Optional[Path]:
        """Resolve the on-disk path for an uploaded kind, or None if missing."""
        meta = self.load(sid)
        fname = meta.uploads.get(kind)
        if not fname:
            return None
        return self.upload_dir(sid) / fname

    def get_affine(self, sid: str) -> Optional[np.ndarray]:
        p = self.affine_path(sid)
        if not p.exists():
            return None
        return np.load(p)

    # ----- predictions -----
    def store_prediction(
        self,
        sid: str,
        model_key: str,
        binary_mask: np.ndarray,         # (3, H, W, D) uint8
        probs: np.ndarray,                # (3, H, W, D) float32
        metrics: dict,                    # serialisable
        affine: Optional[np.ndarray] = None,
    ) -> Path:
        out = self.prediction_dir(sid, model_key)
        out.mkdir(parents=True, exist_ok=True)

        # NIfTI mask. We pack the 3 channels as the 4th axis so the file is a
        # standard 4D NIfTI (3 binary channels), viewable in any NIfTI viewer.
        mask_4d = np.transpose(binary_mask.astype(np.uint8), (1, 2, 3, 0))
        aff = affine if affine is not None else np.eye(4, dtype=np.float32)
        nib.save(nib.Nifti1Image(mask_4d, aff), str(out / "mask.nii.gz"))

        np.savez_compressed(out / "probs.npz", probs=probs.astype(np.float32))
        (out / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        # Touch the session so cleanup leaves it alone.
        meta = self.load(sid)
        self._write_meta(meta)
        return out

    def has_prediction(self, sid: str, model_key: str) -> bool:
        return (self.prediction_dir(sid, model_key) / "metrics.json").exists()

    def load_prediction_metrics(self, sid: str, model_key: str) -> dict:
        p = self.prediction_dir(sid, model_key) / "metrics.json"
        return json.loads(p.read_text(encoding="utf-8"))

    def load_prediction_probs(self, sid: str, model_key: str) -> np.ndarray:
        p = self.prediction_dir(sid, model_key) / "probs.npz"
        with np.load(p) as f:
            return f["probs"]

    def load_prediction_mask(self, sid: str, model_key: str) -> np.ndarray:
        """Return the binary mask as (3, H, W, D)."""
        p = self.prediction_dir(sid, model_key) / "mask.nii.gz"
        data = nib.load(str(p)).get_fdata().astype(np.uint8)
        # We saved it as (H, W, D, 3); restore (3, H, W, D) for code consistency.
        return np.transpose(data, (3, 0, 1, 2))

    def list_predictions(self, sid: str) -> list[str]:
        pdir = self._dir(sid) / "predictions"
        if not pdir.exists():
            return []
        return sorted(p.name for p in pdir.iterdir() if p.is_dir())

    # ----- cleanup -----
    def sweep_expired(self) -> int:
        cutoff = time.time() - SESSION_TTL_HOURS * 3600
        removed = 0
        for d in self.root.iterdir():
            if not d.is_dir():
                continue
            mp = d / "meta.json"
            if not mp.exists():
                continue
            try:
                meta = SessionMeta.from_json(json.loads(mp.read_text(encoding="utf-8")))
            except Exception:
                continue
            if meta.updated_at < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        return removed


sessions = SessionStore()
