"""Session lifecycle + uploads + slice viewer.

POST   /sessions                       create
GET    /sessions/{sid}                 read state
DELETE /sessions/{sid}                 delete
POST   /sessions/{sid}/upload          multipart upload (one or more kinds)
GET    /sessions/{sid}/slice           PNG slice of an uploaded modality
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

import nibabel as nib
import numpy as np

from interface.backend.config import MAX_UPLOAD_BYTES, UPLOAD_KINDS
from interface.backend.schemas import SessionInfo, UploadAck
from interface.backend.sessions import sessions
from interface.backend.slices import slice_to_png_bytes, take_slice

router = APIRouter(tags=["sessions"])


# ---------------------------------------------------------------------------
@router.post("/sessions", response_model=SessionInfo, status_code=201)
def create_session() -> SessionInfo:
    meta = sessions.create()
    return SessionInfo(
        sid=meta.sid,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
        uploads={},
        predictions=[],
    )


@router.get("/sessions/{sid}", response_model=SessionInfo)
def read_session(sid: str) -> SessionInfo:
    if not sessions.exists(sid):
        raise HTTPException(404, f"session {sid} not found")
    meta = sessions.load(sid)
    return SessionInfo(
        sid=meta.sid,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
        uploads=meta.uploads,
        predictions=sessions.list_predictions(sid),
    )


@router.delete("/sessions/{sid}", status_code=204)
def delete_session(sid: str) -> Response:
    sessions.delete(sid)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
@router.post("/sessions/{sid}/upload", response_model=UploadAck)
async def upload_modality(
    sid: str,
    kind: Literal["t1n", "t1c", "t2w", "t2f", "seg"] = Form(...),
    file: UploadFile = File(...),
) -> UploadAck:
    """Upload a single NIfTI volume. The ``kind`` form field tells the
    backend which channel the file represents — the on-disk filename is
    rewritten to BraTS conventions so the shared preprocessing globs find it.
    """
    if not sessions.exists(sid):
        raise HTTPException(404, f"session {sid} not found")
    if kind not in UPLOAD_KINDS:
        raise HTTPException(400, f"unknown kind '{kind}'")

    name = (file.filename or "upload.nii.gz").lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise HTTPException(400, "only .nii / .nii.gz are accepted")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file exceeds max upload size ({MAX_UPLOAD_BYTES} bytes)")
    if len(content) == 0:
        raise HTTPException(400, "uploaded file is empty")

    target_name = sessions.store_upload(sid, kind, content, file.filename or name)
    return UploadAck(sid=sid, kind=kind, filename=target_name, size_bytes=len(content))


# ---------------------------------------------------------------------------
# Raw NIfTI download — the frontend's NiiVue viewer consumes this URL directly.
# Without this endpoint the WebGL viewer has no way to render the uploaded
# volume after a page refresh (the original File object is gone).
# ---------------------------------------------------------------------------
@router.get("/sessions/{sid}/uploads/{kind}")
def download_upload(
    sid: str,
    kind: Literal["t1n", "t1c", "t2w", "t2f", "seg"],
) -> FileResponse:
    if not sessions.exists(sid):
        raise HTTPException(404, f"session {sid} not found")
    path = sessions.upload_path(sid, kind)
    if path is None:
        raise HTTPException(404, f"no {kind} uploaded for this session")
    # NiiVue inspects the URL suffix to pick a parser, so the filename must
    # end in .nii / .nii.gz — we serve the original on-disk name.
    return FileResponse(
        path=path,
        media_type="application/gzip" if path.suffix == ".gz" else "application/octet-stream",
        filename=path.name,
    )


# ---------------------------------------------------------------------------
# Viewer slice — render any uploaded modality as a PNG (fallback when NiiVue
# isn't an option, e.g. for the PDF report or non-WebGL clients).
# ---------------------------------------------------------------------------
@router.get("/sessions/{sid}/slice")
def viewer_slice(
    sid: str,
    kind: Literal["t1n", "t1c", "t2w", "t2f", "seg"] = Query("t1c"),
    view: Literal["axial", "coronal", "sagittal"] = Query("axial"),
    index: int = Query(..., ge=0),
    preprocessed: bool = Query(
        True,
        description="If True, run the shared preprocessing pipeline first so "
                    "the slice matches what a model sees. If False, slice the "
                    "raw upload (different voxel grid).",
    ),
) -> Response:
    """Return a PNG of one slice of an uploaded modality."""
    if not sessions.exists(sid):
        raise HTTPException(404, f"session {sid} not found")

    path = sessions.upload_path(sid, kind)
    if path is None:
        raise HTTPException(404, f"no {kind} uploaded for this session")

    if preprocessed and kind != "seg":
        # Re-run the per-modality pipeline to land on the (128, 128, 128)
        # grid every model uses. This costs ~1s of CPU.
        from shared.preprocessing import preprocess_patient
        vol, _ = preprocess_patient(str(sessions.upload_dir(sid)), kind)
        volume = vol[0]
    elif preprocessed and kind == "seg":
        # For the seg mask we don't z-score normalise; we just want the
        # WT/TC/ET overlay on the preprocessed grid. Load directly + resize
        # using nearest-neighbour to keep integer labels.
        from shared.preprocessing import resize_volume
        raw = nib.load(str(path)).get_fdata().astype(np.int16)
        volume = resize_volume(raw, (128, 128, 128), order=0).astype(np.float32)
    else:
        volume = nib.load(str(path)).get_fdata().astype(np.float32)

    img_slice = take_slice(volume, view, index)
    png = slice_to_png_bytes(img_slice)
    return Response(content=png, media_type="image/png")
