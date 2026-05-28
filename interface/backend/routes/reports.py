"""GET /sessions/{sid}/report.pdf — render a PDF report.

Query parameter ``models`` (comma-separated) selects which predictions to
include. If omitted, every prediction in the session is included.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from interface.backend.reports import build_report
from interface.backend.sessions import sessions

router = APIRouter(tags=["reports"])


@router.get("/sessions/{sid}/report.pdf")
def get_report(
    sid: str,
    models: str | None = Query(None, description="Comma-separated model keys; defaults to all in session"),
) -> Response:
    if not sessions.exists(sid):
        raise HTTPException(404, f"session {sid} not found")

    if models:
        keys = [k.strip() for k in models.split(",") if k.strip()]
    else:
        keys = sessions.list_predictions(sid)

    if not keys:
        raise HTTPException(400, "no predictions in this session — run /predict first")

    pdf_bytes = build_report(sid, keys)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="brats_report_{sid}.pdf"',
        },
    )
