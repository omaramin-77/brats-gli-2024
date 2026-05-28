"""PDF report generation — a single self-contained PDF the user can hand to
a supervisor or print for a viva binder.

Contents:
1. Header with session ID, model used, generation timestamp, disclaimer.
2. Tumour volume summary (WT / TC / ET in mm^3).
3. Metrics table — Dice, IoU, HD95 per sub-region.
4. Three orthogonal slice triplets (axial / coronal / sagittal) at the centre
   of each region with the prediction overlay.
5. Footer with the disclaimer that this is a research tool.
"""
from __future__ import annotations

import datetime as _dt
import io
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak,
)

from interface.backend.sessions import sessions
from interface.backend.slices import slice_to_png_bytes
from shared.config import MODALITIES, TARGET_CHANNEL_NAMES


_DISCLAIMER = (
    "This output is produced by a research prototype trained on a subset of "
    "the BraTS GLI-2024 dataset. It is not a medical device and must not be "
    "used for clinical decision-making."
)


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title":   ParagraphStyle("title", parent=base["Title"], fontSize=18, spaceAfter=12),
        "h2":      ParagraphStyle("h2",    parent=base["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=6),
        "body":    ParagraphStyle("body",  parent=base["BodyText"], fontSize=10, leading=13),
        "small":   ParagraphStyle("small", parent=base["BodyText"], fontSize=8, textColor=colors.grey),
        "warn":    ParagraphStyle("warn",  parent=base["BodyText"], fontSize=9, textColor=colors.HexColor("#8a4a00")),
    }


def _metrics_table(metrics: dict) -> Table:
    rows = [["Sub-region", "Dice", "IoU", "HD95 (mm)", "Volume (mm^3)"]]
    for name in TARGET_CHANNEL_NAMES:
        rows.append([
            name.upper(),
            f"{metrics.get(f'dice_{name}', float('nan')):.4f}",
            f"{metrics.get(f'iou_{name}', float('nan')):.4f}",
            _fmt_hd(metrics.get(f"hd95_{name}", float("nan"))),
            f"{metrics.get(f'volume_mm3_{name}', 0.0):,.0f}",
        ])

    tbl = Table(rows, colWidths=[3 * cm, 3 * cm, 3 * cm, 3 * cm, 4 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a52")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN",      (1, 1), (-1, -1), "RIGHT"),
        ("GRID",       (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ]))
    return tbl


def _fmt_hd(v) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if x != x:    # NaN
        return "n/a"
    return f"{x:.2f}"


def _png_image(png_bytes: bytes, width: float) -> RLImage:
    img = Image.open(io.BytesIO(png_bytes))
    aspect = img.height / img.width
    return RLImage(io.BytesIO(png_bytes), width=width, height=width * aspect)


def _pick_modality_for_overlay(sid: str) -> str:
    """Use T1c for the overlay backdrop if available, else fall back."""
    meta = sessions.load(sid)
    for k in ("t1c", "t1n", "t2f", "t2w"):
        if k in meta.uploads:
            return k
    raise ValueError("session has no MRI modalities uploaded")


def _load_volume(sid: str, kind: str) -> np.ndarray:
    """Preprocessed-equivalent 128**3 volume for slicing. We re-run the shared
    pipeline so the slices in the PDF line up pixel-for-pixel with what was
    shown in the viewer."""
    from shared.preprocessing import preprocess_patient
    vol, _ = preprocess_patient(str(sessions.upload_dir(sid)), kind)
    return vol[0]


def _centre_indices(mask: np.ndarray) -> tuple[int, int, int]:
    """Pick a slice index per view that goes through the centre of mass of
    the prediction. If the mask is empty, fall back to the volume centre."""
    if mask.any():
        coords = np.argwhere(mask)
        c = coords.mean(axis=0)
    else:
        c = np.array(mask.shape) / 2.0
    return int(c[0]), int(c[1]), int(c[2])


def build_report(sid: str, model_keys: Iterable[str]) -> bytes:
    """Render the PDF and return its bytes."""
    model_keys = list(model_keys)
    if not model_keys:
        raise ValueError("at least one model is required")

    backdrop_kind = _pick_modality_for_overlay(sid)
    volume = _load_volume(sid, backdrop_kind)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.5 * cm,  bottomMargin=1.5 * cm,
    )
    styles = _styles()
    flow: list = []

    flow.append(Paragraph("Brain MRI Tumour Segmentation — Report", styles["title"]))
    flow.append(Paragraph(
        f"Session <b>{sid}</b> &nbsp;|&nbsp; generated "
        f"{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; "
        f"backdrop modality: {backdrop_kind.upper()}",
        styles["small"],
    ))
    flow.append(Paragraph(_DISCLAIMER, styles["warn"]))
    flow.append(Spacer(1, 0.4 * cm))

    for idx, key in enumerate(model_keys):
        if not sessions.has_prediction(sid, key):
            flow.append(Paragraph(
                f"<i>Model {key} has no prediction in this session — skipping.</i>",
                styles["body"],
            ))
            continue
        metrics = sessions.load_prediction_metrics(sid, key)
        mask = sessions.load_prediction_mask(sid, key)   # (3, H, W, D)

        flow.append(Paragraph(metrics.get("model_display_name", key), styles["h2"]))
        flow.append(Paragraph(
            f"Architecture: {metrics.get('architecture', '?')}  &nbsp;|&nbsp;  "
            f"Modality input: {metrics.get('modality', '?').upper()}  &nbsp;|&nbsp;  "
            f"Inference: {metrics.get('inference_time_s', 0.0):.2f} s",
            styles["small"],
        ))
        flow.append(Spacer(1, 0.2 * cm))
        flow.append(_metrics_table(metrics))
        flow.append(Spacer(1, 0.3 * cm))

        # Pick centre-of-mass on the largest sub-region (WT) for backdrop slices.
        cx, cy, cz = _centre_indices(mask[0].astype(bool))
        triplet = [
            ("Sagittal", "sagittal", cx),
            ("Coronal",  "coronal",  cy),
            ("Axial",    "axial",    cz),
        ]

        cell_imgs = []
        for label, view, ind in triplet:
            from interface.backend.slices import take_slice
            img_slice = take_slice(volume, view, ind)
            mask_slice = np.stack([
                take_slice(mask[i].astype(np.uint8), view, ind) for i in range(3)
            ], axis=0)
            png = slice_to_png_bytes(img_slice, mask_slice, region="all")
            cell_imgs.append((label, _png_image(png, width=5.0 * cm)))

        img_row = [img for _, img in cell_imgs]
        label_row = [Paragraph(label, styles["small"]) for label, _ in cell_imgs]
        tbl = Table([img_row, label_row], colWidths=[5.5 * cm] * 3)
        tbl.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ]))
        flow.append(tbl)

        if idx != len(model_keys) - 1:
            flow.append(PageBreak())

    flow.append(Spacer(1, 0.6 * cm))
    flow.append(Paragraph(_DISCLAIMER, styles["warn"]))

    doc.build(flow)
    return buf.getvalue()
