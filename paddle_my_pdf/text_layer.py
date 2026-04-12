"""Invisible text layer computation for searchable PDF creation.

Converts OCR bounding-box polygons (in pixel coordinates) into precisely
positioned, invisible PDF text operations.  Font metrics (ascender /
descender) are used so that the selectable text region matches the
detected bounding-box height exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import fitz
import numpy as np

from .config import CJK_FONT_PATH

# Smallest font size we will ever emit (PDF points).
MIN_FONT_SIZE = 4.0


@dataclass
class TextOp:
    """One invisible-text insertion destined for a PDF page."""

    origin: fitz.Point
    text: str
    font_size: float
    h_scale: float


@lru_cache(maxsize=1)
def _font_metrics() -> Tuple[fitz.Font, float, float, float]:
    """Return ``(font, ascender, descender, total_height)``, cached.

    *ascender* is a positive fraction of ``font_size`` (distance from
    baseline to the top of the tallest glyph).  *descender* is negative
    (distance from baseline to the lowest descender).  *total_height* is
    ``ascender - descender``.
    """
    font = fitz.Font(fontfile=CJK_FONT_PATH)
    asc = font.ascender
    desc = font.descender
    return font, asc, desc, asc - desc


def compute_text_ops(
    ocr_items: List[Tuple[str, np.ndarray]],
    page_w: float,
    page_h: float,
    img_w: int,
    img_h: int,
) -> List[TextOp]:
    """Convert OCR results to positioned invisible-text operations.

    Parameters
    ----------
    ocr_items:
        Each element is ``(text, polygon)`` where *polygon* is a 4x2
        numpy array of corner coordinates in **pixel** space of the
        rendered page image.
    page_w, page_h:
        Target PDF page dimensions in points.
    img_w, img_h:
        Pixel dimensions of the rendered image that was fed to OCR.

    Returns
    -------
    list[TextOp]
        Ready-to-insert text operations whose font size and baseline
        are derived from the font's ascender / descender metrics so
        that the selectable region covers the full bounding box.
    """
    font, ascender, _descender, total_height = _font_metrics()

    sx = page_w / img_w
    sy = page_h / img_h

    ops: List[TextOp] = []
    for text, poly in ocr_items:
        pts = poly.copy()
        pts[:, 0] *= sx
        pts[:, 1] *= sy

        x0 = float(pts[:, 0].min())
        y0 = float(pts[:, 1].min())
        x1 = float(pts[:, 0].max())
        y1 = float(pts[:, 1].max())

        box_w = x1 - x0
        box_h = y1 - y0
        if box_w <= 0 or box_h <= 0:
            continue

        # Size the font so the full ascender-to-descender span equals
        # the bounding-box height.  This ensures the selectable region
        # in PDF viewers covers the entire detected text area.
        font_size = max(box_h / total_height, MIN_FONT_SIZE)

        # Horizontal scale: stretch / compress the text to match the
        # detected bounding-box width exactly.
        natural_w = font.text_length(text, fontsize=font_size)
        h_scale = box_w / natural_w if natural_w > 0 else 1.0

        # Place the baseline so the ascender aligns with the box top
        # and the descender aligns with the box bottom.
        baseline_y = y0 + ascender * font_size
        origin = fitz.Point(x0, baseline_y)

        ops.append(TextOp(origin, text, font_size, h_scale))

    return ops
