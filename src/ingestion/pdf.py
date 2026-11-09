"""PDF page rendering."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf

from .config import IngestionConfig
from .errors import CorruptFileError, InvalidPageError, ResourceLimitError

logger = logging.getLogger(__name__)

_PDF_POINTS_PER_INCH = 72.0


@dataclass(frozen=True)
class RenderedPage:
    image: np.ndarray  # BGR uint8
    dpi: float  # actual render DPI (may be below the requested DPI)
    page_index: int
    page_count: int


def render_pdf_page(path: Path, config: IngestionConfig) -> RenderedPage:
    """Render one page, lowering DPI as needed to respect ``max_render_pixels``."""
    try:
        with pymupdf.open(str(path)) as doc:
            if doc.needs_pass:
                raise CorruptFileError("PDF is password protected.")
            page_count = doc.page_count
            if page_count == 0:
                raise CorruptFileError("PDF document contains no pages.")
            if config.pdf_page >= page_count:
                raise InvalidPageError(
                    f"Page {config.pdf_page} requested but PDF has {page_count} page(s)."
                )

            page = doc[config.pdf_page]
            dpi = _fit_dpi(page.rect.width, page.rect.height, config)
            zoom = dpi / _PDF_POINTS_PER_INCH
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False
            )
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    except (CorruptFileError, InvalidPageError, ResourceLimitError):
        raise
    except Exception as exc:  # PyMuPDF raises assorted RuntimeError/FileDataError types
        raise CorruptFileError(f"Could not render PDF {path.name}: {exc}") from exc

    return RenderedPage(bgr, dpi, config.pdf_page, page_count)


def _fit_dpi(width_pt: float, height_pt: float, config: IngestionConfig) -> float:
    """Return the DPI to render at, capped so the bitmap stays within budget."""
    if width_pt <= 0 or height_pt <= 0:
        raise CorruptFileError("PDF page has invalid dimensions.")
    dpi = float(config.target_dpi)
    pixels = (width_pt * dpi / _PDF_POINTS_PER_INCH) * (height_pt * dpi / _PDF_POINTS_PER_INCH)
    if pixels > config.max_render_pixels:
        dpi *= math.sqrt(config.max_render_pixels / pixels)
        logger.warning("Page too large for %d DPI; rendering at %.1f DPI", config.target_dpi, dpi)
    return dpi
