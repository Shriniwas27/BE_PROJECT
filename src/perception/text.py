"""Text detection, classification and removal."""

from __future__ import annotations

import logging
import re
from typing import Protocol

import cv2
import numpy as np

from .config import PerceptionConfig, TextRemoval
from .errors import DependencyError, InferenceError
from .imageops import REFERENCE_DIMENSION, binarize
from .models import AnnotationKind, Box, DetectedAnnotation, Polygon

logger = logging.getLogger(__name__)

_UNIT = r"(?:mm|cm|m|ft|in)?"
_NUMBER = r"\d+(?:[.,]\d+)?"
_DIMENSION_PATTERNS = (
    re.compile(rf"^{_NUMBER}\s*{_UNIT}$", re.IGNORECASE),  # 3500, 3.5m, 12,5
    re.compile(rf"^{_NUMBER}\s*{_UNIT}\s*[x×*]\s*{_NUMBER}\s*{_UNIT}$", re.IGNORECASE),  # 3.5x4
    re.compile(r"""^\d+'\s*-?\s*(?:\d+(?:\s+\d+/\d+)?")?$"""),  # 12'-6"
    re.compile(r"""^\d+(?:\s+\d+/\d+)?"$"""),  # 6"
)


class TextDetector(Protocol):
    """Anything that can find text in a grayscale image."""

    def detect(self, image: np.ndarray) -> list[DetectedAnnotation]: ...


def classify_text(text: str) -> AnnotationKind:
    """Label text as a dimension (measurement) or a plain label."""
    cleaned = text.strip()
    if any(pattern.match(cleaned) for pattern in _DIMENSION_PATTERNS):
        return AnnotationKind.DIMENSION
    return AnnotationKind.LABEL


def polygon_to_box(polygon: Polygon, width: int, height: int) -> Box:
    """Axis-aligned bounding box of a polygon, clipped to the image."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (
        max(0, min(xs)),
        max(0, min(ys)),
        min(width, max(xs)),
        min(height, max(ys)),
    )


class EasyOcrTextDetector:
    """EasyOCR adapter. The model is loaded once, at construction."""

    def __init__(self, languages: tuple[str, ...], use_gpu: bool) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise DependencyError(
                "easyocr is not installed. Install it with: pip install easyocr"
            ) from exc
        self._reader = easyocr.Reader(list(languages), gpu=use_gpu)

    def detect(self, image: np.ndarray) -> list[DetectedAnnotation]:
        height, width = image.shape[:2]
        try:
            raw = self._reader.readtext(image)
        except Exception as exc:
            raise InferenceError(f"OCR failed: {exc}") from exc

        found: list[DetectedAnnotation] = []
        for bbox, text, confidence in raw:
            polygon: Polygon = tuple((int(round(x)), int(round(y))) for x, y in bbox)
            box = polygon_to_box(polygon, width, height)
            found.append(
                DetectedAnnotation(
                    text=str(text),
                    box=box,
                    confidence=float(confidence),
                    kind=classify_text(str(text)),
                    polygon=polygon,
                )
            )
        return found


class TiledTextDetector:
    """Runs another detector on overlapping tiles, so small text on large sheets is read.

    Detections touching an *interior* tile edge are discarded (the neighbouring
    tile sees them whole thanks to the overlap); duplicates from the overlap are
    merged, keeping the most confident reading.
    """

    _EDGE_MARGIN_PX = 3
    _DUPLICATE_IOU = 0.3

    def __init__(self, inner: TextDetector, tile: int, overlap: int) -> None:
        if tile <= 0 or not 0 <= overlap < tile:
            raise ValueError("tile must be positive and overlap must be in [0, tile)")
        self._inner = inner
        self._tile = tile
        self._step = tile - overlap

    def detect(self, image: np.ndarray) -> list[DetectedAnnotation]:
        height, width = image.shape[:2]
        if max(height, width) <= self._tile:
            return self._inner.detect(image)
        found: list[DetectedAnnotation] = []
        for y0 in _tile_starts(height, self._tile, self._step):
            for x0 in _tile_starts(width, self._tile, self._step):
                y1, x1 = min(height, y0 + self._tile), min(width, x0 + self._tile)
                for item in self._inner.detect(image[y0:y1, x0:x1]):
                    if self._cut_by_tile_edge(item, x0, y0, x1 - x0, y1 - y0, width, height):
                        continue
                    found.append(_shift(item, x0, y0))
        return _merge_duplicates(found, self._DUPLICATE_IOU)

    def _cut_by_tile_edge(
        self,
        item: DetectedAnnotation,
        x0: int,
        y0: int,
        tile_w: int,
        tile_h: int,
        width: int,
        height: int,
    ) -> bool:
        bx1, by1, bx2, by2 = item.box
        margin = self._EDGE_MARGIN_PX
        return (
            (x0 > 0 and bx1 <= margin)
            or (y0 > 0 and by1 <= margin)
            or (x0 + tile_w < width and bx2 >= tile_w - margin)
            or (y0 + tile_h < height and by2 >= tile_h - margin)
        )


def _tile_starts(total: int, tile: int, step: int) -> list[int]:
    starts = list(range(0, max(total - tile, 0) + 1, step))
    if starts[-1] + tile < total:
        starts.append(total - tile)
    return starts


def _shift(item: DetectedAnnotation, dx: int, dy: int) -> DetectedAnnotation:
    x1, y1, x2, y2 = item.box
    return DetectedAnnotation(
        text=item.text,
        box=(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
        confidence=item.confidence,
        kind=item.kind,
        polygon=tuple((px + dx, py + dy) for px, py in item.polygon),
    )


def _box_iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _merge_duplicates(
    items: list[DetectedAnnotation], threshold: float
) -> list[DetectedAnnotation]:
    kept: list[DetectedAnnotation] = []
    for item in sorted(items, key=lambda a: -a.confidence):
        if all(_box_iou(item.box, other.box) <= threshold for other in kept):
            kept.append(item)
    return kept


def build_text_mask(
    shape: tuple[int, int],
    annotations: list[DetectedAnnotation],
    padding: int,
    extra_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize annotation outlines (polygon when known, else box) into a mask.

    ``extra_mask`` (e.g. heuristic glyphs) is merged in before padding is applied.
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if extra_mask is not None:
        mask = extra_mask.astype(np.uint8, copy=True)
    for annotation in annotations:
        if annotation.polygon:
            points = np.array(annotation.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [points], 255)
        else:
            x1, y1, x2, y2 = annotation.box
            cv2.rectangle(mask, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), 255, -1)
    if padding > 0:
        size = 2 * padding + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        grown: np.ndarray = cv2.dilate(mask, kernel)
        return grown
    return mask


def heuristic_glyph_mask(gray: np.ndarray, config: PerceptionConfig) -> np.ndarray:
    """Mask isolated, glyph-sized specks (fallback when no OCR is available).

    Thresholds are scaled from a 2048 px reference to the actual image size.
    """
    scale = max(gray.shape[:2]) / REFERENCE_DIMENSION
    min_area, max_area, max_side = 20 * scale**2, 400 * scale**2, 30 * scale
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binarize(gray, config))
    areas = stats[:, cv2.CC_STAT_AREA]
    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    keep = (areas > min_area) & (areas < max_area) & (widths < max_side) & (heights < max_side)
    keep[0] = False
    glyphs: np.ndarray = keep[labels].astype(np.uint8) * 255
    return glyphs


def remove_text(gray: np.ndarray, mask: np.ndarray, config: PerceptionConfig) -> np.ndarray:
    """Return a copy of ``gray`` with the masked regions removed."""
    if not mask.any():
        return np.array(gray, copy=True)
    if config.text_removal is TextRemoval.FILL:
        cleaned: np.ndarray = np.array(gray, copy=True)
        cleaned[mask > 0] = 255
        return cleaned
    inpainted: np.ndarray = cv2.inpaint(gray, mask, config.inpaint_radius, cv2.INPAINT_TELEA)
    return inpainted
