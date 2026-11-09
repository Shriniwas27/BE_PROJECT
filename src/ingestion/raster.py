"""Raster normalization: decoding, dewarping, polarity, contrast, resizing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import IngestionConfig
from .errors import CorruptFileError

logger = logging.getLogger(__name__)

_APPROX_EPSILON = 0.02  # fraction of contour perimeter for polygon approximation


@dataclass(frozen=True)
class RasterOutput:
    image: np.ndarray  # uint8 grayscale
    original_size: tuple[int, int]  # (width, height) of the input
    scale_factor: float
    dewarped: bool


def load_image(path: Path) -> np.ndarray:
    """Decode an image preserving bit depth and alpha.

    Uses ``imdecode`` so that non-ASCII paths work on every platform.
    """
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise CorruptFileError(f"Could not read {path}: {exc}") from exc
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise CorruptFileError(f"Could not decode image: {path}")
    return image


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert any bit depth to uint8."""
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return (image // 257).astype(np.uint8)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(image.max(initial=0.0)) <= 1.0 else 1.0
        return np.clip(image * scale, 0, 255).astype(np.uint8)
    low, high = float(image.min()), float(image.max())
    if high == low:
        return np.zeros(image.shape, dtype=np.uint8)
    return ((image.astype(np.float64) - low) * (255.0 / (high - low))).astype(np.uint8)


def flatten_alpha(image: np.ndarray) -> np.ndarray:
    """Composite BGRA/GA images over white; return 2-D gray or 3-channel BGR."""
    if image.ndim == 2:
        return image
    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    if channels == 3:
        return image
    if channels == 2:  # gray + alpha
        color, alpha = image[:, :, 0:1], image[:, :, 1:2]
    elif channels == 4:
        color, alpha = image[:, :, :3], image[:, :, 3:4]
    else:
        raise CorruptFileError(f"Unsupported channel count: {channels}")
    weight = alpha.astype(np.float32) / 255.0
    blended = color.astype(np.float32) * weight + 255.0 * (1.0 - weight)
    out: np.ndarray = blended.astype(np.uint8)
    return out[:, :, 0] if channels == 2 else out


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = points.reshape(4, 2).astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(sums)], pts[np.argmin(diffs)], pts[np.argmax(sums)], pts[np.argmax(diffs)]],
        dtype=np.float32,
    )


class RasterNormalizer:
    """Turns arbitrary raster input into dark-on-light, bounded grayscale."""

    def __init__(self, config: IngestionConfig) -> None:
        self._cfg = config

    def normalize(self, image: np.ndarray, *, allow_dewarp: bool) -> RasterOutput:
        image = flatten_alpha(to_uint8(image))
        height, width = image.shape[:2]

        dewarped = False
        if allow_dewarp:
            corrected = self._dewarp(image)
            if corrected is not None:
                image, dewarped = corrected, True

        gray = self._normalize_polarity(to_gray(image))
        if self._cfg.apply_clahe:
            grid = self._cfg.clahe_tile_grid
            clahe = cv2.createCLAHE(clipLimit=self._cfg.clahe_clip_limit, tileGridSize=(grid, grid))
            gray = clahe.apply(gray)

        gray, scale = self._resize(gray)
        return RasterOutput(gray, (width, height), scale, dewarped)

    # -- steps ------------------------------------------------------------

    def _find_paper_quad(self, image: np.ndarray) -> np.ndarray | None:
        height, width = image.shape[:2]
        total = float(width * height)
        blurred = cv2.GaussianBlur(to_gray(image), (5, 5), 0)
        edges = cv2.Canny(blurred, self._cfg.canny_low, self._cfg.canny_high)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            ratio = cv2.contourArea(contour) / total
            if ratio < self._cfg.dewarp_min_area_ratio:
                break  # sorted: nothing larger remains
            if ratio > self._cfg.dewarp_max_area_ratio:
                continue  # image frame / page border, not paper
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, _APPROX_EPSILON * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                if self._is_axis_aligned(approx):
                    continue  # a straight frame needs no correction and would only crop
                return approx
        return None

    def _is_axis_aligned(self, quad: np.ndarray) -> bool:
        pts = order_corners(quad)
        edges = np.roll(pts, -1, axis=0) - pts
        angles = np.degrees(np.arctan2(edges[:, 1], edges[:, 0]))
        deviation = np.abs((angles + 45.0) % 90.0 - 45.0)  # distance to nearest multiple of 90
        return bool(deviation.max() < self._cfg.dewarp_min_skew_degrees)

    def _dewarp(self, image: np.ndarray) -> np.ndarray | None:
        """Return the perspective-corrected image, or None if no paper found."""
        quad = self._find_paper_quad(image)
        if quad is None:
            return None
        tl, tr, br, bl = rect = order_corners(quad)
        out_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        out_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if out_w < 2 or out_h < 2:
            return None
        dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        logger.debug("Applying perspective correction to %dx%d", out_w, out_h)
        return cv2.warpPerspective(image, matrix, (out_w, out_h))

    def _normalize_polarity(self, gray: np.ndarray) -> np.ndarray:
        """Invert dark-background drawings (e.g. blueprints).

        Line drawings are mostly background, so the global median is a more
        robust background estimate than border pixels (which title blocks and
        scan shadows corrupt).
        """
        if float(np.median(gray)) < self._cfg.polarity_threshold:
            logger.debug("Dark background detected; inverting")
            return cv2.bitwise_not(gray)
        return gray

    def _resize(self, gray: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = gray.shape[:2]
        longest = max(height, width)
        if longest <= self._cfg.max_dimension:
            return gray, 1.0
        scale = self._cfg.max_dimension / float(longest)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return cv2.resize(gray, size, interpolation=cv2.INTER_AREA), scale
