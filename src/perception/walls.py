"""Wall segmentation and opening carving."""

from __future__ import annotations

import cv2
import numpy as np

from .config import PerceptionConfig
from .imageops import binarize, remove_small_components
from .models import Box

_MIN_LINE_PX = 5


def line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Structuring element: a 1-px line of ``length`` at ``angle_deg``."""
    angle = angle_deg % 180.0
    if angle == 0.0:
        return np.ones((1, length), dtype=np.uint8)
    if angle == 90.0:
        return np.ones((length, 1), dtype=np.uint8)
    size = length | 1  # odd
    kernel = np.zeros((size, size), dtype=np.uint8)
    centre = size // 2
    dx = int(round(np.cos(np.radians(angle)) * centre))
    dy = int(round(-np.sin(np.radians(angle)) * centre))
    cv2.line(kernel, (centre - dx, centre - dy), (centre + dx, centre + dy), 1, 1)
    return kernel


class WallSegmenter:
    """Classical wall extraction: binarize, keep long straight strokes, clean up."""

    def __init__(self, config: PerceptionConfig) -> None:
        self._cfg = config

    def segment(self, gray: np.ndarray) -> np.ndarray:
        """Return a uint8 mask (255 = wall) for a dark-on-light grayscale image."""
        height, width = gray.shape[:2]
        binary = binarize(gray, self._cfg)

        length = max(_MIN_LINE_PX, round(self._cfg.wall_min_length_ratio * max(height, width)))
        lines = np.zeros_like(binary)
        for angle in self._cfg.wall_angles_deg:
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, line_kernel(length, angle))
            lines = cv2.bitwise_or(lines, opened)

        if self._cfg.wall_close_px > 0 and self._cfg.wall_close_iterations > 0:
            kernel = np.ones((self._cfg.wall_close_px, self._cfg.wall_close_px), np.uint8)
            lines = cv2.morphologyEx(
                lines, cv2.MORPH_CLOSE, kernel, iterations=self._cfg.wall_close_iterations
            )

        min_area = int(self._cfg.min_component_area_ratio * height * width)
        return remove_small_components(lines, min_area)


def carve_openings(
    wall_mask: np.ndarray, boxes: list[Box], padding: int
) -> tuple[np.ndarray, np.ndarray]:
    """Erase ``boxes`` from the wall mask; return (carved_walls, opening_mask)."""
    height, width = wall_mask.shape[:2]
    carved = wall_mask.copy()
    openings = np.zeros_like(wall_mask)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
        if x2 <= x1 or y2 <= y1:
            continue
        carved[y1:y2, x1:x2] = 0
        openings[y1:y2, x1:x2] = 255
    return carved, openings
