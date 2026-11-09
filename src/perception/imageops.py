"""Shared low-level image helpers."""

from __future__ import annotations

import cv2
import numpy as np

from .config import Binarization, PerceptionConfig
from .errors import InvalidImageError

REFERENCE_DIMENSION = 2048  # resolution at which pixel-size constants were tuned
_MIN_ADAPTIVE_BLOCK = 31


def to_gray_uint8(image: object) -> np.ndarray:
    """Validate the input and return a 2-D uint8 grayscale array."""
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"Expected a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise InvalidImageError(f"Expected uint8 image, got {image.dtype}")
    if image.size == 0 or image.ndim not in (2, 3):
        raise InvalidImageError(f"Unusable image shape: {image.shape}")
    if image.ndim == 3:
        if image.shape[2] != 3:
            raise InvalidImageError(f"Expected 3 channels (BGR), got {image.shape[2]}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def binarize(gray: np.ndarray, config: PerceptionConfig) -> np.ndarray:
    """Return a mask with ink (dark pixels) = 255."""
    if config.binarization is Binarization.OTSU:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    longest = max(gray.shape[:2])
    block = max(_MIN_ADAPTIVE_BLOCK, int(config.adaptive_block_fraction * longest))
    block += 1 - block % 2  # must be odd
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        config.adaptive_c,
    )


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components smaller than ``min_area`` pixels."""
    if min_area <= 1:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # background
    out: np.ndarray = (keep[labels].astype(np.uint8)) * 255
    return out
