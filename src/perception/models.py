"""Data models returned by the perception pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

Box = tuple[int, int, int, int]  # (x1, y1, x2, y2) in input-image pixels
Polygon = tuple[tuple[int, int], ...]


class SymbolCategory(str, Enum):
    DOOR = "door"
    WINDOW = "window"
    OPENING = "opening"


class AnnotationKind(str, Enum):
    DIMENSION = "dimension"  # looks like a measurement, e.g. "3.5m", "12'-6\""
    LABEL = "label"  # any other text, e.g. "BEDROOM"


@dataclass(frozen=True)
class DetectedSymbol:
    label: str  # raw class name from the detector
    category: SymbolCategory
    box: Box
    confidence: float
    source: str = "model"  # "model" or "heuristic"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class DetectedAnnotation:
    text: str
    box: Box
    confidence: float
    kind: AnnotationKind = AnnotationKind.LABEL
    polygon: Polygon = ()  # exact (possibly rotated) text outline; empty -> use box


@dataclass
class PerceptionResult:
    """Everything is expressed in the pixel space of the input image.

    Attributes:
        wall_mask: uint8, 255 = wall. Detected openings are carved out.
        opening_mask: uint8, 255 where openings were carved from the walls.
        text_mask: uint8, 255 where text was removed.
        cleaned_image: uint8 grayscale with text removed.
        image_size: (width, height) of the processed image.
        warnings: Non-fatal issues (e.g. a detector was unavailable).
    """

    wall_mask: np.ndarray
    opening_mask: np.ndarray
    text_mask: np.ndarray
    cleaned_image: np.ndarray
    image_size: tuple[int, int]
    doors: list[DetectedSymbol] = field(default_factory=list)
    windows: list[DetectedSymbol] = field(default_factory=list)
    openings: list[DetectedSymbol] = field(default_factory=list)
    annotations: list[DetectedAnnotation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
