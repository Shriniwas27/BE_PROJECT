"""Data models returned by the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

Point = tuple[float, float]


class InputType(str, Enum):
    RASTER = "raster"
    VECTOR = "vector"


@dataclass(frozen=True)
class VectorPrimitive:
    """A polyline extracted from CAD data (curves are already flattened)."""

    layer: str
    entity_type: str  # original DXF type, e.g. 'LINE', 'ARC', 'SPLINE'
    points: tuple[Point, ...]
    closed: bool = False


@dataclass(frozen=True)
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y


@dataclass
class IngestionMetadata:
    """Provenance and scale information needed to interpret the result.

    Attributes:
        source_format: Detected file format (e.g. 'pdf', 'png', 'dxf').
        dpi: For PDF input, the resolution of the returned image relative to
            the physical page (pixels per inch). ``None`` when unknown.
        scale_factor: Processed pixels divided by source pixels (<= 1 when the
            image was downscaled). 1.0 for vector input.
        dewarped: True only if a perspective transform was actually applied.
        page_index / page_count: PDF page information.
        units: DXF drawing units (e.g. 'Millimeters'), if declared.
        bounds: Extents of the extracted vector geometry.
        entity_count: Number of vector primitives extracted.
        skipped_entities: DXF entity types that were ignored, with counts.
        warnings: Non-fatal issues encountered during processing.
    """

    source_format: str
    dpi: float | None = None
    scale_factor: float = 1.0
    dewarped: bool = False
    page_index: int | None = None
    page_count: int | None = None
    units: str | None = None
    bounds: Bounds | None = None
    entity_count: int = 0
    skipped_entities: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class IngestionResult:
    input_type: InputType
    metadata: IngestionMetadata
    image: np.ndarray | None = None  # uint8 grayscale, dark lines on light bg
    vectors: list[VectorPrimitive] | None = None
    original_dimensions: tuple[int, int] | None = None  # (width, height) px
