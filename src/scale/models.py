"""Data models for the scale package."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import InvalidScaleInputError
from .units import SQUARE_FEET_PER_SQUARE_METER

Point = tuple[float, float]
Box = tuple[int, int, int, int]


class ScaleSource(str, Enum):
    MANUAL = "manual"  # tier 1: user anchor(s)
    DIMENSION_TEXT = "dimension_text"  # tier 2: OCR dimensions matched to walls
    DRAWING_RATIO = "drawing_ratio"  # tier 2: "1:100" plus the page DPI
    DOOR_PRIOR = "door_prior"  # tier 3: standard door width
    VECTOR_UNITS = "vector_units"  # exact: DXF drawing units


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class WallKind(str, Enum):
    EXTERIOR = "exterior"  # bounds exactly one room
    INTERIOR = "interior"  # shared by two or more rooms
    UNKNOWN = "unknown"  # touches no closed room


class OpeningKind(str, Enum):
    DOOR = "door"
    WINDOW = "window"
    OPENING = "opening"


@dataclass(frozen=True)
class TextObservation:
    """A piece of text found on the plan, in image pixels.

    ``polygon`` is the exact (possibly rotated) outline; when empty, ``box`` is used.
    """

    text: str
    box: Box
    confidence: float = 1.0
    polygon: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ManualAnchor:
    """Two image points a known real-world distance apart (tier 1, ground truth).

    Points are in the pixel space of the image given to Module 2.
    """

    p1: Point
    p2: Point
    length_m: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.length_m) and self.length_m > 0):
            raise InvalidScaleInputError("ManualAnchor.length_m must be a positive number")
        if self.pixel_length <= 0:
            raise InvalidScaleInputError("ManualAnchor points must be distinct")

    @property
    def pixel_length(self) -> float:
        return math.dist(self.p1, self.p2)

    @property
    def meters_per_pixel(self) -> float:
        return self.length_m / self.pixel_length


@dataclass(frozen=True)
class ScaleEvidence:
    """One piece of evidence considered by the resolver (kept for transparency).

    ``group`` identifies the originating text/door set so that several
    hypotheses from one text count as a single vote. ``convention`` records
    what the dimension was assumed to measure (wall centrelines, clear
    interior faces, or outside faces).
    """

    source: ScaleSource
    meters_per_pixel: float
    weight: float
    group: int
    text: str | None = None
    note: str = ""
    used: bool = False
    convention: str = ""  # "centerline" | "interior" | "exterior"; "" = convention-neutral


@dataclass(frozen=True)
class ScaleResult:
    """The resolved real-world scale.

    Attributes:
        meters_per_pixel: The global scale factor S.
        source: Which tier produced it.
        confidence: 0-1 heuristic score (not a calibrated probability).
        level: Coarse bucket of ``confidence``.
        relative_uncertainty: Heuristic 1-sigma relative error of S.
        support: Independent observations that agree with S.
        evidence: Everything considered, with ``used`` marking agreeing items.
        warnings: Non-fatal issues (disagreements, implausible values, ...).
    """

    meters_per_pixel: float
    source: ScaleSource
    confidence: float
    level: ConfidenceLevel
    relative_uncertainty: float
    support: int = 0
    evidence: list[ScaleEvidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def pixels_per_meter(self) -> float:
        return 1.0 / self.meters_per_pixel

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = _plain(self)
        return result


@dataclass(frozen=True)
class PlanTransform:
    """Mapping between image pixels and plan metres.

    ``x_m = (x_px - origin_x) * s``; ``y_m = (origin_y - y_px) * s`` when
    ``flip_y`` else ``(y_px - origin_y) * s``.
    """

    meters_per_pixel: float
    flip_y: bool
    origin_x_px: float
    origin_y_px: float

    def to_meters(self, point: Point) -> Point:
        s = self.meters_per_pixel
        x = (point[0] - self.origin_x_px) * s
        y = (self.origin_y_px - point[1]) * s if self.flip_y else (point[1] - self.origin_y_px) * s
        return (x, y)

    def to_pixels(self, point: Point) -> Point:
        s = self.meters_per_pixel
        x = point[0] / s + self.origin_x_px
        y = self.origin_y_px - point[1] / s if self.flip_y else point[1] / s + self.origin_y_px
        return (x, y)


@dataclass(frozen=True)
class ScaledOpening:
    """A door/window/opening on a wall, in metres.

    ``start_m``/``end_m`` are distances from the wall's start point; heights are
    above the floor.
    """

    kind: OpeningKind
    start_m: float
    end_m: float
    width_m: float
    sill_height_m: float
    head_height_m: float
    center: Point

    @property
    def height_m(self) -> float:
        return self.head_height_m - self.sill_height_m


@dataclass(frozen=True)
class ScaledWall:
    id: int
    start: Point
    end: Point
    length_m: float
    thickness_m: float
    height_m: float
    kind: WallKind
    thickness_clamped: bool = False
    openings: list[ScaledOpening] = field(default_factory=list)


@dataclass(frozen=True)
class ScaledRoom:
    """Room footprint in metres. Exteriors are counter-clockwise, holes clockwise."""

    id: int
    polygon: list[Point]
    centroid: Point
    area_m2: float  # along wall centrelines, net of holes
    holes: list[list[Point]] = field(default_factory=list)
    interior_polygon: list[Point] | None = None
    interior_area_m2: float = 0.0  # usable floor area

    @property
    def area_sqft(self) -> float:
        return self.area_m2 * SQUARE_FEET_PER_SQUARE_METER

    @property
    def interior_area_sqft(self) -> float:
        return self.interior_area_m2 * SQUARE_FEET_PER_SQUARE_METER


@dataclass(frozen=True)
class ScaledPlan:
    """The wall graph in real-world metres, with parametric elevations applied."""

    scale: ScaleResult
    transform: PlanTransform
    walls: list[ScaledWall]
    rooms: list[ScaledRoom]
    ceiling_height_m: float
    bounds_m: tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = _plain(self)
        result["units"] = "meters"
        return result


def _plain(value: Any) -> Any:
    """Recursively convert dataclasses/enums/tuples into JSON-friendly types."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {name: _plain(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
