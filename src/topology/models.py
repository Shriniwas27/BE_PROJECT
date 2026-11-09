"""Data models returned by the topology pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

Point = tuple[float, float]
Box = tuple[int, int, int, int]


class OpeningType(str, Enum):
    DOOR = "door"
    WINDOW = "window"
    OPENING = "opening"


@dataclass(frozen=True)
class OpeningInterval:
    """An opening on a wall edge, in parametric coordinates along the edge."""

    opening_type: OpeningType
    t1: float  # [0, 1] start along the edge (u -> v)
    t2: float  # [0, 1] end along the edge
    width_px: float
    center_px: Point  # point on the wall axis at the middle of the opening


@dataclass(frozen=True)
class GraphVertex:
    id: int
    x: float
    y: float


@dataclass
class GraphEdge:
    """A straight wall segment between two vertices.

    ``support`` is the fraction of the edge lying on wall pixels. Values well
    below 1 indicate an edge that was synthesised, e.g. bridging a door gap.
    """

    id: int
    u: int
    v: int
    length: float
    thickness: float
    support: float = 1.0
    openings: list[OpeningInterval] = field(default_factory=list)


@dataclass(frozen=True)
class RoomFace:
    id: int
    polygon: list[Point]  # exterior ring along wall centrelines (closed)
    area_px: float  # centreline area, net of holes
    centroid: Point
    holes: list[list[Point]] = field(default_factory=list)
    interior_polygon: list[Point] | None = None  # inset by half a wall thickness
    interior_area_px: float = 0.0


@dataclass
class PlanarGraph:
    """Wall network and rooms, in the pixel space of the input mask."""

    vertices: dict[int, GraphVertex]
    edges: list[GraphEdge]
    rooms: list[RoomFace]
    thickness_px: float = 0.0  # median measured wall thickness
    rotation_deg: float = 0.0  # tilt of the plan's own axes relative to the image axes
    dangling_vertices: list[int] = field(default_factory=list)  # degree-1 vertices
    unattached_openings: list[tuple[OpeningType, Box]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        result: dict[str, Any] = _plain(self)
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
