"""Apply a resolved scale and parametric elevations to a pixel-space wall graph."""

from __future__ import annotations

import statistics

from shapely import LineString, Point

from topology import GraphEdge, OpeningInterval, OpeningType, PlanarGraph, RoomFace

from .config import ScaleConfig
from .models import (
    OpeningKind,
    PlanTransform,
    ScaledOpening,
    ScaledPlan,
    ScaledRoom,
    ScaledWall,
    ScaleResult,
    WallKind,
)
from .models import (
    Point as Pt,
)

_BOUNDARY_TOLERANCE_PX = 1.0
_BOUNDARY_TOLERANCE_THICKNESSES = 0.25
_DOOR_WIDTH_SANITY_M = (0.5, 2.5)
_WINDOW_MAX_WIDTH_M = 4.0

_KIND_BY_TYPE = {
    OpeningType.DOOR: OpeningKind.DOOR,
    OpeningType.WINDOW: OpeningKind.WINDOW,
    OpeningType.OPENING: OpeningKind.OPENING,
}


def apply_scale(graph: PlanarGraph, scale: ScaleResult, config: ScaleConfig) -> ScaledPlan:
    """Convert ``graph`` to metres, extrude to ``config.ceiling_height_m`` and set sills."""
    s = scale.meters_per_pixel
    transform = _make_transform(graph, s, config)
    kinds = _classify_walls(graph)
    warnings = list(graph.warnings)

    walls: list[ScaledWall] = []
    clamped: list[float] = []
    odd_openings = 0
    for edge in graph.edges:
        a, b = graph.vertices[edge.u], graph.vertices[edge.v]
        length_m = edge.length * s
        measured = (edge.thickness if edge.thickness > 0 else graph.thickness_px) * s
        thickness = min(max(measured, config.min_wall_thickness_m), config.max_wall_thickness_m)
        was_clamped = thickness != measured
        if was_clamped:
            clamped.append(measured)
        openings = [_scale_opening(o, edge, length_m, transform, config) for o in edge.openings]
        odd_openings += sum(1 for o in openings if _is_implausible(o))
        walls.append(
            ScaledWall(
                id=edge.id,
                start=transform.to_meters((a.x, a.y)),
                end=transform.to_meters((b.x, b.y)),
                length_m=length_m,
                thickness_m=thickness,
                height_m=config.ceiling_height_m,
                kind=kinds.get(edge.id, WallKind.UNKNOWN),
                thickness_clamped=was_clamped,
                openings=openings,
            )
        )

    if clamped:
        warnings.append(
            f"{len(clamped)} wall(s) had thickness clamped to "
            f"[{config.min_wall_thickness_m:g}, {config.max_wall_thickness_m:g}] m "
            f"(measured {min(clamped):.2f}-{max(clamped):.2f} m, median "
            f"{statistics.median(clamped):.2f} m). Double-line walls under-measure."
        )
    if odd_openings:
        warnings.append(
            f"{odd_openings} opening(s) have an implausible width at this scale; "
            "check the scale and the detections."
        )

    rooms = [_scale_room(room, transform, s) for room in graph.rooms]
    return ScaledPlan(
        scale=scale,
        transform=transform,
        walls=walls,
        rooms=rooms,
        ceiling_height_m=config.ceiling_height_m,
        bounds_m=_bounds(walls, rooms),
        warnings=warnings,
    )


# -- transform ----------------------------------------------------------------


def _make_transform(graph: PlanarGraph, s: float, config: ScaleConfig) -> PlanTransform:
    points = [(v.x, v.y) for v in graph.vertices.values()]
    for room in graph.rooms:
        points.extend(room.polygon)
    if not config.translate_to_origin or not points:
        return PlanTransform(s, config.flip_y, 0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    origin_y = max(ys) if config.flip_y else min(ys)
    return PlanTransform(s, config.flip_y, min(xs), origin_y)


# -- walls --------------------------------------------------------------------


def _classify_walls(graph: PlanarGraph) -> dict[int, WallKind]:
    """Exterior = bounds exactly one room, interior = shared by two or more."""
    if not graph.rooms or not graph.edges:
        return {}
    rings = [_room_rings(room) for room in graph.rooms]
    tolerance = max(_BOUNDARY_TOLERANCE_PX, _BOUNDARY_TOLERANCE_THICKNESSES * graph.thickness_px)
    kinds: dict[int, WallKind] = {}
    for edge in graph.edges:
        a, b = graph.vertices[edge.u], graph.vertices[edge.v]
        midpoint = Point((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
        touching = sum(
            1 for room_rings in rings if any(r.distance(midpoint) <= tolerance for r in room_rings)
        )
        kinds[edge.id] = (
            WallKind.UNKNOWN
            if touching == 0
            else WallKind.EXTERIOR
            if touching == 1
            else WallKind.INTERIOR
        )
    return kinds


def _room_rings(room: RoomFace) -> list[LineString]:
    return [LineString(room.polygon)] + [LineString(h) for h in room.holes if len(h) >= 2]


def _scale_opening(
    opening: OpeningInterval,
    edge: GraphEdge,
    length_m: float,
    transform: PlanTransform,
    config: ScaleConfig,
) -> ScaledOpening:
    kind = _KIND_BY_TYPE[opening.opening_type]
    sill, head = {
        OpeningKind.DOOR: (config.door_sill_m, config.door_head_m),
        OpeningKind.WINDOW: (config.window_sill_m, config.window_head_m),
        OpeningKind.OPENING: (config.opening_sill_m, config.opening_head_m),
    }[kind]
    start_m, end_m = opening.t1 * length_m, opening.t2 * length_m
    return ScaledOpening(
        kind=kind,
        start_m=start_m,
        end_m=end_m,
        width_m=end_m - start_m,
        sill_height_m=sill,
        head_height_m=head,
        center=transform.to_meters(opening.center_px),
    )


def _is_implausible(opening: ScaledOpening) -> bool:
    if opening.kind is OpeningKind.DOOR:
        return not _DOOR_WIDTH_SANITY_M[0] <= opening.width_m <= _DOOR_WIDTH_SANITY_M[1]
    if opening.kind is OpeningKind.WINDOW:
        return opening.width_m > _WINDOW_MAX_WIDTH_M
    return False


# -- rooms --------------------------------------------------------------------


def _scale_room(room: RoomFace, transform: PlanTransform, s: float) -> ScaledRoom:
    interior = (
        _orient(_ring(room.interior_polygon, transform), counter_clockwise=True)
        if room.interior_polygon
        else None
    )
    return ScaledRoom(
        id=room.id,
        polygon=_orient(_ring(room.polygon, transform), counter_clockwise=True),
        centroid=transform.to_meters(room.centroid),
        area_m2=room.area_px * s * s,
        holes=[_orient(_ring(h, transform), counter_clockwise=False) for h in room.holes],
        interior_polygon=interior,
        interior_area_m2=room.interior_area_px * s * s,
    )


def _ring(points: list[Pt], transform: PlanTransform) -> list[Pt]:
    return [transform.to_meters(p) for p in points]


def _orient(ring: list[Pt], counter_clockwise: bool) -> list[Pt]:
    """Return ``ring`` wound the requested way (flipping Y reverses winding)."""
    area = sum(
        x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1], strict=True)
    )
    return ring if (area > 0) == counter_clockwise else ring[::-1]


def _bounds(walls: list[ScaledWall], rooms: list[ScaledRoom]) -> tuple[float, float, float, float]:
    points = [p for w in walls for p in (w.start, w.end)] + [p for r in rooms for p in r.polygon]
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))
