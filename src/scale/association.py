"""Associate dimension texts with wall spans and rooms to produce scale candidates.

A dimension string such as ``3600`` is only evidence once we know *which
pixels* it measures. For each text we enumerate span hypotheses on the wall
graph:

* a contiguous run of collinear wall edges, parallel to the text baseline,
  near the text, and whose middle sits next to the text (dimension strings are
  centred on the span they describe);
* the whole plan extent along that direction;

and each span is tried as centreline-to-centreline, clear interior
(``L - t``) and outside-face (``L + t``) length. ``A x B`` labels are matched
to the rectangular room that contains the text. Every hypothesis becomes a
:class:`ScaleEvidence`; the consensus step later picks what most texts agree
on, so wrong hypotheses simply lose the vote.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from shapely import Point as ShapelyPoint
from shapely import Polygon

from topology import GraphVertex, PlanarGraph, RoomFace

from .config import ScaleConfig
from .dimensions import LengthReading, ParsedText, TextKind
from .models import ScaleEvidence, ScaleSource, TextObservation

_LINE_ANGLE_TOL_RAD = math.radians(3.0)  # edges this parallel and close form one wall line
_LINE_OFFSET_THICKNESSES = 0.75
_CHAIN_GAP_THICKNESSES = 1.0  # edges closer than this are one contiguous chain
_MAX_RUN_EDGES = 10  # longest run of consecutive edges enumerated as one span
_PROXIMITY_PENALTY = 0.4
_CENTERING_PENALTY = 0.4
_NEAR_SQUARE_RATIO = 1.15
_CENTERLINE_ROOM_FACTOR = 0.7  # room sizes read off centreline polygons are less exact


@dataclass(frozen=True)
class _WallLine:
    """A group of collinear wall edges and the contiguous runs they form."""

    origin: tuple[float, float]
    direction: tuple[float, float]  # unit vector, angle in [0, pi)
    angle: float
    chains: tuple[tuple[tuple[float, float], ...], ...]  # intervals along ``direction``


@dataclass(frozen=True)
class _Span:
    t0: float
    t1: float
    is_extent: bool = False

    @property
    def length(self) -> float:
        return self.t1 - self.t0

    @property
    def middle(self) -> float:
        return 0.5 * (self.t0 + self.t1)


@dataclass(frozen=True)
class _RoomShape:
    room: RoomFace
    container: Polygon  # centreline polygon: "is the text inside this room?"
    long_side: float  # measured on the interior polygon when available
    short_side: float
    rectangularity: float
    from_interior: bool


@dataclass
class _LineGroup:
    """Edges accumulated onto one wall line while grouping."""

    angle: float
    direction: tuple[float, float]
    origin: tuple[float, float]
    edges: list[tuple[GraphVertex, GraphVertex]]


def build_wall_lines(graph: PlanarGraph, thickness: float) -> list[_WallLine]:
    """Group graph edges into collinear lines and contiguous chains."""
    groups: list[_LineGroup] = []
    offset_tol = max(_LINE_OFFSET_THICKNESSES * thickness, 2.0)
    for edge in sorted(graph.edges, key=lambda e: -e.length):
        a, b = graph.vertices[edge.u], graph.vertices[edge.v]
        dx, dy = b.x - a.x, b.y - a.y
        if math.hypot(dx, dy) == 0:
            continue
        angle = math.atan2(dy, dx) % math.pi
        mid = ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
        group = next(
            (
                g
                for g in groups
                if _angle_diff(angle, g.angle) <= _LINE_ANGLE_TOL_RAD
                and _perp_distance(mid, g.origin, g.direction) <= offset_tol
            ),
            None,
        )
        if group is None:
            groups.append(
                _LineGroup(angle, (math.cos(angle), math.sin(angle)), (a.x, a.y), [(a, b)])
            )
        else:
            group.edges.append((a, b))

    gap = _CHAIN_GAP_THICKNESSES * thickness
    lines: list[_WallLine] = []
    for group in groups:
        intervals = []
        for a, b in group.edges:
            ta = _project((a.x, a.y), group.origin, group.direction)
            tb = _project((b.x, b.y), group.origin, group.direction)
            intervals.append((min(ta, tb), max(ta, tb)))
        intervals.sort()
        chains: list[list[tuple[float, float]]] = []
        for interval in intervals:
            if chains and interval[0] - chains[-1][-1][1] <= gap:
                chains[-1].append(interval)
            else:
                chains.append([interval])
        lines.append(
            _WallLine(
                origin=group.origin,
                direction=group.direction,
                angle=group.angle,
                chains=tuple(tuple(c) for c in chains),
            )
        )
    return lines


def collect_candidates(
    graph: PlanarGraph,
    observations: Sequence[tuple[int, TextObservation, ParsedText]],
    config: ScaleConfig,
) -> list[ScaleEvidence]:
    """Scale hypotheses for all LINEAR and ROOM_SIZE texts, strongest first."""
    if not graph.edges:
        return []
    xs = [v.x for v in graph.vertices.values()]
    ys = [v.y for v in graph.vertices.values()]
    extent_px = max(max(xs) - min(xs), max(ys) - min(ys))
    if extent_px <= 0:
        return []
    thickness = graph.thickness_px if graph.thickness_px > 0 else 1.0
    lines = build_wall_lines(graph, thickness)
    rooms = _build_room_shapes(graph, config)
    max_offset = config.max_text_offset_ratio * extent_px

    found: list[ScaleEvidence] = []
    for index, observation, parsed in observations:
        if parsed.kind is TextKind.LINEAR:
            found.extend(
                _span_hypotheses(
                    index, observation, parsed, graph, lines, thickness, max_offset, config
                )
            )
        elif parsed.kind is TextKind.ROOM_SIZE:
            found.extend(_room_hypotheses(index, observation, parsed, rooms, config))
    found.sort(key=lambda e: -e.weight)
    return found[: config.max_candidates]


# -- span hypotheses ----------------------------------------------------------


def _span_hypotheses(
    index: int,
    observation: TextObservation,
    parsed: ParsedText,
    graph: PlanarGraph,
    lines: Sequence[_WallLine],
    thickness: float,
    max_offset: float,
    config: ScaleConfig,
) -> list[ScaleEvidence]:
    centre, baseline = text_axis(observation)
    angle_tol = math.radians(config.angle_tolerance_deg)
    scored: list[tuple[float, _Span, float, float, float]] = []  # score, span, d, off, tol_len

    for line in lines:
        if _angle_diff(baseline, line.angle) > angle_tol:
            continue
        distance = _perp_distance(centre, line.origin, line.direction)
        if distance > max_offset:
            continue
        t_centre = _project(centre, line.origin, line.direction)
        spans = list(_runs(line))
        spans.append(_extent_span(graph, line))
        for span in spans:
            if span.length <= 0:
                continue
            tolerance = max(config.span_center_tolerance_ratio * span.length, thickness)
            offset = abs(t_centre - span.middle)
            if offset > tolerance:
                continue
            score = distance / max_offset + offset / tolerance
            scored.append((score, span, distance, offset, tolerance))

    scored.sort(key=lambda item: item[0])
    evidence: list[ScaleEvidence] = []
    seen: set[tuple[int, int]] = set()
    kept = 0
    for _, span, distance, offset, tolerance in scored:
        signature = (round(span.t0), round(span.t1))
        if signature in seen:
            continue
        seen.add(signature)
        kept += 1
        if kept > config.max_hypotheses_per_text:
            break
        proximity = 1.0 - _PROXIMITY_PENALTY * distance / max_offset
        centering = 1.0 - _CENTERING_PENALTY * offset / tolerance
        base = observation.confidence * proximity * centering
        for reading in parsed.lengths[0]:
            evidence.extend(
                _length_variants(index, observation, reading, span, thickness, base, config)
            )
    return evidence


def _length_variants(
    index: int,
    observation: TextObservation,
    reading: LengthReading,
    span: _Span,
    thickness: float,
    base_weight: float,
    config: ScaleConfig,
) -> list[ScaleEvidence]:
    variants = (
        ("centerline", span.length, config.weight_centerline),
        ("interior", span.length - thickness, config.weight_interior),
        ("exterior", span.length + thickness, config.weight_exterior),
    )
    out: list[ScaleEvidence] = []
    for name, pixels, prior in variants:
        weight = base_weight * reading.weight * prior
        if pixels <= 0 or weight <= 0:
            continue
        kind = "plan extent" if span.is_extent else "wall span"
        out.append(
            ScaleEvidence(
                source=ScaleSource.DIMENSION_TEXT,
                meters_per_pixel=reading.meters / pixels,
                weight=weight,
                group=index,
                text=observation.text,
                note=f"{kind} ({name}), read as {reading.unit}",
                convention=name,
            )
        )
    return out


def _runs(line: _WallLine) -> list[_Span]:
    spans: list[_Span] = []
    for chain in line.chains:
        for i in range(len(chain)):
            for j in range(i, min(len(chain), i + _MAX_RUN_EDGES)):
                spans.append(_Span(chain[i][0], chain[j][1]))
    return spans


def _extent_span(graph: PlanarGraph, line: _WallLine) -> _Span:
    values = [_project((v.x, v.y), line.origin, line.direction) for v in graph.vertices.values()]
    return _Span(min(values), max(values), is_extent=True)


# -- room-size hypotheses -----------------------------------------------------


def _build_room_shapes(graph: PlanarGraph, config: ScaleConfig) -> list[_RoomShape]:
    shapes: list[_RoomShape] = []
    for room in graph.rooms:
        container = Polygon(room.polygon)
        measured_ring = room.interior_polygon or room.polygon
        measured = Polygon(measured_ring)
        if not container.is_valid or not measured.is_valid or measured.area <= 0:
            continue
        rectangle = measured.minimum_rotated_rectangle
        if rectangle.is_empty or rectangle.area <= 0:
            continue
        corners = list(rectangle.exterior.coords)
        side_a = math.dist(corners[0], corners[1])
        side_b = math.dist(corners[1], corners[2])
        rectangularity = measured.area / rectangle.area
        if rectangularity < config.room_min_rectangularity:
            continue
        shapes.append(
            _RoomShape(
                room=room,
                container=container,
                long_side=max(side_a, side_b),
                short_side=min(side_a, side_b),
                rectangularity=rectangularity,
                from_interior=room.interior_polygon is not None,
            )
        )
    return shapes


def _room_hypotheses(
    index: int,
    observation: TextObservation,
    parsed: ParsedText,
    rooms: Sequence[_RoomShape],
    config: ScaleConfig,
) -> list[ScaleEvidence]:
    centre, _ = text_axis(observation)
    point = ShapelyPoint(centre)
    shape = next((r for r in rooms if r.container.contains(point)), None)
    if shape is None or shape.short_side <= 0:
        return []
    factor = 1.0 if shape.from_interior else _CENTERLINE_ROOM_FACTOR
    near_square = shape.long_side / shape.short_side < _NEAR_SQUARE_RATIO

    out: list[ScaleEvidence] = []
    for first in parsed.lengths[0]:
        for second in parsed.lengths[1]:
            big, small = max(first.meters, second.meters), min(first.meters, second.meters)
            weight = (
                observation.confidence
                * config.room_size_weight
                * first.weight
                * second.weight
                * factor
                * shape.rectangularity
            )
            pairs = [(big, shape.long_side, "long"), (small, shape.short_side, "short")]
            if near_square:  # which label side maps to which wall pair is arbitrary
                pairs += [
                    (big, shape.short_side, "long/short"),
                    (small, shape.long_side, "short/long"),
                ]
            for meters, pixels, side in pairs:
                out.append(
                    ScaleEvidence(
                        source=ScaleSource.DIMENSION_TEXT,
                        meters_per_pixel=meters / pixels,
                        weight=weight * (0.5 if "/" in side else 1.0),
                        group=index,
                        text=observation.text,
                        note=f"room {shape.room.id} {side} side",
                        convention="interior" if shape.from_interior else "centerline",
                    )
                )
    return out


# -- text geometry ------------------------------------------------------------


def text_axis(observation: TextObservation) -> tuple[tuple[float, float], float]:
    """Centre and baseline angle (radians, in [0, pi)) of a text observation."""
    if len(observation.polygon) >= 4:
        pts = observation.polygon[:4]
        centre = (sum(p[0] for p in pts) / 4.0, sum(p[1] for p in pts) / 4.0)
        first = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
        second = (pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
        longer = first if math.hypot(*first) >= math.hypot(*second) else second
        return centre, math.atan2(longer[1], longer[0]) % math.pi
    x1, y1, x2, y2 = observation.box
    centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return centre, 0.0 if (x2 - x1) >= (y2 - y1) else math.pi / 2.0


def _angle_diff(a: float, b: float) -> float:
    """Smallest difference between two undirected angles in [0, pi)."""
    diff = abs(a - b) % math.pi
    return min(diff, math.pi - diff)


def _project(
    point: tuple[float, float], origin: tuple[float, float], direction: tuple[float, float]
) -> float:
    return (point[0] - origin[0]) * direction[0] + (point[1] - origin[1]) * direction[1]


def _perp_distance(
    point: tuple[float, float], origin: tuple[float, float], direction: tuple[float, float]
) -> float:
    return abs((point[0] - origin[0]) * direction[1] - (point[1] - origin[1]) * direction[0])
