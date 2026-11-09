"""Opening handling: bridging door gaps and attaching openings to wall edges."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely import LineString, Point, STRtree
from shapely import box as shapely_box

from .config import TopologyConfig
from .geometry import Pt, Segment
from .models import Box, GraphEdge, GraphVertex, OpeningInterval, OpeningType

_MIN_WIDTH_PX = 1.0
_NODE_EPS = 1e-6


@dataclass(frozen=True)
class OpeningInput:
    kind: OpeningType
    box: Box


@dataclass(frozen=True)
class _End:
    segment: int
    point: Pt
    outward: Pt  # unit vector pointing away from the wall body


def bridge_openings(
    segments: list[Segment],
    openings: list[OpeningInput],
    thickness: float,
    config: TopologyConfig,
) -> tuple[list[Segment], int]:
    """Add one connecting segment across each opening gap.

    Module 2 carves openings out of the wall mask, so the wall network is
    broken exactly where doors are. For every opening we look for two wall ends
    inside its (expanded) box that face each other along a common line, and
    join them. Returns the extended segment list and the number of bridges.
    """
    ends = _collect_ends(segments)
    used: set[tuple[int, Pt]] = set()
    cos_tol = math.cos(math.radians(config.bridge_angle_tol_deg))
    offset_tol = config.bridge_offset_ratio * thickness
    bridges: list[Segment] = []

    for opening in openings:
        x1, y1, x2, y2 = opening.box
        margin = thickness
        inside = [
            e
            for e in ends
            if x1 - margin <= e.point[0] <= x2 + margin and y1 - margin <= e.point[1] <= y2 + margin
        ]
        max_gap = max(x2 - x1, y2 - y1) + 4.0 * thickness
        best: tuple[float, _End, _End] | None = None
        for i, p in enumerate(inside):
            for q in inside[i + 1 :]:
                if (
                    p.segment == q.segment
                    or (p.segment, p.point) in used
                    or (q.segment, q.point) in used
                ):
                    continue
                dx, dy = q.point[0] - p.point[0], q.point[1] - p.point[1]
                gap = math.hypot(dx, dy)
                if gap < _NODE_EPS or gap > max_gap:
                    continue
                ux, uy = dx / gap, dy / gap
                faces = p.outward[0] * ux + p.outward[1] * uy >= cos_tol
                faces_back = q.outward[0] * -ux + q.outward[1] * -uy >= cos_tol
                offset = abs(p.outward[0] * dy - p.outward[1] * dx)
                if (
                    faces
                    and faces_back
                    and offset <= offset_tol
                    and (best is None or gap < best[0])
                ):
                    best = (gap, p, q)
        if best is not None:
            _, p, q = best
            used.add((p.segment, p.point))
            used.add((q.segment, q.point))
            bridges.append((p.point, q.point))
    return segments + bridges, len(bridges)


def _collect_ends(segments: list[Segment]) -> list[_End]:
    ends: list[_End] = []
    for i, (a, b) in enumerate(segments):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length == 0:
            continue
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        ends.append(_End(i, a, (-ux, -uy)))
        ends.append(_End(i, b, (ux, uy)))
    return ends


def attach_openings(
    edges: list[GraphEdge],
    vertices: dict[int, GraphVertex],
    openings: list[OpeningInput],
    config: TopologyConfig,
) -> list[OpeningInput]:
    """Project each opening onto its host edge; returns the openings that found none.

    The host is the nearby edge that overlaps the opening box the most (a wall
    running through the box beats a perpendicular wall merely touching it).
    Distance is measured to the box, not its centre, because door-swing boxes
    extend well into the room.
    """
    if not edges or not openings:
        return list(openings)
    lines = [
        LineString([(vertices[e.u].x, vertices[e.u].y), (vertices[e.v].x, vertices[e.v].y)])
        for e in edges
    ]
    tree = STRtree(lines)
    reach = config.opening_max_distance_ratio * max(e.thickness for e in edges)
    unattached: list[OpeningInput] = []

    for opening in openings:
        x1, y1, x2, y2 = opening.box
        rect = shapely_box(x1, y1, x2, y2)
        best: tuple[float, float, int] | None = None
        for j in tree.query(rect.buffer(reach)):
            j = int(j)
            distance = float(rect.distance(lines[j]))
            if distance > config.opening_max_distance_ratio * edges[j].thickness:
                continue
            overlap = float(rect.intersection(lines[j]).length)
            score = (overlap, -distance)
            if best is None or score > (best[0], best[1]):
                best = (overlap, -distance, j)
        if best is None:
            unattached.append(opening)
            continue

        edge, line = edges[best[2]], lines[best[2]]
        corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        along = [float(line.project(Point(c))) for c in corners]
        lo, hi = min(along), max(along)
        if hi - lo < _MIN_WIDTH_PX:
            unattached.append(opening)
            continue
        centre = line.interpolate((lo + hi) / 2.0)
        edge.openings.append(
            OpeningInterval(
                opening_type=opening.kind,
                t1=round(lo / edge.length, 4),
                t2=round(hi / edge.length, 4),
                width_px=round(hi - lo, 2),
                center_px=(round(centre.x, 2), round(centre.y, 2)),
            )
        )

    for edge, line in zip(edges, lines, strict=True):
        edge.openings = _merge_overlaps(edge, line)
    return unattached


def _merge_overlaps(edge: GraphEdge, line: LineString) -> list[OpeningInterval]:
    """Merge overlapping intervals of the same type on one edge."""
    merged: list[OpeningInterval] = []
    for item in sorted(edge.openings, key=lambda o: (o.opening_type.value, o.t1)):
        last = merged[-1] if merged else None
        if last is not None and last.opening_type is item.opening_type and item.t1 <= last.t2:
            t1, t2 = last.t1, max(last.t2, item.t2)
            centre = line.interpolate((t1 + t2) / 2.0 * edge.length)
            merged[-1] = OpeningInterval(
                item.opening_type,
                t1,
                t2,
                round((t2 - t1) * edge.length, 2),
                (round(centre.x, 2), round(centre.y, 2)),
            )
        else:
            merged.append(item)
    return sorted(merged, key=lambda o: o.t1)


def as_array(points: list[Pt]) -> np.ndarray:  # pragma: no cover - small helper for callers
    return np.asarray(points, dtype=float)
