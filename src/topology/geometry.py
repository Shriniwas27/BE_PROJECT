"""Pure geometric regularization of line segments (no image dependencies)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import pairwise

import numpy as np
from scipy.spatial import cKDTree
from shapely import LineString, Point, STRtree
from shapely.ops import unary_union

Pt = tuple[float, float]
Segment = tuple[Pt, Pt]

_KEY_DECIMALS = 6
_EPS = 1e-9
_PARALLEL_SKIP_DEG = 15.0


def key(point: Pt) -> Pt:
    """Hashable identity for a point (tolerates float noise)."""
    return (round(point[0], _KEY_DECIMALS), round(point[1], _KEY_DECIMALS))


def seg_length(seg: Segment) -> float:
    return math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1])


def seg_angle(seg: Segment) -> float:
    """Orientation in [0, pi)."""
    return math.atan2(seg[1][1] - seg[0][1], seg[1][0] - seg[0][0]) % math.pi


def angular_distance(a: float, b: float) -> float:
    """Distance between two orientations (period pi)."""
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


# -- orientation estimation ----------------------------------------------------


def estimate_dominant_angle(segments: list[Segment], min_length: float) -> tuple[float, float]:
    """Dominant wall orientation modulo 90 degrees.

    Returns (angle_rad in (-pi/4, pi/4], concentration in [0, 1]). Concentration
    is 1 when every (long) segment is parallel or perpendicular to that angle.
    """
    total = 0.0
    cx = cy = 0.0
    for seg in segments:
        length = seg_length(seg)
        if length < min_length:
            continue
        phase = 4.0 * seg_angle(seg)
        cx += length * math.cos(phase)
        cy += length * math.sin(phase)
        total += length
    if total == 0.0:
        return 0.0, 0.0
    return math.atan2(cy, cx) / 4.0, math.hypot(cx, cy) / total


def rotate_segments(segments: list[Segment], angle: float, centre: Pt) -> list[Segment]:
    """Rotate by ``angle`` (counter-clockwise in image-y-down coordinates) about ``centre``."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def rot(p: Pt) -> Pt:
        dx, dy = p[0] - centre[0], p[1] - centre[1]
        return (centre[0] + dx * cos_a - dy * sin_a, centre[1] + dx * sin_a + dy * cos_a)

    return [(rot(a), rot(b)) for a, b in segments]


# -- direction snapping --------------------------------------------------------


def manhattan_snap(seg: Segment, snap_angles: tuple[float, ...], tol: float) -> Segment:
    """Rotate ``seg`` about its midpoint onto the nearest snap angle, if within ``tol``."""
    (x1, y1), (x2, y2) = seg
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return seg
    angle = math.atan2(dy, dx) % math.pi
    best, best_diff = None, tol
    for target in snap_angles:
        target = target % math.pi
        diff = angular_distance(angle, target)
        if diff <= best_diff:
            best, best_diff = target, diff
    if best is None:
        return seg
    ux, uy = math.cos(best), math.sin(best)
    if abs(ux) < 1e-12:
        ux = 0.0
    if abs(uy) < 1e-12:
        uy = 0.0
    if ux * dx + uy * dy < 0:
        ux, uy = -ux, -uy
    mx, my, half = (x1 + x2) / 2.0, (y1 + y2) / 2.0, length / 2.0
    return ((mx - ux * half, my - uy * half), (mx + ux * half, my + uy * half))


def contract_short_diagonals(
    segments: list[Segment], max_length: float, snap_angles: tuple[float, ...], tol: float
) -> list[Segment]:
    """Collapse short segments that match no snap angle into a single point.

    Skeletonizing a thick wall corner leaves a small diagonal "chamfer" piece.
    It carries no information, and left alone it would dangle once the
    neighbouring walls are aligned. Requires segments that still share exact
    endpoints (i.e. run before any snapping).
    """
    segs = list(segments)

    def is_diagonal(seg: Segment) -> bool:
        angle = seg_angle(seg)
        return all(angular_distance(angle, a % math.pi) > tol for a in snap_angles)

    while True:
        candidates = [
            i for i, s in enumerate(segs) if seg_length(s) < max_length and is_diagonal(s)
        ]
        if not candidates:
            return segs
        target = min(candidates, key=lambda i: seg_length(segs[i]))
        a, b = segs[target]
        merged = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        collapsed = {key(a), key(b)}
        rebuilt: list[Segment] = []
        for i, (p, q) in enumerate(segs):
            if i == target:
                continue
            p2 = merged if key(p) in collapsed else p
            q2 = merged if key(q) in collapsed else q
            if key(p2) != key(q2):
                rebuilt.append((p2, q2))
        segs = rebuilt


# -- endpoint clustering -------------------------------------------------------


def cluster_endpoints(segments: list[Segment], tol: float, min_len: float) -> list[Segment]:
    """Merge endpoints within ``tol`` (single linkage) into their centroid."""
    if not segments:
        return []
    points = np.array([p for seg in segments for p in seg], dtype=float)
    parent = list(range(len(points)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(points).query_pairs(tol):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(points)):
        members[find(index)].append(index)
    centroid = {root: points[idx].mean(axis=0) for root, idx in members.items()}

    out: list[Segment] = []
    for n in range(len(segments)):
        a = centroid[find(2 * n)]
        b = centroid[find(2 * n + 1)]
        seg = ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
        if seg_length(seg) >= min_len:
            out.append(seg)
    return out


# -- orthogonal regularization --------------------------------------------------


def _align_1d(values: list[float], weights: list[float], tol: float) -> list[float]:
    """Single-linkage 1-D clustering; returns each value's length-weighted cluster mean."""
    order = np.argsort(values)
    result = [0.0] * len(values)
    group: list[int] = []

    def flush() -> None:
        if not group:
            return
        w = np.array([weights[i] for i in group])
        v = np.array([values[i] for i in group])
        centre = float(np.average(v, weights=w if w.sum() > 0 else None))
        for i in group:
            result[i] = centre

    for idx in order:
        if group and values[idx] - values[group[-1]] > tol:
            flush()
            group = []
        group.append(int(idx))
    flush()
    return result


def ortho_regularize(
    segments: list[Segment], axis_tol: float, line_tol: float, snap_tol: float
) -> list[Segment]:
    """Make near-horizontal/vertical walls exactly axis-aligned and mutually consistent.

    1. Parallel walls within ``line_tol`` are moved onto one shared line.
    2. Wall ends within ``snap_tol`` of a perpendicular wall line are moved onto
       it, so corners and T-junctions share exact coordinates.
    Other orientations are left untouched.
    """
    h_idx: list[int] = []
    v_idx: list[int] = []
    for i, seg in enumerate(segments):
        angle = seg_angle(seg)
        if angular_distance(angle, 0.0) <= axis_tol:
            h_idx.append(i)
        elif angular_distance(angle, math.pi / 2) <= axis_tol:
            v_idx.append(i)

    out = [[list(seg[0]), list(seg[1])] for seg in segments]
    lengths = [seg_length(s) for s in segments]

    h_y = _align_1d(
        [(segments[i][0][1] + segments[i][1][1]) / 2 for i in h_idx],
        [lengths[i] for i in h_idx],
        line_tol,
    )
    v_x = _align_1d(
        [(segments[i][0][0] + segments[i][1][0]) / 2 for i in v_idx],
        [lengths[i] for i in v_idx],
        line_tol,
    )
    for i, y in zip(h_idx, h_y, strict=True):
        out[i][0][1] = out[i][1][1] = y
    for i, x in zip(v_idx, v_x, strict=True):
        out[i][0][0] = out[i][1][0] = x

    # Snapshot the aligned lines before moving any endpoints.
    v_lines = np.array(
        [
            [out[i][0][0], min(out[i][0][1], out[i][1][1]), max(out[i][0][1], out[i][1][1])]
            for i in v_idx
        ]
    ).reshape(-1, 3)
    h_lines = np.array(
        [
            [out[i][0][1], min(out[i][0][0], out[i][1][0]), max(out[i][0][0], out[i][1][0])]
            for i in h_idx
        ]
    ).reshape(-1, 3)

    for i in h_idx:
        for end in (0, 1):
            x, y = out[i][end]
            out[i][end][0] = _snap_to_line(x, y, v_lines, snap_tol)
    for i in v_idx:
        for end in (0, 1):
            x, y = out[i][end]
            out[i][end][1] = _snap_to_line(y, x, h_lines, snap_tol)

    return [((a[0], a[1]), (b[0], b[1])) for a, b in out]


def _snap_to_line(coord: float, other: float, lines: np.ndarray, tol: float) -> float:
    """Snap ``coord`` to the nearest perpendicular line whose extent covers ``other``."""
    if lines.size == 0:
        return coord
    position, low, high = lines[:, 0], lines[:, 1], lines[:, 2]
    distance = np.abs(position - coord)
    valid = (distance <= tol) & (low - tol <= other) & (other <= high + tol)
    if not valid.any():
        return coord
    return float(position[int(np.argmin(np.where(valid, distance, np.inf)))])


# -- T-junction snapping -------------------------------------------------------


def snap_dangling_to_segments(segments: list[Segment], tol: float) -> list[Segment]:
    """Attach dangling ends to a nearby wall, splitting that wall at the contact point.

    Splitting inserts an explicit shared vertex, so the later noding step never
    depends on floating-point near-misses.
    """
    if not segments:
        return []
    degree = Counter(key(p) for seg in segments for p in seg)
    tree = STRtree([LineString(seg) for seg in segments])
    parallel_skip = math.radians(_PARALLEL_SKIP_DEG)

    new_end: dict[tuple[int, int], Pt] = {}
    splits: dict[int, list[tuple[float, Pt]]] = defaultdict(list)

    for i, seg in enumerate(segments):
        for end in (0, 1):
            point = seg[end]
            if degree[key(point)] > 1:
                continue
            best: tuple[float, int, float, Pt] | None = None
            for j in tree.query(Point(point).buffer(tol)):
                j = int(j)
                if (
                    j == i
                    or angular_distance(seg_angle(seg), seg_angle(segments[j])) < parallel_skip
                ):
                    continue
                t, dist, proj = _project(point, segments[j])
                if dist <= tol and (best is None or dist < best[0]):
                    best = (dist, j, t, proj)
            if best is None:
                continue
            _, j, t, proj = best
            a, b = segments[j]
            length = seg_length(segments[j])
            if t * length < tol:
                target = a
            elif (1 - t) * length < tol:
                target = b
            else:
                target = _reuse_split(splits[j], t, proj, length, tol)
            new_end[(i, end)] = target

    out: list[Segment] = []
    for j, seg in enumerate(segments):
        start = new_end.get((j, 0), seg[0])
        stop = new_end.get((j, 1), seg[1])
        chain = [start] + [p for _, p in sorted(splits[j])] + [stop]
        for p, q in pairwise(chain):
            if key(p) != key(q):
                out.append((p, q))
    return out


def _reuse_split(
    existing: list[tuple[float, Pt]], t: float, proj: Pt, length: float, tol: float
) -> Pt:
    for t_other, p_other in existing:
        if abs(t_other - t) * length < tol * 0.5:
            return p_other
    existing.append((t, proj))
    return proj


def _project(point: Pt, seg: Segment) -> tuple[float, float, Pt]:
    """Project ``point`` onto ``seg``; returns (t in [0,1], distance, projected point)."""
    (ax, ay), (bx, by) = seg
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    t = (
        0.0
        if denom == 0
        else max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / denom))
    )
    px, py = ax + t * dx, ay + t * dy
    return t, math.hypot(point[0] - px, point[1] - py), (px, py)


# -- noding and cleanup ---------------------------------------------------------


def dedupe(segments: list[Segment]) -> list[Segment]:
    seen: set[tuple[Pt, Pt]] = set()
    out: list[Segment] = []
    for a, b in segments:
        ka, kb = key(a), key(b)
        if ka == kb:
            continue
        ident = (ka, kb) if ka <= kb else (kb, ka)
        if ident not in seen:
            seen.add(ident)
            out.append((a, b))
    return out


def planarize(segments: list[Segment], decimals: int = 3) -> list[Segment]:
    """Node all crossings/overlaps and return two-point pieces with rounded coordinates."""
    if not segments:
        return []
    merged = unary_union([LineString(seg) for seg in segments])
    pieces: list[Segment] = []
    for geom in getattr(merged, "geoms", [merged]):
        if geom.geom_type != "LineString":
            continue
        coords = [(round(x, decimals), round(y, decimals)) for x, y in geom.coords]
        pieces.extend((p, q) for p, q in pairwise(coords) if p != q)
    return sorted(dedupe(pieces), key=lambda s: (min(s[0], s[1]), max(s[0], s[1])))


def merge_collinear(segments: list[Segment], angle_tol: float) -> list[Segment]:
    """Join edges meeting at a degree-2 vertex when they are (almost) collinear."""
    segs: dict[int, Segment] = dict(enumerate(segments))
    next_id = len(segs)
    changed = True
    while changed:
        changed = False
        incident: dict[Pt, list[int]] = defaultdict(list)
        for i, (a, b) in segs.items():
            incident[key(a)].append(i)
            incident[key(b)].append(i)
        for node, ids in incident.items():
            if len(ids) != 2 or ids[0] == ids[1] or ids[0] not in segs or ids[1] not in segs:
                continue
            first, second = segs[ids[0]], segs[ids[1]]
            far_a = first[1] if key(first[0]) == node else first[0]
            far_b = second[1] if key(second[0]) == node else second[0]
            if key(far_a) == key(far_b):
                continue
            if angular_distance(seg_angle((far_a, node)), seg_angle((node, far_b))) > angle_tol:
                continue
            # also require the walk to continue in the same direction (not fold back)
            va = (node[0] - far_a[0], node[1] - far_a[1])
            vb = (far_b[0] - node[0], far_b[1] - node[1])
            if va[0] * vb[0] + va[1] * vb[1] <= 0:
                continue
            del segs[ids[0]], segs[ids[1]]
            segs[next_id] = (far_a, far_b)
            next_id += 1
            changed = True
    return list(segs.values())
