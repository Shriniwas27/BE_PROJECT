"""Build a JSON scene (metres, Three.js axes: X right, Y up, Z = -plan Y) from a ScaledPlan.

Walls are emitted as axis-aligned-in-local-space boxes, split around openings so doors and
windows are real holes (no CSG needed in the browser): solid runs between openings, a lintel
above every opening and a sill block below windows. Wall ends are extended by half the wall
thickness so corners and T-junctions close without gaps. Windows also get a glass pane.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scale import OpeningKind, ScaledPlan, ScaledWall

_EPS = 1e-4


@dataclass(frozen=True)
class SceneConfig:
    extend_ends: bool = True  # extend wall ends by thickness/2 to close corners
    floor_thickness_m: float = 0.1
    glass_thickness_m: float = 0.02
    min_piece_m: float = 0.01


def _point_segment_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    denom = dx * dx + dy * dy
    t = 0.0 if denom < _EPS else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / denom))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def _connected_ends(wall: ScaledWall, walls: list[ScaledWall]) -> tuple[bool, bool]:
    """Whether each end of ``wall`` touches another wall (corner / T-junction)."""
    flags = []
    for pt in (wall.start, wall.end):
        reach = max(wall.thickness_m, 0.05)
        flags.append(
            any(
                other.id != wall.id
                and _point_segment_distance(pt, other.start, other.end)
                <= reach / 2.0 + other.thickness_m / 2.0 + 1e-3
                for other in walls
            )
        )
    return flags[0], flags[1]


def build_scene(plan: ScaledPlan, config: SceneConfig | None = None) -> dict[str, Any]:
    cfg = config or SceneConfig()
    boxes: list[dict[str, Any]] = []
    for index, wall in enumerate(plan.walls):
        boxes.extend(_wall_boxes(wall, cfg, _connected_ends(wall, plan.walls), index))
    floors = [
        {
            "id": room.id,
            "polygon": [[x, -y] for x, y in (room.interior_polygon or room.polygon)],
            "holes": [[[x, -y] for x, y in h] for h in room.holes],
            "area_m2": round(room.interior_area_m2 or room.area_m2, 3),
            "centroid": [room.centroid[0], -room.centroid[1]],
        }
        for room in plan.rooms
    ]
    min_x, min_y, max_x, max_y = plan.bounds_m
    return {
        "version": 1,
        "units": "meters",
        "axes": "three (Y up, plan Y -> -Z)",
        "ceiling_height_m": plan.ceiling_height_m,
        "floor_thickness_m": cfg.floor_thickness_m,
        "bounds": {"min": [min_x, -max_y], "max": [max_x, -min_y]},
        "scale": {
            "meters_per_pixel": plan.scale.meters_per_pixel,
            "source": plan.scale.source.value,
            "level": plan.scale.level.value,
        },
        "walls": boxes,
        "floors": floors,
        "warnings": list(plan.warnings) + list(plan.scale.warnings),
    }


def write_scene(plan: ScaledPlan, path: str | Path, config: SceneConfig | None = None) -> Path:
    out = Path(path)
    out.write_text(json.dumps(build_scene(plan, config), indent=2), encoding="utf-8")
    return out


def _wall_boxes(
    wall: ScaledWall,
    cfg: SceneConfig,
    connected: tuple[bool, bool] = (True, True),
    index: int = 0,
) -> list[dict[str, Any]]:
    (x1, y1), (x2, y2) = wall.start, wall.end
    length = math.hypot(x2 - x1, y2 - y1)
    if length < _EPS:
        return []
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    half = wall.thickness_m / 2.0 if cfg.extend_ends else 0.0
    ext0, ext1 = (half if connected[0] else 0.0), (half if connected[1] else 0.0)
    ext = ext0  # local s runs from -ext0 .. length + ext1
    # Overlapping boxes at corners share coplanar tops; a sub-millimetre height stagger per wall
    # stops the GPU z-fighting (flicker) without any visible change.
    height = wall.height_m + (index % 7) * 0.0004
    ox, oy = x1 - ux * ext0, y1 - uy * ext0
    total = length + ext0 + ext1
    rot_y = math.atan2(uy, ux)  # plan angle; three Z = -plan Y so rotation about Y is +angle

    spans = sorted(
        (
            (max(0.0, o.start_m) + ext, min(length, o.end_m) + ext, o)
            for o in wall.openings
            if o.end_m - o.start_m > cfg.min_piece_m
        ),
        key=lambda t: t[0],
    )

    def piece(s0: float, s1: float, z0: float, z1: float, kind: str, thick: float) -> dict[str, Any]:
        mid = (s0 + s1) / 2.0
        return {
            "wall": wall.id,
            "kind": kind,
            "center": [ox + ux * mid, (z0 + z1) / 2.0, -(oy + uy * mid)],
            "size": [s1 - s0, z1 - z0, thick],
            "rotation_y": rot_y,
            "wall_kind": wall.kind.value,
        }

    out: list[dict[str, Any]] = []
    cursor = 0.0
    for s0, s1, op in spans:
        if s0 < cursor:  # overlapping detections: merge into the previous opening
            s0 = cursor
        if s1 <= s0 + cfg.min_piece_m:
            continue
        if s0 - cursor > cfg.min_piece_m:
            out.append(piece(cursor, s0, 0.0, height, "solid", wall.thickness_m))
        if op.head_height_m < height - _EPS:
            out.append(piece(s0, s1, op.head_height_m, height, "lintel", wall.thickness_m))
        if op.sill_height_m > _EPS:
            out.append(piece(s0, s1, 0.0, op.sill_height_m, "sill", wall.thickness_m))
        if op.kind is OpeningKind.WINDOW:
            out.append(
                piece(s0, s1, op.sill_height_m, op.head_height_m, "glass", cfg.glass_thickness_m)
            )
        cursor = s1
    if total - cursor > cfg.min_piece_m:
        out.append(piece(cursor, total, 0.0, height, "solid", wall.thickness_m))
    return out
