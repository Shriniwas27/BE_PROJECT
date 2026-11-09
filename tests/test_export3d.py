from __future__ import annotations

import math

import numpy as np
import cv2

from export3d import build_scene
from perception import PerceptionService, PerceptionConfig, OcrMode
from scale import ManualAnchor, ScaleEngine
from topology import TopologyEngine


def _plan(double: bool) -> np.ndarray:
    img = np.full((1000, 1400), 255, np.uint8)
    for a, b in [((100, 100), (1100, 100)), ((1100, 100), (1100, 800)), ((1100, 800), (100, 800)),
                 ((100, 800), (100, 100)), ((600, 100), (600, 800)), ((600, 450), (1100, 450))]:
        if double:
            cv2.line(img, a, b, 0, 18); cv2.line(img, a, b, 255, 10)
        else:
            cv2.line(img, a, b, 0, 12)
    return img


def _scene(double: bool, doors=None):
    p = PerceptionService(PerceptionConfig(ocr_mode=OcrMode.OFF)).extract(_plan(double))
    g = TopologyEngine().build_topology(p.wall_mask, doors=doors or [])
    plan = ScaleEngine().build_from_perception(
        g, p, manual_anchors=[ManualAnchor((100, 100), (1100, 100), 10.0)])
    return plan, build_scene(plan)


def test_hollow_and_solid_walls_give_same_rooms() -> None:
    solid, _ = _scene(False)
    hollow, _ = _scene(True)
    assert len(solid.rooms) == len(hollow.rooms) == 3
    assert sorted(round(r.area_m2) for r in solid.rooms) == sorted(round(r.area_m2) for r in hollow.rooms)


def test_scene_is_metric_and_door_is_a_real_hole() -> None:
    plan, scene = _scene(False, doors=[(590, 600, 610, 690)])
    assert math.isclose(scene["scale"]["meters_per_pixel"], 0.01, rel_tol=1e-6)
    wall = next(w for w in plan.walls if w.openings)
    door = wall.openings[0]
    assert math.isclose(door.width_m, 0.90, abs_tol=0.03)

    pieces = [b for b in scene["walls"] if b["wall"] == wall.id]
    (x1, y1), (x2, y2) = wall.start, wall.end
    length = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length

    def span(b: dict) -> tuple[float, float]:  # position along the wall in metres
        cx, _, cz = b["center"]
        s_mid = (cx - x1) * ux + (-cz - y1) * uy
        return s_mid - b["size"][0] / 2, s_mid + b["size"][0] / 2

    for b in pieces:
        if b["kind"] == "solid":
            lo, hi = span(b)
            assert hi <= door.start_m + 0.02 or lo >= door.end_m - 0.02, "solid wall blocks the door"
    lintel = next(b for b in pieces if b["kind"] == "lintel")
    assert math.isclose(lintel["center"][1] - lintel["size"][1] / 2, 2.1, abs_tol=1e-6)
    assert math.isclose(scene["bounds"]["max"][0] - scene["bounds"]["min"][0], 10.0, abs_tol=0.3)


def test_free_wall_end_is_not_extended() -> None:
    from scale.models import ScaledWall, WallKind
    from export3d.builder import SceneConfig, _connected_ends, _wall_boxes

    def wall(i, a, b):
        return ScaledWall(i, a, b, math.dist(a, b), 0.2, 2.7, list(WallKind)[0])

    walls = [wall(0, (0, 0), (4, 0)), wall(1, (1, 0), (1, -2))]
    assert _connected_ends(walls[1], walls) == (True, False)
    box = _wall_boxes(walls[1], SceneConfig(), (True, False), 1)[0]
    assert math.isclose(box["size"][0], 2.1, abs_tol=1e-6)
