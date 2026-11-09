from __future__ import annotations

import json
import math
from typing import Any

import cv2
import numpy as np
import pytest

from perception import DetectedSymbol, OcrMode, PerceptionConfig, PerceptionService, SymbolCategory
from topology import (
    InvalidMaskError,
    OpeningType,
    PlanarGraph,
    TopologyConfig,
    TopologyEngine,
)
from topology.geometry import (
    cluster_endpoints,
    manhattan_snap,
    merge_collinear,
    ortho_regularize,
    planarize,
    snap_dangling_to_segments,
)

T = 12  # wall thickness used by the fixtures


def house(width: int = 1000, height: int = 700) -> np.ndarray:
    """Outer rectangle (centrelines x 100..900, y 100..600) plus an interior wall at x=500."""
    mask = np.zeros((height, width), np.uint8)
    cv2.rectangle(mask, (100, 100), (900, 600), 255, T)
    cv2.line(mask, (500, 100), (500, 600), 255, T)
    return mask


def engine(**kwargs: Any) -> TopologyEngine:
    return TopologyEngine(TopologyConfig(**kwargs))


def all_openings(graph: PlanarGraph) -> list[Any]:
    return [o for e in graph.edges for o in e.openings]


# -- basic topology -----------------------------------------------------------


def test_two_room_house() -> None:
    graph = engine().build_topology(house())
    assert len(graph.rooms) == 2
    assert len(graph.vertices) == 6 and len(graph.edges) == 7
    for room in graph.rooms:
        assert room.area_px == pytest.approx(400 * 500, rel=0.03)
    assert graph.warnings == [] and graph.dangling_vertices == []
    assert graph.thickness_px == pytest.approx(T, abs=2)


def test_edges_are_exactly_axis_aligned_and_connected() -> None:
    graph = engine().build_topology(house())
    for edge in graph.edges:
        a, b = graph.vertices[edge.u], graph.vertices[edge.v]
        assert a.x == b.x or a.y == b.y
        assert edge.support == pytest.approx(1.0, abs=0.05)
        assert edge.thickness == pytest.approx(T, abs=2)
    degree: dict[int, int] = {}
    for edge in graph.edges:
        degree[edge.u] = degree.get(edge.u, 0) + 1
        degree[edge.v] = degree.get(edge.v, 0) + 1
    assert sorted(degree.values()) == [2, 2, 2, 2, 3, 3]  # 4 corners + 2 T-junctions


def test_interior_area_is_smaller_by_wall_thickness() -> None:
    room = engine().build_topology(house()).rooms[0]
    assert room.interior_polygon is not None
    assert room.interior_area_px == pytest.approx((400 - T) * (500 - T), rel=0.04)
    assert engine(compute_interior=False).build_topology(house()).rooms[0].interior_polygon is None


def test_rotated_plan_is_straightened_in_its_own_frame() -> None:
    rows, cols = house().shape
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 2.5, 1.0)
    graph = engine().build_topology(cv2.warpAffine(house(), matrix, (cols, rows)))
    assert len(graph.rooms) == 2
    assert abs(abs(graph.rotation_deg) - 2.5) < 0.3
    # every edge is parallel or perpendicular to the plan frame
    offsets = []
    for edge in graph.edges:
        a, b = graph.vertices[edge.u], graph.vertices[edge.v]
        angle = math.degrees(math.atan2(b.y - a.y, b.x - a.x)) % 90.0
        offsets.append(angle)
    spread = max(min(o, 90 - o) if False else o for o in offsets) - min(offsets)
    assert spread < 0.1 or spread > 89.9


def test_deskew_can_be_disabled() -> None:
    assert engine(deskew=False).build_topology(house()).rotation_deg == 0.0


def test_noise_specks_and_short_spurs_are_ignored() -> None:
    mask = house()
    cv2.rectangle(mask, (300, 300), (306, 306), 255, -1)  # isolated speck
    cv2.line(mask, (500, 300), (512, 300), 255, T)  # tiny stub on the interior wall
    graph = engine().build_topology(mask)
    assert len(graph.rooms) == 2 and len(graph.edges) == 7


def test_wall_end_that_stops_short_still_closes_rooms() -> None:
    mask = np.zeros((700, 1000), np.uint8)
    cv2.rectangle(mask, (100, 100), (900, 600), 255, T)
    cv2.line(mask, (500, 112), (500, 600), 255, T)  # starts 12px below the top wall
    graph = engine().build_topology(mask)
    assert len(graph.rooms) == 2


def test_l_shaped_plan_has_one_room_with_correct_area() -> None:
    mask = np.zeros((700, 1000), np.uint8)
    points = np.array(
        [[100, 100], [600, 100], [600, 300], [900, 300], [900, 600], [100, 600]], np.int32
    )
    cv2.polylines(mask, [points], True, 255, T)
    graph = engine().build_topology(mask)
    assert len(graph.rooms) == 1
    assert graph.rooms[0].area_px == pytest.approx(500 * 200 + 800 * 300 - 0, rel=0.03) or (
        graph.rooms[0].area_px == pytest.approx(500 * 500 + 300 * 300, rel=0.03)
    )


def test_room_with_hole_is_reported() -> None:
    mask = house()
    cv2.rectangle(mask, (600, 300), (800, 450), 255, T)  # closed closet inside the right room
    graph = engine().build_topology(mask)
    assert len(graph.rooms) == 3
    holed = [r for r in graph.rooms if r.holes]
    assert len(holed) == 1 and holed[0].area_px < 400 * 500 - 150 * 200 + 5000


def test_double_line_wall_slivers_need_explicit_min_width() -> None:
    mask = np.zeros((700, 1000), np.uint8)
    for offset in (-12, 12):  # two thin parallel lines 24 px apart
        cv2.rectangle(mask, (100 + offset, 100 + offset), (900 - offset, 600 - offset), 255, 3)
    assert len(engine().build_topology(mask).rooms) == 2  # stroke is thin: sliver survives
    graph = engine(min_room_width_px=40).build_topology(mask)
    assert len(graph.rooms) == 1  # the sliver between the lines is discarded


def test_empty_mask_returns_empty_graph() -> None:
    graph = engine().build_topology(np.zeros((100, 100), np.uint8))
    assert graph.edges == [] and graph.rooms == [] and graph.warnings


# -- openings -----------------------------------------------------------------


def door_gap_mask() -> np.ndarray:
    mask = house()
    mask[560:640, 250:330] = 0  # 80 px door carved out of the bottom wall (y=600)
    return mask


DOOR_BOX = (250, 560, 330, 640)


def test_gap_without_bridging_leaks_rooms() -> None:
    graph = engine(bridge_openings=False).build_topology(door_gap_mask(), doors=[DOOR_BOX])
    assert len(graph.rooms) < 2  # left room opens to the outside


def test_door_gap_is_bridged_and_opening_attached() -> None:
    graph = engine().build_topology(door_gap_mask(), doors=[DOOR_BOX])
    assert len(graph.rooms) == 2
    doors = all_openings(graph)
    assert len(doors) == 1
    door = doors[0]
    assert door.opening_type is OpeningType.DOOR
    assert door.width_px == pytest.approx(80, abs=6)
    assert door.center_px[0] == pytest.approx(290, abs=6) and door.center_px[1] == pytest.approx(
        600, abs=3
    )
    host = next(e for e in graph.edges if e.openings)
    assert host.support < 0.9  # synthesised bridge: only partly backed by wall pixels
    assert 0.0 <= door.t1 < door.t2 <= 1.0


def test_swing_shaped_box_extending_into_room_still_matches_wall() -> None:
    swing_box = (250, 520, 340, 604)  # arc sector: extends 80px into the room above the wall
    mask = house()
    mask[560:640, 250:340] = 0
    graph = engine().build_topology(mask, doors=[swing_box])
    assert len(graph.rooms) == 2
    assert all_openings(graph)[0].width_px == pytest.approx(90, abs=8)


def test_window_in_continuous_wall_attaches_without_bridging() -> None:
    graph = engine().build_topology(house(), windows=[(700, 90, 780, 112)])
    windows = all_openings(graph)
    assert len(windows) == 1 and windows[0].opening_type is OpeningType.WINDOW
    assert windows[0].width_px == pytest.approx(80, abs=4)
    assert len(graph.rooms) == 2


def test_overlapping_openings_are_merged() -> None:
    graph = engine().build_topology(house(), windows=[(700, 90, 780, 112), (760, 90, 840, 112)])
    windows = all_openings(graph)
    assert len(windows) == 1 and windows[0].width_px == pytest.approx(140, abs=6)


def test_opening_far_from_any_wall_is_reported_unattached() -> None:
    graph = engine().build_topology(house(), doors=[(300, 300, 360, 360)])
    assert all_openings(graph) == []
    assert graph.unattached_openings == [(OpeningType.DOOR, (300, 300, 360, 360))]
    assert any("could not be matched" in w for w in graph.warnings)


def test_accepts_symbol_objects_and_raw_boxes() -> None:
    symbol = DetectedSymbol("door", SymbolCategory.DOOR, DOOR_BOX, 0.9)
    graph = engine().build_topology(door_gap_mask(), doors=[symbol], windows=[(700, 90, 780, 112)])
    kinds = {o.opening_type for o in all_openings(graph)}
    assert kinds == {OpeningType.DOOR, OpeningType.WINDOW}
    with pytest.raises(ValueError):
        engine().build_topology(house(), doors=["nonsense"])


# -- integration with perception ----------------------------------------------


class _FakeSymbols:
    def __init__(self, symbols: list[DetectedSymbol]) -> None:
        self.symbols = symbols

    def detect(self, image: np.ndarray, wall_mask: np.ndarray) -> list[DetectedSymbol]:
        return self.symbols


def test_end_to_end_with_perception() -> None:
    page = np.full((700, 1000), 255, np.uint8)
    page[house() > 0] = 0
    door = DetectedSymbol("door", SymbolCategory.DOOR, DOOR_BOX, 0.9)
    perception = PerceptionService(
        PerceptionConfig(ocr_mode=OcrMode.OFF), symbol_detector=_FakeSymbols([door])
    ).extract(page)
    assert perception.opening_mask[600, 290] == 255  # perception carved the door

    graph = engine().build_from_perception(perception)
    assert len(graph.rooms) == 2 and len(all_openings(graph)) == 1


# -- serialisation and validation ---------------------------------------------


def test_to_dict_is_json_serialisable() -> None:
    graph = engine().build_topology(door_gap_mask(), doors=[DOOR_BOX])
    payload = json.loads(json.dumps(graph.to_dict()))
    assert len(payload["rooms"]) == 2
    opening = next(o for e in payload["edges"] for o in e["openings"])
    assert opening["opening_type"] == "door"


@pytest.mark.parametrize(
    "bad",
    [
        "x",
        np.zeros((10, 10), np.float32),
        np.zeros((0, 5), np.uint8),
        np.zeros((5, 5, 3), np.uint8),
    ],
)
def test_invalid_masks_rejected(bad: Any) -> None:
    with pytest.raises(InvalidMaskError):
        engine().build_topology(bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"snap_angles_deg": ()},
        {"rdp_epsilon_px": 0},
        {"snap_angle_tol_deg": 50},
        {"min_room_area_ratio": -1},
    ],
)
def test_invalid_config_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TopologyConfig(**kwargs)


def test_input_mask_not_mutated() -> None:
    mask = house()
    before = mask.copy()
    engine().build_topology(mask)
    assert np.array_equal(mask, before)


# -- geometry unit tests ------------------------------------------------------


def test_manhattan_snap_and_diagonals() -> None:
    angles = (0.0, math.pi / 2)
    tol = math.radians(6)
    snapped = manhattan_snap(((0, 0), (100, 4)), angles, tol)
    assert snapped[0][1] == snapped[1][1]  # exactly horizontal
    assert math.hypot(snapped[1][0] - snapped[0][0], 0) == pytest.approx(math.hypot(100, 4))
    assert manhattan_snap(((0, 0), (100, 30)), angles, tol) == ((0, 0), (100, 30))  # too skewed
    diag = manhattan_snap(((0, 0), (100, 97)), (*angles, math.pi / 4), tol)
    assert diag[1][0] - diag[0][0] == pytest.approx(diag[1][1] - diag[0][1])


def test_cluster_endpoints_merges_nearby_and_drops_collapsed() -> None:
    segs = [
        ((0.0, 0.0), (100.0, 0.0)),
        ((103.0, 2.0), (103.0, 100.0)),
        ((50.0, 50.0), (52.0, 51.0)),
    ]
    out = cluster_endpoints(segs, tol=6.0, min_len=3.0)
    assert len(out) == 2 and out[0][1] == out[1][0]


def test_ortho_regularize_aligns_parallels_and_corners() -> None:
    segs = [((0.0, 0.0), (100.0, 1.0)), ((0.0, 1.0), (50.0, 0.0)), ((101.0, 0.0), (101.0, 80.0))]
    out = ortho_regularize(segs, axis_tol=math.radians(6), line_tol=3.0, snap_tol=6.0)
    assert out[0][0][1] == out[0][1][1] == out[1][0][1] == out[1][1][1]
    assert out[0][1] == (out[2][0][0], out[0][0][1]) or out[0][1][0] == out[2][0][0]


def test_snap_dangling_splits_target_and_attaches() -> None:
    segs = [((0.0, 0.0), (200.0, 0.0)), ((100.0, 5.0), (100.0, 100.0))]
    out = snap_dangling_to_segments(segs, tol=8.0)
    assert ((100.0, 0.0), (100.0, 100.0)) in out
    assert ((0.0, 0.0), (100.0, 0.0)) in out and ((100.0, 0.0), (200.0, 0.0)) in out


def test_planarize_nodes_crossings_and_merge_collinear_rejoins() -> None:
    cross = planarize([((0.0, 0.0), (100.0, 0.0)), ((50.0, -50.0), (50.0, 50.0))])
    assert len(cross) == 4
    chain = merge_collinear(
        [((0.0, 0.0), (50.0, 0.0)), ((50.0, 0.0), (120.0, 0.0))], math.radians(1.5)
    )
    assert chain == [((0.0, 0.0), (120.0, 0.0))]
    fold = merge_collinear(
        [((0.0, 0.0), (50.0, 0.0)), ((50.0, 0.0), (0.0, 0.0))], math.radians(1.5)
    )
    assert len(fold) == 2


def test_six_room_apartment_with_doors_at_full_resolution() -> None:
    thick = 14
    mask = np.zeros((1500, 2048), np.uint8)
    left, right, top, bottom = 150, 1900, 150, 1350
    cv2.rectangle(mask, (left, top), (right, bottom), 255, thick)
    columns = [left + (right - left) * i // 3 for i in (1, 2)]
    middle = (top + bottom) // 2
    for x in columns:
        cv2.line(mask, (x, top), (x, bottom), 255, thick)
    cv2.line(mask, (left, middle), (right, middle), 255, thick)

    boxes: list[tuple[int, int, int, int]] = []

    def door_h(x: int, y: int, w: int = 110) -> None:
        mask[y - thick : y + thick, x : x + w] = 0
        boxes.append((x, y - 40, x + w, y + 40))

    def door_v(x: int, y: int, w: int = 110) -> None:
        mask[y : y + w, x - thick : x + thick] = 0
        boxes.append((x - 40, y, x + 40, y + w))

    for x in (300, 1000, 1500):
        door_h(x, middle)
    door_h(400, bottom)
    door_v(columns[0], 300)
    door_v(columns[0], 900)
    door_v(columns[1], 400)
    door_v(columns[1], 1000)

    graph = engine().build_topology(mask, doors=boxes)
    assert len(graph.rooms) == 6 and graph.warnings == []
    assert all(r.area_px == pytest.approx(583 * 600, rel=0.03) for r in graph.rooms)
    widths = [o.width_px for e in graph.edges for o in e.openings]
    assert len(widths) == len(boxes) and all(w == pytest.approx(110, abs=4) for w in widths)


def test_door_where_a_partition_meets_the_wall_is_reported_not_silently_wrong() -> None:
    mask = house()
    mask[560:640, 480:540] = 0  # doorway exactly where the interior wall meets the bottom wall
    graph = engine().build_topology(mask, doors=[(470, 560, 550, 640)])
    assert graph.warnings  # dangling end and/or unclosed rooms are surfaced
