from __future__ import annotations

import json
import math
from typing import Any

import cv2
import numpy as np
import pytest

from perception import DetectedAnnotation, PerceptionResult
from scale import (
    ConfidenceLevel,
    InvalidScaleInputError,
    ManualAnchor,
    OpeningKind,
    ScaleConfig,
    ScaleEngine,
    ScaleResolutionError,
    ScaleSource,
    TextKind,
    TextObservation,
    WallKind,
    observations_from_perception,
    parse_text,
)
from scale.consensus import find_best_consensus, find_consensus, weighted_median
from scale.models import ScaleEvidence
from scale.units import drawing_units_to_meters
from topology import PlanarGraph, TopologyEngine

T = 12
TRUE_SCALE = 0.0125  # metres per pixel for the dimension-text fixtures (800 px == 10 m)


# -- fixtures ------------------------------------------------------------------


def house_mask() -> np.ndarray:
    """Centrelines x 100..900, y 100..600; interior wall at x=500."""
    mask = np.zeros((700, 1000), np.uint8)
    cv2.rectangle(mask, (100, 100), (900, 600), 255, T)
    cv2.line(mask, (500, 100), (500, 600), 255, T)
    return mask


def house() -> PlanarGraph:
    return TopologyEngine().build_topology(house_mask())


def house_with_doors() -> PlanarGraph:
    """Three 85 px doors; with 0.85 m doors the scale is exactly 0.01 m/px."""
    mask = house_mask()
    boxes = [(200, 90, 285, 112), (600, 90, 685, 112), (494, 300, 507, 385)]
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 0
    return TopologyEngine().build_topology(mask, doors=boxes, windows=[(700, 590, 800, 612)])


def text(value: str, cx: float, cy: float, vertical: bool = False, conf: float = 0.9) -> Any:
    w, h = (16, 60) if vertical else (60, 16)
    box = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
    return TextObservation(value, box, conf)


def centerline_texts() -> list[TextObservation]:
    return [
        text("10000", 500, 70),
        text("5000", 300, 75),
        text("5000", 700, 75),
        text("6250", 60, 350, vertical=True),
    ]


def distractors() -> list[TextObservation]:
    return [
        text("D1", 300, 300),
        text("2400", 420, 420),
        text("BEDROOM", 300, 350),
        text("12.5 m2", 700, 350),
    ]


def rel_error(value: float, truth: float) -> float:
    return abs(value / truth - 1.0)


# -- dimension parsing ---------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "meters"),
    [
        ("3.60m", 3.6),
        ("3,60 m", 3.6),
        ("3.6 METERS", 3.6),
        ("3600mm", 3.6),
        ("360cm", 3.6),
        ("12 ft", 3.6576),
        ("12'", 3.6576),
        ("12'-6\"", 3.81),
        ("12' 6 1/2\"", 3.82296),
        ("12\u2019-6\u201d", 3.81),  # typographic quotes
        ("3.6O m", 3.6),  # OCR: letter O for zero
    ],
)
def test_explicit_units_parse_to_meters(raw: str, meters: float) -> None:
    parsed = parse_text(raw)
    assert parsed.kind is TextKind.LINEAR
    (reading,) = parsed.lengths[0]
    assert reading.meters == pytest.approx(meters, rel=1e-4)
    assert reading.weight == 1.0


def test_bare_numbers_return_weighted_alternatives() -> None:
    mm = parse_text("3600").lengths[0]
    assert mm[0].unit == "mm" and mm[0].meters == pytest.approx(3.6)
    ambiguous = {r.unit: r.meters for r in parse_text("360").lengths[0]}
    assert ambiguous["cm"] == pytest.approx(3.6)
    assert ambiguous["mm"] == pytest.approx(0.36)
    assert parse_text("3.5").lengths[0][0].unit == "m"
    assert parse_text("3 600").lengths[0][0].meters == pytest.approx(3.6)


@pytest.mark.parametrize("raw", ["12.5 m\u00b2", "12.5 m2", "120 sq ft", "BEDROOM", "D1", "", "  "])
def test_areas_and_labels_are_not_lengths(raw: str) -> None:
    assert parse_text(raw).kind in (TextKind.AREA, TextKind.OTHER)
    assert parse_text(raw).lengths == ()


def test_scale_ratio_and_room_size() -> None:
    assert parse_text("1:100").ratio == 100.0
    assert parse_text("SCALE 1:50").ratio == 50.0
    assert parse_text("1:0").kind is TextKind.OTHER

    size = parse_text("3.5 x 4.0 m")
    assert size.kind is TextKind.ROOM_SIZE
    assert [s[0].meters for s in size.lengths] == [3.5, 4.0]
    assert parse_text("3500 \u00d7 4200").kind is TextKind.ROOM_SIZE
    feet = parse_text("12'-0\" x 14'-6\"")
    assert feet.lengths[1][0].meters == pytest.approx(4.4196, rel=1e-4)


def test_implausible_values_are_rejected_by_window() -> None:
    assert parse_text("0.1").kind is TextKind.OTHER
    assert parse_text("12'-16\"").kind is TextKind.OTHER  # 16 inches is not a remainder
    custom = parse_text("0.1", plausible_range_m=(0.05, 60.0))
    assert custom.kind is TextKind.LINEAR


def test_drawing_units() -> None:
    assert drawing_units_to_meters("Millimeters") == 0.001
    assert drawing_units_to_meters("Feet") == pytest.approx(0.3048)
    assert drawing_units_to_meters(None) is None
    assert drawing_units_to_meters("Parsecs") is None


# -- consensus -----------------------------------------------------------------


def evidence(scale: float, weight: float, group: int, convention: str = "") -> ScaleEvidence:
    return ScaleEvidence(ScaleSource.DIMENSION_TEXT, scale, weight, group, convention=convention)


def test_weighted_median() -> None:
    assert weighted_median([1.0, 2.0, 10.0], [1.0, 1.0, 1.0]) == 2.0
    assert weighted_median([1.0, 10.0], [1.0, 5.0]) == 10.0
    with pytest.raises(ValueError):
        weighted_median([], [])


def test_consensus_outvotes_outliers() -> None:
    items = [
        evidence(0.0100, 0.6, 0),
        evidence(0.0101, 0.6, 1),
        evidence(0.0099, 0.6, 2),
        evidence(0.0500, 1.0, 3),  # one strong but isolated reading
    ]
    result = find_consensus(items, 0.03)
    assert result is not None
    assert result.support == 3
    assert result.meters_per_pixel == pytest.approx(0.0100, rel=0.01)
    assert 3 not in result.members


def test_each_text_votes_once() -> None:
    many_from_one = [evidence(0.0100 + i * 1e-6, 1.0, 0) for i in range(10)]
    two_texts = [evidence(0.0200, 0.6, 1), evidence(0.0201, 0.6, 2)]
    result = find_consensus(many_from_one + two_texts, 0.03)
    assert result is not None and result.support == 2
    assert result.meters_per_pixel == pytest.approx(0.0200, rel=0.01)


def test_tight_agreement_and_convention_coherence_win() -> None:
    # Interior hypotheses agree loosely; centreline hypotheses agree exactly.
    items = [
        evidence(0.01040, 1.0, 0, "interior"),
        evidence(0.01000, 1.0, 1, "interior"),
        evidence(0.01020, 1.0, 2, "interior"),
        evidence(0.01000, 1.0, 0, "centerline"),
        evidence(0.01000, 1.0, 1, "centerline"),
        evidence(0.01000, 1.0, 2, "centerline"),
    ]
    result = find_best_consensus(items, 0.05, coherence_bonus=1.15)
    assert result is not None
    assert result.convention == "centerline"
    assert result.meters_per_pixel == pytest.approx(0.0100)


def test_consensus_handles_empty_input() -> None:
    assert find_consensus([], 0.03) is None
    assert find_best_consensus([evidence(-1.0, 1.0, 0)], 0.03, 1.1) is None


# -- tier 2: dimension text ----------------------------------------------------


def test_centerline_dimensions_resolve_accurately() -> None:
    result = ScaleEngine().resolve_scale(house(), centerline_texts() + distractors())
    assert result.source is ScaleSource.DIMENSION_TEXT
    assert rel_error(result.meters_per_pixel, TRUE_SCALE) < 0.015
    assert result.support >= 3
    assert result.level is ConfidenceLevel.HIGH
    assert any(e.used for e in result.evidence)


def test_interior_dimensions_resolve_accurately() -> None:
    clear = round((400 - 13) * TRUE_SCALE * 1000)
    tall = round((500 - 13) * TRUE_SCALE * 1000)
    texts = [
        text(str(clear), 300, 75),
        text(str(clear), 700, 75),
        text(str(tall), 60, 350, vertical=True),
    ]
    result = ScaleEngine().resolve_scale(house(), texts + distractors())
    assert rel_error(result.meters_per_pixel, TRUE_SCALE) < 0.015


def test_imperial_dimensions() -> None:
    texts = [
        text("32'-10\"", 500, 70),  # 10.0 m
        text("16'-5\"", 300, 75),  # 5.0 m
        text("16'-5\"", 700, 75),
    ]
    result = ScaleEngine().resolve_scale(house(), texts)
    assert rel_error(result.meters_per_pixel, TRUE_SCALE) < 0.015


def test_room_size_label_is_used() -> None:
    texts = [text("4.85 x 6.1 m", 300, 350), text("4.85 x 6.1 m", 700, 350)]
    result = ScaleEngine().resolve_scale(house(), texts)
    assert rel_error(result.meters_per_pixel, TRUE_SCALE) < 0.02


def test_misassociated_text_does_not_derail() -> None:
    noise = [text("3000", 500, 70), text("1800", 300, 75)]  # wrong numbers near good spans
    result = ScaleEngine().resolve_scale(house(), centerline_texts() + noise)
    assert rel_error(result.meters_per_pixel, TRUE_SCALE) < 0.015


def test_low_confidence_and_area_text_ignored() -> None:
    weak = [text("5000", 300, 75, conf=0.1), text("12.5 m2", 700, 75)]
    with pytest.raises(ScaleResolutionError):
        ScaleEngine().resolve_scale(house(), weak)


def test_text_far_from_any_wall_is_ignored() -> None:
    far = [text("5000", 500, 690)]  # 90 px below the bottom wall
    config = ScaleConfig(max_text_offset_ratio=0.05)  # search radius 40 px
    with pytest.raises(ScaleResolutionError):
        ScaleEngine(config).resolve_scale(house(), far)
    assert ScaleEngine().resolve_scale(house(), far).support == 1  # inside the default radius


def test_single_text_is_low_confidence_with_warning() -> None:
    result = ScaleEngine().resolve_scale(house(), [text("10000", 500, 70)])
    assert result.support == 1
    assert result.level is ConfidenceLevel.LOW
    assert any("ManualAnchor" in w for w in result.warnings)


# -- drawing ratio -------------------------------------------------------------


def test_drawing_ratio_uses_dpi() -> None:
    dpi = 25.4 / (0.01 * 1000 / 100)  # 1:100 at 0.01 m/px
    result = ScaleEngine().resolve_scale(house(), [text("SCALE 1:100", 100, 20)], image_dpi=dpi)
    assert result.source is ScaleSource.DRAWING_RATIO
    assert result.meters_per_pixel == pytest.approx(0.01, rel=1e-6)


def test_drawing_ratio_without_dpi_is_ignored_with_warning() -> None:
    result = ScaleEngine().resolve_scale(house_with_doors(), [text("1:100", 100, 20)])
    assert result.source is ScaleSource.DOOR_PRIOR
    assert any("DPI" in w for w in result.warnings)


def test_ratio_corroborates_dimensions() -> None:
    dpi = 25.4 / (TRUE_SCALE * 1000 / 100)
    with_ratio = centerline_texts()[:2] + [text("1:100", 100, 20)]
    result = ScaleEngine().resolve_scale(house(), with_ratio, image_dpi=dpi)
    assert result.source is ScaleSource.DIMENSION_TEXT
    assert result.support >= 3
    assert rel_error(result.meters_per_pixel, TRUE_SCALE) < 0.01


# -- tier 3: door prior --------------------------------------------------------


def test_door_prior_is_last_resort() -> None:
    result = ScaleEngine().resolve_scale(house_with_doors())
    assert result.source is ScaleSource.DOOR_PRIOR
    assert result.meters_per_pixel == pytest.approx(0.01, rel=1e-3)
    assert result.level is ConfidenceLevel.MEDIUM
    assert result.relative_uncertainty >= 0.12
    assert any("+/-" in w for w in result.warnings)


def test_custom_door_width_prior() -> None:
    config = ScaleConfig(door_width_m=0.90)
    result = ScaleEngine(config).resolve_scale(house_with_doors())
    assert result.meters_per_pixel == pytest.approx(0.90 / 85, rel=1e-3)


def test_single_text_contradicting_doors_is_rejected() -> None:
    # "2000" over an 800 px wall implies 0.0025 m/px; doors would be 0.21 m wide.
    result = ScaleEngine().resolve_scale(house_with_doors(), [text("2000", 500, 70)])
    assert result.source is ScaleSource.DOOR_PRIOR
    assert any("misread" in w for w in result.warnings)


def test_strong_text_contradicting_doors_is_kept_with_warning() -> None:
    # Texts imply 0.03125 m/px, so the 85 px doors would be 2.66 m wide (3.1x standard).
    texts = [
        text("25000", 500, 70),
        text("12500", 300, 75),
        text("12500", 700, 75),
        text("15625", 60, 350, vertical=True),
    ]
    result = ScaleEngine().resolve_scale(house_with_doors(), texts)
    assert result.source is ScaleSource.DIMENSION_TEXT
    assert any("3.1" in w and "door widths" in w for w in result.warnings)
    clean = ScaleEngine().resolve_scale(house(), texts)
    assert result.confidence < clean.confidence  # penalised for the contradiction


def test_agreeing_doors_boost_a_single_text() -> None:
    # 800 px at 0.01 m/px = 8 m, consistent with the doors.
    boosted = ScaleEngine().resolve_scale(house_with_doors(), [text("8000", 500, 70)])
    alone = ScaleEngine().resolve_scale(house(), [text("8000", 500, 70)])
    assert boosted.confidence > alone.confidence
    assert boosted.source is ScaleSource.DIMENSION_TEXT


def test_implausible_plan_extent_is_rejected() -> None:
    config = ScaleConfig(plan_extent_m=(1.5, 5.0))  # a 10 m plan is "implausible"
    with pytest.raises(ScaleResolutionError):
        ScaleEngine(config).resolve_scale(house(), centerline_texts())


def test_no_evidence_raises() -> None:
    with pytest.raises(ScaleResolutionError, match="ManualAnchor"):
        ScaleEngine().resolve_scale(house())


# -- tier 1: manual ------------------------------------------------------------


def test_manual_anchor_overrides_everything() -> None:
    anchor = ManualAnchor((100, 100), (900, 100), 8.0)  # 0.01 m/px
    result = ScaleEngine().resolve_scale(house_with_doors(), centerline_texts(), [anchor])
    assert result.source is ScaleSource.MANUAL
    assert result.meters_per_pixel == pytest.approx(0.01)
    assert result.confidence == 1.0 and result.level is ConfidenceLevel.HIGH
    assert any("differs" in w for w in result.warnings)  # text says 0.0125


def test_manual_anchor_uncertainty_reflects_pixel_error() -> None:
    long_anchor = ManualAnchor((0, 0), (1000, 0), 10.0)
    short_anchor = ManualAnchor((0, 0), (50, 0), 0.5)
    engine = ScaleEngine()
    long_u = engine.resolve_scale(house(), manual_anchors=[long_anchor]).relative_uncertainty
    short_u = engine.resolve_scale(house(), manual_anchors=[short_anchor]).relative_uncertainty
    assert long_u < short_u


def test_multiple_anchors_use_median_and_flag_disagreement() -> None:
    good = [ManualAnchor((0, 0), (800, 0), 8.0), ManualAnchor((0, 0), (0, 500), 5.0)]
    ok = ScaleEngine().resolve_scale(house(), manual_anchors=good)
    assert ok.support == 2 and not any("disagree" in w for w in ok.warnings)

    bad = good + [ManualAnchor((0, 0), (800, 0), 12.0)]
    result = ScaleEngine().resolve_scale(house(), manual_anchors=bad)
    assert result.meters_per_pixel == pytest.approx(0.01)
    assert any("disagree" in w for w in result.warnings)


def test_manual_anchor_works_with_empty_graph() -> None:
    empty = PlanarGraph({}, [], [])
    plan = ScaleEngine().build(empty, manual_anchors=[ManualAnchor((0, 0), (100, 0), 1.0)])
    assert plan.walls == [] and plan.rooms == [] and plan.bounds_m == (0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "args",
    [
        ((0, 0), (0, 0), 1.0),
        ((0, 0), (1, 0), 0.0),
        ((0, 0), (1, 0), -2.0),
        ((0, 0), (1, 0), math.nan),
    ],
)
def test_invalid_manual_anchor(args: Any) -> None:
    with pytest.raises(InvalidScaleInputError):
        ManualAnchor(*args)


def test_invalid_inputs() -> None:
    engine = ScaleEngine()
    with pytest.raises(InvalidScaleInputError):
        engine.resolve_scale("not a graph")  # type: ignore[arg-type]
    with pytest.raises(InvalidScaleInputError):
        engine.resolve_scale(house(), image_dpi=0)
    with pytest.raises(InvalidScaleInputError):
        engine.resolve_scale(house(), image_dpi=math.inf)


def test_vector_units_are_exact() -> None:
    result = ScaleEngine.scale_from_drawing_units("Millimeters")
    assert result.meters_per_pixel == 0.001
    assert result.source is ScaleSource.VECTOR_UNITS and result.level is ConfidenceLevel.HIGH
    with pytest.raises(ScaleResolutionError):
        ScaleEngine.scale_from_drawing_units(None)


# -- scaled plan ---------------------------------------------------------------


def test_scaled_plan_geometry() -> None:
    graph = house_with_doors()
    anchor = ManualAnchor((100, 100), (900, 100), 8.0)
    plan = ScaleEngine().build(graph, manual_anchors=[anchor])

    min_x, min_y, max_x, max_y = plan.bounds_m
    assert (min_x, min_y) == pytest.approx((0.0, 0.0))
    assert max_x == pytest.approx(8.0, abs=0.05) and max_y == pytest.approx(5.0, abs=0.05)
    assert plan.ceiling_height_m == 2.70
    assert len(plan.walls) == len(graph.edges)
    for wall in plan.walls:
        assert wall.length_m == pytest.approx(math.dist(wall.start, wall.end), rel=1e-6)
        assert wall.height_m == 2.70
        assert 0.10 <= wall.thickness_m <= 0.35

    kinds = [w.kind for w in plan.walls]
    assert kinds.count(WallKind.INTERIOR) >= 1 and kinds.count(WallKind.EXTERIOR) >= 4
    assert len(plan.rooms) == 2
    assert sum(r.area_m2 for r in plan.rooms) == pytest.approx(8.0 * 5.0, rel=0.04)
    assert plan.rooms[0].area_sqft == pytest.approx(plan.rooms[0].area_m2 * 10.7639, rel=1e-4)
    assert all(r.interior_area_m2 < r.area_m2 for r in plan.rooms)


def test_opening_elevations_and_offsets() -> None:
    plan = ScaleEngine().build(
        house_with_doors(), manual_anchors=[ManualAnchor((100, 100), (900, 100), 8.0)]
    )
    openings = [o for w in plan.walls for o in w.openings]
    doors = [o for o in openings if o.kind is OpeningKind.DOOR]
    windows = [o for o in openings if o.kind is OpeningKind.WINDOW]
    assert len(doors) == 3 and len(windows) == 1
    for door in doors:
        assert (door.sill_height_m, door.head_height_m, door.height_m) == (0.0, 2.10, 2.10)
        assert door.width_m == pytest.approx(0.85, abs=0.03)
    (window,) = windows
    assert (window.sill_height_m, window.head_height_m) == (0.90, 2.10)
    assert window.height_m == pytest.approx(1.20)
    for wall in plan.walls:
        for o in wall.openings:
            assert 0 <= o.start_m < o.end_m <= wall.length_m + 1e-6


def test_thickness_is_clamped_and_reported() -> None:
    config = ScaleConfig(min_wall_thickness_m=0.20, max_wall_thickness_m=0.35)
    plan = ScaleEngine(config).build(
        house(), manual_anchors=[ManualAnchor((100, 100), (900, 100), 8.0)]
    )
    assert all(w.thickness_clamped and w.thickness_m == 0.20 for w in plan.walls)
    assert any("clamped" in w for w in plan.warnings)


def test_room_winding_and_y_up() -> None:
    plan = ScaleEngine().build(house(), manual_anchors=[ManualAnchor((100, 100), (900, 100), 8.0)])

    def signed_area(ring: list[tuple[float, float]]) -> float:
        return 0.5 * sum(
            x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1], strict=True)
        )

    assert all(signed_area(r.polygon) > 0 for r in plan.rooms)  # counter-clockwise
    top_wall_y = max(w.start[1] for w in plan.walls)
    assert top_wall_y == pytest.approx(5.0, abs=0.05)  # image top is plan "north" (y up)


def test_output_frame_options_and_transform_roundtrip() -> None:
    anchor = ManualAnchor((100, 100), (900, 100), 8.0)
    kept = ScaleEngine(ScaleConfig(flip_y=False, translate_to_origin=False)).build(
        house(), manual_anchors=[anchor]
    )
    assert kept.bounds_m[0] == pytest.approx(1.0, abs=0.1)  # x=100px kept at 1 m
    assert kept.bounds_m[1] == pytest.approx(1.0, abs=0.1)

    default = ScaleEngine().build(house(), manual_anchors=[anchor])
    for point in [(300.0, 200.0), (100.0, 600.0)]:
        back = default.transform.to_pixels(default.transform.to_meters(point))
        assert back == pytest.approx(point)


def test_plan_serialises_to_json() -> None:
    plan = ScaleEngine().build(house_with_doors(), centerline_texts())
    payload = json.loads(json.dumps(plan.to_dict()))
    assert payload["units"] == "meters"
    assert payload["scale"]["source"] == "dimension_text"
    assert payload["walls"][0]["kind"] in {"exterior", "interior", "unknown"}


def test_build_from_perception_adapter() -> None:
    annotations = [DetectedAnnotation(o.text, o.box, o.confidence) for o in centerline_texts()]
    perception = PerceptionResult(
        wall_mask=house_mask(),
        opening_mask=np.zeros((700, 1000), np.uint8),
        text_mask=np.zeros((700, 1000), np.uint8),
        cleaned_image=np.zeros((700, 1000), np.uint8),
        image_size=(1000, 700),
        annotations=annotations,
    )
    assert [o.text for o in observations_from_perception(perception)] == [
        "10000",
        "5000",
        "5000",
        "6250",
    ]
    plan = ScaleEngine().build_from_perception(house(), perception)
    assert rel_error(plan.scale.meters_per_pixel, TRUE_SCALE) < 0.015


# -- configuration ---------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"agreement_tolerance": 0.0},
        {"agreement_tolerance": 0.6},
        {"min_support": 0},
        {"door_width_m": -1},
        {"plausible_length_m": (5.0, 1.0)},
        {"min_wall_thickness_m": 0.5, "max_wall_thickness_m": 0.3},
        {"door_head_m": 3.0},  # above the ceiling
        {"window_sill_m": 2.5},  # sill above head
        {"min_ocr_confidence": 1.5},
        {"door_disagreement_factor": 1.0},
        {"convention_coherence_bonus": 0.5},
    ],
)
def test_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ScaleConfig(**kwargs)
