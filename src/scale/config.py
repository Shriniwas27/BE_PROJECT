"""Configuration for the scale package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleConfig:
    """Immutable, validated settings for :class:`scale.ScaleEngine`.

    Dimension text (tier 2):
        min_ocr_confidence: Observations below this are ignored.
        plausible_length_m: Measurements outside (min, max) metres are treated
            as not-a-span (door tags, ceiling heights in mm, years, ...).
        max_text_offset_ratio: How far (x longest plan side) a dimension text
            may sit from the wall it measures.
        angle_tolerance_deg: Max angle between the text baseline and a wall.
        span_center_tolerance_ratio: A dimension text must sit near the middle
            of the span it measures; allowed offset as a ratio of span length
            (never less than one wall thickness).
        max_hypotheses_per_text: Best span hypotheses kept per text.
        weight_centerline / weight_interior / weight_exterior: Prior that a
            dimension measures wall centrelines, clear interior faces, or
            outside faces. Each span is tried under all three; consensus then
            prefers a single convention shared by all texts.
        convention_coherence_bonus: Score multiplier for a consensus in which
            every text follows the same convention (drawings normally do).
        room_size_weight: Prior for ``A x B`` labels matched to a room.
        room_min_rectangularity: Rooms less rectangular than this (area /
            minimum bounding rectangle) are not used for ``A x B`` labels.
        ratio_notation_weight: Prior for a ``1:N`` drawing scale.
        agreement_tolerance: Relative tolerance (e.g. 0.03 = 3 %) within which
            two scale estimates count as agreeing.
        min_support: Independent texts that must agree for a *confident*
            dimension-based scale. Weaker evidence must pass the door-prior
            cross-check.
        max_candidates: Hard cap on candidates entering the consensus search.

    Cross-checks and priors (tier 3):
        door_width_m: Typical clear width of a single interior door.
        door_prior_uncertainty: Heuristic 1-sigma relative error of that prior.
        door_disagreement_factor: Text and door scales differing by more than
            this factor disagree.
        plan_extent_m: Plausible (min, max) length of the plan's longest side.
        plausible_wall_thickness_m / plausible_door_width_m: Soft sanity bands
            that only produce warnings.
        anchor_pick_error_px: Assumed per-point click error of manual anchors,
            used for their reported uncertainty.
        anchor_disagreement_tolerance: Manual anchors disagreeing by more than
            this fraction produce a warning.

    Elevation defaults (the plan has no Z data):
        ceiling_height_m: Clear wall height.
        door_sill_m / door_head_m, window_sill_m / window_head_m,
        opening_sill_m / opening_head_m: Heights of each opening type.
        min_wall_thickness_m / max_wall_thickness_m: Measured wall thickness is
            clamped into this band (double-line walls under-measure).

    Output frame:
        flip_y: Output Y points up (image Y points down).
        translate_to_origin: Move the plan's minimum corner to (0, 0).
    """

    min_ocr_confidence: float = 0.4
    plausible_length_m: tuple[float, float] = (0.3, 60.0)
    max_text_offset_ratio: float = 0.12
    angle_tolerance_deg: float = 12.0
    span_center_tolerance_ratio: float = 0.2
    max_hypotheses_per_text: int = 12
    weight_centerline: float = 1.0
    weight_interior: float = 1.0
    weight_exterior: float = 0.9
    room_size_weight: float = 0.7
    room_min_rectangularity: float = 0.9
    ratio_notation_weight: float = 0.6
    agreement_tolerance: float = 0.03
    min_support: int = 2
    max_candidates: int = 1500
    convention_coherence_bonus: float = 1.15

    door_width_m: float = 0.85
    door_prior_uncertainty: float = 0.12
    door_disagreement_factor: float = 1.6
    plan_extent_m: tuple[float, float] = (1.5, 300.0)
    plausible_wall_thickness_m: tuple[float, float] = (0.04, 0.8)
    plausible_door_width_m: tuple[float, float] = (0.55, 1.6)
    anchor_pick_error_px: float = 1.0
    anchor_disagreement_tolerance: float = 0.02

    ceiling_height_m: float = 2.70
    door_sill_m: float = 0.0
    door_head_m: float = 2.10
    window_sill_m: float = 0.90
    window_head_m: float = 2.10
    opening_sill_m: float = 0.0
    opening_head_m: float = 2.10
    min_wall_thickness_m: float = 0.10
    max_wall_thickness_m: float = 0.35

    flip_y: bool = True
    translate_to_origin: bool = True

    def __post_init__(self) -> None:
        _check_range("plausible_length_m", self.plausible_length_m)
        _check_range("plan_extent_m", self.plan_extent_m)
        _check_range("plausible_wall_thickness_m", self.plausible_wall_thickness_m)
        _check_range("plausible_door_width_m", self.plausible_door_width_m)

        unit_interval = (
            "min_ocr_confidence",
            "room_min_rectangularity",
            "weight_centerline",
            "weight_interior",
            "weight_exterior",
            "room_size_weight",
            "ratio_notation_weight",
        )
        for name in unit_interval:
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")

        positive = (
            "max_text_offset_ratio",
            "angle_tolerance_deg",
            "span_center_tolerance_ratio",
            "agreement_tolerance",
            "door_width_m",
            "door_prior_uncertainty",
            "anchor_pick_error_px",
            "anchor_disagreement_tolerance",
            "ceiling_height_m",
            "min_wall_thickness_m",
            "max_wall_thickness_m",
            "door_head_m",
            "window_head_m",
            "opening_head_m",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.agreement_tolerance >= 0.5:
            raise ValueError("agreement_tolerance must be < 0.5")
        if self.door_disagreement_factor <= 1.0:
            raise ValueError("door_disagreement_factor must be > 1")
        if self.convention_coherence_bonus < 1.0:
            raise ValueError("convention_coherence_bonus must be >= 1")
        if self.min_support < 1:
            raise ValueError("min_support must be >= 1")
        if self.max_hypotheses_per_text < 1 or self.max_candidates < 1:
            raise ValueError("max_hypotheses_per_text and max_candidates must be >= 1")
        if self.min_wall_thickness_m > self.max_wall_thickness_m:
            raise ValueError("min_wall_thickness_m must be <= max_wall_thickness_m")
        for sill, head, label in (
            (self.door_sill_m, self.door_head_m, "door"),
            (self.window_sill_m, self.window_head_m, "window"),
            (self.opening_sill_m, self.opening_head_m, "opening"),
        ):
            if sill < 0 or sill >= head:
                raise ValueError(f"{label}: need 0 <= sill < head")
            if head > self.ceiling_height_m:
                raise ValueError(f"{label}_head_m must not exceed ceiling_height_m")


def _check_range(name: str, bounds: tuple[float, float]) -> None:
    low, high = bounds
    if not 0.0 < low < high:
        raise ValueError(f"{name} must satisfy 0 < min < max")
