"""Scale arbitration.

Strict priority between tiers, but each tier is *validated* by the others:

1. **Manual anchor** - ground truth. Always wins; automated evidence is only
   used to warn about contradictions.
2. **Text evidence** - OCR dimensions matched to wall spans and room sizes,
   plus ``1:N`` drawing notation when the page DPI is known. Pooled and reduced
   by consensus, so misread numbers and mismatched spans lose the vote.
3. **Architectural prior** - median detected door width vs. a standard door.

Weak text evidence (fewer than ``min_support`` agreeing texts) is rejected when
it contradicts the door prior, and any automated result whose implied plan size
is absurd is rejected, falling through to the next tier.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace

from topology import OpeningType, PlanarGraph

from .association import collect_candidates
from .config import ScaleConfig
from .consensus import Consensus, find_best_consensus
from .dimensions import ParsedText, TextKind, parse_text
from .errors import InvalidScaleInputError, ScaleResolutionError
from .models import (
    ConfidenceLevel,
    ManualAnchor,
    ScaleEvidence,
    ScaleResult,
    ScaleSource,
    TextObservation,
)

logger = logging.getLogger(__name__)

_INCH_METERS = 0.0254
_CONFIDENCE_BY_SUPPORT = {1: 0.40, 2: 0.70}
_CONFIDENCE_THREE_OR_MORE = 0.85
_RATIO_ONLY_CONFIDENCE = 0.55
_SPREAD_PENALTY_PER_UNIT = 5.0
_MAX_SPREAD_PENALTY = 0.5
_RATIO_CORROBORATION_BONUS = 0.08
_DOOR_AGREEMENT_BONUS_WEAK = 0.10
_DOOR_AGREEMENT_BONUS_STRONG = 0.07
_DOOR_AGREEMENT_BAND = 1.25
_DOOR_DISAGREEMENT_PENALTY = 0.75
_SOFT_WARNING_PENALTY = 0.9
_CONFIDENCE_CAP = 0.97
_MIN_UNCERTAINTY = 0.03
_MANUAL_CONTRADICTION = 0.10
_MAX_UNUSED_EVIDENCE = 20
_HIGH_CONFIDENCE = 0.75
_MEDIUM_CONFIDENCE = 0.50

_DOOR_CONFIDENCE_STRONG = 0.50  # >= 3 doors, tight spread
_DOOR_CONFIDENCE_MEDIUM = 0.42
_DOOR_CONFIDENCE_WEAK = 0.35
_DOOR_TIGHT_SPREAD = 0.10

_Parsed = tuple[int, TextObservation, ParsedText]


@dataclass(frozen=True)
class _DoorPrior:
    meters_per_pixel: float
    count: int
    spread: float  # median absolute deviation / median


class ScaleResolver:
    """Chooses the global scale factor (metres per pixel) for a plan."""

    def __init__(self, config: ScaleConfig) -> None:
        self._cfg = config

    def resolve(
        self,
        graph: PlanarGraph,
        observations: Sequence[TextObservation] = (),
        manual_anchors: Sequence[ManualAnchor] = (),
        image_dpi: float | None = None,
    ) -> ScaleResult:
        if image_dpi is not None and not (math.isfinite(image_dpi) and image_dpi > 0):
            raise InvalidScaleInputError("image_dpi must be a positive number")

        warnings: list[str] = []
        parsed = self._parse(observations)
        door = self._door_prior(graph)
        pool = collect_candidates(graph, parsed, self._cfg) + self._ratio_evidence(
            parsed, image_dpi, warnings
        )
        consensus = (
            find_best_consensus(
                pool, self._cfg.agreement_tolerance, self._cfg.convention_coherence_bonus
            )
            if pool
            else None
        )

        if manual_anchors:
            return self._manual(manual_anchors, graph, consensus, pool, door, warnings)

        result = self._from_text(consensus, pool, graph, door, warnings)
        if result is None and door is not None:
            result = self._from_door(door, graph, warnings)
        if result is None:
            raise ScaleResolutionError(
                "No scale evidence found (no usable dimension text, drawing ratio or door "
                "openings). Provide a ManualAnchor."
            )
        logger.info(
            "Scale %.6f m/px from %s (confidence %.2f, %s)",
            result.meters_per_pixel,
            result.source.value,
            result.confidence,
            result.level.value,
        )
        return result

    # -- tier 1 -----------------------------------------------------------

    def _manual(
        self,
        anchors: Sequence[ManualAnchor],
        graph: PlanarGraph,
        consensus: Consensus | None,
        pool: Sequence[ScaleEvidence],
        door: _DoorPrior | None,
        warnings: list[str],
    ) -> ScaleResult:
        cfg = self._cfg
        scales = [a.meters_per_pixel for a in anchors]
        scale = statistics.median(scales)
        spread = (max(scales) - min(scales)) / scale
        if len(anchors) > 1 and spread > cfg.anchor_disagreement_tolerance:
            warnings.append(
                f"Manual anchors disagree by {spread:.1%}; the median was used. "
                "Re-check the anchor points."
            )
        pick = math.hypot(cfg.anchor_pick_error_px, cfg.anchor_pick_error_px)
        inverse_variance = sum((a.pixel_length / pick) ** 2 for a in anchors)
        uncertainty = max(1.0 / math.sqrt(inverse_variance), spread / 2.0)

        if consensus is not None:
            difference = abs(consensus.meters_per_pixel / scale - 1.0)
            if difference > _MANUAL_CONTRADICTION:
                warnings.append(
                    f"Manual scale differs {difference:.0%} from dimension text "
                    f"({consensus.support} text(s) agree on {consensus.meters_per_pixel:.6f} m/px)."
                )
        if door is not None and self._disagrees(scale, door.meters_per_pixel):
            warnings.append("Manual scale implies unusual door widths; verify the anchor length.")
        warnings.extend(self._soft_warnings(graph, scale)[0])

        evidence = [
            ScaleEvidence(
                ScaleSource.MANUAL,
                a.meters_per_pixel,
                1.0,
                group=-(i + 1),
                note=f"anchor {a.length_m:g} m over {a.pixel_length:.1f}px",
                used=True,
            )
            for i, a in enumerate(anchors)
        ]
        return ScaleResult(
            meters_per_pixel=scale,
            source=ScaleSource.MANUAL,
            confidence=1.0,
            level=ConfidenceLevel.HIGH,
            relative_uncertainty=uncertainty,
            support=len(anchors),
            evidence=evidence + self._trim(pool, set()),
            warnings=warnings,
        )

    # -- tier 2 -----------------------------------------------------------

    def _from_text(
        self,
        consensus: Consensus | None,
        pool: Sequence[ScaleEvidence],
        graph: PlanarGraph,
        door: _DoorPrior | None,
        warnings: list[str],
    ) -> ScaleResult | None:
        if consensus is None:
            return None
        cfg = self._cfg
        scale = consensus.meters_per_pixel
        members = [pool[i] for i in consensus.members]
        n_dimension = sum(m.source is ScaleSource.DIMENSION_TEXT for m in members)
        n_ratio = len(members) - n_dimension
        strong = consensus.support >= cfg.min_support

        if not self._extent_plausible(graph, scale):
            warnings.append(
                f"Text-derived scale {scale:.6f} m/px implies an implausible plan size; rejected."
            )
            return None

        door_agrees = False
        door_contradicts = False
        if door is not None:
            ratio = scale / door.meters_per_pixel
            door_contradicts = self._disagrees(scale, door.meters_per_pixel)
            door_agrees = 1.0 / _DOOR_AGREEMENT_BAND <= ratio <= _DOOR_AGREEMENT_BAND
            if door_contradicts and not strong:
                warnings.append(
                    "Single dimension text contradicts the door-width prior; "
                    "treated as a misread and rejected."
                )
                return None
            if door_contradicts:
                warnings.append(
                    f"Dimension-derived scale implies door widths {ratio:.2f}x the "
                    "standard; check units and the dimension text."
                )

        if n_dimension == 0:
            confidence = _RATIO_ONLY_CONFIDENCE
            source = ScaleSource.DRAWING_RATIO
        else:
            confidence = (
                _CONFIDENCE_THREE_OR_MORE
                if consensus.support >= 3
                else _CONFIDENCE_BY_SUPPORT[consensus.support]
            )
            source = ScaleSource.DIMENSION_TEXT
        confidence *= 1.0 - min(_MAX_SPREAD_PENALTY, _SPREAD_PENALTY_PER_UNIT * consensus.spread)
        if n_dimension and n_ratio:
            confidence += _RATIO_CORROBORATION_BONUS
        if door_agrees:
            confidence += _DOOR_AGREEMENT_BONUS_STRONG if strong else _DOOR_AGREEMENT_BONUS_WEAK
        if door_contradicts:
            confidence *= _DOOR_DISAGREEMENT_PENALTY

        soft, soft_count = self._soft_warnings(graph, scale)
        warnings.extend(soft)
        confidence *= _SOFT_WARNING_PENALTY**soft_count
        confidence = min(_CONFIDENCE_CAP, confidence)

        uncertainty = max(
            _MIN_UNCERTAINTY, math.hypot(consensus.spread, 0.02) / math.sqrt(consensus.support)
        )
        if not strong:
            warnings.append(
                f"Scale rests on {consensus.support} text(s); add a ManualAnchor to confirm."
            )
        return ScaleResult(
            meters_per_pixel=scale,
            source=source,
            confidence=confidence,
            level=_level(confidence),
            relative_uncertainty=uncertainty,
            support=consensus.support,
            evidence=self._trim(pool, set(consensus.members)),
            warnings=warnings,
        )

    def _parse(self, observations: Sequence[TextObservation]) -> list[_Parsed]:
        parsed: list[_Parsed] = []
        for index, observation in enumerate(observations):
            if observation.confidence < self._cfg.min_ocr_confidence:
                continue
            result = parse_text(observation.text, self._cfg.plausible_length_m)
            if result.kind in (TextKind.OTHER, TextKind.AREA):
                continue
            parsed.append((index, observation, result))
        return parsed

    def _ratio_evidence(
        self, parsed: Sequence[_Parsed], dpi: float | None, warnings: list[str]
    ) -> list[ScaleEvidence]:
        evidence: list[ScaleEvidence] = []
        for index, observation, result in parsed:
            if result.kind is not TextKind.SCALE_RATIO or result.ratio is None:
                continue
            if dpi is None:
                warnings.append(
                    f"Drawing scale '{observation.text.strip()}' found but the page DPI is "
                    "unknown; it was not used."
                )
                continue
            evidence.append(
                ScaleEvidence(
                    source=ScaleSource.DRAWING_RATIO,
                    meters_per_pixel=result.ratio * _INCH_METERS / dpi,
                    weight=self._cfg.ratio_notation_weight * observation.confidence,
                    group=index,
                    text=observation.text,
                    note=f"1:{result.ratio:g} at {dpi:g} dpi",
                )
            )
        return evidence

    # -- tier 3 -----------------------------------------------------------

    def _door_prior(self, graph: PlanarGraph) -> _DoorPrior | None:
        widths = [
            o.width_px
            for e in graph.edges
            for o in e.openings
            if o.opening_type is OpeningType.DOOR and o.width_px > 0
        ]
        if not widths:
            return None
        median = statistics.median(widths)
        deviation = statistics.median(abs(w - median) for w in widths) / median
        return _DoorPrior(self._cfg.door_width_m / median, len(widths), deviation)

    def _from_door(self, door: _DoorPrior, graph: PlanarGraph, warnings: list[str]) -> ScaleResult:
        if door.count >= 3 and door.spread <= _DOOR_TIGHT_SPREAD:
            confidence = _DOOR_CONFIDENCE_STRONG
        elif door.count >= 2:
            confidence = _DOOR_CONFIDENCE_MEDIUM
        else:
            confidence = _DOOR_CONFIDENCE_WEAK
        if not self._extent_plausible(graph, door.meters_per_pixel):
            warnings.append("Door-prior scale implies an implausible plan size; verify manually.")
            confidence = min(confidence, _DOOR_CONFIDENCE_WEAK) * _SOFT_WARNING_PENALTY
        soft, soft_count = self._soft_warnings(graph, door.meters_per_pixel, skip_doors=True)
        warnings.extend(soft)
        confidence *= _SOFT_WARNING_PENALTY**soft_count
        warnings.append(
            f"Scale estimated from {door.count} door(s) assuming {self._cfg.door_width_m:g} m "
            "wide; accuracy is roughly +/-10-15%. Add a ManualAnchor for exact dimensions."
        )
        evidence = ScaleEvidence(
            ScaleSource.DOOR_PRIOR,
            door.meters_per_pixel,
            1.0,
            group=-1,
            note=f"{door.count} door(s), spread {door.spread:.0%}",
            used=True,
        )
        return ScaleResult(
            meters_per_pixel=door.meters_per_pixel,
            source=ScaleSource.DOOR_PRIOR,
            confidence=confidence,
            level=_level(confidence),
            relative_uncertainty=max(
                self._cfg.door_prior_uncertainty, door.spread / math.sqrt(door.count)
            ),
            support=door.count,
            evidence=[evidence],
            warnings=warnings,
        )

    # -- validation helpers -------------------------------------------------

    def _disagrees(self, scale_a: float, scale_b: float) -> bool:
        ratio = scale_a / scale_b
        factor = self._cfg.door_disagreement_factor
        return ratio > factor or ratio < 1.0 / factor

    def _extent_plausible(self, graph: PlanarGraph, scale: float) -> bool:
        extent = _extent_px(graph)
        if extent <= 0:
            return True
        low, high = self._cfg.plan_extent_m
        return low <= extent * scale <= high

    def _soft_warnings(
        self, graph: PlanarGraph, scale: float, skip_doors: bool = False
    ) -> tuple[list[str], int]:
        cfg = self._cfg
        messages: list[str] = []
        if graph.thickness_px > 0:
            thickness = graph.thickness_px * scale
            low, high = cfg.plausible_wall_thickness_m
            if not low <= thickness <= high:
                messages.append(
                    f"Median wall thickness is {thickness:.2f} m at this scale "
                    f"(expected {low:g}-{high:g} m)."
                )
        if not skip_doors:
            widths = [
                o.width_px
                for e in graph.edges
                for o in e.openings
                if o.opening_type is OpeningType.DOOR and o.width_px > 0
            ]
            if widths:
                width = statistics.median(widths) * scale
                low, high = cfg.plausible_door_width_m
                if not low <= width <= high:
                    messages.append(
                        f"Median door width is {width:.2f} m at this scale "
                        f"(expected {low:g}-{high:g} m)."
                    )
        return messages, len(messages)

    @staticmethod
    def _trim(pool: Sequence[ScaleEvidence], used: set[int]) -> list[ScaleEvidence]:
        """Used evidence plus the strongest unused items (bounded for readability)."""
        marked = [replace(e, used=True) if i in used else e for i, e in enumerate(pool)]
        kept = [e for e in marked if e.used]
        rest = sorted((e for e in marked if not e.used), key=lambda e: -e.weight)
        return kept + rest[:_MAX_UNUSED_EVIDENCE]


def _level(confidence: float) -> ConfidenceLevel:
    if confidence >= _HIGH_CONFIDENCE:
        return ConfidenceLevel.HIGH
    if confidence >= _MEDIUM_CONFIDENCE:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def _extent_px(graph: PlanarGraph) -> float:
    if not graph.vertices:
        return 0.0
    xs = [v.x for v in graph.vertices.values()]
    ys = [v.y for v in graph.vertices.values()]
    return max(max(xs) - min(xs), max(ys) - min(ys))
