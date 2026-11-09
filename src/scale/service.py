"""Public entry point of the scale package."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from perception import PerceptionResult
from topology import PlanarGraph

from .apply import apply_scale
from .config import ScaleConfig
from .errors import InvalidScaleInputError, ScaleResolutionError
from .models import (
    ConfidenceLevel,
    ManualAnchor,
    ScaledPlan,
    ScaleResult,
    ScaleSource,
    TextObservation,
)
from .resolver import ScaleResolver
from .units import drawing_units_to_meters

logger = logging.getLogger(__name__)


class ScaleEngine:
    """Resolves real-world scale and turns a pixel wall graph into a metric plan.

    Typical use::

        plan = ScaleEngine().build_from_perception(graph, perception_result)

    Scale priority: manual anchor > dimension text / drawing ratio (consensus)
    > standard door width. See :mod:`scale.resolver` for the arbitration rules.
    """

    def __init__(self, config: ScaleConfig | None = None) -> None:
        self._cfg = config or ScaleConfig()
        self._resolver = ScaleResolver(self._cfg)

    def resolve_scale(
        self,
        graph: PlanarGraph,
        observations: Sequence[TextObservation] = (),
        manual_anchors: Sequence[ManualAnchor] = (),
        image_dpi: float | None = None,
    ) -> ScaleResult:
        """Resolve metres-per-pixel for ``graph``.

        Args:
            graph: Wall graph from the topology module (pixel space).
            observations: Text found on the plan (pixel space of the same image).
            manual_anchors: User-measured references; override everything.
            image_dpi: Resolution of the processed image relative to the
                physical page (``IngestionResult.metadata.dpi``). Enables
                ``1:N`` drawing-scale notation.

        Raises:
            InvalidScaleInputError: malformed arguments.
            ScaleResolutionError: no evidence at all (supply a manual anchor).
        """
        self._validate_graph(graph)
        return self._resolver.resolve(graph, observations, manual_anchors, image_dpi)

    def apply_scale(self, graph: PlanarGraph, scale: ScaleResult) -> ScaledPlan:
        """Convert ``graph`` to metres using an already-resolved ``scale``."""
        self._validate_graph(graph)
        return apply_scale(graph, scale, self._cfg)

    def build(
        self,
        graph: PlanarGraph,
        observations: Sequence[TextObservation] = (),
        manual_anchors: Sequence[ManualAnchor] = (),
        image_dpi: float | None = None,
    ) -> ScaledPlan:
        """Resolve the scale, then scale the graph."""
        scale = self.resolve_scale(graph, observations, manual_anchors, image_dpi)
        return self.apply_scale(graph, scale)

    def build_from_perception(
        self,
        graph: PlanarGraph,
        perception: PerceptionResult,
        manual_anchors: Sequence[ManualAnchor] = (),
        image_dpi: float | None = None,
    ) -> ScaledPlan:
        """Convenience wrapper using the OCR text from the perception module."""
        return self.build(
            graph, observations_from_perception(perception), manual_anchors, image_dpi
        )

    @staticmethod
    def scale_from_drawing_units(units: str | None) -> ScaleResult:
        """Exact scale for vector input, where one drawing unit is a known length.

        ``units`` is ``IngestionMetadata.units`` (e.g. ``"Millimeters"``). The
        returned factor converts drawing units (not pixels) to metres, so apply
        it to a graph built directly from DXF geometry.
        """
        factor = drawing_units_to_meters(units)
        if factor is None:
            raise ScaleResolutionError(
                f"Drawing units {units!r} are unknown or unitless; provide a ManualAnchor."
            )
        return ScaleResult(
            meters_per_pixel=factor,
            source=ScaleSource.VECTOR_UNITS,
            confidence=1.0,
            level=ConfidenceLevel.HIGH,
            relative_uncertainty=0.0,
            support=1,
        )

    @staticmethod
    def _validate_graph(graph: object) -> None:
        if not isinstance(graph, PlanarGraph):
            raise InvalidScaleInputError("graph must be a topology.PlanarGraph")


def observations_from_perception(perception: PerceptionResult) -> list[TextObservation]:
    """Adapt Module 2 annotations (all text, not only 'dimension') for scale resolution."""
    return [
        TextObservation(
            text=a.text,
            box=a.box,
            confidence=a.confidence if math.isfinite(a.confidence) else 0.0,
            polygon=a.polygon,
        )
        for a in perception.annotations
    ]
