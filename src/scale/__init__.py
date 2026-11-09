"""Floor-plan scale and dimensionality: metres-per-pixel, metric walls, rooms, elevations."""

from .config import ScaleConfig
from .dimensions import LengthReading, ParsedText, TextKind, parse_text
from .errors import InvalidScaleInputError, ScaleError, ScaleResolutionError
from .models import (
    ConfidenceLevel,
    ManualAnchor,
    OpeningKind,
    PlanTransform,
    ScaledOpening,
    ScaledPlan,
    ScaledRoom,
    ScaledWall,
    ScaleEvidence,
    ScaleResult,
    ScaleSource,
    TextObservation,
    WallKind,
)
from .service import ScaleEngine, observations_from_perception

__all__ = [
    "ConfidenceLevel",
    "InvalidScaleInputError",
    "LengthReading",
    "ManualAnchor",
    "OpeningKind",
    "ParsedText",
    "PlanTransform",
    "ScaleConfig",
    "ScaleEngine",
    "ScaleError",
    "ScaleEvidence",
    "ScaleResolutionError",
    "ScaleResult",
    "ScaleSource",
    "ScaledOpening",
    "ScaledPlan",
    "ScaledRoom",
    "ScaledWall",
    "TextKind",
    "TextObservation",
    "WallKind",
    "observations_from_perception",
    "parse_text",
]
