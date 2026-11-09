"""Floor-plan perception: walls, openings and text from a normalized image."""

from .config import Binarization, OcrMode, PerceptionConfig, TextRemoval, WallFusion
from .errors import (
    DependencyError,
    InferenceError,
    InvalidImageError,
    ModelLoadError,
    PerceptionError,
)
from .learned import HybridWallSegmenter, OnnxWallModel, WallSegmentationModel
from .models import (
    AnnotationKind,
    DetectedAnnotation,
    DetectedSymbol,
    PerceptionResult,
    SymbolCategory,
)
from .service import PerceptionService
from .symbols import SymbolDetector
from .text import TextDetector

__all__ = [
    "AnnotationKind",
    "Binarization",
    "DependencyError",
    "DetectedAnnotation",
    "DetectedSymbol",
    "HybridWallSegmenter",
    "InferenceError",
    "InvalidImageError",
    "ModelLoadError",
    "OnnxWallModel",
    "OcrMode",
    "PerceptionConfig",
    "PerceptionError",
    "PerceptionResult",
    "PerceptionService",
    "SymbolCategory",
    "SymbolDetector",
    "TextDetector",
    "TextRemoval",
    "WallFusion",
    "WallSegmentationModel",
]
