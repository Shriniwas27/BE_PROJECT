"""Exception hierarchy for the perception package."""

from __future__ import annotations


class PerceptionError(Exception):
    """Base class for all perception failures."""


class InvalidImageError(PerceptionError, ValueError):
    """The input array is not a usable 8-bit grayscale/BGR image."""


class DependencyError(PerceptionError):
    """An optional dependency (easyocr, ultralytics) is required but missing."""


class ModelLoadError(PerceptionError):
    """A model file is missing or could not be loaded."""


class InferenceError(PerceptionError):
    """A detector failed while processing an image."""
