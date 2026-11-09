"""Exception hierarchy for the scale package."""

from __future__ import annotations


class ScaleError(Exception):
    """Base class for all scale-resolution failures."""


class InvalidScaleInputError(ScaleError, ValueError):
    """An argument (graph, anchor, observation, DPI, ...) is malformed."""


class ScaleResolutionError(ScaleError):
    """No usable scale evidence was found.

    Recover by supplying a :class:`scale.ManualAnchor`.
    """
