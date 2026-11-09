"""Exception hierarchy for the topology package."""

from __future__ import annotations


class TopologyError(Exception):
    """Base class for all topology failures."""


class InvalidMaskError(TopologyError, ValueError):
    """The wall mask is not a usable 2-D uint8 array."""
