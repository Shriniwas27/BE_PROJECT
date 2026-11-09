"""Exception hierarchy for the ingestion package."""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for all ingestion failures."""


class UnsupportedFormatError(IngestionError):
    """The file type is not supported (or could not be identified)."""


class CorruptFileError(IngestionError):
    """The file is unreadable, malformed, or empty."""


class ResourceLimitError(IngestionError):
    """The input exceeds a configured safety limit (size, entity count, ...)."""


class InvalidPageError(IngestionError):
    """The requested PDF page does not exist."""
