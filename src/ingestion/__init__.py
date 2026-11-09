"""Floor-plan ingestion: normalize raster, PDF and DXF inputs."""

from .config import IngestionConfig
from .errors import (
    CorruptFileError,
    IngestionError,
    InvalidPageError,
    ResourceLimitError,
    UnsupportedFormatError,
)
from .models import (
    Bounds,
    IngestionMetadata,
    IngestionResult,
    InputType,
    VectorPrimitive,
)
from .service import IngestionService

__all__ = [
    "Bounds",
    "CorruptFileError",
    "IngestionConfig",
    "IngestionError",
    "IngestionMetadata",
    "IngestionResult",
    "IngestionService",
    "InputType",
    "InvalidPageError",
    "ResourceLimitError",
    "UnsupportedFormatError",
    "VectorPrimitive",
]
