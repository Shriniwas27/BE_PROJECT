"""File validation and format detection (magic bytes first, extension second)."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from .errors import CorruptFileError, ResourceLimitError, UnsupportedFormatError

_HEADER_BYTES = 32


class FileFormat(str, Enum):
    PDF = "pdf"
    DXF = "dxf"
    PNG = "png"
    JPEG = "jpeg"
    BMP = "bmp"
    TIFF = "tiff"
    WEBP = "webp"

    @property
    def is_raster(self) -> bool:
        return self not in (FileFormat.PDF, FileFormat.DXF)


_EXTENSIONS = {
    ".pdf": FileFormat.PDF,
    ".dxf": FileFormat.DXF,
    ".png": FileFormat.PNG,
    ".jpg": FileFormat.JPEG,
    ".jpeg": FileFormat.JPEG,
    ".bmp": FileFormat.BMP,
    ".tif": FileFormat.TIFF,
    ".tiff": FileFormat.TIFF,
    ".webp": FileFormat.WEBP,
}


def _sniff(header: bytes) -> FileFormat | None:
    if header.startswith(b"%PDF"):
        return FileFormat.PDF
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return FileFormat.PNG
    if header.startswith(b"\xff\xd8\xff"):
        return FileFormat.JPEG
    if header.startswith(b"BM"):
        return FileFormat.BMP
    if header[:4] in (b"II*\x00", b"MM\x00*"):
        return FileFormat.TIFF
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return FileFormat.WEBP
    if header.startswith(b"AutoCAD Binary DXF"):
        return FileFormat.DXF
    return None


def validate_file(path: Path, max_bytes: int) -> None:
    """Ensure the path is a readable, non-empty file within the size limit."""
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if not path.is_file():
        raise CorruptFileError(f"Not a regular file: {path}")
    size = path.stat().st_size
    if size == 0:
        raise CorruptFileError(f"File is empty: {path}")
    if size > max_bytes:
        raise ResourceLimitError(f"File is {size} bytes; limit is {max_bytes}")


def detect_format(path: Path) -> FileFormat:
    """Identify the format, preferring file content over the extension."""
    with path.open("rb") as handle:
        header = handle.read(_HEADER_BYTES)

    if header.startswith(b"AC10") or path.suffix.lower() == ".dwg":
        raise UnsupportedFormatError("DWG is not supported; export the drawing as DXF.")

    by_content = _sniff(header)
    by_extension = _EXTENSIONS.get(path.suffix.lower())

    if by_content is not None:
        return by_content
    if by_extension is FileFormat.DXF:
        return FileFormat.DXF  # ASCII DXF has no reliable magic number
    if by_extension is not None:
        raise CorruptFileError(
            f"File content does not match its '{path.suffix}' extension: {path.name}"
        )
    raise UnsupportedFormatError(f"Unsupported file format: '{path.suffix}'")
