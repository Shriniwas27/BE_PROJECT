"""Public entry point of the ingestion package."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import IngestionConfig
from .dxf import DxfParser
from .formats import FileFormat, detect_format, validate_file
from .models import IngestionMetadata, IngestionResult, InputType
from .pdf import render_pdf_page
from .raster import RasterNormalizer, load_image

logger = logging.getLogger(__name__)


class IngestionService:
    """Normalizes raster, PDF and DXF floor plans into a uniform result.

    Raises subclasses of :class:`ingestion.errors.IngestionError` for bad
    input, and ``FileNotFoundError`` when the path does not exist.
    """

    def __init__(self, config: IngestionConfig | None = None) -> None:
        self._cfg = config or IngestionConfig()
        self._raster = RasterNormalizer(self._cfg)
        self._dxf = DxfParser(self._cfg)

    def process(self, file_path: str | Path) -> IngestionResult:
        path = Path(file_path)
        validate_file(path, self._cfg.max_file_bytes)
        fmt = detect_format(path)
        logger.info("Ingesting %s as %s", path.name, fmt.value)

        if fmt is FileFormat.DXF:
            return self._dxf.parse(path)
        if fmt is FileFormat.PDF:
            return self._process_pdf(path)
        return self._process_image(path, fmt)

    def _process_image(self, path: Path, fmt: FileFormat) -> IngestionResult:
        output = self._raster.normalize(load_image(path), allow_dewarp=self._cfg.dewarp_photos)
        metadata = IngestionMetadata(
            source_format=fmt.value,
            scale_factor=output.scale_factor,
            dewarped=output.dewarped,
        )
        return IngestionResult(
            InputType.RASTER, metadata, image=output.image, original_dimensions=output.original_size
        )

    def _process_pdf(self, path: Path) -> IngestionResult:
        page = render_pdf_page(path, self._cfg)
        # PDF renders are axis-aligned: dewarping would only risk cropping the page frame.
        output = self._raster.normalize(page.image, allow_dewarp=False)
        metadata = IngestionMetadata(
            source_format=FileFormat.PDF.value,
            dpi=page.dpi * output.scale_factor,
            scale_factor=output.scale_factor,
            page_index=page.page_index,
            page_count=page.page_count,
        )
        if page.page_count > 1:
            metadata.warnings.append(
                f"PDF has {page.page_count} pages; only page {page.page_index} was processed."
            )
        return IngestionResult(
            InputType.RASTER, metadata, image=output.image, original_dimensions=output.original_size
        )
