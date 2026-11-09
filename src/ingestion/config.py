"""Configuration for the ingestion pipeline. All tunables live here."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestionConfig:
    """Immutable, validated settings for :class:`ingestion.IngestionService`.

    Attributes:
        target_dpi: Desired PDF render resolution.
        max_dimension: Longest side (px) of the returned raster image.
        max_file_bytes: Reject input files larger than this.
        max_render_pixels: Hard cap on PDF render size; DPI is lowered to fit.
        pdf_page: Zero-based PDF page to render.
        dewarp_photos: Apply perspective correction to raster images (photos/
            skewed scans). Never applied to PDF renders, which are already
            axis-aligned.
        dewarp_min_area_ratio: Minimum fraction of the image the paper quad
            must cover to be accepted.
        dewarp_max_area_ratio: Quads covering more than this are treated as
            the image frame/border, not paper, and ignored.
        dewarp_min_skew_degrees: Quads whose edges are all within this angle of
            horizontal/vertical are axis-aligned frames (e.g. a drawing border)
            and are left untouched.
        canny_low / canny_high: Canny edge thresholds used for paper detection.
        polarity_threshold: Median gray level below which the image is
            considered dark-background and inverted.
        apply_clahe: Enable local contrast enhancement.
        clahe_clip_limit / clahe_tile_grid: CLAHE parameters.
        max_dxf_entities: Safety cap on extracted DXF primitives.
        max_block_depth: Maximum nesting depth when exploding block INSERTs.
        curve_tolerance_mm: Max deviation (in mm) when flattening curves
            (arcs, circles, splines) into polylines. Converted to drawing
            units using the DXF ``$INSUNITS`` header when available.
        skip_hidden_layers: Ignore entities on frozen or switched-off layers.
    """

    target_dpi: int = 300
    max_dimension: int = 2048
    max_file_bytes: int = 200 * 1024 * 1024
    max_render_pixels: int = 120_000_000
    pdf_page: int = 0

    dewarp_photos: bool = True
    dewarp_min_area_ratio: float = 0.3
    dewarp_max_area_ratio: float = 0.98
    dewarp_min_skew_degrees: float = 2.0
    canny_low: int = 50
    canny_high: int = 200

    polarity_threshold: int = 128
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8

    max_dxf_entities: int = 2_000_000
    max_block_depth: int = 8
    curve_tolerance_mm: float = 1.0
    skip_hidden_layers: bool = True

    def __post_init__(self) -> None:
        positive = {
            "target_dpi": self.target_dpi,
            "max_dimension": self.max_dimension,
            "max_file_bytes": self.max_file_bytes,
            "max_render_pixels": self.max_render_pixels,
            "clahe_tile_grid": self.clahe_tile_grid,
            "max_dxf_entities": self.max_dxf_entities,
            "curve_tolerance_mm": self.curve_tolerance_mm,
            "clahe_clip_limit": self.clahe_clip_limit,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.pdf_page < 0:
            raise ValueError("pdf_page must be >= 0")
        if self.max_block_depth < 0:
            raise ValueError("max_block_depth must be >= 0")
        if not 0 < self.dewarp_min_area_ratio < self.dewarp_max_area_ratio <= 1:
            raise ValueError("require 0 < dewarp_min_area_ratio < dewarp_max_area_ratio <= 1")
        if self.dewarp_min_skew_degrees < 0:
            raise ValueError("dewarp_min_skew_degrees must be >= 0")
        if not 0 <= self.polarity_threshold <= 255:
            raise ValueError("polarity_threshold must be within [0, 255]")
