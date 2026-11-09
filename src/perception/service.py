"""Public entry point of the perception package."""

from __future__ import annotations

import logging

import numpy as np

from ingestion import IngestionResult, InputType

from .config import OcrMode, PerceptionConfig
from .errors import DependencyError, InferenceError, PerceptionError
from .imageops import to_gray_uint8
from .learned import HybridWallSegmenter, OnnxWallModel, WallSegmentationModel
from .models import (
    Box,
    DetectedAnnotation,
    DetectedSymbol,
    PerceptionResult,
    SymbolCategory,
)
from .segsymbols import SegmentationSymbolDetector
from .symbols import HeuristicDoorDetector, SymbolDetector, YoloSymbolDetector
from .text import (
    EasyOcrTextDetector,
    TextDetector,
    TiledTextDetector,
    build_text_mask,
    heuristic_glyph_mask,
    polygon_to_box,
    remove_text,
)
from .walls import carve_openings

logger = logging.getLogger(__name__)


class PerceptionService:
    """Extracts walls, openings and text from a normalized floor-plan image.

    Detectors can be injected (``text_detector``, ``symbol_detector``) for
    testing or to plug in custom models; otherwise they are built from the
    configuration. Input must be dark lines on a light background, as produced
    by :class:`ingestion.IngestionService`.
    """

    def __init__(
        self,
        config: PerceptionConfig | None = None,
        *,
        text_detector: TextDetector | None = None,
        symbol_detector: SymbolDetector | None = None,
        wall_model: WallSegmentationModel | None = None,
    ) -> None:
        self._cfg = config or PerceptionConfig()
        text = text_detector or self._build_text_detector()
        if text is not None and self._cfg.ocr_tile_px > 0:
            text = TiledTextDetector(text, self._cfg.ocr_tile_px, self._cfg.ocr_tile_overlap_px)
        self._text = text
        self._symbols = symbol_detector or self._build_symbol_detector()
        self._walls = HybridWallSegmenter(self._cfg, wall_model or self._build_wall_model())

        self._static_warnings: list[str] = []
        if self._text is None and not self._cfg.heuristic_text_fallback:
            self._static_warnings.append("No text detector: text was not removed.")
        if self._symbols is None:
            self._static_warnings.append("No symbol detector: doors/windows were not detected.")

    # -- public API -------------------------------------------------------

    def extract(self, image: np.ndarray) -> PerceptionResult:
        gray = to_gray_uint8(image)
        height, width = gray.shape[:2]
        warnings = list(self._static_warnings)

        annotations = self._detect_text(gray)
        glyphs = (
            heuristic_glyph_mask(gray, self._cfg)
            if self._text is None and self._cfg.heuristic_text_fallback
            else None
        )
        text_mask = build_text_mask(gray.shape, annotations, self._cfg.text_padding_px, glyphs)
        cleaned = remove_text(gray, text_mask, self._cfg)

        raw_walls, wall_warnings = self._walls.segment_with_warnings(cleaned)
        warnings.extend(wall_warnings)
        symbols = self._detect_symbols(cleaned, raw_walls)

        wall_mask, opening_mask = carve_openings(
            raw_walls, self._boxes_to_carve(symbols), self._cfg.carve_padding_px
        )
        logger.info(
            "Perception: %d annotations, %d symbols, wall coverage %.2f%%",
            len(annotations),
            len(symbols),
            100.0 * float(np.count_nonzero(wall_mask)) / (height * width),
        )
        return PerceptionResult(
            wall_mask=wall_mask,
            opening_mask=opening_mask,
            text_mask=text_mask,
            cleaned_image=cleaned,
            image_size=(width, height),
            doors=[s for s in symbols if s.category is SymbolCategory.DOOR],
            windows=[s for s in symbols if s.category is SymbolCategory.WINDOW],
            openings=[s for s in symbols if s.category is SymbolCategory.OPENING],
            annotations=annotations,
            warnings=warnings,
        )

    def extract_from_ingestion(self, result: IngestionResult) -> PerceptionResult:
        """Convenience wrapper for the output of the ingestion module."""
        if result.input_type is not InputType.RASTER or result.image is None:
            raise ValueError("Perception requires a raster IngestionResult (got vector input).")
        return self.extract(result.image)

    # -- construction -----------------------------------------------------

    def _build_text_detector(self) -> TextDetector | None:
        mode = self._cfg.ocr_mode
        if mode is OcrMode.OFF:
            return None
        try:
            return EasyOcrTextDetector(self._cfg.ocr_languages, self._cfg.device.startswith("cuda"))
        except DependencyError:
            if mode is OcrMode.REQUIRED:
                raise
            logger.warning("EasyOCR unavailable; continuing without text detection.")
            return None

    def _build_wall_model(self) -> WallSegmentationModel | None:
        if self._cfg.wall_model_path is None:
            return None
        return OnnxWallModel(
            self._cfg.wall_model_path,
            self._cfg.device,
            self._cfg.wall_model_channel,
            self._cfg.wall_model_logits,
        )

    def _build_symbol_detector(self) -> SymbolDetector | None:
        cfg = self._cfg
        if cfg.wall_model_path and (
            cfg.wall_model_door_channel is not None or cfg.wall_model_window_channel is not None
        ):
            from .learned import _load_session

            session = _load_session(cfg.wall_model_path, cfg.device)

            def head(ch: int | None) -> OnnxWallModel | None:
                return None if ch is None else OnnxWallModel(None, cfg.device, ch, session=session)

            return SegmentationSymbolDetector(
                head(cfg.wall_model_door_channel), head(cfg.wall_model_window_channel), cfg
            )
        if self._cfg.symbol_model_path is not None:
            return YoloSymbolDetector(
                self._cfg.symbol_model_path,
                self._cfg.symbol_confidence,
                self._cfg.device,
                self._cfg.symbol_class_map,
            )
        if self._cfg.heuristic_symbols:
            return HeuristicDoorDetector(self._cfg)
        return None

    # -- pipeline steps ---------------------------------------------------

    def _detect_text(self, gray: np.ndarray) -> list[DetectedAnnotation]:
        if self._text is None:
            return []
        height, width = gray.shape[:2]
        try:
            raw = self._text.detect(gray)
        except PerceptionError:
            raise
        except Exception as exc:
            raise InferenceError(f"Text detection failed: {exc}") from exc

        kept: list[DetectedAnnotation] = []
        for annotation in raw:
            if annotation.confidence < self._cfg.ocr_min_confidence:
                continue
            box = (
                polygon_to_box(annotation.polygon, width, height)
                if annotation.polygon
                else _clip_box(annotation.box, width, height)
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            kept.append(
                DetectedAnnotation(
                    annotation.text,
                    box,
                    annotation.confidence,
                    annotation.kind,
                    annotation.polygon,
                )
            )
        return kept

    def _detect_symbols(self, cleaned: np.ndarray, raw_walls: np.ndarray) -> list[DetectedSymbol]:
        if self._symbols is None:
            return []
        try:
            return self._symbols.detect(cleaned, raw_walls)
        except PerceptionError:
            raise
        except Exception as exc:
            raise InferenceError(f"Symbol detection failed: {exc}") from exc

    def _boxes_to_carve(self, symbols: list[DetectedSymbol]) -> list[Box]:
        flags = {
            SymbolCategory.DOOR: self._cfg.carve_doors,
            SymbolCategory.WINDOW: self._cfg.carve_windows,
            SymbolCategory.OPENING: self._cfg.carve_openings,
        }
        return [s.box for s in symbols if flags[s.category]]


def _clip_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
