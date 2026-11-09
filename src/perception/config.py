"""Configuration for the perception pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OcrMode(str, Enum):
    OFF = "off"
    AUTO = "auto"  # use OCR if installed, otherwise continue with a warning
    REQUIRED = "required"  # fail at construction if OCR is unavailable


class Binarization(str, Enum):
    OTSU = "otsu"  # global; best for clean, evenly lit drawings
    ADAPTIVE = "adaptive"  # local; for uneven illumination (large block size)


class WallFusion(str, Enum):
    CLASSICAL = "classical"  # morphology only; the wall model is ignored
    LEARNED = "learned"  # the wall model's mask as-is (soft edges)
    GUIDED = "guided"  # classical strokes kept only where the model says "wall"


class TextRemoval(str, Enum):
    INPAINT = "inpaint"  # Telea inpainting; reconstructs lines crossing text
    FILL = "fill"  # paint white; faster, assumes dark-on-light input


@dataclass(frozen=True)
class PerceptionConfig:
    """Immutable, validated settings for :class:`perception.PerceptionService`.

    All size thresholds that depend on drawing scale are expressed as ratios of
    the image's longest side, so behaviour is consistent across resolutions.

    Text:
        ocr_mode: See :class:`OcrMode`.
        ocr_languages: EasyOCR language codes.
        ocr_min_confidence: Detections below this are ignored. Low values
            erase real wall pixels on false positives, so keep it moderate.
        text_padding_px: Mask growth around each text region.
        text_removal: See :class:`TextRemoval`.
        ocr_tile_px / ocr_tile_overlap_px: Run OCR on overlapping tiles of this
            size (0 = whole image). Recovers small dimension text on large
            sheets; the overlap must exceed the longest text string.
        inpaint_radius: Telea radius in pixels.
        heuristic_text_fallback: When no OCR is available, remove isolated
            glyph-sized specks. Off by default: it can also remove tiny
            genuine features.

    Symbols:
        symbol_model_path: YOLO weights. ``None`` disables the neural detector.
        symbol_confidence: YOLO confidence threshold.
        symbol_class_map: ``(substring, category)`` pairs mapping model class
            names to ``door`` / ``window`` / ``opening``. First match wins.
        heuristic_symbols: Use the quarter-circle door-swing detector when no
            model is configured. Experimental.
        heuristic_door_size_ratio: (min, max) door swing radius / longest side.
        device: Inference device, e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``.

    Walls:
        binarization / adaptive_block_fraction / adaptive_c: See
            :class:`Binarization`. The adaptive block is
            ``adaptive_block_fraction * longest_side`` (min 31, forced odd) so
            thick solid walls are not hollowed out.
        wall_angles_deg: Orientations of wall lines to keep. Add 45/135 for
            diagonal walls.
        wall_min_length_ratio: Minimum straight run kept, as a ratio.
        wall_close_px / wall_close_iterations: Gap closing.
        min_component_area_ratio: Drop specks smaller than this fraction of
            the image area.

    Learned walls (optional; needs ``onnxruntime``):
        wall_model_path: ONNX wall-segmentation network (see
            :mod:`perception.learned` for the I/O contract). ``None`` keeps the
            classical segmenter.
        wall_model_threshold: Probability above which a pixel is wall.
        wall_model_tile_px / wall_model_overlap_px: Tiled inference geometry.
        wall_model_channel: Wall class index when the model outputs K classes.
        wall_model_logits: The single-channel output is logits, not probabilities.
        wall_fusion: See :class:`WallFusion`. Applies only when a model is set.
        wall_guide_dilate_px: How far (px) a classical stroke may sit from the
            model's mask and still be kept in ``GUIDED`` fusion.

    Openings:
        carve_doors / carve_windows / carve_openings: Erase detected symbol
            boxes from the wall mask so downstream code sees the gap.
        carve_padding_px: Extra margin around carved boxes.
    """

    ocr_mode: OcrMode = OcrMode.AUTO
    ocr_languages: tuple[str, ...] = ("en",)
    ocr_min_confidence: float = 0.4
    ocr_tile_px: int = 0
    ocr_tile_overlap_px: int = 160
    text_padding_px: int = 2
    text_removal: TextRemoval = TextRemoval.INPAINT
    inpaint_radius: int = 3
    heuristic_text_fallback: bool = False

    symbol_model_path: str | None = None
    symbol_confidence: float = 0.35
    symbol_class_map: tuple[tuple[str, str], ...] = (
        ("door", "door"),
        ("window", "window"),
        ("opening", "opening"),
    )
    heuristic_symbols: bool = False
    heuristic_door_size_ratio: tuple[float, float] = (0.015, 0.08)
    device: str = "cpu"

    binarization: Binarization = Binarization.OTSU
    adaptive_block_fraction: float = 0.0625
    adaptive_c: int = 10
    wall_angles_deg: tuple[float, ...] = (0.0, 90.0)
    wall_min_length_ratio: float = 0.0075
    wall_close_px: int = 3
    wall_close_iterations: int = 2
    min_component_area_ratio: float = 1e-5

    wall_model_path: str | None = None
    wall_model_threshold: float = 0.5
    wall_model_tile_px: int = 512
    wall_model_overlap_px: int = 64
    wall_model_channel: int = 0
    wall_model_logits: bool = False
    wall_fusion: WallFusion = WallFusion.GUIDED
    wall_guide_dilate_px: int = 4
    # Multi-class model: channels of the same network that hold door / window scores (None = off)
    wall_model_door_channel: int | None = None
    wall_model_window_channel: int | None = None
    symbol_seg_threshold: float = 0.5
    symbol_seg_min_area_ratio: float = 2e-6

    carve_doors: bool = True
    carve_windows: bool = False
    carve_openings: bool = True
    carve_padding_px: int = 0

    def __post_init__(self) -> None:
        if not self.ocr_languages:
            raise ValueError("ocr_languages must not be empty")
        if not 0.0 <= self.ocr_min_confidence <= 1.0:
            raise ValueError("ocr_min_confidence must be within [0, 1]")
        if not 0.0 <= self.symbol_confidence <= 1.0:
            raise ValueError("symbol_confidence must be within [0, 1]")
        if not self.device:
            raise ValueError("device must not be empty")
        if not self.wall_angles_deg:
            raise ValueError("wall_angles_deg must not be empty")
        low, high = self.heuristic_door_size_ratio
        if not 0.0 < low < high <= 1.0:
            raise ValueError("heuristic_door_size_ratio must satisfy 0 < min < max <= 1")
        if not 0.0 < self.adaptive_block_fraction <= 1.0:
            raise ValueError("adaptive_block_fraction must be within (0, 1]")
        if self.wall_min_length_ratio <= 0:
            raise ValueError("wall_min_length_ratio must be positive")
        if self.min_component_area_ratio < 0:
            raise ValueError("min_component_area_ratio must be >= 0")
        for name in (
            "text_padding_px",
            "inpaint_radius",
            "wall_close_px",
            "wall_close_iterations",
            "carve_padding_px",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 < self.wall_model_threshold < 1.0:
            raise ValueError("wall_model_threshold must be within (0, 1)")
        if self.wall_model_tile_px < 64:
            raise ValueError("wall_model_tile_px must be >= 64")
        if not 0 <= self.wall_model_overlap_px < self.wall_model_tile_px // 2:
            raise ValueError("wall_model_overlap_px must be in [0, wall_model_tile_px / 2)")
        if self.wall_model_channel < 0 or self.wall_guide_dilate_px < 0:
            raise ValueError("wall_model_channel and wall_guide_dilate_px must be >= 0")
        if self.ocr_tile_px != 0 and self.ocr_tile_px < 256:
            raise ValueError("ocr_tile_px must be 0 (off) or >= 256")
        if self.ocr_tile_px and not 0 <= self.ocr_tile_overlap_px < self.ocr_tile_px // 2:
            raise ValueError("ocr_tile_overlap_px must be in [0, ocr_tile_px / 2)")
        for ch in (self.wall_model_door_channel, self.wall_model_window_channel):
            if ch is not None and ch < 0:
                raise ValueError("door/window channel must be >= 0")
        if not 0.0 < self.symbol_seg_threshold < 1.0:
            raise ValueError("symbol_seg_threshold must be within (0, 1)")
        valid = {"door", "window", "opening"}
        for _, category in self.symbol_class_map:
            if category not in valid:
                raise ValueError(f"symbol_class_map category must be one of {sorted(valid)}")
