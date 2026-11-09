from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from ingestion import IngestionService
from perception import (
    AnnotationKind,
    Binarization,
    DependencyError,
    DetectedAnnotation,
    DetectedSymbol,
    InferenceError,
    InvalidImageError,
    ModelLoadError,
    OcrMode,
    PerceptionConfig,
    PerceptionService,
    SymbolCategory,
)
from perception.symbols import YoloSymbolDetector, map_class_name
from perception.text import classify_text
from perception.walls import carve_openings, line_kernel

# -- helpers ------------------------------------------------------------------


def plan(width: int = 1000, height: int = 700) -> np.ndarray:
    """White page, thick rectangular outer walls and one interior wall."""
    img = np.full((height, width), 255, np.uint8)
    cv2.rectangle(img, (100, 100), (width - 100, height - 100), 0, 10)
    cv2.line(img, (width // 2, 100), (width // 2, height - 100), 0, 10)
    return img


class FakeText:
    def __init__(self, annotations: list[DetectedAnnotation]) -> None:
        self.annotations = annotations

    def detect(self, image: np.ndarray) -> list[DetectedAnnotation]:
        return self.annotations


class FakeSymbols:
    def __init__(self, symbols: list[DetectedSymbol]) -> None:
        self.symbols = symbols

    def detect(self, image: np.ndarray, wall_mask: np.ndarray) -> list[DetectedSymbol]:
        return self.symbols


def make_service(**kwargs: Any) -> PerceptionService:
    cfg = PerceptionConfig(ocr_mode=OcrMode.OFF, **kwargs.pop("cfg", {}))
    return PerceptionService(cfg, **kwargs)


# -- walls --------------------------------------------------------------------


def test_walls_detected_and_mask_is_binary() -> None:
    result = make_service().extract(plan())
    mask = result.wall_mask
    assert mask.shape == (700, 1000) and mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[100, 500] == 255 and mask[300, 500] == 255  # outer + interior wall
    assert mask[300, 250] == 0  # room interior stays empty


def test_solid_thick_walls_are_not_hollowed_out() -> None:
    img = np.full((600, 800), 255, np.uint8)
    cv2.rectangle(img, (100, 100), (700, 130), 0, -1)  # 30 px solid wall
    result = make_service().extract(img)
    assert result.wall_mask[115, 400] == 255  # centre of the stroke is filled


def test_speckle_noise_is_removed() -> None:
    img = plan()
    rng = np.random.default_rng(0)
    for _ in range(60):
        x, y = int(rng.integers(150, 450)), int(rng.integers(150, 570))
        img[y, x] = 0
    mask = make_service().extract(img).wall_mask
    assert mask[150:570, 150:450].sum() < 255 * 50


def test_diagonal_walls_only_with_diagonal_angles() -> None:
    img = np.full((600, 800), 255, np.uint8)
    cv2.line(img, (100, 100), (500, 500), 0, 3)
    assert make_service().extract(img).wall_mask.sum() == 0
    cfg = {"wall_angles_deg": (0.0, 45.0, 90.0, 135.0)}
    assert make_service(cfg=cfg).extract(img).wall_mask.sum() > 0


def test_adaptive_binarization_handles_uneven_lighting() -> None:
    img = plan().astype(np.float32)
    gradient = np.linspace(0.55, 1.0, img.shape[1], dtype=np.float32)[None, :]
    shaded = (img * gradient).astype(np.uint8)
    cfg = {"binarization": Binarization.ADAPTIVE}
    mask = make_service(cfg=cfg).extract(shaded).wall_mask
    assert mask[300, 500] == 255 and mask[100, 300] == 255


def test_line_kernel_shapes() -> None:
    assert line_kernel(15, 0).shape == (1, 15)
    assert line_kernel(15, 90).shape == (15, 1)
    assert line_kernel(15, 45).sum() > 1


# -- text ---------------------------------------------------------------------


def test_text_is_removed_and_walls_untouched() -> None:
    img = plan()
    cv2.putText(img, "BEDROOM", (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    polygon = ((240, 350), (440, 350), (440, 410), (240, 410))
    annotation = DetectedAnnotation(
        "BEDROOM", (240, 350, 440, 410), 0.9, AnnotationKind.LABEL, polygon
    )
    service = make_service(text_detector=FakeText([annotation]))
    result = service.extract(img)
    assert result.cleaned_image[350:410, 240:440].min() > 150
    assert result.text_mask[380, 300] == 255
    assert result.wall_mask[300, 500] == 255
    assert result.annotations[0].text == "BEDROOM"


def test_low_confidence_text_is_ignored() -> None:
    annotation = DetectedAnnotation("x", (200, 200, 260, 240), 0.1)
    result = make_service(text_detector=FakeText([annotation])).extract(plan())
    assert result.annotations == [] and result.text_mask.sum() == 0


def test_fill_removal_and_box_only_annotation() -> None:
    img = plan()
    img[300:320, 200:260] = 0
    annotation = DetectedAnnotation("A", (200, 300, 260, 320), 0.9)
    service = make_service(text_detector=FakeText([annotation]), cfg={"text_removal": "fill"})
    result = service.extract(img)
    assert result.cleaned_image[310, 230] == 255


def test_out_of_bounds_annotation_is_clipped() -> None:
    annotation = DetectedAnnotation("A", (-50, -50, 40, 40), 0.9)
    result = make_service(text_detector=FakeText([annotation])).extract(plan())
    assert result.annotations[0].box == (0, 0, 40, 40)


def test_heuristic_text_fallback_removes_glyph_sized_specks_only() -> None:
    img = plan()
    cv2.putText(img, "8", (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    cfg = {"heuristic_text_fallback": True}
    result = make_service(cfg=cfg).extract(img)
    assert result.cleaned_image[380:410, 298:320].min() > 150
    assert result.wall_mask[300, 500] == 255
    assert "No text detector" not in " ".join(result.warnings)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("3500", AnnotationKind.DIMENSION),
        ("3.5m", AnnotationKind.DIMENSION),
        ("3,5 m", AnnotationKind.DIMENSION),
        ("12'-6\"", AnnotationKind.DIMENSION),
        ("3.5 x 4.0", AnnotationKind.DIMENSION),
        ("BEDROOM", AnnotationKind.LABEL),
        ("Bath 2", AnnotationKind.LABEL),
        ("", AnnotationKind.LABEL),
    ],
)
def test_classify_text(text: str, kind: AnnotationKind) -> None:
    assert classify_text(text) is kind


# -- symbols ------------------------------------------------------------------


def door_symbol(box: tuple[int, int, int, int]) -> DetectedSymbol:
    return DetectedSymbol("door", SymbolCategory.DOOR, box, 0.9)


def test_doors_are_carved_windows_are_not_by_default() -> None:
    door = door_symbol((480, 90, 520, 130))
    window = DetectedSymbol("window", SymbolCategory.WINDOW, (200, 90, 260, 110), 0.8)
    service = make_service(symbol_detector=FakeSymbols([door, window]))
    result = service.extract(plan())
    assert result.wall_mask[100, 500] == 0 and result.opening_mask[100, 500] == 255
    assert result.wall_mask[100, 230] == 255  # window region still wall
    assert len(result.doors) == 1 and len(result.windows) == 1 and not result.openings


def test_carve_windows_flag_and_padding() -> None:
    window = DetectedSymbol("window", SymbolCategory.WINDOW, (200, 95, 260, 105), 0.8)
    cfg = {"carve_windows": True, "carve_padding_px": 5}
    result = make_service(symbol_detector=FakeSymbols([window]), cfg=cfg).extract(plan())
    assert result.wall_mask[100, 230] == 0
    assert result.opening_mask[92, 230] == 255  # padding applied


def test_carve_openings_clips_to_image() -> None:
    mask = np.full((50, 50), 255, np.uint8)
    carved, openings = carve_openings(mask, [(-10, -10, 20, 20), (60, 60, 70, 70)], padding=0)
    assert carved[10, 10] == 0 and carved[30, 30] == 255
    assert openings[10, 10] == 255 and openings.sum() == 255 * 20 * 20


def _door_plan(radius: int = 90) -> np.ndarray:
    img = np.full((1000, 1500), 255, np.uint8)
    hinge = (700, 500)
    cv2.line(img, (100, 500), (hinge[0], 500), 0, 10)  # wall left of door
    cv2.line(img, (hinge[0] + radius, 500), (1400, 500), 0, 10)  # wall right of door
    cv2.ellipse(img, hinge, (radius, radius), 0, 0, 90, 0, 2)  # swing arc
    return img


def test_heuristic_detector_finds_quarter_circle_door_swing() -> None:
    result = make_service(cfg={"heuristic_symbols": True}).extract(_door_plan())
    assert len(result.doors) == 1
    door = result.doors[0]
    assert door.source == "heuristic" and 0.5 <= door.confidence <= 0.9
    x1, y1, x2, y2 = door.box
    assert x1 <= 700 <= x2 and y1 <= 500 <= y2  # box covers the hinge
    assert abs(x2 - 790) < 20  # ...and the far end of the arc
    assert result.wall_mask[500, 740] == 0  # wall gap was carved


def test_heuristic_detector_ignores_plain_walls_blobs_and_full_circles() -> None:
    img = plan()
    cv2.rectangle(img, (300, 300), (360, 360), 0, -1)  # solid square
    cv2.circle(img, (700, 450), 40, 0, 2)  # full circle (e.g. a column), not a swing
    result = make_service(cfg={"heuristic_symbols": True}).extract(img)
    assert result.doors == []


def test_class_map_and_yolo_adapter_with_fake_model() -> None:
    class Model:
        names = {0: "Door_single", 1: "window", 2: "sofa", 3: "opening"}

        def __call__(self, *args: Any, **kwargs: Any) -> list[Any]:
            def box(cls_id: int, conf: float, xyxy: list[float]) -> Any:
                return SimpleNamespace(cls=[cls_id], conf=[conf], xyxy=[xyxy])

            boxes = [
                box(0, 0.9, [10.4, 10.6, 50.2, 80.0]),
                box(1, 0.8, [-20, 5, 40, 30]),
                box(2, 0.99, [0, 0, 5, 5]),  # unmapped class
                box(3, 0.7, [300, 300, 300, 300]),  # degenerate
            ]
            return [SimpleNamespace(boxes=boxes)]

    detector = YoloSymbolDetector(
        None, 0.35, "cpu", PerceptionConfig().symbol_class_map, model=Model()
    )
    symbols = detector.detect(np.full((100, 100), 255, np.uint8), np.zeros((100, 100), np.uint8))
    assert [s.category for s in symbols] == [SymbolCategory.DOOR, SymbolCategory.WINDOW]
    assert symbols[0].box == (10, 11, 50, 80) and all(isinstance(v, int) for v in symbols[0].box)
    assert symbols[1].box == (0, 5, 40, 30)
    assert map_class_name("Sofa", PerceptionConfig().symbol_class_map) is None


def test_missing_model_file_raises(tmp_path: Path) -> None:
    cfg = PerceptionConfig(ocr_mode=OcrMode.OFF, symbol_model_path=str(tmp_path / "nope.pt"))
    with pytest.raises(ModelLoadError):
        PerceptionService(cfg)


def test_missing_ultralytics_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    cfg = PerceptionConfig(ocr_mode=OcrMode.OFF, symbol_model_path=str(weights))
    with pytest.raises(DependencyError):
        PerceptionService(cfg)


# -- OCR availability ---------------------------------------------------------


def test_ocr_required_fails_fast_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "easyocr", None)
    with pytest.raises(DependencyError):
        PerceptionService(PerceptionConfig(ocr_mode=OcrMode.REQUIRED))


def test_ocr_auto_degrades_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "easyocr", None)
    result = PerceptionService(PerceptionConfig(ocr_mode=OcrMode.AUTO)).extract(plan())
    assert any("No text detector" in w for w in result.warnings)
    assert any("No symbol detector" in w for w in result.warnings)


# -- errors and validation ----------------------------------------------------


class Exploding:
    def detect(self, *args: Any) -> list[Any]:
        raise RuntimeError("boom")


def test_detector_failures_are_wrapped() -> None:
    with pytest.raises(InferenceError):
        make_service(text_detector=Exploding()).extract(plan())  # type: ignore[arg-type]
    with pytest.raises(InferenceError):
        make_service(symbol_detector=Exploding()).extract(plan())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        "not an array",
        np.zeros((10, 10), np.float32),
        np.zeros((0, 10), np.uint8),
        np.zeros((10, 10, 4), np.uint8),
        np.zeros((2, 2, 2, 2), np.uint8),
    ],
)
def test_invalid_images_rejected(bad: Any) -> None:
    with pytest.raises(InvalidImageError):
        make_service().extract(bad)


def test_bgr_input_accepted_and_input_not_mutated() -> None:
    gray = plan()
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    before = bgr.copy()
    result = make_service().extract(bgr)
    assert result.image_size == (1000, 700)
    assert np.array_equal(bgr, before)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ocr_min_confidence": 1.5},
        {"ocr_languages": ()},
        {"wall_angles_deg": ()},
        {"heuristic_door_size_ratio": (0.5, 0.1)},
        {"symbol_class_map": (("door", "banana"),)},
        {"carve_padding_px": -1},
        {"device": ""},
    ],
)
def test_invalid_config_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PerceptionConfig(**kwargs)


# -- integration with the ingestion module -----------------------------------


def test_end_to_end_with_ingestion(tmp_path: Path) -> None:
    path = tmp_path / "plan.png"
    cv2.imwrite(str(path), plan(1600, 1100))
    ingested = IngestionService().process(path)
    result = make_service().extract_from_ingestion(ingested)
    assert result.wall_mask.sum() > 0 and result.image_size == (1600, 1100)


def test_vector_ingestion_result_rejected(tmp_path: Path) -> None:
    import ezdxf

    path = tmp_path / "a.dxf"
    doc = ezdxf.new()
    doc.modelspace().add_line((0, 0), (10, 10))
    doc.saveas(str(path))
    with pytest.raises(ValueError, match="raster"):
        make_service().extract_from_ingestion(IngestionService().process(path))
