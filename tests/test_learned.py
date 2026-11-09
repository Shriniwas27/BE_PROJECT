from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from perception import (
    DetectedAnnotation,
    InferenceError,
    ModelLoadError,
    OcrMode,
    OnnxWallModel,
    PerceptionConfig,
    PerceptionService,
    WallFusion,
)
from perception.learned import HybridWallSegmenter, predict_tiled
from perception.text import TiledTextDetector


class ThickInkModel:
    """Pixel-local fake network: 'wall' = thick dark strokes (thin lines are not walls)."""

    def predict(self, tile: np.ndarray) -> np.ndarray:
        dark = (tile < 128).astype(np.uint8)
        thick = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return thick.astype(np.float32)


class BlankModel:
    def predict(self, tile: np.ndarray) -> np.ndarray:
        return np.zeros(tile.shape, np.float32)


def plan_with_dimension_line() -> np.ndarray:
    image = np.full((400, 500), 255, np.uint8)
    cv2.rectangle(image, (60, 60), (440, 300), 0, 10)  # thick walls
    cv2.line(image, (60, 360), (440, 360), 0, 1)  # thin dimension line below the plan
    return image


def tiny_onnx(path: Path, fixed: int | None = None) -> Path:
    """y = sigmoid((0.5 - x) * 20): dark pixels -> wall probability ~1."""
    shape = [1, 1, fixed, fixed] if fixed else [1, 1, "h", "w"]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)
    nodes = [
        helper.make_node("Sub", ["half", "x"], ["d"]),
        helper.make_node("Mul", ["gain", "d"], ["g"]),
        helper.make_node("Sigmoid", ["g"], ["y"]),
    ]
    consts = [
        helper.make_tensor("half", TensorProto.FLOAT, [], [0.5]),
        helper.make_tensor("gain", TensorProto.FLOAT, [], [20.0]),
    ]
    graph = helper.make_graph(nodes, "wall", [x], [y], initializer=consts)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())
    return path


# -- ONNX adapter -------------------------------------------------------------


def test_onnx_model_roundtrip_dynamic_shape(tmp_path: Path) -> None:
    model = OnnxWallModel(str(tiny_onnx(tmp_path / "m.onnx")))
    tile = np.full((96, 128), 255, np.uint8)
    tile[40:50, :] = 0
    probability = model.predict(tile)
    assert probability.shape == tile.shape and probability.dtype == np.float32
    assert probability[45, 10] > 0.99 and probability[10, 10] < 0.01


def test_onnx_model_resizes_for_fixed_input(tmp_path: Path) -> None:
    model = OnnxWallModel(str(tiny_onnx(tmp_path / "fixed.onnx", fixed=64)))
    tile = np.full((200, 300), 255, np.uint8)
    tile[90:110, :] = 0
    probability = model.predict(tile)
    assert probability.shape == (200, 300)  # mapped back to the tile size
    assert probability[100, 150] > 0.9 and probability[20, 150] < 0.1


class FakeSession:
    """Minimal onnxruntime.InferenceSession stand-in."""

    def __init__(self, output: np.ndarray | Exception, shape: list[Any] | None = None) -> None:
        self._output = output
        self._shape = shape or [1, 1, "h", "w"]

    def get_inputs(self) -> list[Any]:
        return [type("Input", (), {"name": "in", "shape": self._shape})()]

    def run(self, _names: Any, _feed: Any) -> list[np.ndarray]:
        if isinstance(self._output, Exception):
            raise self._output
        return [self._output]


def test_multiclass_output_uses_softmax_channel() -> None:
    logits = np.zeros((1, 3, 8, 8), np.float32)
    logits[0, 2] = 10.0  # class 2 dominates
    tile = np.zeros((8, 8), np.uint8)
    wall = OnnxWallModel(None, channel=2, session=FakeSession(logits)).predict(tile)
    other = OnnxWallModel(None, channel=0, session=FakeSession(logits)).predict(tile)
    assert wall.min() > 0.99 and other.max() < 0.01


def test_logits_flag_applies_sigmoid() -> None:
    raw = np.full((1, 1, 4, 4), 5.0, np.float32)
    tile = np.zeros((4, 4), np.uint8)
    assert OnnxWallModel(None, logits=True, session=FakeSession(raw)).predict(tile)[0, 0] > 0.99
    assert OnnxWallModel(None, logits=False, session=FakeSession(raw)).predict(tile)[0, 0] == 1.0


def test_onnx_errors() -> None:
    with pytest.raises(ModelLoadError):
        OnnxWallModel("/no/such/model.onnx")
    with pytest.raises(ModelLoadError):
        OnnxWallModel(None, session=FakeSession(np.zeros(1), shape=[1, 256]))
    with pytest.raises(ModelLoadError):
        OnnxWallModel(None, session=FakeSession(np.zeros(1), shape=[1, 5, "h", "w"]))
    with pytest.raises(InferenceError):
        OnnxWallModel(None, session=FakeSession(RuntimeError("boom"))).predict(
            np.zeros((8, 8), np.uint8)
        )
    with pytest.raises(InferenceError):
        OnnxWallModel(
            None, channel=9, session=FakeSession(np.zeros((1, 3, 4, 4), np.float32))
        ).predict(np.zeros((4, 4), np.uint8))


# -- tiled inference -----------------------------------------------------------


@pytest.mark.parametrize("shape", [(700, 1000), (300, 200), (256, 256), (513, 257)])
def test_tiled_equals_whole_image_prediction(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(0)
    image = np.full(shape, 255, np.uint8)
    for _ in range(12):
        x, y = int(rng.integers(0, shape[1] - 30)), int(rng.integers(0, shape[0] - 30))
        cv2.rectangle(image, (x, y), (x + 25, y + 20), 0, -1)
    model = ThickInkModel()
    tiled = predict_tiled(model, image, tile=256, overlap=32)
    whole = model.predict(image)
    assert tiled.shape == shape
    # Identical except at tile borders where the 5x5 opening sees padding/cut context.
    assert np.mean(np.abs(tiled - whole) > 0.5) < 0.005


def test_tiled_rejects_wrong_output_shape() -> None:
    class Wrong:
        def predict(self, tile: np.ndarray) -> np.ndarray:
            return np.zeros((3, 3), np.float32)

    with pytest.raises(InferenceError):
        predict_tiled(Wrong(), np.zeros((300, 300), np.uint8), tile=128, overlap=16)


# -- fusion --------------------------------------------------------------------


def segment(fusion: WallFusion, model: Any, dilate: int = 4) -> tuple[np.ndarray, list[str]]:
    config = PerceptionConfig(
        wall_fusion=fusion,
        wall_guide_dilate_px=dilate,
        wall_model_tile_px=512,
        wall_min_length_ratio=0.02,
    )
    return HybridWallSegmenter(config, model).segment_with_warnings(plan_with_dimension_line())


def test_classical_only_keeps_dimension_line() -> None:
    mask, warnings = segment(WallFusion.CLASSICAL, ThickInkModel())
    assert mask[360, 100:400].any()  # the thin dimension line is (wrongly) a "wall"
    assert warnings == []


def test_guided_fusion_removes_non_walls_and_keeps_crisp_edges() -> None:
    classical, _ = segment(WallFusion.CLASSICAL, None)
    guided, warnings = segment(WallFusion.GUIDED, ThickInkModel())
    assert not guided[355:365, :].any()  # dimension line gone
    walls = classical[:320] > 0  # everything above the dimension line
    assert walls.any()
    kept = np.count_nonzero(guided[:320][walls]) / np.count_nonzero(walls)
    assert kept > 0.95  # wall pixels survive the gate
    # Guided output is a subset of the classical one, so edges are not blurred or grown.
    assert not np.any((guided > 0) & (classical == 0))
    assert warnings == []


def test_learned_fusion_returns_model_mask() -> None:
    mask, _ = segment(WallFusion.LEARNED, ThickInkModel())
    assert not mask[355:365, :].any()
    assert mask[62:68, 100:400].any()


@pytest.mark.parametrize("fusion", [WallFusion.GUIDED, WallFusion.LEARNED])
def test_blank_model_falls_back_to_classical_with_warning(fusion: WallFusion) -> None:
    classical, _ = segment(WallFusion.CLASSICAL, None)
    mask, warnings = segment(fusion, BlankModel())
    assert np.array_equal(mask, classical)
    assert any("fell back" in w for w in warnings)


def test_service_uses_injected_wall_model() -> None:
    config = PerceptionConfig(ocr_mode=OcrMode.OFF, wall_min_length_ratio=0.02)
    plain = PerceptionService(config).extract(plan_with_dimension_line())
    guided = PerceptionService(config, wall_model=ThickInkModel()).extract(
        plan_with_dimension_line()
    )
    assert plain.wall_mask[360, 100:400].any()
    assert not guided.wall_mask[355:365, :].any()


def test_service_loads_model_from_config(tmp_path: Path) -> None:
    path = tiny_onnx(tmp_path / "wall.onnx")
    config = PerceptionConfig(
        ocr_mode=OcrMode.OFF, wall_model_path=str(path), wall_min_length_ratio=0.02
    )
    result = PerceptionService(config).extract(plan_with_dimension_line())
    # The tiny model flags every dark pixel, so nothing is gated out, but it ran end to end.
    assert result.wall_mask[62:68, 100:400].any()
    with pytest.raises(ModelLoadError):
        PerceptionService(PerceptionConfig(ocr_mode=OcrMode.OFF, wall_model_path="/missing.onnx"))


# -- tiled OCR -----------------------------------------------------------------


class BlobTextDetector:
    """Fake OCR: every dark connected blob is one 'word'; sees only the image it is given."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[DetectedAnnotation]:
        self.calls += 1
        count, _, stats, _ = cv2.connectedComponentsWithStats((image < 128).astype(np.uint8))
        found = []
        for i in range(1, count):
            x, y, w, h, _area = (int(v) for v in stats[i])
            found.append(DetectedAnnotation(f"{w}x{h}", (x, y, x + w, y + h), 0.9))
        return found


def test_tiled_ocr_finds_each_word_once_with_absolute_coordinates() -> None:
    image = np.full((1000, 1000), 255, np.uint8)
    words = [(20, 20), (390, 100), (700, 300), (395, 395), (880, 900), (500, 610)]
    for x, y in words:
        cv2.rectangle(image, (x, y), (x + 40, y + 12), 0, -1)  # (390,100) straddles a tile edge
    inner = BlobTextDetector()
    found = TiledTextDetector(inner, tile=400, overlap=150).detect(image)
    assert inner.calls > 1
    assert len(found) == len(words)
    assert sorted((a.box[0], a.box[1]) for a in found) == sorted(words)
    assert all(a.text == "41x13" for a in found)


def test_tiled_ocr_small_image_uses_single_pass() -> None:
    inner = BlobTextDetector()
    TiledTextDetector(inner, tile=400, overlap=100).detect(np.full((300, 300), 255, np.uint8))
    assert inner.calls == 1
    with pytest.raises(ValueError):
        TiledTextDetector(inner, tile=100, overlap=100)


def test_service_wraps_text_detector_when_tiling_enabled() -> None:
    image = np.full((900, 900), 255, np.uint8)
    cv2.rectangle(image, (390, 100), (430, 112), 0, -1)
    inner = BlobTextDetector()
    config = PerceptionConfig(ocr_tile_px=400, ocr_tile_overlap_px=150, ocr_min_confidence=0.1)
    result = PerceptionService(config, text_detector=inner).extract(image)
    assert inner.calls > 1
    assert len(result.annotations) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wall_model_threshold": 0.0},
        {"wall_model_threshold": 1.0},
        {"wall_model_tile_px": 32},
        {"wall_model_overlap_px": 300},
        {"wall_model_channel": -1},
        {"wall_guide_dilate_px": -1},
        {"ocr_tile_px": 100},
        {"ocr_tile_px": 512, "ocr_tile_overlap_px": 400},
    ],
)
def test_new_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PerceptionConfig(**kwargs)
