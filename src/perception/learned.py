"""Learned wall segmentation and its fusion with the classical segmenter.

A trained network (e.g. a U-Net fine-tuned on CubiCasa5k) is far better than
morphology at deciding *what is a wall* (versus dimension lines, furniture,
hatching, title blocks) but its mask edges are soft. The default ``GUIDED``
fusion therefore uses the network only as a semantic gate: classical strokes are
kept where the network says "wall", so edges stay pixel-exact for the topology
stage.

Model contract for :class:`OnnxWallModel`
-----------------------------------------
* Input: ``float32`` ``[1, C, H, W]`` with ``C`` in {1, 3}, values in ``[0, 1]``
  (``0`` = black ink, ``1`` = white paper). Bake any mean/std normalisation into
  the exported graph. ``H`` and ``W`` may be fixed or dynamic.
* Output: ``[1, 1, H, W]`` / ``[1, H, W]`` wall probability, or logits when
  ``logits=True``; or ``[1, K, H, W]`` class scores, softmaxed, with the wall
  class at index ``channel``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .config import PerceptionConfig, WallFusion
from .errors import DependencyError, InferenceError, ModelLoadError
from .imageops import remove_small_components
from .walls import WallSegmenter

logger = logging.getLogger(__name__)

_MIN_COVERAGE = 1e-4  # a model mask covering less than this fraction is treated as "found nothing"
_WEIGHT_EPSILON = 1e-6


class WallSegmentationModel(Protocol):
    """Anything that maps a grayscale tile to a per-pixel wall probability."""

    def predict(self, tile: np.ndarray) -> np.ndarray:
        """``uint8`` (H, W) in, ``float32`` (H, W) in [0, 1] out."""
        ...


class OnnxWallModel:
    """ONNX Runtime adapter. The model is loaded once, at construction."""

    def __init__(
        self,
        model_path: str | None,
        device: str = "cpu",
        channel: int = 0,
        logits: bool = False,
        *,
        session: Any | None = None,
    ) -> None:
        self._channel = channel
        self._logits = logits
        if session is None:
            session = _load_session(model_path, device)
        self._session = session
        inp = session.get_inputs()[0]
        shape = list(inp.shape)
        if len(shape) != 4:
            raise ModelLoadError(f"Wall model input must be 4-D NCHW, got shape {shape}")
        self._input_name: str = inp.name
        self._channels = shape[1] if isinstance(shape[1], int) else 1
        if self._channels not in (1, 3):
            raise ModelLoadError(f"Wall model must take 1 or 3 channels, got {self._channels}")
        self._fixed_height = shape[2] if isinstance(shape[2], int) else None
        self._fixed_width = shape[3] if isinstance(shape[3], int) else None

    def predict(self, tile: np.ndarray) -> np.ndarray:
        height, width = tile.shape[:2]
        resized = self._fit_to_input(tile)
        data = resized.astype(np.float32) / 255.0
        batch = np.repeat(data[None, None], self._channels, axis=1)
        try:
            output = np.asarray(self._session.run(None, {self._input_name: batch})[0])
        except Exception as exc:
            raise InferenceError(f"Wall model inference failed: {exc}") from exc

        probability = self._to_probability(output)
        if probability.shape != (height, width):
            probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
        clipped: np.ndarray = np.clip(probability, 0.0, 1.0).astype(np.float32)
        return clipped

    def _fit_to_input(self, tile: np.ndarray) -> np.ndarray:
        """Resize to the network's fixed input size (no-op for dynamic shapes)."""
        height, width = tile.shape[:2]
        target_h, target_w = self._fixed_height, self._fixed_width
        if not target_h or not target_w or (height, width) == (target_h, target_w):
            return tile
        interpolation = cv2.INTER_AREA if height * width > target_h * target_w else cv2.INTER_LINEAR
        resized: np.ndarray = cv2.resize(tile, (target_w, target_h), interpolation=interpolation)
        return resized

    def _to_probability(self, output: np.ndarray) -> np.ndarray:
        if output.ndim == 3:  # [1, H, W]
            output = output[:, None]
        if output.ndim != 4 or output.shape[0] != 1:
            raise InferenceError(f"Unexpected wall model output shape {output.shape}")
        scores = output[0]
        if scores.shape[0] == 1:
            plane = scores[0].astype(np.float32)
            return _sigmoid(plane) if self._logits else plane
        if not 0 <= self._channel < scores.shape[0]:
            raise InferenceError(f"wall_model_channel {self._channel} out of range")
        shifted = scores - scores.max(axis=0, keepdims=True)
        exp = np.exp(shifted)
        softmax: np.ndarray = exp[self._channel] / exp.sum(axis=0)
        return softmax.astype(np.float32)


def predict_tiled(
    model: WallSegmentationModel, gray: np.ndarray, tile: int, overlap: int
) -> np.ndarray:
    """Probability map of ``gray`` by overlapping tiles blended with a smooth window.

    Tiling keeps memory bounded and runs the network at native resolution
    (the scale it was trained for); the window removes seams between tiles.
    """
    height, width = gray.shape[:2]
    pad_h, pad_w = max(0, tile - height), max(0, tile - width)
    padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255)
    full_h, full_w = padded.shape
    step = tile - overlap

    window_1d = np.hanning(tile + 2)[1:-1].astype(np.float32)
    window = np.outer(window_1d, window_1d)
    accumulated = np.zeros((full_h, full_w), np.float32)
    weights = np.zeros((full_h, full_w), np.float32)
    for y in _starts(full_h, tile, step):
        for x in _starts(full_w, tile, step):
            patch = padded[y : y + tile, x : x + tile]
            probability = model.predict(patch)
            if probability.shape != patch.shape:
                raise InferenceError(f"Model returned {probability.shape} for a {patch.shape} tile")
            accumulated[y : y + tile, x : x + tile] += probability * window
            weights[y : y + tile, x : x + tile] += window
    blended: np.ndarray = accumulated / np.maximum(weights, _WEIGHT_EPSILON)
    return blended[:height, :width]


class HybridWallSegmenter:
    """Wall stage: classical, learned, or learned-guided classical segmentation."""

    def __init__(
        self,
        config: PerceptionConfig,
        model: WallSegmentationModel | None = None,
        classical: WallSegmenter | None = None,
    ) -> None:
        self._cfg = config
        self._model = model
        self._classical = classical or WallSegmenter(config)

    def segment(self, gray: np.ndarray) -> np.ndarray:
        mask, _ = self.segment_with_warnings(gray)
        return mask

    def segment_with_warnings(self, gray: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Return (wall_mask, warnings) for a dark-on-light grayscale image."""
        cfg = self._cfg
        if self._model is None or cfg.wall_fusion is WallFusion.CLASSICAL:
            return self._classical.segment(gray), []

        learned = self._learned_mask(gray)
        coverage = float(np.count_nonzero(learned)) / learned.size
        if coverage < _MIN_COVERAGE:
            message = "Wall model found (almost) no walls; fell back to classical segmentation."
            logger.warning(message)
            return self._classical.segment(gray), [message]

        if cfg.wall_fusion is WallFusion.LEARNED:
            return learned, []

        guide = _dilate(learned, cfg.wall_guide_dilate_px)
        fused = cv2.bitwise_and(self._classical.segment(gray), guide)
        min_area = int(cfg.min_component_area_ratio * gray.shape[0] * gray.shape[1])
        return remove_small_components(fused, min_area), []

    def _learned_mask(self, gray: np.ndarray) -> np.ndarray:
        assert self._model is not None
        cfg = self._cfg
        probability = predict_tiled(
            self._model, gray, cfg.wall_model_tile_px, cfg.wall_model_overlap_px
        )
        mask = (probability >= cfg.wall_model_threshold).astype(np.uint8) * 255
        min_area = int(cfg.min_component_area_ratio * gray.shape[0] * gray.shape[1])
        return remove_small_components(mask, min_area)


# -- helpers ------------------------------------------------------------------


def _load_session(model_path: str | None, device: str) -> Any:
    if model_path is None or not Path(model_path).is_file():
        raise ModelLoadError(f"Wall model not found: {model_path}")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise DependencyError(
            "onnxruntime is not installed. Install it with: pip install onnxruntime"
        ) from exc
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.startswith("cuda")
        else ["CPUExecutionProvider"]
    )
    try:
        return ort.InferenceSession(model_path, providers=providers)
    except Exception as exc:
        raise ModelLoadError(f"Could not load wall model {model_path}: {exc}") from exc


def _starts(total: int, tile: int, step: int) -> list[int]:
    """Tile origins covering ``total`` pixels; the last tile is flush with the end."""
    starts = list(range(0, max(total - tile, 0) + 1, step))
    if starts[-1] + tile < total:
        starts.append(total - tile)
    return starts


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result: np.ndarray = 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))
    return result


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    size = 2 * radius + 1
    grown: np.ndarray = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return grown
