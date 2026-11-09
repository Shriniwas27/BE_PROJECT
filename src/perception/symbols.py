"""Door / window / opening detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .config import PerceptionConfig
from .errors import DependencyError, InferenceError, ModelLoadError
from .imageops import binarize
from .models import DetectedSymbol, SymbolCategory

logger = logging.getLogger(__name__)

_INLIER_TOLERANCE_FRACTION = 0.05  # of radius
_INLIER_TOLERANCE_MIN_PX = 1.5
_MIN_INLIER_FRACTION = 0.85
_SPAN_RANGE_DEG = (45.0, 135.0)  # a door swing is ~90 deg; ends get trimmed by wall removal
_MIN_COMPONENT_PIXELS = 20
_REFINE_ITERATIONS = 20
_NEIGHBOUR_REACH = 2.0  # in radii
_MAX_DOOR_NEIGHBOURS = 1
_BOX_MARGIN_FRACTION = 0.1  # of radius, around arc + hinge


class SymbolDetector(Protocol):
    """Anything that can find door/window/opening symbols.

    ``wall_mask`` is the raw wall segmentation, supplied for detectors that
    can use it (the heuristic one); neural detectors may ignore it.
    """

    def detect(self, image: np.ndarray, wall_mask: np.ndarray) -> list[DetectedSymbol]: ...


def map_class_name(name: str, class_map: tuple[tuple[str, str], ...]) -> SymbolCategory | None:
    lowered = name.lower()
    for substring, category in class_map:
        if substring in lowered:
            return SymbolCategory(category)
    return None


class YoloSymbolDetector:
    """Ultralytics YOLO adapter trained on architectural symbols."""

    def __init__(
        self,
        model_path: str | None,
        confidence: float,
        device: str,
        class_map: tuple[tuple[str, str], ...],
        *,
        model: Any | None = None,
    ) -> None:
        self._confidence = confidence
        self._device = device
        self._class_map = class_map
        if model is not None:  # injected (tests / custom runtimes)
            self._model = model
            return
        if model_path is None or not Path(model_path).is_file():
            raise ModelLoadError(f"Symbol model not found: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DependencyError(
                "ultralytics is not installed. Install it with: pip install ultralytics"
            ) from exc
        try:
            self._model = YOLO(model_path)
        except Exception as exc:
            raise ModelLoadError(f"Could not load model {model_path}: {exc}") from exc

    def detect(self, image: np.ndarray, wall_mask: np.ndarray) -> list[DetectedSymbol]:
        height, width = image.shape[:2]
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        try:
            result = self._model(bgr, conf=self._confidence, device=self._device, verbose=False)[0]
        except Exception as exc:
            raise InferenceError(f"Symbol detection failed: {exc}") from exc

        names = self._model.names
        symbols: list[DetectedSymbol] = []
        for box in result.boxes:
            label = str(names[int(box.cls[0])])
            category = map_class_name(label, self._class_map)
            if category is None:
                continue
            x1, y1, x2, y2 = (int(round(float(v))) for v in box.xyxy[0])
            clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            symbols.append(
                DetectedSymbol(label, category, clipped, float(box.conf[0]), source="model")
            )
        return symbols


class HeuristicDoorDetector:
    """Experimental quarter-circle door-swing detector (no model required).

    Removes straight wall lines from the ink mask, then fits a circle to each
    remaining component. Components that lie on a circle of plausible radius
    and span roughly a quarter turn are reported as doors; the box covers the
    arc and its hinge so carving it opens the wall. Expect misses and false
    positives on complex drawings; prefer a trained model for production use.
    """

    def __init__(self, config: PerceptionConfig) -> None:
        self._cfg = config

    def detect(self, image: np.ndarray, wall_mask: np.ndarray) -> list[DetectedSymbol]:
        height, width = image.shape[:2]
        longest = max(height, width)
        min_radius = self._cfg.heuristic_door_size_ratio[0] * longest
        max_radius = self._cfg.heuristic_door_size_ratio[1] * longest

        walls = cv2.dilate(wall_mask, np.ones((3, 3), np.uint8))
        residual = cv2.bitwise_and(binarize(image, self._cfg), cv2.bitwise_not(walls))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)

        arcs: list[_Arc] = []
        for index in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[index])
            if area < _MIN_COMPONENT_PIXELS or max(w, h) > 2.5 * max_radius:
                continue
            ys, xs = np.nonzero(labels[y : y + h, x : x + w] == index)
            points = np.stack([xs + x, ys + y], axis=1).astype(np.float64)
            fit = _fit_arc(points)
            if fit is None:
                continue
            cx, cy, radius, inliers = fit
            if not min_radius <= radius <= max_radius or len(points) < radius:
                continue
            margin = _BOX_MARGIN_FRACTION * radius  # the fitted hinge is only approximate
            box = (
                max(0, int(min(x, cx) - margin)),
                max(0, int(min(y, cy) - margin)),
                min(width, int(max(x + w, cx) + margin) + 1),
                min(height, int(max(y + h, cy) + margin) + 1),
            )
            arcs.append(_Arc((x + w / 2.0, y + h / 2.0), radius, inliers, box))

        return [
            DetectedSymbol(
                "door_swing",
                SymbolCategory.DOOR,
                arc.box,
                float(min(arc.inliers, 0.9)),
                "heuristic",
            )
            for arc in arcs
            if not _is_circle_fragment(arc, arcs)
        ]


@dataclass(frozen=True)
class _Arc:
    centre: tuple[float, float]  # middle of the arc's bounding box
    radius: float
    inliers: float
    box: tuple[int, int, int, int]


def _is_circle_fragment(arc: _Arc, arcs: list[_Arc]) -> bool:
    """A full circle (column, fixture) is cut into several neighbouring arcs by wall
    removal. A real door has at most one neighbour (a double door), so arcs with
    two or more close neighbours are rejected."""
    reach = _NEIGHBOUR_REACH * arc.radius
    neighbours = sum(
        1
        for other in arcs
        if other is not arc
        and np.hypot(arc.centre[0] - other.centre[0], arc.centre[1] - other.centre[1]) < reach
    )
    return neighbours >= _MAX_DOOR_NEIGHBOURS + 1


def _refine_circle(
    xs: np.ndarray, ys: np.ndarray, cx: float, cy: float, radius: float
) -> tuple[float, float, float]:
    """Gauss-Newton minimisation of geometric error; the algebraic fit is biased on short arcs."""
    for _ in range(_REFINE_ITERATIONS):
        dx, dy = xs - cx, ys - cy
        dist = np.hypot(dx, dy)
        if np.any(dist < 1e-9):
            break
        jacobian = np.stack([-dx / dist, -dy / dist, -np.ones_like(dist)], axis=1)
        step, *_ = np.linalg.lstsq(jacobian, -(dist - radius), rcond=None)
        cx, cy, radius = cx + step[0], cy + step[1], radius + step[2]
        if np.abs(step).max() < 1e-3:
            break
    return cx, cy, abs(radius)


def _fit_arc(points: np.ndarray) -> tuple[float, float, float, float] | None:
    """Least-squares circle fit (Kasa). Returns (cx, cy, radius, inlier_fraction).

    Returns None unless the points hug a circle and cover a quarter-ish arc.
    """
    xs, ys = points[:, 0], points[:, 1]
    design = np.stack([2 * xs, 2 * ys, np.ones_like(xs)], axis=1)
    target = xs**2 + ys**2
    try:
        (cx, cy, c), *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    radius_sq = c + cx**2 + cy**2
    if radius_sq <= 0:
        return None
    cx, cy, radius = _refine_circle(xs, ys, float(cx), float(cy), float(np.sqrt(radius_sq)))

    distances = np.hypot(xs - cx, ys - cy)
    tolerance = max(_INLIER_TOLERANCE_MIN_PX, _INLIER_TOLERANCE_FRACTION * radius)
    inliers = float(np.mean(np.abs(distances - radius) <= tolerance))
    if inliers < _MIN_INLIER_FRACTION:
        return None

    angles = np.sort(np.degrees(np.arctan2(ys - cy, xs - cx)))
    gaps = np.diff(np.concatenate([angles, [angles[0] + 360.0]]))
    span = 360.0 - float(gaps.max())
    if not _SPAN_RANGE_DEG[0] <= span <= _SPAN_RANGE_DEG[1]:
        return None
    return float(cx), float(cy), radius, inliers
