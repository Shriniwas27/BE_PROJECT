"""Door/window detection from a multi-class segmentation network.

The same U-Net that segments walls can also output door and window classes. Components of
those class masks become boxes, which is more robust than a swing-arc heuristic and gives
tight boxes along the wall (width = real opening width), unlike a box that includes the swing.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import PerceptionConfig
from .learned import WallSegmentationModel, predict_tiled
from .models import DetectedSymbol, SymbolCategory


class SegmentationSymbolDetector:
    def __init__(
        self,
        door_model: WallSegmentationModel | None,
        window_model: WallSegmentationModel | None,
        config: PerceptionConfig,
    ) -> None:
        self._models = [
            (m, cat, name)
            for m, cat, name in (
                (door_model, SymbolCategory.DOOR, "door"),
                (window_model, SymbolCategory.WINDOW, "window"),
            )
            if m is not None
        ]
        self._cfg = config

    def detect(self, image: np.ndarray, wall_mask: np.ndarray) -> list[DetectedSymbol]:
        cfg = self._cfg
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        min_area = max(12, int(cfg.symbol_seg_min_area_ratio * gray.shape[0] * gray.shape[1]))
        out: list[DetectedSymbol] = []
        for model, category, name in self._models:
            prob = predict_tiled(model, gray, cfg.wall_model_tile_px, cfg.wall_model_overlap_px)
            mask = (prob >= cfg.symbol_seg_threshold).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, count):
                x, y, w, h, area = (int(v) for v in stats[i])
                if area < min_area:
                    continue
                conf = float(prob[labels == i].mean())
                out.append(DetectedSymbol(name, category, (x, y, x + w, y + h), conf, "segmentation"))
        return out
