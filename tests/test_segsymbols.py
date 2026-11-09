from __future__ import annotations

import numpy as np
import pytest

from perception import PerceptionConfig
from perception.segsymbols import SegmentationSymbolDetector


class _Blob:
    def predict(self, tile: np.ndarray) -> np.ndarray:
        m = np.zeros(tile.shape, np.float32)
        m[100:120, 50:90] = 1.0
        return m


def test_segmentation_detector_returns_tight_box() -> None:
    img = np.full((300, 400), 255, np.uint8)
    found = SegmentationSymbolDetector(_Blob(), None, PerceptionConfig()).detect(img, img)
    assert [(s.category.value, s.box) for s in found] == [("door", (50, 100, 90, 120))]


def test_bad_channel_rejected() -> None:
    with pytest.raises(ValueError):
        PerceptionConfig(wall_model_door_channel=-1)
