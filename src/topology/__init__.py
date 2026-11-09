"""Floor-plan topology: wall graph, rooms and openings from a wall mask."""

from .config import TopologyConfig
from .errors import InvalidMaskError, TopologyError
from .models import (
    GraphEdge,
    GraphVertex,
    OpeningInterval,
    OpeningType,
    PlanarGraph,
    RoomFace,
)
from .service import TopologyEngine

__all__ = [
    "GraphEdge",
    "GraphVertex",
    "InvalidMaskError",
    "OpeningInterval",
    "OpeningType",
    "PlanarGraph",
    "RoomFace",
    "TopologyConfig",
    "TopologyEngine",
    "TopologyError",
]
