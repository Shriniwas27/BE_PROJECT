"""Public entry point of the topology package."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from perception import PerceptionResult

from .config import TopologyConfig
from .errors import InvalidMaskError
from .geometry import (
    Segment,
    cluster_endpoints,
    contract_short_diagonals,
    dedupe,
    estimate_dominant_angle,
    key,
    manhattan_snap,
    merge_collinear,
    ortho_regularize,
    planarize,
    rotate_segments,
    seg_length,
    snap_dangling_to_segments,
)
from .models import (
    Box,
    GraphEdge,
    GraphVertex,
    OpeningType,
    PlanarGraph,
)
from .openings import OpeningInput, attach_openings, bridge_openings
from .rooms import extract_rooms
from .skeleton import SkeletonInfo, analyse_mask, extract_segments

logger = logging.getLogger(__name__)

_SAMPLE_STEP_PX = 3.0


class TopologyEngine:
    """Turns a wall mask (plus detected openings) into a planar wall graph and rooms.

    Pipeline: skeletonize -> trace/simplify -> Manhattan snap -> endpoint
    clustering -> axis alignment -> T-junction snapping -> opening bridging ->
    noding -> collinear merge -> rooms -> graph -> opening attachment.

    All coordinates are in the pixel space of the mask.
    """

    def __init__(self, config: TopologyConfig | None = None) -> None:
        self._cfg = config or TopologyConfig()

    # -- public API -------------------------------------------------------

    def build_topology(
        self,
        wall_mask: np.ndarray,
        doors: Sequence[Any] | None = None,
        windows: Sequence[Any] | None = None,
        openings: Sequence[Any] | None = None,
    ) -> PlanarGraph:
        """Build the graph. Symbols may be objects with a ``.box`` or raw (x1,y1,x2,y2) boxes."""
        mask = self._validate(wall_mask)
        inputs = (
            _normalise(doors, OpeningType.DOOR)
            + _normalise(windows, OpeningType.WINDOW)
            + _normalise(openings, OpeningType.OPENING)
        )
        return self._build(mask, inputs)

    def build_from_perception(self, perception: PerceptionResult) -> PlanarGraph:
        """Convenience wrapper for the output of the perception module."""
        return self.build_topology(
            perception.wall_mask,
            doors=perception.doors,
            windows=perception.windows,
            openings=perception.openings,
        )

    # -- pipeline ---------------------------------------------------------

    def _build(self, mask: np.ndarray, openings: list[OpeningInput]) -> PlanarGraph:
        cfg = self._cfg
        warnings: list[str] = []
        mask = _fuse_hollow_walls(mask, cfg.hollow_wall_close_ratio)
        info = analyse_mask(mask)
        if not info.skeleton.any():
            warnings.append("Wall mask is empty; no topology produced.")
            return PlanarGraph({}, [], [], 0.0, warnings=warnings)

        thickness = info.thickness or cfg.default_thickness_px
        snap_tol = max(cfg.min_endpoint_snap_px, cfg.endpoint_snap_ratio * thickness)
        line_tol = max(cfg.min_line_align_px, cfg.line_align_ratio * thickness)
        axis_tol = math.radians(cfg.snap_angle_tol_deg)

        segments = extract_segments(
            info.skeleton, cfg.rdp_epsilon_px, cfg.spur_length_ratio * thickness
        )
        rotation = self._plan_rotation(segments, thickness, mask.shape)
        centre = (mask.shape[1] / 2.0, mask.shape[0] / 2.0)
        if rotation:
            segments = rotate_segments(segments, -rotation, centre)
        angles = tuple(math.radians(a) for a in cfg.snap_angles_deg)
        segments = contract_short_diagonals(
            segments, cfg.chamfer_length_ratio * thickness, angles, axis_tol
        )
        segments = [manhattan_snap(s, angles, axis_tol) for s in segments]
        segments = cluster_endpoints(segments, snap_tol, cfg.min_segment_length_px)
        segments = ortho_regularize(segments, axis_tol, line_tol, snap_tol)
        segments = [s for s in dedupe(segments) if seg_length(s) >= cfg.min_segment_length_px]
        segments = snap_dangling_to_segments(segments, snap_tol)
        if rotation:
            segments = rotate_segments(segments, rotation, centre)

        if cfg.bridge_openings and openings:
            segments, bridged = bridge_openings(segments, openings, thickness, cfg)
            logger.debug("Bridged %d of %d openings", bridged, len(openings))

        pieces = planarize(segments)
        pieces = merge_collinear(pieces, math.radians(cfg.collinear_merge_angle_deg))

        height, width = mask.shape
        rooms, dangles = extract_rooms(pieces, float(height * width), thickness, cfg)
        vertices, edges = self._assemble(pieces, mask, info, thickness)

        degree = Counter(v for e in edges for v in (e.u, e.v))
        dangling = sorted(v for v, d in degree.items() if d == 1)
        if dangles:
            warnings.append(f"{dangles} dangling wall end(s); some rooms may be unclosed.")
        if not rooms:
            warnings.append("No closed rooms were found.")

        unattached = attach_openings(edges, vertices, openings, cfg)
        if unattached:
            warnings.append(f"{len(unattached)} opening(s) could not be matched to a wall.")
        logger.info(
            "Topology: %d vertices, %d edges, %d rooms, thickness %.1fpx",
            len(vertices),
            len(edges),
            len(rooms),
            thickness,
        )
        return PlanarGraph(
            vertices=vertices,
            edges=edges,
            rooms=rooms,
            thickness_px=thickness,
            rotation_deg=math.degrees(rotation),
            dangling_vertices=dangling,
            unattached_openings=[(o.kind, o.box) for o in unattached],
            warnings=warnings,
        )

    def _plan_rotation(
        self, segments: list[Segment], thickness: float, shape: tuple[int, ...]
    ) -> float:
        """Dominant tilt (radians) to regularize against; 0 when negligible or ambiguous."""
        cfg = self._cfg
        if not cfg.deskew:
            return 0.0
        angle, concentration = estimate_dominant_angle(segments, 3.0 * thickness)
        if abs(math.degrees(angle)) < cfg.deskew_min_angle_deg:
            return 0.0
        if concentration < cfg.deskew_min_concentration:
            logger.debug("No dominant orientation (concentration %.2f)", concentration)
            return 0.0
        return angle

    def _assemble(
        self, pieces: list[Segment], mask: np.ndarray, info: SkeletonInfo, thickness: float
    ) -> tuple[dict[int, GraphVertex], list[GraphEdge]]:
        registry: dict[tuple[float, float], int] = {}
        vertices: dict[int, GraphVertex] = {}
        edges: list[GraphEdge] = []

        def vertex_id(point: tuple[float, float]) -> int:
            ident = key(point)
            if ident not in registry:
                registry[ident] = len(registry)
                vertices[registry[ident]] = GraphVertex(registry[ident], point[0], point[1])
            return registry[ident]

        for a, b in pieces:
            support, edge_thickness = _measure_edge(a, b, mask, info.distance, thickness)
            edges.append(
                GraphEdge(
                    id=len(edges),
                    u=vertex_id(a),
                    v=vertex_id(b),
                    length=seg_length((a, b)),
                    thickness=edge_thickness,
                    support=support,
                )
            )
        return vertices, edges

    @staticmethod
    def _validate(mask: object) -> np.ndarray:
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
            raise InvalidMaskError("wall_mask must be a non-empty 2-D numpy array")
        if mask.dtype != np.uint8:
            raise InvalidMaskError(f"wall_mask must be uint8, got {mask.dtype}")
        return mask


def _fuse_hollow_walls(mask: np.ndarray, ratio: float) -> np.ndarray:
    """Fuse the two parallel strokes of double-line walls into one solid wall.

    Directional closing (1xk, then kx1) fills only gaps between parallel axis-aligned strokes.
    Without it each hollow wall skeletonises into two centrelines and produces sliver rooms.
    """
    if ratio <= 0:
        return mask
    k = max(3, round(ratio * max(mask.shape)))
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, k), np.uint8))
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((k, 1), np.uint8))


def _normalise(symbols: Sequence[Any] | None, kind: OpeningType) -> list[OpeningInput]:
    items: list[OpeningInput] = []
    for symbol in symbols or []:
        raw = getattr(symbol, "box", symbol)
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cannot interpret {symbol!r} as a box") from exc
        box: Box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        items.append(OpeningInput(kind, box))
    return items


def _measure_edge(
    a: tuple[float, float],
    b: tuple[float, float],
    mask: np.ndarray,
    distance: np.ndarray,
    fallback: float,
) -> tuple[float, float]:
    """Fraction of the edge on wall pixels, and the local wall thickness."""
    length = seg_length((a, b))
    count = max(2, int(length / _SAMPLE_STEP_PX) + 1)
    xs = np.clip(np.rint(np.linspace(a[0], b[0], count)).astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(np.rint(np.linspace(a[1], b[1], count)).astype(int), 0, mask.shape[0] - 1)
    on_wall = mask[ys, xs] > 0
    support = float(on_wall.mean())
    if on_wall.any():
        thickness = float(max(1.0, 2.0 * np.median(distance[ys, xs][on_wall]) - 1.0))
    else:
        thickness = fallback
    return round(support, 3), round(thickness, 2)
