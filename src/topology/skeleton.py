"""Wall centreline extraction: skeletonize, trace branches, simplify."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .geometry import Segment

_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_MIN_BRANCH_PIXELS = 2
_PEAK_WINDOW = 5


@dataclass(frozen=True)
class SkeletonInfo:
    skeleton: np.ndarray  # bool, 1-px wide
    distance: np.ndarray  # distance transform, max-filtered (for sampling thickness)
    thickness: float  # median wall thickness in px (0 if no walls)


@dataclass(frozen=True)
class _Branch:
    points: list[tuple[float, float]]  # ordered (x, y)
    free_start: bool  # start is a true skeleton endpoint
    free_end: bool


def analyse_mask(mask: np.ndarray) -> SkeletonInfo:
    """Skeletonize the mask and measure wall thickness from the distance transform."""
    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary.astype(bool))
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if not skeleton.any():
        return SkeletonInfo(skeleton, distance, 0.0)
    counts = _neighbour_count(skeleton)
    on_line = skeleton & (counts == 2)  # ignore ends and junctions, where the transform is biased
    samples = distance[on_line] if on_line.any() else distance[skeleton]
    thickness = float(max(1.0, 2.0 * np.median(samples) - 1.0))
    # A skeleton can sit a pixel or two off-centre; a local maximum is robust to that.
    peak = cv2.dilate(distance, np.ones((_PEAK_WINDOW, _PEAK_WINDOW), np.uint8))
    return SkeletonInfo(skeleton, peak, thickness)


def extract_segments(skeleton: np.ndarray, rdp_epsilon: float, spur_length: float) -> list[Segment]:
    """Trace the skeleton into straight segments, removing short free-ended spurs."""
    segments: list[Segment] = []
    for branch in _trace(skeleton):
        vertices = cv2.approxPolyDP(
            np.array(branch.points, dtype=np.float32).reshape(-1, 1, 2), rdp_epsilon, False
        ).reshape(-1, 2)
        if len(vertices) < 2:
            continue
        length = float(np.sum(np.hypot(*np.diff(vertices, axis=0).T)))
        free_ends = int(branch.free_start) + int(branch.free_end)
        if free_ends >= 1 and length < spur_length:
            continue
        for p, q in pairwise(vertices):
            if p[0] != q[0] or p[1] != q[1]:
                segments.append(((float(p[0]), float(p[1])), (float(q[0]), float(q[1]))))
    return segments


def _neighbour_count(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = 0
    sk = skeleton.astype(np.float32)
    counts = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return (counts * sk).astype(np.int32)


def _trace(skeleton: np.ndarray) -> list[_Branch]:
    counts = _neighbour_count(skeleton)
    junction = skeleton & (counts >= 3)
    free_end = skeleton & (counts <= 1)
    branch_px = skeleton & ~junction

    _, j_labels, _, j_centroids = cv2.connectedComponentsWithStats(
        junction.astype(np.uint8), connectivity=8
    )
    b_count, b_labels = cv2.connectedComponents(branch_px.astype(np.uint8), connectivity=8)

    ys, xs = np.nonzero(branch_px)
    labels = b_labels[ys, xs]
    order = np.argsort(labels, kind="stable")
    ys, xs, labels = ys[order], xs[order], labels[order]
    boundaries = np.flatnonzero(np.diff(labels)) + 1

    height, width = skeleton.shape
    branches: list[_Branch] = []
    for group_y, group_x in zip(np.split(ys, boundaries), np.split(xs, boundaries), strict=True):
        if len(group_y) < _MIN_BRANCH_PIXELS:
            continue
        pixels = set(zip(group_y.tolist(), group_x.tolist(), strict=True))
        path = _walk(pixels, free_end)
        if len(path) < _MIN_BRANCH_PIXELS:
            continue
        start_free = bool(free_end[path[0]])
        end_free = bool(free_end[path[-1]])
        points = [(float(x), float(y)) for y, x in path]
        if not start_free:
            hit = _adjacent_junction(path[0], j_labels, j_centroids, height, width)
            if hit is not None:
                points.insert(0, hit)
        if not end_free:
            hit = _adjacent_junction(path[-1], j_labels, j_centroids, height, width)
            if hit is not None:
                points.append(hit)
        branches.append(_Branch(points, start_free, end_free))
    return branches


def _walk(pixels: set[tuple[int, int]], free_end: np.ndarray) -> list[tuple[int, int]]:
    """Order the pixels of one branch, starting from a free end when there is one."""
    start = next((p for p in sorted(pixels) if free_end[p]), None)
    if start is None:
        start = min(
            pixels,
            key=lambda p: sum((p[0] + dy, p[1] + dx) in pixels for dy, dx in _NEIGHBOURS),
        )
    path = [start]
    visited = {start}
    current = start
    while True:
        candidates = [
            (current[0] + dy, current[1] + dx)
            for dy, dx in _NEIGHBOURS
            if (current[0] + dy, current[1] + dx) in pixels
            and (current[0] + dy, current[1] + dx) not in visited
        ]
        if not candidates:
            break
        # prefer 4-connected steps for a cleaner path
        candidates.sort(key=lambda p: abs(p[0] - current[0]) + abs(p[1] - current[1]))
        current = candidates[0]
        visited.add(current)
        path.append(current)
    return path


def _adjacent_junction(
    pixel: tuple[int, int], labels: np.ndarray, centroids: np.ndarray, height: int, width: int
) -> tuple[float, float] | None:
    y, x = pixel
    window = labels[max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)]
    found = window[window > 0]
    if found.size == 0:
        return None
    label = int(np.argmax(np.bincount(found)))
    return float(centroids[label][0]), float(centroids[label][1])
