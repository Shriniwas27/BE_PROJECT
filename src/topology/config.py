"""Configuration for the topology pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopologyConfig:
    """Immutable, validated settings for :class:`topology.TopologyEngine`.

    Tolerances are expressed as multiples of the wall thickness measured from
    the mask, so they adapt to image resolution and drawing style. ``*_px``
    values are absolute floors/fallbacks.

    Geometry:
        deskew: Estimate the plan's dominant orientation (from long segments)
            and regularize in that frame, so a slightly rotated scan still
            yields straight, connected walls. Edges are returned in the image
            frame; ``PlanarGraph.rotation_deg`` reports the tilt.
        deskew_min_angle_deg: Tilts smaller than this are ignored.
        deskew_min_concentration: Minimum agreement (0-1) between segment
            orientations for a dominant frame to be trusted.
        snap_angles_deg: Orientations segments may be snapped to. Add 45 and
            135 for diagonal walls.
        snap_angle_tol_deg: Max deviation from a snap angle to force snapping.
        rdp_epsilon_px: Polyline simplification tolerance for skeleton paths.
        endpoint_snap_ratio / min_endpoint_snap_px: Endpoints closer than this
            are merged; dangling ends this close to another wall attach to it.
        line_align_ratio / min_line_align_px: Parallel axis-aligned walls whose
            centrelines are closer than this are moved onto one common line.
        spur_length_ratio: Skeleton branches shorter than this (x thickness)
            that end freely are treated as noise and removed.
        chamfer_length_ratio: Short off-axis pieces (x thickness) left by
            skeletonizing thick corners are collapsed into a point.
        min_segment_length_px: Segments shorter than this are dropped.
        collinear_merge_angle_deg: Degree-2 junctions straighter than this are
            merged into a single edge.
        default_thickness_px: Used only if thickness cannot be measured.

    Openings:
        bridge_openings: Reconnect wall ends across detected door/opening gaps
            so rooms close; the bridge becomes the host edge of the opening.
        bridge_angle_tol_deg / bridge_offset_ratio: How collinear the two wall
            ends must be (angle, and lateral offset x thickness).
        opening_max_distance_ratio: Max distance (x thickness) between an
            opening box and its host wall.

    Rooms:
        min_room_area_ratio: Faces smaller than this fraction of the image
            are discarded.
        min_room_width_ratio: Faces narrower than this (x thickness) are
            discarded; removes the slivers inside double-line walls.
        min_room_width_px: Absolute floor for the width test. Set it for
            double-line (outline) walls, where the measured stroke thickness is
            much smaller than the wall itself.
        compute_interior: Also compute each room inset by half a wall
            thickness (the usable floor area).
    """

    deskew: bool = True
    deskew_min_angle_deg: float = 0.25
    deskew_min_concentration: float = 0.5
    hollow_wall_close_ratio: float = 0.01  # fuse double-line walls before skeletonising; 0 = off
    snap_angles_deg: tuple[float, ...] = (0.0, 90.0)
    snap_angle_tol_deg: float = 6.0
    rdp_epsilon_px: float = 2.0
    endpoint_snap_ratio: float = 1.0
    min_endpoint_snap_px: float = 4.0
    line_align_ratio: float = 0.5
    min_line_align_px: float = 2.0
    spur_length_ratio: float = 1.0
    chamfer_length_ratio: float = 2.5
    min_segment_length_px: float = 3.0
    collinear_merge_angle_deg: float = 1.5
    default_thickness_px: float = 12.0

    bridge_openings: bool = True
    bridge_angle_tol_deg: float = 15.0
    bridge_offset_ratio: float = 1.0
    opening_max_distance_ratio: float = 1.5

    min_room_area_ratio: float = 5e-4
    min_room_width_ratio: float = 2.0
    min_room_width_px: float = 0.0
    compute_interior: bool = True

    def __post_init__(self) -> None:
        if not self.snap_angles_deg:
            raise ValueError("snap_angles_deg must not be empty")
        positive = (
            "snap_angle_tol_deg",
            "deskew_min_angle_deg",
            "rdp_epsilon_px",
            "endpoint_snap_ratio",
            "min_endpoint_snap_px",
            "line_align_ratio",
            "min_line_align_px",
            "spur_length_ratio",
            "chamfer_length_ratio",
            "min_segment_length_px",
            "collinear_merge_angle_deg",
            "default_thickness_px",
            "bridge_angle_tol_deg",
            "bridge_offset_ratio",
            "opening_max_distance_ratio",
            "min_room_width_ratio",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.hollow_wall_close_ratio < 0.1:
            raise ValueError("hollow_wall_close_ratio must be within [0, 0.1)")
        if not 0.0 <= self.deskew_min_concentration <= 1.0:
            raise ValueError("deskew_min_concentration must be within [0, 1]")
        if self.min_room_width_px < 0:
            raise ValueError("min_room_width_px must be >= 0")
        if self.min_room_area_ratio < 0:
            raise ValueError("min_room_area_ratio must be >= 0")
        if self.snap_angle_tol_deg >= 45:
            raise ValueError("snap_angle_tol_deg must be < 45")
