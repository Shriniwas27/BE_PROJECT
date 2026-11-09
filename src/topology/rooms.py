"""Room extraction from the noded wall network."""

from __future__ import annotations

from shapely import LineString, Polygon
from shapely.ops import polygonize_full

from .config import TopologyConfig
from .geometry import Segment
from .models import RoomFace

_MITRE = 2  # shapely join_style for mitred offsets


def extract_rooms(
    pieces: list[Segment], image_area: float, thickness: float, config: TopologyConfig
) -> tuple[list[RoomFace], int]:
    """Return (rooms, number_of_dangling_wall_ends).

    Faces are bounded cycles of the wall network. Tiny faces and faces thinner
    than ``min_room_width_ratio`` x thickness (slivers inside double-line
    walls) are discarded.
    """
    if not pieces:
        return [], 0
    faces, _cuts, dangles, _invalid = polygonize_full([LineString(p) for p in pieces])

    min_area = config.min_room_area_ratio * image_area
    min_width = max(config.min_room_width_ratio * thickness, config.min_room_width_px)
    kept: list[Polygon] = []
    for poly in faces.geoms:
        if not poly.is_valid or poly.area < min_area:
            continue
        if poly.buffer(-min_width / 2.0, join_style=_MITRE).is_empty:
            continue
        kept.append(poly)
    kept.sort(key=lambda p: (round(p.centroid.y, 1), round(p.centroid.x, 1)))

    rooms = [_to_room(i, poly, thickness, config) for i, poly in enumerate(kept)]
    return rooms, len(dangles.geoms)


def _to_room(index: int, poly: Polygon, thickness: float, config: TopologyConfig) -> RoomFace:
    interior_ring = None
    interior_area = 0.0
    if config.compute_interior:
        inset = poly.buffer(-thickness / 2.0, join_style=_MITRE)
        if not inset.is_empty:
            largest = max(getattr(inset, "geoms", [inset]), key=lambda g: g.area)
            interior_ring = [(float(x), float(y)) for x, y in largest.exterior.coords]
            interior_area = float(inset.area)
    centroid = poly.centroid
    return RoomFace(
        id=index,
        polygon=[(float(x), float(y)) for x, y in poly.exterior.coords],
        area_px=float(poly.area),
        centroid=(float(centroid.x), float(centroid.y)),
        holes=[[(float(x), float(y)) for x, y in ring.coords] for ring in poly.interiors],
        interior_polygon=interior_ring,
        interior_area_px=interior_area,
    )
