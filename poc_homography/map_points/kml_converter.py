"""Convert KML geographic points to map points with pixel coordinates."""

from __future__ import annotations

from poc_homography.kml import GeoPointRegistry
from poc_homography.map_points.map_point import MapPoint
from poc_homography.map_points.map_point_registry import MapPointRegistry


def convert_geo_registry_to_map_points(
    geo_registry: GeoPointRegistry, map_id: str
) -> MapPointRegistry:
    """Convert a GeoPointRegistry to MapPointRegistry.

    This extracts only the pixel coordinates and point IDs from a georeferenced
    point registry, discarding all geographic (lat/lon) information.

    Args:
        geo_registry: Source registry with geographic and pixel coordinates.
        map_id: Identifier for the target map (e.g., "map_valte").

    Returns:
        New MapPointRegistry containing only pixel coordinates and IDs.
    """
    map_points: dict[str, MapPoint] = {}

    for point_id, (pixel_point, _kml_point) in geo_registry.points.items():
        map_point = MapPoint(
            id=point_id,
            pixel_x=pixel_point.x,
            pixel_y=pixel_point.y,
            map_id=map_id,
        )
        map_points[point_id] = map_point

    return MapPointRegistry(map_id=map_id, points=map_points)
