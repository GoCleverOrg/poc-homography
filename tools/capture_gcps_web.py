#!/usr/bin/env python3
"""
DEPRECATED: Web-based GCP Capture Tool - No longer supported after MapPoint migration.

This tool was designed for GPS-based ground control point capture, which has been
replaced by the MapPoint system that uses map-relative pixel coordinates instead of
GPS coordinates (latitude/longitude).

The MapPoint migration (issue #156) removed the following GPS-dependent components
that this tool relied on:
- WorldPoint (GPS coordinates)
- MapCoordinate
- GPSPositionMixin
- HomographyProviderExtended
- gps_to_local_xy() and local_xy_to_gps() functions

Replacement workflow:
--------------------
Instead of capturing GPS coordinates and projecting them to images, the new workflow is:

1. Use a map image with reference points marked at known pixel coordinates
2. Create a MapPoint registry with point IDs and their pixel coordinates on the map
3. Manually identify correspondences between camera pixels and map point IDs
4. Use MapPointHomography to compute transformations

Example:
    from poc_homography.map_points import MapPointRegistry
    from poc_homography.homography_map_points import MapPointHomography

    # Load map points
    registry = MapPointRegistry.load("map_points.json")

    # Define GCPs: camera pixel -> map point ID correspondences
    gcps = [
        {"pixel_x": 800, "pixel_y": 580, "map_point_id": "A7"},
        {"pixel_x": 1082, "pixel_y": 390, "map_point_id": "A6"},
        # ... more GCPs
    ]

    # Compute homography
    homography = MapPointHomography(map_id="map_valte")
    result = homography.compute_from_gcps(gcps, registry)

    # Project camera pixel to map
    map_point = homography.camera_to_map((960, 540))

See also:
    - examples/demo_map_point_homography.py - Complete MapPoint workflow demo
    - poc_homography/homography_map_points.py - MapPoint homography implementation
    - poc_homography/map_points/ - MapPoint data structures

Migration completed: 2026-01-10
"""

import sys


def main():
    """Display deprecation notice and exit."""
    print("=" * 80)
    print("DEPRECATED TOOL: capture_gcps_web")
    print("=" * 80)
    print()
    print("This tool is no longer supported after the MapPoint migration.")
    print()
    print("The GPS-based GCP capture workflow has been replaced with a MapPoint")
    print("system that uses map-relative pixel coordinates instead of GPS coordinates.")
    print()
    print("For the new workflow, see:")
    print("  - examples/demo_map_point_homography.py")
    print("  - poc_homography/homography_map_points.py")
    print()
    print("Reason: Removed GPS dependencies (WorldPoint, MapCoordinate, etc.)")
    print("Migration: Issue #156 - MapPoint system implementation")
    print("=" * 80)
    sys.exit(1)


if __name__ == "__main__":
    main()
