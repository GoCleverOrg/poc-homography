#!/usr/bin/env python3
"""
Debug tool to analyze coordinate transformation discrepancies between:
1. extract_kml_points.py: pyproj UTM (EPSG:25830) ↔ WGS84
2. capture_gcps_web.py: equirectangular projection (spherical Earth)

This helps identify the root cause of translation and scaling mismatches.
"""

import argparse
import math
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pyproj import Transformer

from poc_homography.camera_config import get_camera_by_name, get_camera_configs

# Constants
EARTH_RADIUS_M = 6371000.0  # Spherical approximation used in coordinate_converter.py

# Georeferencing parameters - loaded from camera config at runtime
ORIGIN_EASTING = None
ORIGIN_NORTHING = None
GSD = None
UTM_CRS = None


def pyproj_pixel_to_latlon(px: float, py: float) -> tuple:
    """Convert pixel to lat/lon using pyproj (what extract_kml_points.py does)."""
    transformer = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)

    easting = ORIGIN_EASTING + (px * GSD)
    northing = ORIGIN_NORTHING + (py * -GSD)  # Y is inverted

    lon, lat = transformer.transform(easting, northing)
    return lat, lon, easting, northing


def pyproj_latlon_to_pixel(lat: float, lon: float) -> tuple:
    """Convert lat/lon to pixel using pyproj (reverse of above)."""
    transformer = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)

    easting, northing = transformer.transform(lon, lat)

    px = (easting - ORIGIN_EASTING) / GSD
    py = (northing - ORIGIN_NORTHING) / -GSD

    return px, py, easting, northing


def equirect_gps_to_local_xy(ref_lat: float, ref_lon: float, lat: float, lon: float) -> tuple:
    """Convert GPS to local XY using equirectangular projection (what capture_gcps_web.py uses)."""
    ref_lat_rad = math.radians(ref_lat)
    lat_rad = math.radians(lat)
    delta_lat = math.radians(lat - ref_lat)
    delta_lon = math.radians(lon - ref_lon)

    avg_lat_rad = (ref_lat_rad + lat_rad) / 2

    x = delta_lon * math.cos(avg_lat_rad) * EARTH_RADIUS_M
    y = delta_lat * EARTH_RADIUS_M

    return x, y


def equirect_local_xy_to_gps(ref_lat: float, ref_lon: float, x: float, y: float) -> tuple:
    """Convert local XY to GPS using equirectangular projection (inverse)."""
    ref_lat_rad = math.radians(ref_lat)

    delta_lat_rad = y / EARTH_RADIUS_M
    delta_lon_rad = x / (EARTH_RADIUS_M * math.cos(ref_lat_rad))

    lat = ref_lat + math.degrees(delta_lat_rad)
    lon = ref_lon + math.degrees(delta_lon_rad)

    return lat, lon


def analyze_transformation_discrepancy():
    """Analyze the transformation differences."""

    print("=" * 80)
    print("COORDINATE TRANSFORMATION DEBUG ANALYSIS")
    print("=" * 80)

    # Test points: corners and center of the image
    # Image is 1681 x 916 pixels
    test_pixels = [
        (0, 0, "Top-Left"),
        (1681, 0, "Top-Right"),
        (0, 916, "Bottom-Left"),
        (1681, 916, "Bottom-Right"),
        (840, 458, "Center"),
        (100, 100, "Near Top-Left"),
        (1500, 800, "Near Bottom-Right"),
    ]

    print("\n1. PIXEL → LAT/LON (via pyproj UTM)")
    print("-" * 80)

    points_data = []
    for px, py, name in test_pixels:
        lat, lon, easting, northing = pyproj_pixel_to_latlon(px, py)
        points_data.append(
            {
                "name": name,
                "px": px,
                "py": py,
                "lat": lat,
                "lon": lon,
                "easting": easting,
                "northing": northing,
            }
        )
        print(
            f"{name:20s} pixel({px:4d},{py:4d}) → UTM({easting:.2f}, {northing:.2f}) → ({lat:.8f}, {lon:.8f})"
        )

    # Get reference point (use top-left as reference for equirectangular)
    ref_lat = points_data[0]["lat"]
    ref_lon = points_data[0]["lon"]

    print(f"\n2. USING TOP-LEFT AS REFERENCE: ({ref_lat:.8f}, {ref_lon:.8f})")
    print("-" * 80)

    print("\n3. COMPARE: UTM distances vs Equirectangular distances from reference")
    print("-" * 80)
    print(
        f"{'Point':<20} {'UTM ΔE(m)':<12} {'UTM ΔN(m)':<12} {'Equi ΔX(m)':<12} {'Equi ΔY(m)':<12} {'Err X(m)':<10} {'Err Y(m)':<10} {'Err %':<10}"
    )

    errors = []
    for p in points_data:
        # UTM distance from reference
        utm_dx = p["easting"] - points_data[0]["easting"]
        utm_dy = p["northing"] - points_data[0]["northing"]

        # Equirectangular distance from reference
        equi_x, equi_y = equirect_gps_to_local_xy(ref_lat, ref_lon, p["lat"], p["lon"])

        # Error
        err_x = equi_x - utm_dx
        err_y = equi_y - utm_dy

        utm_dist = math.sqrt(utm_dx**2 + utm_dy**2)
        err_dist = math.sqrt(err_x**2 + err_y**2)
        err_pct = (err_dist / utm_dist * 100) if utm_dist > 0 else 0

        errors.append(
            {
                "name": p["name"],
                "utm_dx": utm_dx,
                "utm_dy": utm_dy,
                "equi_x": equi_x,
                "equi_y": equi_y,
                "err_x": err_x,
                "err_y": err_y,
                "err_pct": err_pct,
            }
        )

        print(
            f"{p['name']:<20} {utm_dx:<12.3f} {utm_dy:<12.3f} {equi_x:<12.3f} {equi_y:<12.3f} {err_x:<10.3f} {err_y:<10.3f} {err_pct:<10.2f}%"
        )

    print("\n4. SCALING ANALYSIS")
    print("-" * 80)

    # Compare diagonal distances
    top_left = points_data[0]
    bottom_right = points_data[3]

    utm_diagonal = math.sqrt(
        (bottom_right["easting"] - top_left["easting"]) ** 2
        + (bottom_right["northing"] - top_left["northing"]) ** 2
    )

    equi_diagonal_x, equi_diagonal_y = equirect_gps_to_local_xy(
        top_left["lat"], top_left["lon"], bottom_right["lat"], bottom_right["lon"]
    )
    equi_diagonal = math.sqrt(equi_diagonal_x**2 + equi_diagonal_y**2)

    scale_factor = equi_diagonal / utm_diagonal if utm_diagonal > 0 else 1

    print(f"Image diagonal (UTM):           {utm_diagonal:.3f} m")
    print(f"Image diagonal (Equirectangular): {equi_diagonal:.3f} m")
    print(f"Scale factor (Equi/UTM):         {scale_factor:.8f}")
    print(f"Scale error:                      {(scale_factor - 1) * 100:.4f}%")

    # Separate X and Y scale factors
    utm_width = bottom_right["easting"] - top_left["easting"]
    utm_height = (
        top_left["northing"] - bottom_right["northing"]
    )  # Note: northing decreases going down

    scale_x = equi_diagonal_x / utm_width if utm_width != 0 else 1
    scale_y = -equi_diagonal_y / utm_height if utm_height != 0 else 1  # Note sign

    print(f"\nScale factor X (East-West):       {scale_x:.8f} ({(scale_x - 1) * 100:.4f}% error)")
    print(f"Scale factor Y (North-South):     {scale_y:.8f} ({(scale_y - 1) * 100:.4f}% error)")

    print("\n5. ROOT CAUSE ANALYSIS")
    print("-" * 80)
    print("""
The discrepancy comes from TWO fundamentally different coordinate systems:

A) extract_kml_points.py uses:
   - pyproj with EPSG:25830 (ETRS89 / UTM Zone 30N)
   - GRS80 ellipsoid (semi-major axis: 6,378,137 m, flattening: 1/298.257222101)
   - Transverse Mercator projection with scale factor 0.9996 at central meridian
   - Proper conformal projection that preserves angles

B) capture_gcps_web.py uses:
   - Equirectangular projection (simple approximation)
   - Spherical Earth model (radius: 6,371,000 m)
   - Simple formula: x = Δλ × cos(φ_avg) × R, y = Δφ × R
   - NOT conformal - distorts at larger distances

KEY DIFFERENCES:
1. Earth model: UTM uses ellipsoid, equirectangular uses sphere
2. Projection: UTM is conformal, equirectangular is not
3. Scale: UTM has 0.9996 scale factor at central meridian
4. X-direction: cos(latitude) correction differs between methods
""")

    print("\n6. ROUND-TRIP TEST: pixel → lat/lon → pixel")
    print("-" * 80)

    print("\nUsing pyproj (should be exact):")
    for px, py, name in test_pixels[:3]:
        lat, lon, _, _ = pyproj_pixel_to_latlon(px, py)
        px2, py2, _, _ = pyproj_latlon_to_pixel(lat, lon)
        err_px = px2 - px
        err_py = py2 - py
        print(
            f"  {name}: ({px},{py}) → ({lat:.8f},{lon:.8f}) → ({px2:.3f},{py2:.3f})  error: ({err_px:.6f}, {err_py:.6f}) pixels"
        )

    print("\n7. SUGGESTED FIX")
    print("-" * 80)
    print("""
To fix the mismatch, you have two options:

OPTION 1: Make capture_gcps_web.py use pyproj UTM
   - Replace gps_to_local_xy() with pyproj transformation
   - This ensures both tools use the same coordinate system

OPTION 2: Make extract_kml_points.py use equirectangular
   - Match what capture_gcps_web.py expects
   - Less accurate but compatible

RECOMMENDED: Option 1 - Use pyproj consistently in both tools
""")

    return errors


def test_specific_point(px: float, py: float):
    """Test a specific pixel coordinate through both systems."""
    print(f"\nTesting pixel ({px}, {py}):")
    print("-" * 40)

    lat, lon, easting, northing = pyproj_pixel_to_latlon(px, py)
    print(f"pyproj:  UTM({easting:.2f}, {northing:.2f}) → ({lat:.8f}, {lon:.8f})")

    # Now if we use this lat/lon in capture_gcps_web with a reference point
    # we get different local XY values
    ref_lat, ref_lon, _, _ = pyproj_pixel_to_latlon(0, 0)  # Use top-left as ref
    equi_x, equi_y = equirect_gps_to_local_xy(ref_lat, ref_lon, lat, lon)

    # Expected local XY from UTM
    ref_easting = ORIGIN_EASTING
    ref_northing = ORIGIN_NORTHING
    utm_x = easting - ref_easting
    utm_y = northing - ref_northing

    print(f"UTM local XY from ref: ({utm_x:.3f}, {utm_y:.3f}) m")
    print(f"Equi local XY from ref: ({equi_x:.3f}, {equi_y:.3f}) m")
    print(f"Difference: ({equi_x - utm_x:.3f}, {equi_y - utm_y:.3f}) m")


def load_geotiff_params(camera_name: str):
    """
    Load georeferencing parameters from camera config.

    Args:
        camera_name: Name of the camera to load config for

    Returns:
        Tuple of (origin_easting, origin_northing, gsd, utm_crs)

    Raises:
        SystemExit: If camera not found or missing geotiff_params
    """
    camera_config = get_camera_by_name(camera_name)

    if camera_config is None:
        available_cameras = [cam["name"] for cam in get_camera_configs()]
        print(f"Error: Camera '{camera_name}' not found in camera configuration.", file=sys.stderr)
        print(f"Available cameras: {', '.join(available_cameras)}", file=sys.stderr)
        sys.exit(1)

    if "geotiff_params" not in camera_config:
        print(
            f"Error: Camera '{camera_name}' does not have 'geotiff_params' configured.",
            file=sys.stderr,
        )
        sys.exit(1)

    geotiff_params = camera_config["geotiff_params"]

    # Handle both new (geotransform) and legacy (origin_easting, etc.) formats
    if "geotransform" in geotiff_params:
        # New format: extract values from geotransform array
        gt = geotiff_params["geotransform"]
        origin_easting = gt[0]
        origin_northing = gt[3]
        gsd = abs(gt[1])  # GSD is always positive (pixel width)
    else:
        # Legacy format: direct key access
        required_params = ["origin_easting", "origin_northing", "pixel_size_x"]
        missing_params = [p for p in required_params if p not in geotiff_params]

        if missing_params:
            print(
                f"Error: Camera '{camera_name}' geotiff_params missing required fields: {', '.join(missing_params)}",
                file=sys.stderr,
            )
            sys.exit(1)

        origin_easting = geotiff_params["origin_easting"]
        origin_northing = geotiff_params["origin_northing"]
        gsd = abs(geotiff_params["pixel_size_x"])  # GSD is always positive

    # Validate utm_crs (required in both formats)
    if "utm_crs" not in geotiff_params:
        print(
            f"Error: Camera '{camera_name}' geotiff_params missing 'utm_crs' field.",
            file=sys.stderr,
        )
        sys.exit(1)

    utm_crs = geotiff_params["utm_crs"]

    return origin_easting, origin_northing, gsd, utm_crs


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Debug tool to analyze coordinate transformation discrepancies"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="Valte",
        help="Camera name to load georeferencing parameters from (default: Valte)",
    )
    parser.add_argument(
        "px", type=float, nargs="?", help="Pixel X coordinate for specific point test (optional)"
    )
    parser.add_argument(
        "py", type=float, nargs="?", help="Pixel Y coordinate for specific point test (optional)"
    )
    args = parser.parse_args()

    # Load georeferencing parameters from camera config
    ORIGIN_EASTING, ORIGIN_NORTHING, GSD, UTM_CRS = load_geotiff_params(args.camera)

    print(f"Loaded georeferencing parameters from camera '{args.camera}':")
    print(f"  Origin Easting: {ORIGIN_EASTING}")
    print(f"  Origin Northing: {ORIGIN_NORTHING}")
    print(f"  GSD: {GSD} m/pixel")
    print(f"  UTM CRS: {UTM_CRS}")
    print()

    errors = analyze_transformation_discrepancy()

    if args.px is not None and args.py is not None:
        test_specific_point(args.px, args.py)
