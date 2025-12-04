#!/usr/bin/env python3
"""
Calculate ground distance between GPS coordinates using Haversine formula.
"""

import math
import numpy as np

from poc_homography.coordinate_converter import (
    EARTH_RADIUS_M,
    gps_to_local_xy,
    local_xy_to_gps
)


def dms_to_dd(dms_str: str) -> float:
    """
    Convert DMS string to decimal degrees.

    Examples:
        "39°38'25.7\"N" -> 39.640472
        "0°13'48.7\"W" -> -0.230194
    """
    # Remove direction letter
    direction = dms_str[-1]
    dms_str = dms_str[:-1]

    # Split degrees, minutes, seconds
    parts = dms_str.replace('°', ' ').replace("'", ' ').replace('"', '').split()

    degrees = float(parts[0])
    minutes = float(parts[1]) if len(parts) > 1 else 0.0
    seconds = float(parts[2]) if len(parts) > 2 else 0.0

    # Calculate decimal degrees
    dd = degrees + minutes/60.0 + seconds/3600.0

    # Apply direction (West and South are negative)
    if direction in ['W', 'S']:
        dd = -dd

    return dd


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two GPS coordinates using Haversine formula.

    Args:
        lat1, lon1: First point (decimal degrees)
        lat2, lon2: Second point (decimal degrees)

    Returns:
        Distance in meters
    """
    R = EARTH_RADIUS_M

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(delta_lat/2)**2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    distance = R * c

    return distance


def bearing_between_points(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate bearing from point 1 to point 2.

    Returns:
        Bearing in degrees (0° = North, 90° = East, 180° = South, 270° = West)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)

    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)

    bearing_rad = math.atan2(x, y)
    bearing_deg = math.degrees(bearing_rad)

    # Normalize to 0-360
    bearing_deg = (bearing_deg + 360) % 360

    return bearing_deg


def dd_to_dms(dd: float, is_latitude: bool = True) -> str:
    """
    Convert decimal degrees to DMS string.

    Args:
        dd: Decimal degrees
        is_latitude: True for lat (N/S), False for lon (E/W)

    Returns:
        DMS string like "39°38'25.7\"N"
    """
    # Determine direction
    if is_latitude:
        direction = 'N' if dd >= 0 else 'S'
    else:
        direction = 'E' if dd >= 0 else 'W'

    # Work with absolute value
    dd_abs = abs(dd)

    # Extract degrees, minutes, seconds
    degrees = int(dd_abs)
    minutes_decimal = (dd_abs - degrees) * 60
    minutes = int(minutes_decimal)
    seconds = (minutes_decimal - minutes) * 60

    return f"{degrees}°{minutes}'{seconds:.1f}\"{direction}"


def compare_distances(camera_gps: dict, point_gps: dict,
                      homography_distance: float, verbose: bool = True):
    """
    Compare homography distance with actual GPS distance.

    Args:
        camera_gps: {"lat": "39°38'25.7\"N", "lon": "0°13'48.7\"W"}
        point_gps: {"lat": "39°38'25.6\"N", "lon": "0°13'48.4\"W"}
        homography_distance: Distance computed by homography (meters)
        verbose: Print detailed output

    Returns:
        dict with comparison results
    """
    # Convert DMS to decimal degrees
    cam_lat_dd = dms_to_dd(camera_gps["lat"])
    cam_lon_dd = dms_to_dd(camera_gps["lon"])
    pt_lat_dd = dms_to_dd(point_gps["lat"])
    pt_lon_dd = dms_to_dd(point_gps["lon"])

    # Calculate actual GPS distance
    gps_distance = haversine_distance(cam_lat_dd, cam_lon_dd, pt_lat_dd, pt_lon_dd)

    # Calculate bearing
    bearing = bearing_between_points(cam_lat_dd, cam_lon_dd, pt_lat_dd, pt_lon_dd)

    # Calculate local X, Y
    x_local, y_local = gps_to_local_xy(cam_lat_dd, cam_lon_dd, pt_lat_dd, pt_lon_dd)

    # Calculate error
    error_m = homography_distance - gps_distance
    error_pct = (error_m / gps_distance * 100) if gps_distance > 0 else 0

    results = {
        "camera_dd": (cam_lat_dd, cam_lon_dd),
        "point_dd": (pt_lat_dd, pt_lon_dd),
        "gps_distance_m": gps_distance,
        "homography_distance_m": homography_distance,
        "error_m": error_m,
        "error_pct": error_pct,
        "bearing_deg": bearing,
        "local_x_m": x_local,
        "local_y_m": y_local,
    }

    if verbose:
        print("="*70)
        print("GPS DISTANCE VALIDATION")
        print("="*70)

        print(f"\nCamera GPS:")
        print(f"  DMS: {camera_gps['lat']}, {camera_gps['lon']}")
        print(f"  DD:  {cam_lat_dd:.6f}°, {cam_lon_dd:.6f}°")

        print(f"\nPoint GPS:")
        print(f"  DMS: {point_gps['lat']}, {point_gps['lon']}")
        print(f"  DD:  {pt_lat_dd:.6f}°, {pt_lon_dd:.6f}°")

        print(f"\nDistance Comparison:")
        print(f"  GPS Distance (Haversine):  {gps_distance:.2f} m")
        print(f"  Homography Distance:       {homography_distance:.2f} m")
        print(f"  Error:                     {error_m:+.2f} m ({error_pct:+.1f}%)")

        print(f"\nGeometry:")
        print(f"  Bearing: {bearing:.1f}° ({get_cardinal_direction(bearing)})")
        print(f"  Local X (East):  {x_local:+.2f} m")
        print(f"  Local Y (North): {y_local:+.2f} m")

        # Assessment
        print(f"\n{'='*70}")
        print("ASSESSMENT:")
        print("="*70)

        abs_error = abs(error_m)
        if abs_error < 0.5:
            status = "✓ EXCELLENT"
            color = "🟢"
        elif abs_error < 1.0:
            status = "✓ GOOD"
            color = "🟢"
        elif abs_error < 2.0:
            status = "⚠ ACCEPTABLE"
            color = "🟡"
        elif abs_error < 5.0:
            status = "⚠ POOR"
            color = "🟠"
        else:
            status = "✗ VERY POOR"
            color = "🔴"

        print(f"{color} {status}")
        print(f"   Error: {abs_error:.2f}m at {gps_distance:.2f}m distance")
        print(f"   Relative error: {abs(error_pct):.1f}%")

        if abs_error > 1.0:
            print(f"\n   Possible causes:")
            if error_m > 0:
                print(f"   • Homography overestimates distance")
                print(f"   • Camera height may be set too high")
                print(f"   • Tilt angle may need calibration")
            else:
                print(f"   • Homography underestimates distance")
                print(f"   • Camera height may be set too low")
                print(f"   • Ground not perfectly flat")

        print("="*70 + "\n")

    return results


def get_cardinal_direction(bearing: float) -> str:
    """Convert bearing to cardinal direction."""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(bearing / 22.5) % 16
    return directions[index]


def main():
    """Example usage."""
    import sys

    if len(sys.argv) < 7:
        print("Usage: python gps_distance_calculator.py CAM_LAT CAM_LON POINT_LAT POINT_LON HOMOGRAPHY_DIST")
        print('Example: python gps_distance_calculator.py "39°38\'25.7\\"N" "0°13\'48.7\\"W" "39°38\'25.6\\"N" "0°13\'48.4\\"W" 3.44')
        sys.exit(1)

    camera_gps = {"lat": sys.argv[1], "lon": sys.argv[2]}
    point_gps = {"lat": sys.argv[3], "lon": sys.argv[4]}
    homography_distance = float(sys.argv[5])

    compare_distances(camera_gps, point_gps, homography_distance)


if __name__ == "__main__":
    # Example from your data
    camera_gps = {"lat": "39°38'25.7\"N", "lon": "0°13'48.7\"W"}
    point_gps = {"lat": "39°38'25.6\"N", "lon": "0°13'48.4\"W"}
    homography_distance = 3.44

    compare_distances(camera_gps, point_gps, homography_distance)
