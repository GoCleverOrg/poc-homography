#!/usr/bin/env python3
"""
GPS and local coordinate conversion utilities.

This module provides coordinate conversion functions between GPS (latitude/longitude)
and local Cartesian (X, Y) coordinate systems using the equirectangular projection
approximation.

The equirectangular projection (also known as Plate Carrée) is a simple and
computationally efficient approximation that works well for small distances
(typically < 10 km). It assumes the Earth is locally flat and uses a spherical
Earth model with a mean radius of 6,371,000 meters.

Coordinate System Convention:
    - X axis: East-West direction (positive = East, negative = West)
    - Y axis: North-South direction (positive = North, negative = South)
    - Reference point: (0, 0) in local coordinates

Accuracy Notes:
    - Best accuracy: distances < 1 km
    - Good accuracy: distances < 10 km
    - Decreasing accuracy: distances > 10 km
    - Not suitable: polar regions (latitude > 80°)

The conversion uses the more accurate Formula B approach with proper radian
conversions and the Earth's mean radius of 6,371,000 meters.
"""

import math
from typing import Tuple


# Earth's mean radius in meters (WGS84 spherical approximation)
EARTH_RADIUS_M = 6371000.0


def gps_to_local_xy(ref_lat: float, ref_lon: float,
                    lat: float, lon: float) -> Tuple[float, float]:
    """
    Convert GPS coordinates to local X, Y coordinates relative to a reference point.

    Uses equirectangular projection approximation, which is accurate for small
    distances (< 10 km). The conversion assumes a spherical Earth model.

    Formula:
        x = Δλ × cos(φ_avg) × R
        y = Δφ × R

    where:
        Δλ = difference in longitude (radians)
        Δφ = difference in latitude (radians)
        φ_avg = average latitude of reference and target points (radians)
        R = Earth's mean radius (6,371,000 meters)

    Args:
        ref_lat: Reference point latitude in decimal degrees
        ref_lon: Reference point longitude in decimal degrees
        lat: Target point latitude in decimal degrees
        lon: Target point longitude in decimal degrees

    Returns:
        Tuple of (x_meters, y_meters) where:
            x_meters: East-West distance (positive = East of reference)
            y_meters: North-South distance (positive = North of reference)

    Raises:
        ValueError: If latitude values are outside valid range [-90, 90] or
                   if near polar regions (absolute latitude > 85 degrees)

    Example:
        >>> ref_lat, ref_lon = 39.640472, -0.230194
        >>> lat, lon = 39.640444, -0.230111
        >>> x, y = gps_to_local_xy(ref_lat, ref_lon, lat, lon)
        >>> print(f"Point is {x:.2f}m East and {y:.2f}m North of reference")
    """
    # Validate latitude range
    if not -90 <= ref_lat <= 90 or not -90 <= lat <= 90:
        raise ValueError(
            f"Latitude must be in range [-90, 90]. "
            f"Got ref_lat={ref_lat}, lat={lat}"
        )

    # Warn about polar regions where equirectangular projection is inaccurate
    if abs(ref_lat) > 85 or abs(lat) > 85:
        raise ValueError(
            f"Equirectangular projection is not accurate near poles. "
            f"Latitude should be within [-85, 85]. "
            f"Got ref_lat={ref_lat}, lat={lat}"
        )

    # Convert to radians
    ref_lat_rad = math.radians(ref_lat)
    lat_rad = math.radians(lat)
    delta_lat = math.radians(lat - ref_lat)
    delta_lon = math.radians(lon - ref_lon)

    # Equirectangular projection
    # Use average latitude for better accuracy across the distance
    avg_lat_rad = (ref_lat_rad + lat_rad) / 2

    x = delta_lon * math.cos(avg_lat_rad) * EARTH_RADIUS_M
    y = delta_lat * EARTH_RADIUS_M

    return x, y


def local_xy_to_gps(ref_lat: float, ref_lon: float,
                    x_meters: float, y_meters: float) -> Tuple[float, float]:
    """
    Convert local X, Y coordinates to GPS coordinates (inverse of gps_to_local_xy).

    Uses the inverse equirectangular projection to convert local Cartesian
    coordinates back to GPS latitude/longitude.

    This is the more accurate Formula B approach:
        Δφ = y / R
        Δλ = x / (R × cos(φ_ref))

    where:
        Δφ = latitude difference (radians)
        Δλ = longitude difference (radians)
        φ_ref = reference latitude (radians)
        R = Earth's mean radius (6,371,000 meters)

    Args:
        ref_lat: Reference point latitude in decimal degrees
        ref_lon: Reference point longitude in decimal degrees
        x_meters: East-West distance in meters (positive = East)
        y_meters: North-South distance in meters (positive = North)

    Returns:
        Tuple of (latitude, longitude) in decimal degrees

    Raises:
        ValueError: If reference latitude is outside valid range [-90, 90] or
                   if near polar regions (absolute latitude > 85 degrees)

    Example:
        >>> ref_lat, ref_lon = 39.640472, -0.230194
        >>> x_meters, y_meters = 5.5, -3.0
        >>> lat, lon = local_xy_to_gps(ref_lat, ref_lon, x_meters, y_meters)
        >>> print(f"GPS: {lat:.6f}°, {lon:.6f}°")

    Note:
        This function uses the reference latitude (not average latitude) for the
        cosine correction term, which is appropriate for the inverse transformation
        and matches Formula B specifications.
    """
    # Validate latitude range
    if not -90 <= ref_lat <= 90:
        raise ValueError(
            f"Reference latitude must be in range [-90, 90]. "
            f"Got ref_lat={ref_lat}"
        )

    # Warn about polar regions where equirectangular projection is inaccurate
    if abs(ref_lat) > 85:
        raise ValueError(
            f"Equirectangular projection is not accurate near poles. "
            f"Reference latitude should be within [-85, 85]. "
            f"Got ref_lat={ref_lat}"
        )

    # Convert reference position to radians
    ref_lat_rad = math.radians(ref_lat)

    # Calculate latitude/longitude deltas in radians (Formula B)
    delta_lat_rad = y_meters / EARTH_RADIUS_M
    delta_lon_rad = x_meters / (EARTH_RADIUS_M * math.cos(ref_lat_rad))

    # Convert to degrees and add to reference position
    lat = ref_lat + math.degrees(delta_lat_rad)
    lon = ref_lon + math.degrees(delta_lon_rad)

    return lat, lon
