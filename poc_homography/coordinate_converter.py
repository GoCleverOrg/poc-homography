#!/usr/bin/env python3
"""
GPS and local coordinate conversion utilities.

This module provides coordinate conversion functions between GPS (latitude/longitude)
and local Cartesian (X, Y) coordinate systems using either:

1. Equirectangular projection (simple spherical approximation)
2. UTM projection via pyproj (accurate ellipsoidal model)

The UTM-based approach is recommended when working with georeferenced imagery
that uses UTM coordinates (e.g., ETRS89 / UTM Zone 30N - EPSG:25830).

Coordinate System Convention:
    - X axis: East-West direction (positive = East, negative = West)
    - Y axis: North-South direction (positive = North, negative = South)
    - Reference point: (0, 0) in local coordinates

Accuracy Notes (Equirectangular):
    - Best accuracy: distances < 1 km
    - Good accuracy: distances < 10 km
    - Decreasing accuracy: distances > 10 km
    - Has anisotropic scale errors (~2% X, ~5% Y vs UTM)

Accuracy Notes (UTM via pyproj):
    - High accuracy at all reasonable distances
    - Matches georeferenced imagery exactly
    - Recommended for GCP work with orthorectified maps
"""

import math
from typing import Tuple, Optional

try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False


# Earth's mean radius in meters (WGS84 spherical approximation)
EARTH_RADIUS_M = 6371000.0

# Default UTM CRS for the Valencia/Spain area
DEFAULT_UTM_CRS = "EPSG:25830"  # ETRS89 / UTM Zone 30N


class UTMConverter:
    """
    UTM-based coordinate converter using pyproj.

    This provides accurate coordinate conversion that matches georeferenced
    imagery using UTM projections. Unlike the equirectangular approximation,
    this uses the proper ellipsoidal Earth model and conformal projection.
    """

    def __init__(self, utm_crs: str = DEFAULT_UTM_CRS):
        """
        Initialize the UTM converter.

        Args:
            utm_crs: The UTM coordinate reference system (e.g., "EPSG:25830")
        """
        if not PYPROJ_AVAILABLE:
            raise ImportError(
                "pyproj is required for UTM conversion. "
                "Install with: pip install pyproj"
            )

        self.utm_crs = utm_crs
        self._to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        self._to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

        # Reference point (set when first conversion is made)
        self._ref_easting: Optional[float] = None
        self._ref_northing: Optional[float] = None
        self._ref_lat: Optional[float] = None
        self._ref_lon: Optional[float] = None

    def set_reference(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Set the reference point for local coordinate conversion.

        Args:
            lat: Reference latitude in decimal degrees
            lon: Reference longitude in decimal degrees

        Returns:
            Tuple of (easting, northing) in UTM coordinates
        """
        self._ref_lat = lat
        self._ref_lon = lon
        self._ref_easting, self._ref_northing = self._to_utm.transform(lon, lat)
        return self._ref_easting, self._ref_northing

    def set_reference_utm(self, easting: float, northing: float):
        """
        Set the reference point directly in UTM coordinates.

        Args:
            easting: Reference easting in meters
            northing: Reference northing in meters
        """
        self._ref_easting = easting
        self._ref_northing = northing
        lon, lat = self._to_wgs84.transform(easting, northing)
        self._ref_lat = lat
        self._ref_lon = lon

    def gps_to_local_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert GPS coordinates to local X, Y relative to reference point.

        Args:
            lat: Target latitude in decimal degrees
            lon: Target longitude in decimal degrees

        Returns:
            Tuple of (x_meters, y_meters) relative to reference point
        """
        if self._ref_easting is None:
            raise ValueError("Reference point not set. Call set_reference() first.")

        easting, northing = self._to_utm.transform(lon, lat)
        x = easting - self._ref_easting
        y = northing - self._ref_northing
        return x, y

    def utm_to_local_xy(self, easting: float, northing: float) -> Tuple[float, float]:
        """
        Convert UTM coordinates to local X, Y relative to reference point.

        Args:
            easting: UTM easting in meters
            northing: UTM northing in meters

        Returns:
            Tuple of (x_meters, y_meters) relative to reference point
        """
        if self._ref_easting is None:
            raise ValueError("Reference point not set. Call set_reference() first.")

        x = easting - self._ref_easting
        y = northing - self._ref_northing
        return x, y

    def local_xy_to_gps(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert local X, Y coordinates to GPS.

        Args:
            x: East-West distance in meters (positive = East)
            y: North-South distance in meters (positive = North)

        Returns:
            Tuple of (latitude, longitude) in decimal degrees
        """
        if self._ref_easting is None:
            raise ValueError("Reference point not set. Call set_reference() first.")

        easting = self._ref_easting + x
        northing = self._ref_northing + y
        lon, lat = self._to_wgs84.transform(easting, northing)
        return lat, lon

    def local_xy_to_utm(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert local X, Y coordinates to UTM.

        Args:
            x: East-West distance in meters (positive = East)
            y: North-South distance in meters (positive = North)

        Returns:
            Tuple of (easting, northing) in UTM coordinates
        """
        if self._ref_easting is None:
            raise ValueError("Reference point not set. Call set_reference() first.")

        easting = self._ref_easting + x
        northing = self._ref_northing + y
        return easting, northing

    def gps_to_utm(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert GPS to UTM coordinates (no reference point needed).

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees

        Returns:
            Tuple of (easting, northing) in UTM coordinates
        """
        easting, northing = self._to_utm.transform(lon, lat)
        return easting, northing

    def utm_to_gps(self, easting: float, northing: float) -> Tuple[float, float]:
        """
        Convert UTM to GPS coordinates.

        Args:
            easting: UTM easting in meters
            northing: UTM northing in meters

        Returns:
            Tuple of (latitude, longitude) in decimal degrees
        """
        lon, lat = self._to_wgs84.transform(easting, northing)
        return lat, lon


# Global UTM converter instance (lazily initialized)
_utm_converter: Optional[UTMConverter] = None


def get_utm_converter(utm_crs: str = DEFAULT_UTM_CRS) -> UTMConverter:
    """
    Get or create a UTM converter instance.

    Args:
        utm_crs: The UTM coordinate reference system

    Returns:
        UTMConverter instance
    """
    global _utm_converter
    if _utm_converter is None or _utm_converter.utm_crs != utm_crs:
        _utm_converter = UTMConverter(utm_crs)
    return _utm_converter


def gps_to_local_xy_utm(ref_lat: float, ref_lon: float,
                        lat: float, lon: float,
                        utm_crs: str = DEFAULT_UTM_CRS) -> Tuple[float, float]:
    """
    Convert GPS to local XY using UTM projection (accurate method).

    This is the recommended method when working with georeferenced imagery.

    Args:
        ref_lat: Reference latitude in decimal degrees
        ref_lon: Reference longitude in decimal degrees
        lat: Target latitude in decimal degrees
        lon: Target longitude in decimal degrees
        utm_crs: UTM coordinate reference system (default: EPSG:25830)

    Returns:
        Tuple of (x_meters, y_meters)
    """
    converter = get_utm_converter(utm_crs)
    converter.set_reference(ref_lat, ref_lon)
    return converter.gps_to_local_xy(lat, lon)


def local_xy_to_gps_utm(ref_lat: float, ref_lon: float,
                        x: float, y: float,
                        utm_crs: str = DEFAULT_UTM_CRS) -> Tuple[float, float]:
    """
    Convert local XY to GPS using UTM projection (accurate method).

    Args:
        ref_lat: Reference latitude in decimal degrees
        ref_lon: Reference longitude in decimal degrees
        x: East-West distance in meters
        y: North-South distance in meters
        utm_crs: UTM coordinate reference system (default: EPSG:25830)

    Returns:
        Tuple of (latitude, longitude)
    """
    converter = get_utm_converter(utm_crs)
    converter.set_reference(ref_lat, ref_lon)
    return converter.local_xy_to_gps(x, y)


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


class GCPCoordinateConverter:
    """
    Unified coordinate converter for GCP operations.

    This class provides a consistent interface for converting between GPS, UTM,
    and local XY coordinates. It automatically uses UTM (via pyproj) when available
    for maximum accuracy, falling back to equirectangular projection otherwise.

    The converter maintains state about:
    - Reference point (camera position or GCP centroid)
    - UTM CRS (if using UTM coordinates)
    - Conversion method (UTM or equirectangular)

    Usage:
        >>> converter = GCPCoordinateConverter()
        >>> converter.set_reference_gps(39.640472, -0.230194)
        >>> x, y = converter.gps_to_local(39.640500, -0.230100)
        >>> lat, lon = converter.local_to_gps(10.0, -5.0)
    """

    def __init__(self, utm_crs: str = DEFAULT_UTM_CRS, prefer_utm: bool = True):
        """
        Initialize the coordinate converter.

        Args:
            utm_crs: UTM coordinate reference system (default: EPSG:25830)
            prefer_utm: If True, use UTM when pyproj is available (default: True)
        """
        self.utm_crs = utm_crs
        self.prefer_utm = prefer_utm and PYPROJ_AVAILABLE
        self._utm_converter: Optional[UTMConverter] = None

        # Reference point
        self._ref_lat: Optional[float] = None
        self._ref_lon: Optional[float] = None
        self._ref_easting: Optional[float] = None
        self._ref_northing: Optional[float] = None

        # Initialize UTM converter if available and preferred
        if self.prefer_utm:
            try:
                self._utm_converter = UTMConverter(utm_crs)
            except Exception:
                self.prefer_utm = False

    @property
    def using_utm(self) -> bool:
        """Return True if using UTM projection, False if using equirectangular."""
        return self.prefer_utm and self._utm_converter is not None

    @property
    def method_name(self) -> str:
        """Return the name of the conversion method being used."""
        return f"UTM ({self.utm_crs})" if self.using_utm else "Equirectangular"

    def set_reference_gps(self, lat: float, lon: float):
        """
        Set the reference point using GPS coordinates.

        Args:
            lat: Reference latitude in decimal degrees
            lon: Reference longitude in decimal degrees
        """
        self._ref_lat = lat
        self._ref_lon = lon

        if self._utm_converter:
            self._ref_easting, self._ref_northing = self._utm_converter.set_reference(lat, lon)
        else:
            self._ref_easting = None
            self._ref_northing = None

    def set_reference_utm(self, easting: float, northing: float):
        """
        Set the reference point using UTM coordinates.

        Args:
            easting: Reference easting in meters
            northing: Reference northing in meters
        """
        self._ref_easting = easting
        self._ref_northing = northing

        if self._utm_converter:
            self._utm_converter.set_reference_utm(easting, northing)
            self._ref_lat = self._utm_converter._ref_lat
            self._ref_lon = self._utm_converter._ref_lon
        else:
            raise ValueError("UTM reference requires pyproj. Install with: pip install pyproj")

    def gps_to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert GPS coordinates to local X, Y relative to reference.

        Args:
            lat: Target latitude in decimal degrees
            lon: Target longitude in decimal degrees

        Returns:
            Tuple of (x_meters, y_meters)
        """
        if self._ref_lat is None:
            raise ValueError("Reference point not set. Call set_reference_gps() first.")

        if self._utm_converter:
            return self._utm_converter.gps_to_local_xy(lat, lon)
        else:
            return gps_to_local_xy(self._ref_lat, self._ref_lon, lat, lon)

    def utm_to_local(self, easting: float, northing: float) -> Tuple[float, float]:
        """
        Convert UTM coordinates to local X, Y relative to reference.

        Args:
            easting: UTM easting in meters
            northing: UTM northing in meters

        Returns:
            Tuple of (x_meters, y_meters)
        """
        if self._ref_easting is None:
            raise ValueError("Reference point not set. Call set_reference_gps() or set_reference_utm() first.")

        if self._utm_converter:
            return self._utm_converter.utm_to_local_xy(easting, northing)
        else:
            # Direct calculation without pyproj
            return easting - self._ref_easting, northing - self._ref_northing

    def local_to_gps(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert local X, Y coordinates to GPS.

        Args:
            x: East-West distance in meters (positive = East)
            y: North-South distance in meters (positive = North)

        Returns:
            Tuple of (latitude, longitude)
        """
        if self._ref_lat is None:
            raise ValueError("Reference point not set. Call set_reference_gps() first.")

        if self._utm_converter:
            return self._utm_converter.local_xy_to_gps(x, y)
        else:
            return local_xy_to_gps(self._ref_lat, self._ref_lon, x, y)

    def local_to_utm(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert local X, Y coordinates to UTM.

        Args:
            x: East-West distance in meters (positive = East)
            y: North-South distance in meters (positive = North)

        Returns:
            Tuple of (easting, northing)
        """
        if self._ref_easting is None:
            raise ValueError("Reference point not set with UTM.")

        if self._utm_converter:
            return self._utm_converter.local_xy_to_utm(x, y)
        else:
            return self._ref_easting + x, self._ref_northing + y

    def convert_point(self, point: dict) -> Tuple[float, float]:
        """
        Convert a GCP point to local X, Y coordinates.

        This method intelligently selects the best conversion path:
        - If point has UTM coordinates and we're using UTM, use utm_to_local
        - Otherwise use gps_to_local

        Args:
            point: Dictionary with either:
                - 'latitude' and 'longitude' (GPS)
                - 'utm_easting' and 'utm_northing' (UTM)
                - Both (UTM preferred when using UTM converter)

        Returns:
            Tuple of (x_meters, y_meters)
        """
        has_utm = 'utm_easting' in point and 'utm_northing' in point
        has_gps = 'latitude' in point and 'longitude' in point

        if not has_utm and not has_gps:
            raise ValueError("Point must have either GPS (latitude/longitude) or UTM (utm_easting/utm_northing) coordinates")

        # Prefer UTM when available and we're using UTM converter
        if has_utm and self.using_utm:
            return self.utm_to_local(point['utm_easting'], point['utm_northing'])
        elif has_gps:
            return self.gps_to_local(point['latitude'], point['longitude'])
        else:
            # Has UTM but not using UTM converter - need to convert via GPS
            if self._utm_converter:
                lat, lon = self._utm_converter.utm_to_gps(point['utm_easting'], point['utm_northing'])
                return self.gps_to_local(lat, lon)
            else:
                raise ValueError("Point has only UTM coordinates but pyproj is not available")


# Global converter instance for convenient access
_global_converter: Optional[GCPCoordinateConverter] = None


def get_gcp_converter(utm_crs: str = DEFAULT_UTM_CRS) -> GCPCoordinateConverter:
    """
    Get or create a global GCP coordinate converter.

    Args:
        utm_crs: UTM coordinate reference system

    Returns:
        GCPCoordinateConverter instance
    """
    global _global_converter
    if _global_converter is None or _global_converter.utm_crs != utm_crs:
        _global_converter = GCPCoordinateConverter(utm_crs)
    return _global_converter
