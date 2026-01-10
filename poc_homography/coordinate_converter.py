#!/usr/bin/env python3
"""
UTM coordinate conversion utilities.

This module provides coordinate conversion functions between GPS (latitude/longitude)
and local Cartesian (X, Y) coordinate systems using UTM projection via pyproj.

The UTM-based approach provides accurate conversions that match georeferenced
imagery using UTM coordinates (e.g., ETRS89 / UTM Zone 30N - EPSG:25830).

Coordinate System Convention:
    - X axis: East-West direction (positive = East, negative = West)
    - Y axis: North-South direction (positive = North, negative = South)
    - Reference point: (0, 0) in local coordinates

Accuracy Notes (UTM via pyproj):
    - High accuracy at all reasonable distances
    - Matches georeferenced imagery exactly
    - Recommended for GCP work with orthorectified maps

Axis Order Convention (CRITICAL):
    All pyproj.Transformer instances MUST use always_xy=True to enforce
    traditional GIS axis ordering:
    - EPSG:4326 (WGS84 GPS): (longitude, latitude) - NOT authority order
    - EPSG:25830 (UTM 30N): (easting, northing)

    Why always_xy=True is required:
    - EPSG:4326 authority definition specifies (latitude, longitude) order
    - GIS tools traditionally use (longitude, latitude) for consistency with (x,y)
    - Without always_xy=True, axis order depends on CRS metadata (unpredictable)
    - Axis swaps cause silent catastrophic calibration failures

    Correct usage:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)
        easting, northing = transformer.transform(longitude, latitude)

    See: https://pyproj4.github.io/pyproj/stable/api/transformer.html#pyproj.transformer.Transformer.from_crs
"""

from __future__ import annotations

from poc_homography.types import Degrees, Meters

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
                "pyproj is required for UTM conversion. Install with: pip install pyproj"
            )

        self.utm_crs = utm_crs
        self._to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        self._to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

        # Reference point (set when first conversion is made)
        self._ref_easting: float | None = None
        self._ref_northing: float | None = None
        self._ref_lat: float | None = None
        self._ref_lon: float | None = None

    def set_reference(self, lat: Degrees, lon: Degrees) -> tuple[Meters, Meters]:
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
        easting, northing = self._to_utm.transform(lon, lat)
        self._ref_easting = Meters(easting)
        self._ref_northing = Meters(northing)
        return self._ref_easting, self._ref_northing

    def set_reference_utm(self, easting: Meters, northing: Meters):
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

    def _require_reference(self) -> tuple[float, float]:
        """Return (easting, northing) or raise ValueError if not set."""
        if self._ref_easting is None or self._ref_northing is None:
            raise ValueError("Reference point not set. Call set_reference() first.")
        return self._ref_easting, self._ref_northing

    def gps_to_local_xy(self, lat: Degrees, lon: Degrees) -> tuple[Meters, Meters]:
        """
        Convert GPS coordinates to local X, Y relative to reference point.

        Args:
            lat: Target latitude in decimal degrees
            lon: Target longitude in decimal degrees

        Returns:
            Tuple of (x_meters, y_meters) relative to reference point
        """
        ref_easting, ref_northing = self._require_reference()
        easting, northing = self._to_utm.transform(lon, lat)
        return Meters(easting - ref_easting), Meters(northing - ref_northing)

    def utm_to_local_xy(self, easting: Meters, northing: Meters) -> tuple[Meters, Meters]:
        """
        Convert UTM coordinates to local X, Y relative to reference point.

        Args:
            easting: UTM easting in meters
            northing: UTM northing in meters

        Returns:
            Tuple of (x_meters, y_meters) relative to reference point
        """
        ref_easting, ref_northing = self._require_reference()
        return Meters(easting - ref_easting), Meters(northing - ref_northing)

    def local_xy_to_gps(self, x: Meters, y: Meters) -> tuple[Degrees, Degrees]:
        """
        Convert local X, Y coordinates to GPS.

        Args:
            x: East-West distance in meters (positive = East)
            y: North-South distance in meters (positive = North)

        Returns:
            Tuple of (latitude, longitude) in decimal degrees
        """
        ref_easting, ref_northing = self._require_reference()
        lon, lat = self._to_wgs84.transform(ref_easting + x, ref_northing + y)
        return Degrees(lat), Degrees(lon)

    def local_xy_to_utm(self, x: Meters, y: Meters) -> tuple[Meters, Meters]:
        """
        Convert local X, Y coordinates to UTM.

        Args:
            x: East-West distance in meters (positive = East)
            y: North-South distance in meters (positive = North)

        Returns:
            Tuple of (easting, northing) in UTM coordinates
        """
        ref_easting, ref_northing = self._require_reference()
        return Meters(ref_easting + x), Meters(ref_northing + y)

    def gps_to_utm(self, lat: Degrees, lon: Degrees) -> tuple[Meters, Meters]:
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

    def utm_to_gps(self, easting: Meters, northing: Meters) -> tuple[Degrees, Degrees]:
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
_utm_converter: UTMConverter | None = None


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
        self._utm_converter: UTMConverter | None = None

        # Reference point
        self._ref_lat: float | None = None
        self._ref_lon: float | None = None
        self._ref_easting: float | None = None
        self._ref_northing: float | None = None

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

    def _require_gps_reference(self) -> None:
        """Raise ValueError if GPS reference point not set."""
        if self._ref_lat is None or self._ref_lon is None:
            raise ValueError("Reference point not set. Call set_reference_gps() first.")

    def _require_utm_reference(self) -> tuple[float, float]:
        """Return (easting, northing) or raise ValueError if not set."""
        if self._ref_easting is None or self._ref_northing is None:
            raise ValueError(
                "Reference point not set. Call set_reference_gps() or set_reference_utm() first."
            )
        return self._ref_easting, self._ref_northing

    def set_reference_gps(self, lat: Degrees, lon: Degrees):
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

    def set_reference_utm(self, easting: Meters, northing: Meters):
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

    def gps_to_local(self, lat: Degrees, lon: Degrees) -> tuple[Meters, Meters]:
        """
        Convert GPS coordinates to local X, Y relative to reference.

        Args:
            lat: Target latitude in decimal degrees
            lon: Target longitude in decimal degrees

        Returns:
            Tuple of (x_meters, y_meters)
        """
        self._require_gps_reference()

        if self._utm_converter:
            return self._utm_converter.gps_to_local_xy(lat, lon)
        else:
            raise ValueError("GPS conversion requires pyproj. Install with: pip install pyproj")

    def utm_to_local(self, easting: Meters, northing: Meters) -> tuple[Meters, Meters]:
        """
        Convert UTM coordinates to local X, Y relative to reference.

        Args:
            easting: UTM easting in meters
            northing: UTM northing in meters

        Returns:
            Tuple of (x_meters, y_meters)
        """
        ref_easting, ref_northing = self._require_utm_reference()
        if self._utm_converter:
            return self._utm_converter.utm_to_local_xy(easting, northing)
        return Meters(easting - ref_easting), Meters(northing - ref_northing)

    def local_to_gps(self, x: Meters, y: Meters) -> tuple[Degrees, Degrees]:
        """
        Convert local X, Y coordinates to GPS.

        Args:
            x: East-West distance in meters (positive = East)
            y: North-South distance in meters (positive = North)

        Returns:
            Tuple of (latitude, longitude)
        """
        self._require_gps_reference()

        if self._utm_converter:
            return self._utm_converter.local_xy_to_gps(x, y)
        else:
            raise ValueError("GPS conversion requires pyproj. Install with: pip install pyproj")

    def local_to_utm(self, x: Meters, y: Meters) -> tuple[Meters, Meters]:
        """
        Convert local X, Y coordinates to UTM.

        Args:
            x: East-West distance in meters (positive = East)
            y: North-South distance in meters (positive = North)

        Returns:
            Tuple of (easting, northing)
        """
        ref_easting, ref_northing = self._require_utm_reference()
        if self._utm_converter:
            return self._utm_converter.local_xy_to_utm(x, y)
        return Meters(ref_easting + x), Meters(ref_northing + y)

    def convert_point(self, point: dict) -> tuple[Meters, Meters]:
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
        has_utm = "utm_easting" in point and "utm_northing" in point
        has_gps = "latitude" in point and "longitude" in point

        if not has_utm and not has_gps:
            raise ValueError(
                "Point must have either GPS (latitude/longitude) or UTM (utm_easting/utm_northing) coordinates"
            )

        # Prefer UTM when available and we're using UTM converter
        if has_utm and self.using_utm:
            return self.utm_to_local(point["utm_easting"], point["utm_northing"])
        elif has_gps:
            return self.gps_to_local(point["latitude"], point["longitude"])
        else:
            # Has UTM but not using UTM converter - need to convert via GPS
            if self._utm_converter:
                lat, lon = self._utm_converter.utm_to_gps(
                    point["utm_easting"], point["utm_northing"]
                )
                return self.gps_to_local(lat, lon)
            else:
                raise ValueError("Point has only UTM coordinates but pyproj is not available")


# Global converter instance for convenient access
_global_converter: GCPCoordinateConverter | None = None


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
