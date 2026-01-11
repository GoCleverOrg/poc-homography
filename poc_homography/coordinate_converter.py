#!/usr/bin/env python3
"""
UTM coordinate conversion utilities.

This module provides coordinate conversion functions between GPS (latitude/longitude)
and local Cartesian (X, Y) coordinate systems using UTM projection via pyproj.

The UTM-based approach provides accurate conversions that match georeferenced
imagery using UTM coordinates (e.g., ETRS89 / UTM Zone 30N - EPSG:25830).

This module uses an IMMUTABLE pattern - converters are created via factory methods
`with_reference()` and `with_reference_utm()` which return fully configured instances.

Coordinate System Convention:
    - X axis: East-West direction (positive = East, negative = West)
    - Y axis: North-South direction (positive = North, negative = South)
    - Reference point: (0, 0) in local coordinates

USAGE:
======
```python
# Create converter with GPS reference (immutable pattern)
converter = UTMConverter.with_reference(lat=39.5, lon=-0.5)
x, y = converter.gps_to_local_xy(39.501, -0.499)

# Create converter with UTM reference
converter = UTMConverter.with_reference_utm(easting=737575.0, northing=4391595.0)
x, y = converter.utm_to_local_xy(737580.0, 4391600.0)
```

Axis Order Convention (CRITICAL):
    All pyproj.Transformer instances MUST use always_xy=True to enforce
    traditional GIS axis ordering:
    - EPSG:4326 (WGS84 GPS): (longitude, latitude) - NOT authority order
    - EPSG:25830 (UTM 30N): (easting, northing)
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class UTMConverterConfig:
    """
    Immutable configuration for UTMConverter.

    This frozen dataclass captures all the state needed to configure a UTMConverter,
    enabling an immutable factory pattern for creating pre-configured converters.

    Attributes:
        utm_crs: The UTM coordinate reference system (e.g., "EPSG:25830")
        reference_lat: Reference latitude in decimal degrees (or None if not set)
        reference_lon: Reference longitude in decimal degrees (or None if not set)
        reference_easting: Reference UTM easting in meters (or None if not set)
        reference_northing: Reference UTM northing in meters (or None if not set)
    """

    utm_crs: str = DEFAULT_UTM_CRS
    reference_lat: Degrees | None = None
    reference_lon: Degrees | None = None
    reference_easting: Meters | None = None
    reference_northing: Meters | None = None

    @classmethod
    def from_gps(
        cls, lat: Degrees, lon: Degrees, utm_crs: str = DEFAULT_UTM_CRS
    ) -> UTMConverterConfig:
        """
        Create a config from GPS coordinates.

        Args:
            lat: Reference latitude in decimal degrees
            lon: Reference longitude in decimal degrees
            utm_crs: The UTM coordinate reference system

        Returns:
            UTMConverterConfig with GPS reference set
        """
        return cls(
            utm_crs=utm_crs,
            reference_lat=lat,
            reference_lon=lon,
            reference_easting=None,
            reference_northing=None,
        )

    @classmethod
    def from_utm(
        cls, easting: Meters, northing: Meters, utm_crs: str = DEFAULT_UTM_CRS
    ) -> UTMConverterConfig:
        """
        Create a config from UTM coordinates.

        Args:
            easting: Reference UTM easting in meters
            northing: Reference UTM northing in meters
            utm_crs: The UTM coordinate reference system

        Returns:
            UTMConverterConfig with UTM reference set
        """
        return cls(
            utm_crs=utm_crs,
            reference_lat=None,
            reference_lon=None,
            reference_easting=easting,
            reference_northing=northing,
        )


class UTMConverter:
    """
    UTM-based coordinate converter using pyproj.

    This provides accurate coordinate conversion that matches georeferenced
    imagery using UTM projections. Unlike the equirectangular approximation,
    this uses the proper ellipsoidal Earth model and conformal projection.

    Use the factory methods `with_reference()` or `with_reference_utm()` to
    create configured instances (immutable pattern).
    """

    def __init__(
        self,
        utm_crs: str = DEFAULT_UTM_CRS,
        *,
        _ref_easting: float | None = None,
        _ref_northing: float | None = None,
        _ref_lat: float | None = None,
        _ref_lon: float | None = None,
    ):
        """
        Initialize the UTM converter.

        Note: For most use cases, prefer the factory methods `with_reference()`
        or `with_reference_utm()` instead of calling __init__ directly.

        Args:
            utm_crs: The UTM coordinate reference system (e.g., "EPSG:25830")
            _ref_easting: Internal - reference UTM easting (use factory methods)
            _ref_northing: Internal - reference UTM northing (use factory methods)
            _ref_lat: Internal - reference latitude (use factory methods)
            _ref_lon: Internal - reference longitude (use factory methods)
        """
        if not PYPROJ_AVAILABLE:
            raise ImportError(
                "pyproj is required for UTM conversion. Install with: pip install pyproj"
            )

        self.utm_crs = utm_crs
        self._to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        self._to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

        # Reference point (immutable after construction)
        self._ref_easting: float | None = _ref_easting
        self._ref_northing: float | None = _ref_northing
        self._ref_lat: float | None = _ref_lat
        self._ref_lon: float | None = _ref_lon

    @classmethod
    def with_reference(
        cls, lat: Degrees, lon: Degrees, utm_crs: str = DEFAULT_UTM_CRS
    ) -> UTMConverter:
        """
        Factory method to create a UTMConverter with GPS reference already set.

        This is the preferred immutable pattern for creating configured converters.
        Returns a new instance with the reference point already configured.

        Args:
            lat: Reference latitude in decimal degrees
            lon: Reference longitude in decimal degrees
            utm_crs: The UTM coordinate reference system (default: EPSG:25830)

        Returns:
            A new UTMConverter instance with the reference point set

        Example:
            >>> converter = UTMConverter.with_reference(lat=39.5, lon=-0.5)
            >>> x, y = converter.gps_to_local_xy(39.501, -0.499)
        """
        # Create temporary converter to compute UTM coordinates
        temp = cls(utm_crs=utm_crs)
        easting, northing = temp._to_utm.transform(lon, lat)

        return cls(
            utm_crs=utm_crs,
            _ref_easting=Meters(easting),
            _ref_northing=Meters(northing),
            _ref_lat=lat,
            _ref_lon=lon,
        )

    @classmethod
    def with_reference_utm(
        cls, easting: Meters, northing: Meters, utm_crs: str = DEFAULT_UTM_CRS
    ) -> UTMConverter:
        """
        Factory method to create a UTMConverter with UTM reference already set.

        This is the preferred immutable pattern for creating configured converters.
        Returns a new instance with the reference point already configured.

        Args:
            easting: Reference UTM easting in meters
            northing: Reference UTM northing in meters
            utm_crs: The UTM coordinate reference system (default: EPSG:25830)

        Returns:
            A new UTMConverter instance with the reference point set

        Example:
            >>> converter = UTMConverter.with_reference_utm(easting=737575.0, northing=4391595.0)
            >>> x, y = converter.utm_to_local_xy(737580.0, 4391600.0)
        """
        # Create temporary converter to compute GPS coordinates
        temp = cls(utm_crs=utm_crs)
        lon, lat = temp._to_wgs84.transform(easting, northing)

        return cls(
            utm_crs=utm_crs,
            _ref_easting=easting,
            _ref_northing=northing,
            _ref_lat=lat,
            _ref_lon=lon,
        )

    @property
    def has_reference(self) -> bool:
        """Check if reference point is set."""
        return self._ref_easting is not None and self._ref_northing is not None

    @property
    def reference_easting(self) -> Meters | None:
        """Get reference UTM easting (read-only)."""
        return Meters(self._ref_easting) if self._ref_easting is not None else None

    @property
    def reference_northing(self) -> Meters | None:
        """Get reference UTM northing (read-only)."""
        return Meters(self._ref_northing) if self._ref_northing is not None else None

    @property
    def reference_lat(self) -> Degrees | None:
        """Get reference latitude (read-only)."""
        return Degrees(self._ref_lat) if self._ref_lat is not None else None

    @property
    def reference_lon(self) -> Degrees | None:
        """Get reference longitude (read-only)."""
        return Degrees(self._ref_lon) if self._ref_lon is not None else None

    def _require_reference(self) -> tuple[float, float]:
        """Return (easting, northing) or raise ValueError if not set."""
        if self._ref_easting is None or self._ref_northing is None:
            raise ValueError(
                "Reference point not set. Use with_reference() or with_reference_utm() "
                "factory methods to create a converter with a reference point."
            )
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


class GCPCoordinateConverter:
    """
    Unified coordinate converter for GCP operations.

    This class provides a consistent interface for converting between GPS, UTM,
    and local XY coordinates. It uses UTM (via pyproj) for maximum accuracy.

    Use the factory methods `with_reference_gps()` or `with_reference_utm()` to
    create configured instances (immutable pattern).

    Usage:
        >>> converter = GCPCoordinateConverter.with_reference_gps(39.640472, -0.230194)
        >>> x, y = converter.gps_to_local(39.640500, -0.230100)
        >>> lat, lon = converter.local_to_gps(10.0, -5.0)
    """

    def __init__(
        self,
        utm_crs: str = DEFAULT_UTM_CRS,
        *,
        _utm_converter: UTMConverter | None = None,
        _ref_lat: float | None = None,
        _ref_lon: float | None = None,
        _ref_easting: float | None = None,
        _ref_northing: float | None = None,
    ):
        """
        Initialize the coordinate converter.

        Note: For most use cases, prefer the factory methods `with_reference_gps()`
        or `with_reference_utm()` instead of calling __init__ directly.

        Args:
            utm_crs: UTM coordinate reference system (default: EPSG:25830)
            _utm_converter: Internal - pre-configured UTM converter
            _ref_lat: Internal - reference latitude
            _ref_lon: Internal - reference longitude
            _ref_easting: Internal - reference UTM easting
            _ref_northing: Internal - reference UTM northing
        """
        self.utm_crs = utm_crs
        self._utm_converter = _utm_converter
        self._ref_lat = _ref_lat
        self._ref_lon = _ref_lon
        self._ref_easting = _ref_easting
        self._ref_northing = _ref_northing

    @classmethod
    def with_reference_gps(
        cls, lat: Degrees, lon: Degrees, utm_crs: str = DEFAULT_UTM_CRS
    ) -> GCPCoordinateConverter:
        """
        Factory method to create a GCPCoordinateConverter with GPS reference already set.

        This is the preferred immutable pattern for creating configured converters.
        Returns a new instance with the reference point already configured.

        Args:
            lat: Reference latitude in decimal degrees
            lon: Reference longitude in decimal degrees
            utm_crs: UTM coordinate reference system (default: EPSG:25830)

        Returns:
            A new GCPCoordinateConverter instance with the reference point set

        Example:
            >>> converter = GCPCoordinateConverter.with_reference_gps(lat=39.640472, lon=-0.230194)
            >>> x, y = converter.gps_to_local(39.640500, -0.230100)
        """
        utm_converter = UTMConverter.with_reference(lat, lon, utm_crs)
        return cls(
            utm_crs=utm_crs,
            _utm_converter=utm_converter,
            _ref_lat=lat,
            _ref_lon=lon,
            _ref_easting=utm_converter.reference_easting,
            _ref_northing=utm_converter.reference_northing,
        )

    @classmethod
    def with_reference_utm(
        cls,
        easting: Meters,
        northing: Meters,
        utm_crs: str = DEFAULT_UTM_CRS,
    ) -> GCPCoordinateConverter:
        """
        Factory method to create a GCPCoordinateConverter with UTM reference already set.

        This is the preferred immutable pattern for creating configured converters.
        Returns a new instance with the reference point already configured.

        Args:
            easting: Reference UTM easting in meters
            northing: Reference UTM northing in meters
            utm_crs: UTM coordinate reference system (default: EPSG:25830)

        Returns:
            A new GCPCoordinateConverter instance with the reference point set

        Example:
            >>> converter = GCPCoordinateConverter.with_reference_utm(
            ...     easting=737575.0, northing=4391595.0
            ... )
            >>> x, y = converter.utm_to_local(737580.0, 4391600.0)
        """
        utm_converter = UTMConverter.with_reference_utm(easting, northing, utm_crs)
        return cls(
            utm_crs=utm_crs,
            _utm_converter=utm_converter,
            _ref_lat=utm_converter.reference_lat,
            _ref_lon=utm_converter.reference_lon,
            _ref_easting=easting,
            _ref_northing=northing,
        )

    @property
    def has_reference(self) -> bool:
        """Check if reference point is set."""
        return self._utm_converter is not None and self._utm_converter.has_reference

    @property
    def using_utm(self) -> bool:
        """Return True if using UTM projection."""
        return self._utm_converter is not None

    def _require_gps_reference(self) -> None:
        """Raise ValueError if GPS reference point not set."""
        if self._ref_lat is None or self._ref_lon is None:
            raise ValueError(
                "Reference point not set. Use with_reference_gps() or with_reference_utm() "
                "factory methods to create a converter with a reference point."
            )

    def _require_utm_reference(self) -> tuple[float, float]:
        """Return (easting, northing) or raise ValueError if not set."""
        if self._ref_easting is None or self._ref_northing is None:
            raise ValueError(
                "Reference point not set. Use with_reference_gps() or with_reference_utm() "
                "factory methods to create a converter with a reference point."
            )
        return self._ref_easting, self._ref_northing

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
