"""GeoTiff value object for coordinate transformations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from pyproj import Transformer

from poc_homography.types import (
    Easting,
    Latitude,
    Longitude,
    Meters,
    Northing,
    PixelsFloat,
    Unitless,
)

# WGS84 geographic CRS — the lat/lon datum used for GPS output.
_WGS84_CRS = "EPSG:4326"


@lru_cache(maxsize=8)
def _to_wgs84_transformer(source_crs: str) -> Transformer:
    """Return a cached pyproj transformer from *source_crs* to WGS84 lat/lon.

    Building a :class:`pyproj.Transformer` is comparatively expensive, so
    instances are memoised per source CRS. ``always_xy=True`` keeps the input
    in (easting, northing) / (lon, lat) axis order regardless of the CRS's
    declared axis ordering.
    """
    return Transformer.from_crs(source_crs, _WGS84_CRS, always_xy=True)


@dataclass(frozen=True)
class GeoTransform:
    """Affine transformation parameters for pixel to geographic coordinate conversion.

    The 6-parameter affine transformation that maps pixel coordinates to
    geographic coordinates (typically UTM).

    Coordinate transformation:
        X_geo = origin_easting + pixel_x * pixel_width + pixel_y * row_rotation
        Y_geo = origin_northing + pixel_x * col_rotation + pixel_y * pixel_height

    Attributes:
        origin_easting: X-coordinate of the upper-left corner (GT[0]).
        pixel_width: Pixel width in ground units, typically meters (GT[1]).
        row_rotation: Row rotation angle (GT[2]), typically 0 for north-up.
        origin_northing: Y-coordinate of the upper-left corner (GT[3]).
        col_rotation: Column rotation angle (GT[4]), typically 0 for north-up.
        pixel_height: Pixel height in ground units, negative for north-up (GT[5]).
    """

    origin_easting: Easting
    pixel_width: Meters
    row_rotation: Unitless
    origin_northing: Northing
    col_rotation: Unitless
    pixel_height: Meters

    def to_list(self) -> list[float]:
        """Return as GDAL-style 6-element list [GT0..GT5]."""
        return [
            float(self.origin_easting),
            float(self.pixel_width),
            float(self.row_rotation),
            float(self.origin_northing),
            float(self.col_rotation),
            float(self.pixel_height),
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "origin_easting": float(self.origin_easting),
            "pixel_width": float(self.pixel_width),
            "row_rotation": float(self.row_rotation),
            "origin_northing": float(self.origin_northing),
            "col_rotation": float(self.col_rotation),
            "pixel_height": float(self.pixel_height),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeoTransform:
        """Create GeoTransform from dictionary."""
        return cls(
            origin_easting=Easting(data["origin_easting"]),
            pixel_width=Meters(data["pixel_width"]),
            row_rotation=Unitless(data["row_rotation"]),
            origin_northing=Northing(data["origin_northing"]),
            col_rotation=Unitless(data["col_rotation"]),
            pixel_height=Meters(data["pixel_height"]),
        )


@dataclass(frozen=True)
class GeoTiff:
    """GeoTIFF metadata for pixel to geographic coordinate transforms.

    Attributes:
        geotransform: Affine transformation parameters.
        crs: Coordinate reference system (e.g., "EPSG:25830").
    """

    geotransform: GeoTransform
    crs: str

    def pixel_to_geo(self, pixel_x: PixelsFloat, pixel_y: PixelsFloat) -> tuple[Easting, Northing]:
        """Convert pixel coordinates to geographic coordinates.

        Args:
            pixel_x: Pixel x-coordinate (column).
            pixel_y: Pixel y-coordinate (row).

        Returns:
            Tuple of (easting, northing) in the CRS units.
        """
        gt = self.geotransform
        easting = Easting(
            gt.origin_easting + pixel_x * float(gt.pixel_width) + pixel_y * gt.row_rotation
        )
        northing = Northing(
            gt.origin_northing + pixel_x * gt.col_rotation + pixel_y * float(gt.pixel_height)
        )
        return (easting, northing)

    def pixel_to_latlon(
        self, pixel_x: PixelsFloat, pixel_y: PixelsFloat
    ) -> tuple[Latitude, Longitude]:
        """Convert map pixel coordinates to WGS84 geographic coordinates.

        Chains :meth:`pixel_to_geo` (pixel → projected easting/northing in this
        GeoTIFF's CRS) with a pyproj reprojection to WGS84 lat/lon, giving the
        GPS coordinate of a point identified on the georeferenced map.

        Args:
            pixel_x: Pixel x-coordinate (column).
            pixel_y: Pixel y-coordinate (row).

        Returns:
            Tuple of (latitude, longitude) in decimal degrees (WGS84).
        """
        easting, northing = self.pixel_to_geo(pixel_x, pixel_y)
        lon, lat = _to_wgs84_transformer(self.crs).transform(float(easting), float(northing))
        return (Latitude(float(lat)), Longitude(float(lon)))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "geotransform": self.geotransform.to_dict(),
            "crs": self.crs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeoTiff:
        """Create GeoTiff from dictionary."""
        return cls(
            geotransform=GeoTransform.from_dict(data["geotransform"]),
            crs=data["crs"],
        )
