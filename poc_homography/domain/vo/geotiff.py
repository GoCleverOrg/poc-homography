"""GeoTiff value object for coordinate transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.types import Easting, Meters, Northing, PixelsFloat, Unitless


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
