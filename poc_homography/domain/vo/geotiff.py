"""GeoTiff value object for coordinate transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.types import Meters


@dataclass(frozen=True)
class GeoTiff:
    """GeoTIFF metadata for pixel to geographic coordinate transforms.

    The geotransform is a 6-parameter affine transformation that maps
    pixel coordinates to geographic coordinates (typically UTM).

    The geotransform parameters are:
        GT[0]: x-coordinate of the upper-left corner
        GT[1]: pixel width (x-resolution)
        GT[2]: row rotation (typically 0 for north-up images)
        GT[3]: y-coordinate of the upper-left corner
        GT[4]: column rotation (typically 0 for north-up images)
        GT[5]: pixel height (y-resolution, typically negative for north-up)

    Coordinate transformation:
        X_geo = GT[0] + pixel_x * GT[1] + pixel_y * GT[2]
        Y_geo = GT[3] + pixel_x * GT[4] + pixel_y * GT[5]

    Attributes:
        geotransform: 6-parameter affine transformation tuple.
        crs: Coordinate reference system (e.g., "EPSG:25830").
    """

    geotransform: tuple[float, float, float, float, float, float]
    crs: str

    @property
    def origin_easting(self) -> float:
        """X-coordinate of the upper-left corner (GT[0])."""
        return self.geotransform[0]

    @property
    def origin_northing(self) -> float:
        """Y-coordinate of the upper-left corner (GT[3])."""
        return self.geotransform[3]

    @property
    def pixel_width(self) -> Meters:
        """Pixel width in ground units, typically meters (GT[1])."""
        return Meters(self.geotransform[1])

    @property
    def pixel_height(self) -> Meters:
        """Pixel height in ground units, typically negative for north-up (GT[5])."""
        return Meters(self.geotransform[5])

    @property
    def row_rotation(self) -> float:
        """Row rotation angle (GT[2]), typically 0 for north-up."""
        return self.geotransform[2]

    @property
    def col_rotation(self) -> float:
        """Column rotation angle (GT[4]), typically 0 for north-up."""
        return self.geotransform[4]

    @property
    def is_north_up(self) -> bool:
        """True if the image is north-up (no rotation)."""
        return self.row_rotation == 0.0 and self.col_rotation == 0.0

    def pixel_to_geo(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """Convert pixel coordinates to geographic coordinates.

        Args:
            pixel_x: Pixel x-coordinate (column).
            pixel_y: Pixel y-coordinate (row).

        Returns:
            Tuple of (easting, northing) in the CRS units.
        """
        gt = self.geotransform
        easting = gt[0] + pixel_x * gt[1] + pixel_y * gt[2]
        northing = gt[3] + pixel_x * gt[4] + pixel_y * gt[5]
        return (easting, northing)

    def geo_to_pixel(self, easting: float, northing: float) -> tuple[float, float]:
        """Convert geographic coordinates to pixel coordinates.

        Args:
            easting: X-coordinate in CRS units.
            northing: Y-coordinate in CRS units.

        Returns:
            Tuple of (pixel_x, pixel_y).

        Raises:
            ValueError: If the geotransform is singular (cannot be inverted).
        """
        gt = self.geotransform
        det = gt[1] * gt[5] - gt[2] * gt[4]
        if det == 0:
            raise ValueError("Geotransform is singular and cannot be inverted")

        dx = easting - gt[0]
        dy = northing - gt[3]
        pixel_x = (gt[5] * dx - gt[2] * dy) / det
        pixel_y = (gt[1] * dy - gt[4] * dx) / det
        return (pixel_x, pixel_y)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "geotransform": list(self.geotransform),
            "crs": self.crs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeoTiff:
        """Create GeoTiff from dictionary."""
        return cls(
            geotransform=tuple(data["geotransform"]),  # type: ignore[arg-type]
            crs=data["crs"],
        )
