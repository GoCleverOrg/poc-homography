"""Georeferencing configuration."""

from dataclasses import dataclass

# Type alias for the 6-parameter GDAL geotransform
Geotransform = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class GeoConfig:
    """Configuration for georeferenced coordinate transformations.

    Attributes:
        crs: Coordinate reference system identifier (e.g., "EPSG:25830").
        geotransform: 6-parameter GDAL affine geotransform as
            (origin_x, pixel_width, row_rotation, origin_y, col_rotation, pixel_height).
            For north-up images, row_rotation and col_rotation are typically 0.
    """

    crs: str
    geotransform: Geotransform
