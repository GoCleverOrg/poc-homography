"""Site model representing a physical location with reference imagery.

A Site represents a physical location that cameras can observe. Multiple cameras
can be associated with the same site. Sites contain the reference imagery and
coordinate system information needed for georeferencing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Site:
    """A physical location with reference imagery and coordinate system.

    Sites represent physical locations like terminals, ports, or facilities.
    Multiple cameras can observe the same site. The site contains the reference
    image and geotransform parameters needed for georeferencing operations.

    Attributes:
        name: Unique site identifier (e.g., "Valte", "Setram").
        image_path: Path to reference PNG/GeoTIFF file, empty string if unconfigured.
        geotransform: GDAL 6-parameter GeoTransform list:
            [origin_easting, pixel_width, row_rotation, origin_northing, col_rotation, pixel_height]
            Empty list if unconfigured.
        utm_crs: CRS identifier string (e.g., "EPSG:25830"), empty string if unconfigured.
    """

    name: str
    image_path: str
    geotransform: list[float]
    utm_crs: str

    @property
    def is_configured(self) -> bool:
        """Check if site has a configured reference image.

        Returns:
            True if image_path is non-empty, False otherwise.
        """
        return bool(self.image_path)
