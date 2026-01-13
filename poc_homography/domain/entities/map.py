"""Map entity representing a georeferenced map image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poc_homography.domain.vo.geotiff import GeoTiff
    from poc_homography.domain.vo.photo import Photo


@dataclass
class Map:
    """A georeferenced map image.

    Represents a map image file with associated GeoTiff metadata for
    converting between pixel coordinates and geographic coordinates.

    Attributes:
        id: Unique identifier for the map (e.g., "valte_site").
        photo: The map image with its dimensions.
        geotiff: GeoTiff metadata for coordinate transformation.
    """

    id: str
    photo: Photo
    geotiff: GeoTiff
