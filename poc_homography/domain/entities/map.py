"""Map entity representing a georeferenced map image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.vo.geotiff import GeoTiff
    from poc_homography.domain.vo.photo import Photo


@dataclass
class Map:
    """A georeferenced map image.

    Represents a map image file (PNG) with associated GeoTiff metadata for
    converting between pixel coordinates and geographic coordinates.

    Attributes:
        id: Unique identifier for the map (e.g., "valte").
        photo: The map image with its dimensions.
        geotiff: GeoTiff metadata for coordinate transformation.
    """

    id: str
    photo: Photo
    geotiff: GeoTiff

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "photo": self.photo.to_dict(),
            "geotiff": self.geotiff.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Map:
        """Create Map from dictionary."""
        from poc_homography.domain.vo.geotiff import GeoTiff
        from poc_homography.domain.vo.photo import Photo

        return cls(
            id=data["id"],
            photo=Photo.from_dict(data["photo"]),
            geotiff=GeoTiff.from_dict(data["geotiff"]),
        )
