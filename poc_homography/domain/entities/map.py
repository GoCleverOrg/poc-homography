"""Map entity representing a georeferenced map image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from poc_homography.domain.vo.geotiff import GeoTiff
    from poc_homography.domain.vo.photo import Photo
    from poc_homography.types import Pixels


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

    def to_dict(self, *, photo_base_path: Path | None = None) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Args:
            photo_base_path: Base directory for making photo path relative.

        Returns:
            Dictionary representation suitable for YAML.
        """
        return {
            "id": self.id,
            "photo": self.photo.to_dict(relative_to=photo_base_path),
            "geotiff": self.geotiff.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        photo_width: Pixels,
        photo_height: Pixels,
        photo_base_path: Path | None = None,
    ) -> Map:
        """Create Map from dictionary.

        Args:
            data: Dictionary with 'id', 'photo', and 'geotiff' keys.
            photo_width: Image width (loaded by caller).
            photo_height: Image height (loaded by caller).
            photo_base_path: Base directory for resolving relative photo paths.

        Returns:
            Map instance.
        """
        from poc_homography.domain.vo.geotiff import GeoTiff
        from poc_homography.domain.vo.photo import Photo

        return cls(
            id=data["id"],
            photo=Photo.from_dict(
                data["photo"],
                width=photo_width,
                height=photo_height,
                base_path=photo_base_path,
            ),
            geotiff=GeoTiff.from_dict(data["geotiff"]),
        )
