"""Map point representation with pixel coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache as cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.pixel_point import PixelPoint


@dataclass(frozen=True)
class MapPoint:
    """A reference point on a map with pixel coordinates.

    This represents a point identified on a map image using pixel coordinates,
    without any geographic (lat/lon) reference. This is the core data structure
    for map-based reference points that don't require geographic transformations.

    Attributes:
        id: Unique identifier for the point (e.g., "Z1", "P5", "A3").
        pixel_x: X coordinate in pixels (column).
        pixel_y: Y coordinate in pixels (row).
        map_id: Identifier of the map this point belongs to (e.g., "map_valte").
    """

    id: str
    pixel_x: float
    pixel_y: float
    map_id: str

    @property
    @cache(maxsize=1)  # noqa: B019 - safe on frozen dataclass
    def pixel(self) -> PixelPoint:
        """Get pixel coordinates as a PixelPoint.

        Returns:
            PixelPoint with x=pixel_x and y=pixel_y.

        Note:
            This property is cached since MapPoint is immutable (frozen dataclass).
        """
        from poc_homography.pixel_point import PixelPoint

        return PixelPoint(self.pixel_x, self.pixel_y)

    def to_dict(self) -> dict[str, Any]:
        """Convert MapPoint to a dictionary for JSON serialization.

        Returns:
            Dictionary with id, pixel_x, pixel_y, and map_id keys.
        """
        return {
            "id": self.id,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "map_id": self.map_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MapPoint:
        """Create MapPoint from a dictionary.

        Args:
            data: Dictionary with id, pixel_x, pixel_y, and map_id keys.

        Returns:
            New MapPoint instance.

        Raises:
            KeyError: If required keys are missing from data.
            ValueError: If data types are invalid.
        """
        return cls(
            id=str(data["id"]),
            pixel_x=float(data["pixel_x"]),
            pixel_y=float(data["pixel_y"]),
            map_id=str(data["map_id"]),
        )
