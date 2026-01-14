"""Map point representation with pixel coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.vo.pixel_point import PixelPoint


@dataclass(frozen=True)
class MapPoint:
    """A reference point on a map with pixel coordinates.

    This represents a point identified on a map image using pixel coordinates,
    without any geographic (lat/lon) reference. This is the core data structure
    for map-based reference points that don't require geographic transformations.

    The ID and map association are managed externally by GroundControlPointCollection,
    which stores points in a dictionary keyed by their IDs.

    Attributes:
        pixel_x: X coordinate in pixels (column).
        pixel_y: Y coordinate in pixels (row).
    """

    map_id: str
    pixel_point: PixelPoint

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "map_id": self.map_id,
            "pixel_point": self.pixel_point.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MapPoint:
        """Create MapPoint from dictionary."""
        from poc_homography.domain.vo.pixel_point import PixelPoint

        return cls(
            map_id=data["map_id"],
            pixel_point=PixelPoint.from_dict(data["pixel_point"]),
        )
