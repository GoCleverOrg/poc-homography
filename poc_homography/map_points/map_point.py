"""Map point representation with pixel coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from poc_homography.pixel_point import PixelPoint


@dataclass(frozen=True)
class MapPoint:
    """A reference point on a map with pixel coordinates.

    This represents a point identified on a map image using pixel coordinates,
    without any geographic (lat/lon) reference. This is the core data structure
    for map-based reference points that don't require geographic transformations.

    The ID and map association are managed externally by MapPointRegistry,
    which stores points in a dictionary keyed by their IDs.

    Attributes:
        pixel_x: X coordinate in pixels (column).
        pixel_y: Y coordinate in pixels (row).
    """

    pixel_x: float
    pixel_y: float

    @cached_property
    def pixel(self) -> PixelPoint:
        """Get pixel coordinates as a PixelPoint."""
        return PixelPoint(self.pixel_x, self.pixel_y)

    def to_dict(self) -> dict[str, Any]:
        """Convert MapPoint to a dictionary for JSON serialization.

        Returns:
            Dictionary with pixel_x and pixel_y keys.
            Note: The id is managed by the MapPointRegistry and added during serialization.
        """
        return {
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MapPoint:
        """Create MapPoint from a dictionary.

        Args:
            data: Dictionary with pixel_x and pixel_y keys.
                  The id key (if present) is ignored as it's managed by MapPointRegistry.

        Returns:
            New MapPoint instance.

        Raises:
            KeyError: If required keys are missing from data.
            ValueError: If data types are invalid.
        """
        return cls(
            pixel_x=float(data["pixel_x"]),
            pixel_y=float(data["pixel_y"]),
        )
