"""Pixel coordinate representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.types import PixelsFloat


@dataclass(frozen=True)
class PixelPoint:
    """Pixel coordinates in an image.

    Use the `create()` factory method to construct instances from float values.
    Direct constructor requires PixelsFloat types.

    Attributes:
        x: Pixel x coordinate (column).
        y: Pixel y coordinate (row).
    """

    x: PixelsFloat
    y: PixelsFloat

    @classmethod
    def create(cls, x: float, y: float) -> PixelPoint:
        """Create PixelPoint from float coordinates.

        Args:
            x: X coordinate as float.
            y: Y coordinate as float.

        Returns:
            New PixelPoint instance.
        """
        return cls(x=PixelsFloat(x), y=PixelsFloat(y))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "x": float(self.x),
            "y": float(self.y),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PixelPoint:
        """Create PixelPoint from dictionary."""
        return cls.create(x=data["x"], y=data["y"])
