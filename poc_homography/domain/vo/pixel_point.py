"""Pixel coordinate representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from poc_homography.types import Pixels, PixelsFloat

if TYPE_CHECKING:
    from poc_homography.domain.vo.vector3 import Vector3


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

    @property
    def pixels_x(self) -> Pixels:
        """X coordinate rounded to integer pixels."""
        return Pixels(round(self.x))

    @property
    def pixels_y(self) -> Pixels:
        """Y coordinate rounded to integer pixels."""
        return Pixels(round(self.y))

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

    def to_homogeneous(self) -> Vector3:
        """Convert to homogeneous coordinates.

        Returns:
            Vector3 with [x, y, 1] representing this point in homogeneous form.
        """
        from poc_homography.domain.vo.vector3 import Vector3

        return Vector3.create(float(self.x), float(self.y), 1.0)

    @classmethod
    def from_homogeneous(cls, v: Vector3) -> PixelPoint:
        """Create PixelPoint from homogeneous coordinates.

        Args:
            v: Vector3 in homogeneous form [x, y, w].

        Returns:
            PixelPoint with coordinates (x/w, y/w).

        Raises:
            ValueError: If w component is near zero (point at infinity).
        """
        normalized = v.normalized()
        return cls.create(normalized.x, normalized.y)
