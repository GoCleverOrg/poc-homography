"""Image dimensions value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.domain.vo.pixel_point import PixelPoint
from poc_homography.types import Pixels, PixelsFloat


@dataclass(frozen=True)
class ImageDimensions:
    """Image dimensions representing width and height in pixels.

    A simple value object that encapsulates image dimensions, providing
    computed properties for common geometric calculations.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
    """

    width: Pixels
    height: Pixels

    @classmethod
    def create(cls, width: int, height: int) -> ImageDimensions:
        """Create ImageDimensions from integer values.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            New ImageDimensions instance.

        Raises:
            ValueError: If width or height is not positive.
        """
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if height <= 0:
            raise ValueError(f"height must be positive, got {height}")
        return cls(width=Pixels(width), height=Pixels(height))

    @property
    def area(self) -> int:
        """Total area in pixels."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Width-to-height aspect ratio."""
        return self.width / self.height

    @property
    def center(self) -> PixelPoint:
        """Center point of the image."""
        return PixelPoint.create(self.center_x, self.center_y)

    @property
    def center_x(self) -> PixelsFloat:
        """X coordinate of image center."""
        return PixelsFloat(self.width / 2.0)

    @property
    def center_y(self) -> PixelsFloat:
        """Y coordinate of image center."""
        return PixelsFloat(self.height / 2.0)

    def contains(self, point: PixelPoint) -> bool:
        """Check if a point is within the image bounds.

        Args:
            point: The pixel point to check.

        Returns:
            True if point is within [0, width) x [0, height).
        """
        return 0 <= point.x < self.width and 0 <= point.y < self.height

    def to_tuple(self) -> tuple[int, int]:
        """Convert to (width, height) tuple."""
        return (int(self.width), int(self.height))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageDimensions:
        """Create ImageDimensions from dictionary."""
        return cls.create(width=data["width"], height=data["height"])
