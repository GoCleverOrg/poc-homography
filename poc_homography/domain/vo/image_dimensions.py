"""Image dimensions value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    def center_x(self) -> PixelsFloat:
        """X coordinate of image center."""
        return PixelsFloat(self.width / 2.0)

    @property
    def center_y(self) -> PixelsFloat:
        """Y coordinate of image center."""
        return PixelsFloat(self.height / 2.0)

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
