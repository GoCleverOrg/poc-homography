"""Pixel coordinate representation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PixelPoint:
    """Pixel coordinates in an image.

    Attributes:
        x: Pixel x coordinate (column).
        y: Pixel y coordinate (row).
    """

    x: float
    y: float

    @property
    def to_pixel(self) -> tuple[int, int]:
        """Convert to integer pixel coordinates.

        Returns:
            Tuple of (x, y) rounded to nearest integer.
        """
        return (round(self.x), round(self.y))
