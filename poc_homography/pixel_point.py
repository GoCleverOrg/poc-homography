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
