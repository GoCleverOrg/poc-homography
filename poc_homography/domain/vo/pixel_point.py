"""Pixel coordinate representation."""

from dataclasses import dataclass

from poc_homography.types import Pixels, PixelsFloat


@dataclass(frozen=True)
class PixelPoint:
    """Pixel coordinates in an image.

    Attributes:
        x: Pixel x coordinate (column).
        y: Pixel y coordinate (row).
    """

    _x: float
    _y: float

    @property
    def _pixel_x(self) -> int:
        return round(self._x)

    @property
    def _pixel_y(self) -> int:
        return round(self._y)

    @property
    def x(self) -> PixelsFloat:
        return PixelsFloat(self._x)

    @property
    def y(self) -> PixelsFloat:
        return PixelsFloat(self._y)

    @property
    def pixels_x(self) -> Pixels:
        return Pixels(self._pixel_x)

    @property
    def pixels_y(self) -> Pixels:
        return Pixels(self._pixel_y)
