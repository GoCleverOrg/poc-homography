"""Camera intrinsic parameters value object."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from poc_homography.types import Millimeters, Pixels, PixelsFloat


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsic parameters.

    Core fields define the physical camera/sensor properties.
    Virtual properties are computed on-the-fly from core fields.
    """

    sensor_width: Millimeters
    """Sensor width in millimeters."""

    base_focal_length: Millimeters
    """Base focal length in millimeters (at 1x zoom)."""

    image_width: Pixels
    """Image width in pixels."""

    image_height: Pixels
    """Image height in pixels."""

    focal_length: Millimeters
    """Focal length in millimeters (zoom-adjusted)."""

    @property
    def focal_length_px(self) -> PixelsFloat:
        """Focal length in pixels (computed from mm and sensor width)."""
        return PixelsFloat(self.focal_length * (self.image_width / self.sensor_width))

    @property
    def cx(self) -> PixelsFloat:
        """Principal point X coordinate (image center)."""
        return PixelsFloat(self.image_width / 2.0)

    @property
    def cy(self) -> PixelsFloat:
        """Principal point Y coordinate (image center)."""
        return PixelsFloat(self.image_height / 2.0)

    @property
    def K(self) -> NDArray[np.float64]:
        """Intrinsic matrix (3x3)."""
        f = self.focal_length_px
        return np.array(
            [
                [f, 0, self.cx],
                [0, f, self.cy],
                [0, 0, 1],
            ]
        )
