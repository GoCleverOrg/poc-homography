"""Camera intrinsic parameters value object."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from poc_homography.domain.vo.matrix3x3 import Matrix3x3
from poc_homography.types import Millimeters, Pixels, PixelsFloat

_PRIVATE_SENTINEL = object()


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsic parameters.

    Encapsulates the physical camera/sensor properties and provides
    the 3x3 intrinsic matrix K for projective geometry calculations.

    The intrinsic matrix K has the form:
        [fx  0  cx]
        [0  fy  cy]
        [0   0   1]

    Where fx=fy (assuming square pixels), cx and cy are the principal point.

    Use the `create()` factory method to construct instances.
    Direct constructor access is reserved for internal use.

    Attributes:
        sensor_width: Sensor width in millimeters.
        base_focal_length: Base focal length in millimeters (at 1x zoom).
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        focal_length: Focal length in millimeters (zoom-adjusted).
    """

    sensor_width: Millimeters
    base_focal_length: Millimeters
    image_width: Pixels
    image_height: Pixels
    focal_length: Millimeters
    _K: Matrix3x3 = field(repr=False)
    _sentinel: object = field(default=None, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        """Verify construction was via create() factory."""
        if self._sentinel is not _PRIVATE_SENTINEL:
            raise TypeError(
                "CameraIntrinsics cannot be instantiated directly. "
                "Use CameraIntrinsics.create() instead."
            )

    @classmethod
    def create(
        cls,
        sensor_width: Millimeters,
        base_focal_length: Millimeters,
        image_width: Pixels,
        image_height: Pixels,
        focal_length: Millimeters,
    ) -> CameraIntrinsics:
        """Create CameraIntrinsics with validation.

        Args:
            sensor_width: Sensor width in millimeters.
            base_focal_length: Base focal length in millimeters (at 1x zoom).
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            focal_length: Focal length in millimeters (zoom-adjusted).

        Returns:
            New CameraIntrinsics instance.

        Raises:
            ValueError: If parameters are invalid (non-positive values).
        """
        if sensor_width <= 0:
            raise ValueError(f"sensor_width must be positive, got {sensor_width}")
        if base_focal_length <= 0:
            raise ValueError(f"base_focal_length must be positive, got {base_focal_length}")
        if image_width <= 0:
            raise ValueError(f"image_width must be positive, got {image_width}")
        if image_height <= 0:
            raise ValueError(f"image_height must be positive, got {image_height}")
        if focal_length <= 0:
            raise ValueError(f"focal_length must be positive, got {focal_length}")

        # Compute derived values
        focal_length_px = focal_length * (image_width / sensor_width)
        cx = image_width / 2.0
        cy = image_height / 2.0

        # Build intrinsic matrix
        K_array = np.array(
            [
                [focal_length_px, 0.0, cx],
                [0.0, focal_length_px, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        K = Matrix3x3.create(K_array)

        return cls(
            sensor_width=sensor_width,
            base_focal_length=base_focal_length,
            image_width=image_width,
            image_height=image_height,
            focal_length=focal_length,
            _K=K,
            _sentinel=_PRIVATE_SENTINEL,
        )

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

    def to_K(self) -> Matrix3x3:
        """Get the 3x3 intrinsic matrix K.

        Returns:
            The intrinsic matrix as a Matrix3x3 value object.
        """
        return self._K
