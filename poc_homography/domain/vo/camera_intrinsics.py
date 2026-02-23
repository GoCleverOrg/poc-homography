"""Camera intrinsic parameters value object."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from poc_homography.domain.vo.image_dimensions import ImageDimensions
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
        dimensions: Image dimensions (width and height in pixels).
        focal_length: Focal length in millimeters (zoom-adjusted).
    """

    sensor_width: Millimeters
    base_focal_length: Millimeters
    dimensions: ImageDimensions
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

    @property
    def image_width(self) -> Pixels:
        """Image width in pixels (backward-compatible property)."""
        return self.dimensions.width

    @property
    def image_height(self) -> Pixels:
        """Image height in pixels (backward-compatible property)."""
        return self.dimensions.height

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

        # Create ImageDimensions
        dimensions = ImageDimensions.create(width=image_width, height=image_height)

        # Compute derived values
        focal_length_px = focal_length * (dimensions.width / sensor_width)
        cx = dimensions.center_x
        cy = dimensions.center_y

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
            dimensions=dimensions,
            focal_length=focal_length,
            _K=K,
            _sentinel=_PRIVATE_SENTINEL,
        )

    @classmethod
    def from_K_matrix(
        cls,
        K: np.ndarray,
        image_width: Pixels,
        image_height: Pixels,
        sensor_width: Millimeters = Millimeters(6.78),
        base_focal_length: Millimeters = Millimeters(5.9),
    ) -> CameraIntrinsics:
        """Create from pre-computed K matrix for legacy code migration.

        This factory allows creating CameraIntrinsics from an existing K matrix
        when the physical sensor parameters are not known precisely.

        Args:
            K: Pre-computed 3x3 intrinsic matrix.
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            sensor_width: Sensor width in mm (default: Hikvision PTZ spec).
            base_focal_length: Base focal length in mm (default: 5.9mm).

        Returns:
            New CameraIntrinsics instance with matching K matrix.

        Raises:
            ValueError: If K matrix is invalid (not 3x3, contains NaN/Inf).
        """
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got shape {K.shape}")
        if not np.all(np.isfinite(K)):
            raise ValueError("K matrix contains NaN or Infinity values")

        focal_length_px = float(K[0, 0])
        focal_length_mm = Millimeters(focal_length_px * sensor_width / image_width)
        dimensions = ImageDimensions.create(width=image_width, height=image_height)

        K_rebuilt = np.array(
            [
                [focal_length_px, 0.0, dimensions.center_x],
                [0.0, focal_length_px, dimensions.center_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        K_matrix = Matrix3x3.create(K_rebuilt)

        return cls(
            sensor_width=sensor_width,
            base_focal_length=base_focal_length,
            dimensions=dimensions,
            focal_length=focal_length_mm,
            _K=K_matrix,
            _sentinel=_PRIVATE_SENTINEL,
        )

    @property
    def focal_length_px(self) -> PixelsFloat:
        """Focal length in pixels (computed from mm and sensor width)."""
        return PixelsFloat(self.focal_length * (self.dimensions.width / self.sensor_width))

    @property
    def cx(self) -> PixelsFloat:
        """Principal point X coordinate (image center)."""
        return self.dimensions.center_x

    @property
    def cy(self) -> PixelsFloat:
        """Principal point Y coordinate (image center)."""
        return self.dimensions.center_y

    def to_K(self) -> Matrix3x3:
        """Get the 3x3 intrinsic matrix K.

        Returns:
            The intrinsic matrix as a Matrix3x3 value object.
        """
        return self._K
