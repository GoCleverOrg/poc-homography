"""Immutable camera parameter and geometry result dataclasses.

This module provides frozen dataclasses for capturing camera configuration
and computed homography state, enabling immutable state management and
easier testing/debugging of the camera geometry pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from poc_homography.types import Degrees, Meters, Pixels, Unitless


@dataclass(frozen=True)
class DistortionCoefficients:
    """Lens distortion coefficients using the OpenCV distortion model.

    The distortion model corrects for radial and tangential lens distortion:
    - Radial distortion (barrel/pincushion): controlled by k1, k2, k3
    - Tangential distortion (decentering): controlled by p1, p2

    For most PTZ cameras, only k1 (and sometimes k2) are significant.
    Positive k1 = barrel distortion (edges curve outward)
    Negative k1 = pincushion distortion (edges curve inward)

    Attributes:
        k1: First radial distortion coefficient (most significant).
        k2: Second radial distortion coefficient.
        p1: First tangential distortion coefficient.
        p2: Second tangential distortion coefficient.
        k3: Third radial distortion coefficient (usually 0).
    """

    k1: Unitless = 0.0  # type: ignore[assignment]
    k2: Unitless = 0.0  # type: ignore[assignment]
    p1: Unitless = 0.0  # type: ignore[assignment]
    p2: Unitless = 0.0  # type: ignore[assignment]
    k3: Unitless = 0.0  # type: ignore[assignment]

    def to_array(self) -> np.ndarray:
        """Convert to numpy array in OpenCV format [k1, k2, p1, p2, k3]."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    def is_zero(self) -> bool:
        """Check if all coefficients are effectively zero."""
        return np.allclose(self.to_array(), 0.0)

    @classmethod
    def from_array(cls, coeffs: np.ndarray) -> DistortionCoefficients:
        """Create from numpy array [k1, k2, p1, p2, k3].

        Args:
            coeffs: Array of 5 distortion coefficients.

        Returns:
            New DistortionCoefficients instance.

        Raises:
            ValueError: If array does not have exactly 5 elements.
        """
        if len(coeffs) != 5:
            raise ValueError(f"Expected 5 distortion coefficients, got {len(coeffs)}")
        return cls(
            k1=Unitless(float(coeffs[0])),
            k2=Unitless(float(coeffs[1])),
            p1=Unitless(float(coeffs[2])),
            p2=Unitless(float(coeffs[3])),
            k3=Unitless(float(coeffs[4])),
        )


@dataclass(frozen=True)
class HeightUncertainty:
    """Height uncertainty bounds for error propagation.

    Represents a confidence interval for camera height, typically from
    height calibration. Used for propagating uncertainty to world
    coordinate projections.

    Attributes:
        lower: Lower bound of height confidence interval in meters.
        upper: Upper bound of height confidence interval in meters.
    """

    lower: Meters
    upper: Meters

    def __post_init__(self) -> None:
        """Validate height uncertainty bounds."""
        if self.lower <= 0:
            raise ValueError(f"Lower bound must be positive, got {self.lower}")
        if self.upper <= 0:
            raise ValueError(f"Upper bound must be positive, got {self.upper}")
        if self.lower > self.upper:
            raise ValueError(f"Lower bound ({self.lower}) cannot exceed upper bound ({self.upper})")


def _validate_matrix_shape(matrix: np.ndarray, expected_shape: tuple[int, ...], name: str) -> None:
    """Validate that a matrix has the expected shape.

    Args:
        matrix: The numpy array to validate.
        expected_shape: Expected shape tuple.
        name: Name of the matrix for error messages.

    Raises:
        ValueError: If shape does not match.
    """
    if matrix.shape != expected_shape:
        raise ValueError(f"{name} must be {expected_shape}, got shape {matrix.shape}")


def _validate_finite(matrix: np.ndarray, name: str) -> None:
    """Validate that all elements in a matrix are finite.

    Args:
        matrix: The numpy array to validate.
        name: Name of the matrix for error messages.

    Raises:
        ValueError: If any element is NaN or Infinity.
    """
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or Infinity values")


@dataclass(frozen=True)
class CameraParameters:
    """Immutable camera configuration for CameraGeometry computation.

    This frozen dataclass captures all camera parameters needed to compute
    the ground plane homography. It provides a snapshot of camera state
    that can be passed around, stored, and compared without mutation concerns.

    Coordinate System Conventions:
        World Frame (Right-Handed):
          - Origin: Arbitrary reference point (typically camera location)
          - X-axis: East (positive = East)
          - Y-axis: North (positive = North)
          - Z-axis: Up (positive = Up, height above ground)
          - Ground plane: Z = 0

        Camera Frame (Right-Handed, standard computer vision):
          - Origin: Camera optical center
          - X-axis: Right (in image)
          - Y-axis: Down (in image)
          - Z-axis: Forward (along optical axis, into the scene)

    Attributes:
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        intrinsic_matrix: 3x3 camera intrinsic matrix K (immutable copy).
        camera_position: Camera world position [X, Y, Z] in meters (immutable copy).
        pan_deg: Pan angle in degrees (positive = right/clockwise from above).
        tilt_deg: Tilt angle in degrees (positive = down, Hikvision convention).
        roll_deg: Roll angle in degrees (positive = clockwise).
        map_width: Width of output map in pixels.
        map_height: Height of output map in pixels.
        pixels_per_meter: Pixels per meter scaling for map projection.
        distortion: Optional lens distortion coefficients.
        height_uncertainty: Optional height uncertainty bounds for error propagation.
        affine_matrix: Optional 3x3 GeoTIFF affine transformation matrix A (immutable copy).
    """

    # Image dimensions
    image_width: Pixels
    image_height: Pixels

    # Intrinsic matrix (stored as bytes for hashability, accessed via property)
    _intrinsic_matrix_data: bytes = field(repr=False)

    # Camera position (stored as bytes for hashability, accessed via property)
    _camera_position_data: bytes = field(repr=False)

    # Orientation angles
    pan_deg: Degrees
    tilt_deg: Degrees
    roll_deg: Degrees

    # Map parameters
    map_width: Pixels
    map_height: Pixels
    pixels_per_meter: Unitless

    # Optional parameters
    distortion: DistortionCoefficients | None = None
    height_uncertainty: HeightUncertainty | None = None

    # Optional GeoTIFF affine matrix (stored as bytes for hashability)
    _affine_matrix_data: bytes | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate all camera parameters."""
        # Validate image dimensions
        if self.image_width <= 0:
            raise ValueError(f"image_width must be positive, got {self.image_width}")
        if self.image_height <= 0:
            raise ValueError(f"image_height must be positive, got {self.image_height}")

        # Validate intrinsic matrix
        K = self.intrinsic_matrix
        _validate_matrix_shape(K, (3, 3), "intrinsic_matrix (K)")
        _validate_finite(K, "intrinsic_matrix (K)")

        # Validate camera position
        pos = self.camera_position
        if len(pos) != 3:
            raise ValueError(f"camera_position must have 3 elements [X, Y, Z], got {len(pos)}")
        _validate_finite(pos, "camera_position")

        # Validate map dimensions
        if self.map_width <= 0:
            raise ValueError(f"map_width must be positive, got {self.map_width}")
        if self.map_height <= 0:
            raise ValueError(f"map_height must be positive, got {self.map_height}")

        # Validate pixels per meter
        if self.pixels_per_meter <= 0:
            raise ValueError(f"pixels_per_meter must be positive, got {self.pixels_per_meter}")

        # Validate affine matrix if present
        if self._affine_matrix_data is not None:
            A = self.affine_matrix
            if A is not None:
                _validate_matrix_shape(A, (3, 3), "affine_matrix (A)")
                _validate_finite(A, "affine_matrix (A)")

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """Get the 3x3 intrinsic matrix K.

        Returns:
            Immutable copy of the intrinsic matrix.
        """
        arr = np.frombuffer(self._intrinsic_matrix_data, dtype=np.float64).reshape(3, 3)
        arr.flags.writeable = False
        return arr

    @property
    def camera_position(self) -> np.ndarray:
        """Get camera world position [X, Y, Z] in meters.

        Returns:
            Immutable copy of camera position array.
        """
        arr = np.frombuffer(self._camera_position_data, dtype=np.float64).copy()
        arr.flags.writeable = False
        return arr

    @property
    def camera_height(self) -> Meters:
        """Get camera height (Z component of position) in meters."""
        return Meters(self.camera_position[2])

    @property
    def affine_matrix(self) -> np.ndarray | None:
        """Get the optional 3x3 GeoTIFF affine transformation matrix A.

        Returns:
            Immutable copy of the affine matrix, or None if not set.
        """
        if self._affine_matrix_data is None:
            return None
        arr = np.frombuffer(self._affine_matrix_data, dtype=np.float64).reshape(3, 3)
        arr.flags.writeable = False
        return arr

    @classmethod
    def create(
        cls,
        image_width: Pixels,
        image_height: Pixels,
        intrinsic_matrix: np.ndarray,
        camera_position: np.ndarray,
        pan_deg: Degrees,
        tilt_deg: Degrees,
        roll_deg: Degrees,
        map_width: Pixels,
        map_height: Pixels,
        pixels_per_meter: Unitless,
        distortion: DistortionCoefficients | None = None,
        height_uncertainty: HeightUncertainty | None = None,
        affine_matrix: np.ndarray | None = None,
    ) -> CameraParameters:
        """Create CameraParameters with numpy arrays.

        This factory method handles the conversion of numpy arrays to their
        internal byte representation for hashability.

        Args:
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            intrinsic_matrix: 3x3 camera intrinsic matrix K.
            camera_position: Camera world position [X, Y, Z] in meters.
            pan_deg: Pan angle in degrees.
            tilt_deg: Tilt angle in degrees.
            roll_deg: Roll angle in degrees.
            map_width: Map width in pixels.
            map_height: Map height in pixels.
            pixels_per_meter: Pixels per meter scaling.
            distortion: Optional distortion coefficients.
            height_uncertainty: Optional height uncertainty bounds.
            affine_matrix: Optional 3x3 GeoTIFF affine matrix A.

        Returns:
            New CameraParameters instance.

        Raises:
            ValueError: If any parameter fails validation.
        """
        # Convert numpy arrays to bytes for hashability
        K_bytes = np.asarray(intrinsic_matrix, dtype=np.float64).tobytes()
        pos_bytes = np.asarray(camera_position, dtype=np.float64).tobytes()
        A_bytes = (
            np.asarray(affine_matrix, dtype=np.float64).tobytes()
            if affine_matrix is not None
            else None
        )

        return cls(
            image_width=image_width,
            image_height=image_height,
            _intrinsic_matrix_data=K_bytes,
            _camera_position_data=pos_bytes,
            pan_deg=pan_deg,
            tilt_deg=tilt_deg,
            roll_deg=roll_deg,
            map_width=map_width,
            map_height=map_height,
            pixels_per_meter=pixels_per_meter,
            distortion=distortion,
            height_uncertainty=height_uncertainty,
            _affine_matrix_data=A_bytes,
        )

    def __hash__(self) -> int:
        """Compute hash for use in sets and as dict keys."""
        return hash(
            (
                self.image_width,
                self.image_height,
                self._intrinsic_matrix_data,
                self._camera_position_data,
                self.pan_deg,
                self.tilt_deg,
                self.roll_deg,
                self.map_width,
                self.map_height,
                self.pixels_per_meter,
                self.distortion,
                self.height_uncertainty,
                self._affine_matrix_data,
            )
        )


@dataclass(frozen=True)
class CameraGeometryResult:
    """Immutable result of camera geometry computation.

    This frozen dataclass captures the computed homography state from
    CameraGeometry, enabling caching, comparison, and immutable state
    management.

    Attributes:
        homography_matrix: 3x3 homography matrix H mapping world ground plane
            coordinates [X, Y, 1] to image pixels [u, v, 1].
        inverse_homography_matrix: 3x3 inverse homography matrix H^-1 mapping
            image pixels [u, v, 1] to world ground plane [X, Y, 1].
        condition_number: Condition number of H, indicating numerical stability.
            Lower is better; values > 1e6 may indicate instability.
        determinant: Determinant of H. Near-zero indicates singular matrix.
        is_valid: Whether the homography passes all validation checks.
        validation_messages: List of validation warnings or errors.
        center_projection_distance: Distance from camera to where image center
            projects onto ground plane, in meters.
    """

    # Homography matrices (stored as bytes for hashability)
    _homography_data: bytes = field(repr=False)
    _inverse_homography_data: bytes = field(repr=False)

    # Numerical quality metrics
    condition_number: float
    determinant: float

    # Validation state
    is_valid: bool
    validation_messages: tuple[str, ...] = field(default_factory=tuple)

    # Projection metrics
    center_projection_distance: Meters | None = None

    def __post_init__(self) -> None:
        """Validate homography matrices."""
        H = self.homography_matrix
        _validate_matrix_shape(H, (3, 3), "homography_matrix (H)")
        _validate_finite(H, "homography_matrix (H)")

        H_inv = self.inverse_homography_matrix
        _validate_matrix_shape(H_inv, (3, 3), "inverse_homography_matrix (H_inv)")
        _validate_finite(H_inv, "inverse_homography_matrix (H_inv)")

    @property
    def homography_matrix(self) -> np.ndarray:
        """Get the 3x3 homography matrix H.

        Returns:
            Immutable copy of the homography matrix.
        """
        arr = np.frombuffer(self._homography_data, dtype=np.float64).reshape(3, 3)
        arr.flags.writeable = False
        return arr

    @property
    def inverse_homography_matrix(self) -> np.ndarray:
        """Get the 3x3 inverse homography matrix H^-1.

        Returns:
            Immutable copy of the inverse homography matrix.
        """
        arr = np.frombuffer(self._inverse_homography_data, dtype=np.float64).reshape(3, 3)
        arr.flags.writeable = False
        return arr

    @classmethod
    def create(
        cls,
        homography_matrix: np.ndarray,
        inverse_homography_matrix: np.ndarray,
        condition_number: float,
        determinant: float,
        is_valid: bool,
        validation_messages: tuple[str, ...] | list[str] = (),
        center_projection_distance: Meters | None = None,
    ) -> CameraGeometryResult:
        """Create CameraGeometryResult with numpy arrays.

        This factory method handles the conversion of numpy arrays to their
        internal byte representation for hashability.

        Args:
            homography_matrix: 3x3 homography matrix H.
            inverse_homography_matrix: 3x3 inverse homography matrix H^-1.
            condition_number: Condition number of H.
            determinant: Determinant of H.
            is_valid: Whether homography is valid.
            validation_messages: List of validation messages.
            center_projection_distance: Distance to center projection in meters.

        Returns:
            New CameraGeometryResult instance.

        Raises:
            ValueError: If matrices fail validation.
        """
        H_bytes = np.asarray(homography_matrix, dtype=np.float64).tobytes()
        H_inv_bytes = np.asarray(inverse_homography_matrix, dtype=np.float64).tobytes()

        # Ensure validation_messages is a tuple
        if isinstance(validation_messages, list):
            validation_messages = tuple(validation_messages)

        return cls(
            _homography_data=H_bytes,
            _inverse_homography_data=H_inv_bytes,
            condition_number=condition_number,
            determinant=determinant,
            is_valid=is_valid,
            validation_messages=validation_messages,
            center_projection_distance=center_projection_distance,
        )

    def __hash__(self) -> int:
        """Compute hash for use in sets and as dict keys."""
        return hash(
            (
                self._homography_data,
                self._inverse_homography_data,
                self.condition_number,
                self.determinant,
                self.is_valid,
                self.validation_messages,
                self.center_projection_distance,
            )
        )
