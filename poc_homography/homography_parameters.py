"""Immutable configuration and result dataclasses for IntrinsicExtrinsicHomography.

This module provides frozen dataclasses for capturing the complete configuration
and computed state of intrinsic/extrinsic homography computation, enabling
immutable state management, caching, and easier testing/debugging.

The classes follow the patterns established in camera_parameters.py for numpy
array handling (bytes storage for hashability).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from poc_homography.homography_interface import HomographyApproach
from poc_homography.types import Degrees, Meters, Millimeters, Pixels, Unitless


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
class IntrinsicExtrinsicConfig:
    """Immutable configuration for IntrinsicExtrinsicHomography computation.

    This frozen dataclass captures all inputs needed for compute_homography(),
    including both the per-frame reference dict parameters and the instance
    configuration. It provides a complete snapshot of the configuration state
    that can be passed around, stored, cached, and compared.

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
        camera_matrix: 3x3 camera intrinsic matrix K (immutable copy).
        camera_position: Camera world position [X, Y, Z] in meters (immutable copy).
        pan_deg: Pan angle in degrees (positive = right/clockwise from above).
        tilt_deg: Tilt angle in degrees (positive = down, Hikvision convention).
        roll_deg: Roll angle in degrees (positive = clockwise).
        map_width: Width of output map in pixels.
        map_height: Height of output map in pixels.
        pixels_per_meter: Pixels per meter scaling for map projection.
        sensor_width_mm: Physical sensor width in millimeters.
        base_focal_length_mm: Base focal length at 1x zoom in millimeters.
        calibration_table: Optional dict mapping zoom_factor to intrinsic parameters.
        map_id: Identifier of the map for generated MapPoints.
    """

    # Camera intrinsic matrix K (stored as bytes for hashability)
    _camera_matrix_data: bytes = field(repr=False)

    # Camera position [X, Y, Z] in meters (stored as bytes for hashability)
    _camera_position_data: bytes = field(repr=False)

    # Orientation angles from reference dict
    pan_deg: Degrees
    tilt_deg: Degrees
    roll_deg: Degrees

    # Map dimensions from reference dict
    map_width: Pixels
    map_height: Pixels

    # Instance configuration
    pixels_per_meter: Unitless
    sensor_width_mm: Millimeters
    base_focal_length_mm: Millimeters
    map_id: str

    # Optional calibration table (stored as tuple of tuples for hashability)
    _calibration_table_data: tuple[tuple[float, tuple[tuple[str, float], ...]], ...] | None = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        """Validate all configuration parameters."""
        # Validate camera matrix
        K = self.camera_matrix
        _validate_matrix_shape(K, (3, 3), "camera_matrix (K)")
        _validate_finite(K, "camera_matrix (K)")

        # Validate camera position
        pos = self.camera_position
        if len(pos) != 3:
            raise ValueError(f"camera_position must have 3 elements [X, Y, Z], got {len(pos)}")
        _validate_finite(pos, "camera_position")

        # Validate camera height (should be positive for ground plane homography)
        if pos[2] <= 0:
            # Warning condition - don't raise, but could log
            pass

        # Validate tilt angle range
        if self.tilt_deg < -90.0 or self.tilt_deg > 90.0:
            raise ValueError(f"tilt_deg must be in range [-90, 90], got {self.tilt_deg}")

        # Validate roll angle range (consistent with IntrinsicExtrinsicHomography)
        ROLL_ERROR_THRESHOLD = 15.0
        if abs(self.roll_deg) > ROLL_ERROR_THRESHOLD:
            raise ValueError(
                f"roll_deg {self.roll_deg:.1f} is outside valid range "
                f"[-{ROLL_ERROR_THRESHOLD}, {ROLL_ERROR_THRESHOLD}]"
            )

        # Validate map dimensions
        if self.map_width <= 0:
            raise ValueError(f"map_width must be positive, got {self.map_width}")
        if self.map_height <= 0:
            raise ValueError(f"map_height must be positive, got {self.map_height}")

        # Validate pixels per meter
        if self.pixels_per_meter <= 0:
            raise ValueError(f"pixels_per_meter must be positive, got {self.pixels_per_meter}")

        # Validate sensor and focal length
        if self.sensor_width_mm <= 0:
            raise ValueError(f"sensor_width_mm must be positive, got {self.sensor_width_mm}")
        if self.base_focal_length_mm <= 0:
            raise ValueError(
                f"base_focal_length_mm must be positive, got {self.base_focal_length_mm}"
            )

        # Validate map_id is non-empty
        if not self.map_id:
            raise ValueError("map_id must be a non-empty string")

    @property
    def camera_matrix(self) -> np.ndarray:
        """Get the 3x3 camera intrinsic matrix K.

        Returns:
            Immutable copy of the intrinsic matrix.
        """
        arr = np.frombuffer(self._camera_matrix_data, dtype=np.float64).reshape(3, 3)
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
    def calibration_table(self) -> dict[float, dict[str, float]] | None:
        """Get the calibration table as a dictionary.

        Returns:
            Dictionary mapping zoom_factor to intrinsic parameters, or None.
        """
        if self._calibration_table_data is None:
            return None
        return {zoom: dict(params) for zoom, params in self._calibration_table_data}

    @classmethod
    def create(
        cls,
        camera_matrix: np.ndarray,
        camera_position: np.ndarray,
        pan_deg: Degrees,
        tilt_deg: Degrees,
        roll_deg: Degrees,
        map_width: Pixels,
        map_height: Pixels,
        pixels_per_meter: Unitless,
        sensor_width_mm: Millimeters,
        base_focal_length_mm: Millimeters,
        map_id: str,
        calibration_table: dict[float, dict[str, float]] | None = None,
    ) -> IntrinsicExtrinsicConfig:
        """Create IntrinsicExtrinsicConfig with numpy arrays.

        This factory method handles the conversion of numpy arrays to their
        internal byte representation for hashability.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix K.
            camera_position: Camera world position [X, Y, Z] in meters.
            pan_deg: Pan angle in degrees.
            tilt_deg: Tilt angle in degrees.
            roll_deg: Roll angle in degrees.
            map_width: Map width in pixels.
            map_height: Map height in pixels.
            pixels_per_meter: Pixels per meter scaling.
            sensor_width_mm: Physical sensor width in millimeters.
            base_focal_length_mm: Base focal length at 1x zoom in mm.
            map_id: Identifier of the map for generated MapPoints.
            calibration_table: Optional calibration table dict.

        Returns:
            New IntrinsicExtrinsicConfig instance.

        Raises:
            ValueError: If any parameter fails validation.
        """
        # Convert numpy arrays to bytes for hashability
        K_bytes = np.asarray(camera_matrix, dtype=np.float64).tobytes()
        pos_bytes = np.asarray(camera_position, dtype=np.float64).tobytes()

        # Convert calibration table to hashable tuple format
        cal_table_data: tuple[tuple[float, tuple[tuple[str, float], ...]], ...] | None = None
        if calibration_table is not None:
            cal_table_data = tuple(
                (float(zoom), tuple(sorted((k, float(v)) for k, v in params.items())))
                for zoom, params in sorted(calibration_table.items())
            )

        return cls(
            _camera_matrix_data=K_bytes,
            _camera_position_data=pos_bytes,
            pan_deg=pan_deg,
            tilt_deg=tilt_deg,
            roll_deg=roll_deg,
            map_width=map_width,
            map_height=map_height,
            pixels_per_meter=pixels_per_meter,
            sensor_width_mm=sensor_width_mm,
            base_focal_length_mm=base_focal_length_mm,
            map_id=map_id,
            _calibration_table_data=cal_table_data,
        )

    @classmethod
    def from_reference_dict(
        cls,
        reference: dict[str, Any],
        pixels_per_meter: Unitless,
        sensor_width_mm: Millimeters,
        base_focal_length_mm: Millimeters,
        map_id: str,
        calibration_table: dict[float, dict[str, float]] | None = None,
    ) -> IntrinsicExtrinsicConfig:
        """Create IntrinsicExtrinsicConfig from reference dict and instance config.

        This factory method creates a config from the reference dict format
        used by IntrinsicExtrinsicHomography.compute_homography(), combined
        with the instance-level configuration.

        Args:
            reference: Dictionary with keys:
                - 'camera_matrix': 3x3 intrinsic camera matrix K
                - 'camera_position': Camera position [X, Y, Z] in meters
                - 'pan_deg': Pan angle in degrees
                - 'tilt_deg': Tilt angle in degrees
                - 'roll_deg': Roll angle in degrees (optional, defaults to 0.0)
                - 'map_width': Output map width in pixels
                - 'map_height': Output map height in pixels
            pixels_per_meter: Scale factor for map visualization.
            sensor_width_mm: Physical sensor width in millimeters.
            base_focal_length_mm: Base focal length at 1x zoom in mm.
            map_id: Identifier of the map for generated MapPoints.
            calibration_table: Optional calibration table dict.

        Returns:
            New IntrinsicExtrinsicConfig instance.

        Raises:
            ValueError: If required keys are missing or validation fails.
        """
        # Validate required keys
        required_keys = [
            "camera_matrix",
            "camera_position",
            "pan_deg",
            "tilt_deg",
            "map_width",
            "map_height",
        ]
        for key in required_keys:
            if key not in reference:
                raise ValueError(f"Missing required reference key: '{key}'")

        return cls.create(
            camera_matrix=reference["camera_matrix"],
            camera_position=reference["camera_position"],
            pan_deg=Degrees(reference["pan_deg"]),
            tilt_deg=Degrees(reference["tilt_deg"]),
            roll_deg=Degrees(reference.get("roll_deg", 0.0)),
            map_width=Pixels(reference["map_width"]),
            map_height=Pixels(reference["map_height"]),
            pixels_per_meter=pixels_per_meter,
            sensor_width_mm=sensor_width_mm,
            base_focal_length_mm=base_focal_length_mm,
            map_id=map_id,
            calibration_table=calibration_table,
        )

    def to_reference_dict(self) -> dict[str, Any]:
        """Convert to reference dict format for compute_homography().

        Returns:
            Dictionary with all reference parameters.
        """
        return {
            "camera_matrix": self.camera_matrix.copy(),
            "camera_position": self.camera_position.copy(),
            "pan_deg": float(self.pan_deg),
            "tilt_deg": float(self.tilt_deg),
            "roll_deg": float(self.roll_deg),
            "map_width": int(self.map_width),
            "map_height": int(self.map_height),
        }

    def __hash__(self) -> int:
        """Compute hash for use in sets and as dict keys."""
        return hash(
            (
                self._camera_matrix_data,
                self._camera_position_data,
                self.pan_deg,
                self.tilt_deg,
                self.roll_deg,
                self.map_width,
                self.map_height,
                self.pixels_per_meter,
                self.sensor_width_mm,
                self.base_focal_length_mm,
                self.map_id,
                self._calibration_table_data,
            )
        )


@dataclass(frozen=True)
class IntrinsicExtrinsicResult:
    """Immutable result of intrinsic/extrinsic homography computation.

    This frozen dataclass captures the complete computed state from
    IntrinsicExtrinsicHomography, including the homography matrices,
    confidence score, and detailed metadata. It extends HomographyResult
    with more detailed information specific to the intrinsic/extrinsic
    approach.

    Attributes:
        homography_matrix: 3x3 homography matrix H mapping world ground plane
            coordinates [X, Y, 1] to image pixels [u, v, 1].
        inverse_homography_matrix: 3x3 inverse homography matrix H^-1 mapping
            image pixels [u, v, 1] to world ground plane [X, Y, 1].
        confidence: Overall confidence score [0.0, 1.0] based on matrix quality
            and camera parameter validity.
        approach: The homography approach used (always INTRINSIC_EXTRINSIC).
        camera_position: Camera world position [X, Y, Z] used for computation.
        pan_deg: Pan angle used for computation.
        tilt_deg: Tilt angle used for computation.
        roll_deg: Roll angle used for computation.
        determinant: Determinant of H. Near-zero indicates singular matrix.
        condition_number: Condition number of H, indicating numerical stability.
        map_dimensions: (width, height) of the output map in pixels.
        is_valid: Whether the homography passes all validation checks.
        validation_messages: List of validation warnings or errors.
    """

    # Homography matrices (stored as bytes for hashability)
    _homography_data: bytes = field(repr=False)
    _inverse_homography_data: bytes = field(repr=False)

    # Core result values
    confidence: float

    # Approach identifier
    approach: HomographyApproach

    # Camera parameters used (stored as bytes for hashability)
    _camera_position_data: bytes = field(repr=False)
    pan_deg: Degrees
    tilt_deg: Degrees
    roll_deg: Degrees

    # Numerical quality metrics
    determinant: float
    condition_number: float

    # Map dimensions
    map_width: Pixels
    map_height: Pixels

    # Validation state
    is_valid: bool
    validation_messages: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate result matrices and parameters."""
        # Validate homography matrices
        H = self.homography_matrix
        _validate_matrix_shape(H, (3, 3), "homography_matrix (H)")
        _validate_finite(H, "homography_matrix (H)")

        H_inv = self.inverse_homography_matrix
        _validate_matrix_shape(H_inv, (3, 3), "inverse_homography_matrix (H_inv)")
        _validate_finite(H_inv, "inverse_homography_matrix (H_inv)")

        # Validate camera position
        pos = self.camera_position
        if len(pos) != 3:
            raise ValueError(f"camera_position must have 3 elements [X, Y, Z], got {len(pos)}")
        _validate_finite(pos, "camera_position")

        # Validate confidence range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in range [0.0, 1.0], got {self.confidence}")

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
    def map_dimensions(self) -> tuple[Pixels, Pixels]:
        """Get map dimensions as (width, height) tuple."""
        return (self.map_width, self.map_height)

    @classmethod
    def create(
        cls,
        homography_matrix: np.ndarray,
        inverse_homography_matrix: np.ndarray,
        confidence: float,
        camera_position: np.ndarray,
        pan_deg: Degrees,
        tilt_deg: Degrees,
        roll_deg: Degrees,
        map_width: Pixels,
        map_height: Pixels,
        determinant: float | None = None,
        condition_number: float | None = None,
        validation_messages: tuple[str, ...] | list[str] = (),
    ) -> IntrinsicExtrinsicResult:
        """Create IntrinsicExtrinsicResult with numpy arrays.

        This factory method handles the conversion of numpy arrays to their
        internal byte representation for hashability. It also computes
        determinant and condition number if not provided.

        Args:
            homography_matrix: 3x3 homography matrix H.
            inverse_homography_matrix: 3x3 inverse homography matrix H^-1.
            confidence: Confidence score [0.0, 1.0].
            camera_position: Camera world position [X, Y, Z] in meters.
            pan_deg: Pan angle in degrees.
            tilt_deg: Tilt angle in degrees.
            roll_deg: Roll angle in degrees.
            map_width: Map width in pixels.
            map_height: Map height in pixels.
            determinant: Optional precomputed determinant.
            condition_number: Optional precomputed condition number.
            validation_messages: List of validation messages.

        Returns:
            New IntrinsicExtrinsicResult instance.

        Raises:
            ValueError: If matrices or parameters fail validation.
        """
        H = np.asarray(homography_matrix, dtype=np.float64)
        H_inv = np.asarray(inverse_homography_matrix, dtype=np.float64)

        # Convert to bytes for hashability
        H_bytes = H.tobytes()
        H_inv_bytes = H_inv.tobytes()
        pos_bytes = np.asarray(camera_position, dtype=np.float64).tobytes()

        # Compute determinant and condition number if not provided
        if determinant is None:
            determinant = float(np.linalg.det(H))
        if condition_number is None:
            condition_number = float(np.linalg.cond(H))

        # Ensure validation_messages is a tuple
        if isinstance(validation_messages, list):
            validation_messages = tuple(validation_messages)

        # Determine validity
        messages_list: list[str] = list(validation_messages)
        is_valid = True

        # Check condition number thresholds
        COND_THRESHOLD_DEGENERATE = 1e10
        COND_THRESHOLD_UNSTABLE = 1e6
        if condition_number > COND_THRESHOLD_DEGENERATE:
            messages_list.append(
                f"Condition number {condition_number:.2e} exceeds degenerate threshold (1e10)"
            )
            is_valid = False
        elif condition_number > COND_THRESHOLD_UNSTABLE:
            messages_list.append(
                f"Condition number {condition_number:.2e} exceeds unstable threshold (1e6)"
            )

        # Check determinant
        MIN_DET_THRESHOLD = 1e-10
        if abs(determinant) < MIN_DET_THRESHOLD:
            messages_list.append(f"Determinant {determinant:.2e} is near zero (singular matrix)")
            is_valid = False

        # Check confidence threshold
        MIN_CONFIDENCE_THRESHOLD = 0.3
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            messages_list.append(
                f"Confidence {confidence:.2f} below minimum threshold ({MIN_CONFIDENCE_THRESHOLD})"
            )
            is_valid = False

        return cls(
            _homography_data=H_bytes,
            _inverse_homography_data=H_inv_bytes,
            confidence=confidence,
            approach=HomographyApproach.INTRINSIC_EXTRINSIC,
            _camera_position_data=pos_bytes,
            pan_deg=pan_deg,
            tilt_deg=tilt_deg,
            roll_deg=roll_deg,
            determinant=determinant,
            condition_number=condition_number,
            map_width=map_width,
            map_height=map_height,
            is_valid=is_valid,
            validation_messages=tuple(messages_list),
        )

    def to_metadata_dict(self) -> dict[str, Any]:
        """Convert to metadata dict format compatible with HomographyResult.

        Returns:
            Dictionary with all metadata for HomographyResult.
        """
        return {
            "approach": self.approach.value,
            "camera_position": self.camera_position.tolist(),
            "pan_deg": float(self.pan_deg),
            "tilt_deg": float(self.tilt_deg),
            "roll_deg": float(self.roll_deg),
            "determinant": self.determinant,
            "condition_number": self.condition_number,
            "map_dimensions": (int(self.map_width), int(self.map_height)),
            "is_valid": self.is_valid,
            "validation_messages": list(self.validation_messages),
        }

    def __hash__(self) -> int:
        """Compute hash for use in sets and as dict keys."""
        return hash(
            (
                self._homography_data,
                self._inverse_homography_data,
                self.confidence,
                self.approach,
                self._camera_position_data,
                self.pan_deg,
                self.tilt_deg,
                self.roll_deg,
                self.determinant,
                self.condition_number,
                self.map_width,
                self.map_height,
                self.is_valid,
                self.validation_messages,
            )
        )
