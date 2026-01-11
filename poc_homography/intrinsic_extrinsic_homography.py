"""
Intrinsic/Extrinsic Homography Provider Implementation.

This module implements the HomographyProvider interface using camera
intrinsic parameters (focal length, principal point) and extrinsic parameters
(rotation, translation) to compute homography transformations.

The homography maps image coordinates to ground plane coordinates using the
pinhole camera model with known camera calibration and pose.

This module uses an IMMUTABLE pattern - all computation is done through the
static `compute_from_config()` classmethod which takes an IntrinsicExtrinsicConfig
and returns an IntrinsicExtrinsicResult.

Coordinate Systems:
    - World Frame: X=East, Y=North, Z=Up (right-handed)
    - Camera Frame: X=Right, Y=Down, Z=Forward (standard CV, right-handed)
    - Image Frame: origin top-left, u=right, v=down (pixels)

USAGE:
======
```python
config = IntrinsicExtrinsicConfig.create(
    camera_matrix=K,
    camera_position=np.array([0, 0, 10]),
    pan_deg=Degrees(45.0),
    tilt_deg=Degrees(30.0),
    roll_deg=Degrees(0.0),
    map_width=Pixels(640),
    map_height=Pixels(640),
    pixels_per_meter=Unitless(100.0),
    sensor_width_mm=Millimeters(7.18),
    base_focal_length_mm=Millimeters(5.9),
    map_id="map_valte",
)
result = IntrinsicExtrinsicHomography.compute_from_config(config)
```
"""

from __future__ import annotations

import logging
import math

import numpy as np

from poc_homography.homography_interface import (
    HomographyProvider,
    HomographyResult,
)
from poc_homography.homography_parameters import (
    IntrinsicExtrinsicConfig,
    IntrinsicExtrinsicResult,
)
from poc_homography.map_points import MapPoint
from poc_homography.pixel_point import PixelPoint  # noqa: TC001 - used at runtime
from poc_homography.types import Degrees, Meters, Millimeters, Pixels, Unitless

logger = logging.getLogger(__name__)


class IntrinsicExtrinsicHomography(HomographyProvider):
    """
    Homography provider using camera intrinsic/extrinsic parameters.

    This implementation computes homography for a ground plane (Z=0) using:
    - Camera intrinsic matrix K (focal length, principal point)
    - Camera extrinsic parameters (position, rotation via pan/tilt/roll)

    This class uses an IMMUTABLE pattern - all computation is done through the
    static `compute_from_config()` classmethod.

    The homography H maps world ground plane points to image pixels:
        [u]       [X_world]
        [v]  = H  [Y_world]
        [1]       [1      ]

    For inverse projection (image to world):
        [X_world]           [u]
        [Y_world]  = H^-1  [v]
        [1      ]           [1]

    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        map_id: Identifier of the map for generated MapPoints
        pixels_per_meter: Scale factor for map visualization (default: 100)
        calibration_table: Optional dict mapping zoom_factor to intrinsic parameters
    """

    # Minimum determinant threshold for valid homography
    MIN_DET_THRESHOLD = 1e-10

    # Minimum confidence threshold for validity
    MIN_CONFIDENCE_THRESHOLD = 0.3

    # Confidence thresholds based on homography matrix determinant
    DET_THRESHOLD_INVALID = 1e-6
    DET_THRESHOLD_LOW = 1e-3
    DET_THRESHOLD_HIGH = 1e3

    # Condition number thresholds for numerical stability assessment
    COND_THRESHOLD_DEGENERATE = 1e10
    COND_THRESHOLD_UNSTABLE = 1e6
    COND_THRESHOLD_MARGINAL = 1e3

    # Confidence penalty multipliers
    CONFIDENCE_PENALTY_UNSTABLE = 0.5
    CONFIDENCE_PENALTY_MARGINAL = 0.9
    CONFIDENCE_PENALTY_BAD_HEIGHT = 0.5
    CONFIDENCE_PENALTY_BAD_TILT = 0.8
    CONFIDENCE_LARGE_DET = 0.7

    # Gimbal lock threshold
    GIMBAL_LOCK_THRESHOLD_DEG = 0.1
    CONFIDENCE_PENALTY_GIMBAL_LOCK = 0.3

    # Edge factor constants for point confidence calculation
    EDGE_FACTOR_CENTER = 1.0
    EDGE_FACTOR_EDGE = 0.7
    EDGE_FACTOR_CORNER_DECAY = 0.2
    EDGE_FACTOR_MIN = 0.3

    # Roll validation thresholds
    ROLL_WARN_THRESHOLD = 5.0
    ROLL_ERROR_THRESHOLD = 15.0

    def __init__(
        self,
        width: Pixels,
        height: Pixels,
        map_id: str,
        pixels_per_meter: Unitless = Unitless(100.0),
        sensor_width_mm: Millimeters = Millimeters(7.18),
        base_focal_length_mm: Millimeters = Millimeters(5.9),
        calibration_table: dict[float, dict[str, float]] | None = None,
    ):
        """
        Initialize intrinsic/extrinsic homography provider.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            map_id: Identifier of the map for generated MapPoints (e.g., "map_valte")
            pixels_per_meter: Scale factor for map visualization (default: 100)
            sensor_width_mm: Physical sensor width in millimeters (default: 7.18)
            base_focal_length_mm: Base focal length at 1x zoom in mm (default: 5.9)
            calibration_table: Optional dictionary mapping zoom_factor (float) to
                intrinsic parameters (dict).
        """
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Image dimensions must be positive, got width={width}, height={height}"
            )

        self.width = width
        self.height = height
        self.map_id = map_id
        self.pixels_per_meter = pixels_per_meter
        self.sensor_width_mm = sensor_width_mm
        self.base_focal_length_mm = base_focal_length_mm
        self.calibration_table = calibration_table

    def _interpolate_calibration_params(self, zoom_factor: float) -> dict[str, float] | None:
        """
        Interpolate intrinsic parameters from calibration table for given zoom factor.
        """
        if self.calibration_table is None or len(self.calibration_table) == 0:
            return None

        try:
            zoom_levels = sorted([float(z) for z in self.calibration_table.keys()])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Calibration table keys must be numeric zoom factors: {e}")

        if len(zoom_levels) == 1:
            single_zoom = zoom_levels[0]
            params = self.calibration_table[single_zoom]
            return {
                "fx": float(params["fx"]),
                "fy": float(params["fy"]),
                "cx": float(params["cx"]),
                "cy": float(params["cy"]),
            }

        if zoom_factor <= zoom_levels[0]:
            params = self.calibration_table[zoom_levels[0]]
            return {
                "fx": float(params["fx"]),
                "fy": float(params["fy"]),
                "cx": float(params["cx"]),
                "cy": float(params["cy"]),
            }

        if zoom_factor >= zoom_levels[-1]:
            params = self.calibration_table[zoom_levels[-1]]
            return {
                "fx": float(params["fx"]),
                "fy": float(params["fy"]),
                "cx": float(params["cx"]),
                "cy": float(params["cy"]),
            }

        for i in range(len(zoom_levels) - 1):
            z_low = zoom_levels[i]
            z_high = zoom_levels[i + 1]

            if z_low <= zoom_factor <= z_high:
                t = (zoom_factor - z_low) / (z_high - z_low)
                params_low = self.calibration_table[z_low]
                params_high = self.calibration_table[z_high]

                return {
                    "fx": params_low["fx"] + (params_high["fx"] - params_low["fx"]) * t,
                    "fy": params_low["fy"] + (params_high["fy"] - params_low["fy"]) * t,
                    "cx": params_low["cx"] + (params_high["cx"] - params_low["cx"]) * t,
                    "cy": params_low["cy"] + (params_high["cy"] - params_low["cy"]) * t,
                }

        return None

    def get_intrinsics(
        self,
        zoom_factor: Unitless,
        width_px: Pixels | None = None,
        height_px: Pixels | None = None,
        sensor_width_mm: Millimeters | None = None,
    ) -> np.ndarray:
        """
        Calculate camera intrinsic matrix K from zoom factor and sensor specs.

        If calibration_table is provided, interpolates K(zoom) from calibrated values.
        Otherwise, uses linear focal length approximation based on base_focal_length_mm.

        Args:
            zoom_factor: Digital or optical zoom multiplier (1.0 = no zoom)
            width_px: Image width in pixels (default: uses instance's width)
            height_px: Image height in pixels (default: uses instance's height)
            sensor_width_mm: Physical sensor width in millimeters

        Returns:
            K: 3x3 camera intrinsic matrix
        """
        if width_px is None:
            width_px = self.width
        if height_px is None:
            height_px = self.height

        if zoom_factor <= 0:
            raise ValueError(f"zoom_factor must be positive, got {zoom_factor}")
        if width_px <= 0 or height_px <= 0:
            raise ValueError("Image dimensions must be positive")

        calibration_params = self._interpolate_calibration_params(zoom_factor)

        if calibration_params is not None:
            fx = calibration_params["fx"]
            fy = calibration_params["fy"]
            cx = calibration_params["cx"]
            cy = calibration_params["cy"]
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
            return K

        if sensor_width_mm is None:
            sensor_width_mm = self.sensor_width_mm

        if sensor_width_mm <= 0:
            raise ValueError("sensor_width_mm must be positive")

        f_mm = self.base_focal_length_mm * zoom_factor
        f_px = f_mm * (width_px / sensor_width_mm)
        cx, cy = width_px / 2.0, height_px / 2.0
        K = np.array([[f_px, 0.0, cx], [0.0, f_px, cy], [0.0, 0.0, 1.0]])
        return K

    def get_distortion_coefficients(self, zoom_factor: Unitless) -> np.ndarray | None:
        """
        Get distortion coefficients for given zoom factor from calibration table.
        """
        if self.calibration_table is None or len(self.calibration_table) == 0:
            return None

        try:
            zoom_levels = sorted([float(z) for z in self.calibration_table.keys()])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Calibration table keys must be numeric zoom factors: {e}")

        if len(zoom_levels) == 1:
            single_zoom = zoom_levels[0]
            params = self.calibration_table[single_zoom]
            return np.array(
                [
                    float(params["k1"]),
                    float(params["k2"]),
                    float(params["p1"]),
                    float(params["p2"]),
                    float(params["k3"]),
                ]
            )

        if zoom_factor <= zoom_levels[0]:
            params = self.calibration_table[zoom_levels[0]]
            return np.array(
                [
                    float(params["k1"]),
                    float(params["k2"]),
                    float(params["p1"]),
                    float(params["p2"]),
                    float(params["k3"]),
                ]
            )

        if zoom_factor >= zoom_levels[-1]:
            params = self.calibration_table[zoom_levels[-1]]
            return np.array(
                [
                    float(params["k1"]),
                    float(params["k2"]),
                    float(params["p1"]),
                    float(params["p2"]),
                    float(params["k3"]),
                ]
            )

        for i in range(len(zoom_levels) - 1):
            z_low = zoom_levels[i]
            z_high = zoom_levels[i + 1]

            if z_low <= zoom_factor <= z_high:
                t = (zoom_factor - z_low) / (z_high - z_low)
                params_low = self.calibration_table[z_low]
                params_high = self.calibration_table[z_high]

                k1 = params_low["k1"] + (params_high["k1"] - params_low["k1"]) * t
                k2 = params_low["k2"] + (params_high["k2"] - params_low["k2"]) * t
                p1 = params_low["p1"] + (params_high["p1"] - params_low["p1"]) * t
                p2 = params_low["p2"] + (params_high["p2"] - params_low["p2"]) * t
                k3 = params_low["k3"] + (params_high["k3"] - params_low["k3"]) * t

                return np.array([k1, k2, p1, p2, k3])

        return None

    @staticmethod
    def _compute_rotation_matrix(
        pan_deg: Degrees, tilt_deg: Degrees, roll_deg: Degrees = Degrees(0.0)
    ) -> np.ndarray:
        """
        Calculate rotation matrix from world to camera coordinates (static version).
        """
        pan_rad = math.radians(pan_deg)
        tilt_rad = math.radians(tilt_deg)
        roll_rad = math.radians(roll_deg)

        R_base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

        Rz_pan = np.array(
            [
                [math.cos(pan_rad), -math.sin(pan_rad), 0.0],
                [math.sin(pan_rad), math.cos(pan_rad), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        Rz_roll = np.array(
            [
                [math.cos(roll_rad), -math.sin(roll_rad), 0.0],
                [math.sin(roll_rad), math.cos(roll_rad), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        Rx_tilt = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(tilt_rad), -math.sin(tilt_rad)],
                [0.0, math.sin(tilt_rad), math.cos(tilt_rad)],
            ]
        )

        R: np.ndarray = Rx_tilt @ Rz_roll @ R_base @ Rz_pan
        return R

    @staticmethod
    def _compute_ground_homography(
        K: np.ndarray,
        camera_position: np.ndarray,
        pan_deg: Degrees,
        tilt_deg: Degrees,
        roll_deg: Degrees = Degrees(0.0),
        min_det_threshold: float = 1e-10,
    ) -> np.ndarray:
        """
        Calculate homography matrix mapping world ground plane (Z=0) to image.
        """
        R = IntrinsicExtrinsicHomography._compute_rotation_matrix(pan_deg, tilt_deg, roll_deg)
        C = camera_position
        t = -R @ C

        r1 = R[:, 0]
        r2 = R[:, 1]

        H_extrinsic = np.column_stack([r1, r2, t])
        if H_extrinsic.shape != (3, 3):
            raise ValueError(f"H_extrinsic must be 3x3, got {H_extrinsic.shape}")

        H = K @ H_extrinsic
        if H.shape != (3, 3):
            raise ValueError(f"Homography must be 3x3, got {H.shape}")

        if abs(H[2, 2]) < min_det_threshold:
            logger.warning(
                f"Homography normalization failed (H[2,2]={H[2, 2]:.2e}). Returning identity."
            )
            return np.eye(3)

        H = H / H[2, 2]
        return np.asarray(H)

    @staticmethod
    def _compute_confidence(
        H: np.ndarray,
        camera_position: np.ndarray | None = None,
        tilt_deg: float | None = None,
        *,
        min_det_threshold: float = 1e-10,
        det_threshold_invalid: float = 1e-6,
        det_threshold_low: float = 1e-3,
        det_threshold_high: float = 1e3,
        confidence_large_det: float = 0.7,
        cond_threshold_degenerate: float = 1e10,
        cond_threshold_unstable: float = 1e6,
        cond_threshold_marginal: float = 1e3,
        confidence_penalty_unstable: float = 0.5,
        confidence_penalty_marginal: float = 0.9,
        confidence_penalty_bad_height: float = 0.5,
        confidence_penalty_bad_tilt: float = 0.8,
        gimbal_lock_threshold_deg: float = 0.1,
        confidence_penalty_gimbal_lock: float = 0.3,
    ) -> float:
        """
        Calculate confidence score for the homography matrix (static version).
        """
        det_H = np.linalg.det(H)

        if abs(det_H) < min_det_threshold:
            return 0.0

        det_abs = abs(det_H)

        if det_abs < det_threshold_invalid:
            confidence = 0.0
        elif det_abs < det_threshold_low:
            confidence = 0.5
        elif det_abs < det_threshold_high:
            confidence = 1.0
        else:
            confidence = confidence_large_det

        cond_H = np.linalg.cond(H)
        if cond_H > cond_threshold_degenerate:
            confidence = 0.0
        elif cond_H > cond_threshold_unstable:
            confidence *= confidence_penalty_unstable
        elif cond_H > cond_threshold_marginal:
            confidence *= confidence_penalty_marginal

        if camera_position is not None:
            camera_height = camera_position[2]
            if camera_height <= 0:
                confidence *= confidence_penalty_bad_height

        if tilt_deg is not None:
            if tilt_deg < -90.0 or tilt_deg > 90.0:
                confidence *= confidence_penalty_bad_tilt
            elif abs(abs(tilt_deg) - 90.0) < gimbal_lock_threshold_deg:
                confidence *= confidence_penalty_gimbal_lock

        return confidence

    @classmethod
    def compute_from_config(cls, config: IntrinsicExtrinsicConfig) -> IntrinsicExtrinsicResult:
        """
        Compute homography from config as a pure function (classmethod).

        This method performs all homography computation without modifying any
        instance state. It takes an immutable IntrinsicExtrinsicConfig and
        returns an immutable IntrinsicExtrinsicResult.

        This is the recommended entry point for the immutable pattern, enabling:
        - Pure functional computation with no side effects
        - Easy caching based on config hash
        - Thread-safe parallel computation
        - Simplified testing with explicit inputs/outputs

        Args:
            config: Immutable configuration containing all parameters needed
                for homography computation.

        Returns:
            IntrinsicExtrinsicResult containing the computed homography matrices,
            confidence score, and metadata.

        Example:
            >>> config = IntrinsicExtrinsicConfig.create(
            ...     camera_matrix=K,
            ...     camera_position=np.array([0, 0, 10]),
            ...     pan_deg=Degrees(45.0),
            ...     tilt_deg=Degrees(30.0),
            ...     roll_deg=Degrees(0.0),
            ...     map_width=Pixels(640),
            ...     map_height=Pixels(640),
            ...     pixels_per_meter=Unitless(100.0),
            ...     sensor_width_mm=Millimeters(7.18),
            ...     base_focal_length_mm=Millimeters(5.9),
            ...     map_id="map_valte",
            ... )
            >>> result = IntrinsicExtrinsicHomography.compute_from_config(config)
            >>> print(result.confidence)
            1.0
        """
        K = config.camera_matrix
        camera_position = config.camera_position
        pan_deg = config.pan_deg
        tilt_deg = config.tilt_deg
        roll_deg = config.roll_deg
        map_width = config.map_width
        map_height = config.map_height

        validation_messages: list[str] = []

        if camera_position[2] <= 0:
            validation_messages.append(
                f"Camera height (Z={camera_position[2]}) should be positive for ground plane homography"
            )

        if abs(abs(tilt_deg) - 90.0) < cls.GIMBAL_LOCK_THRESHOLD_DEG:
            validation_messages.append(
                f"Tilt angle ({tilt_deg:.2f} degrees) is near gimbal lock zone"
            )

        if abs(roll_deg) > cls.ROLL_WARN_THRESHOLD:
            validation_messages.append(
                f"Roll angle ({roll_deg:.2f} degrees) is unusually large (>{cls.ROLL_WARN_THRESHOLD} degrees)"
            )

        H = cls._compute_ground_homography(
            K,
            camera_position,
            Degrees(pan_deg),
            Degrees(tilt_deg),
            Degrees(roll_deg),
            cls.MIN_DET_THRESHOLD,
        )

        det_H = float(np.linalg.det(H))
        if abs(det_H) < cls.MIN_DET_THRESHOLD:
            H_inv = np.eye(3)
            confidence = 0.0
            validation_messages.append(
                f"Homography is singular (det={det_H:.2e}). Inverse may be unstable."
            )
        else:
            H_inv = np.asarray(np.linalg.inv(H))
            confidence = cls._compute_confidence(
                H,
                camera_position,
                float(tilt_deg),
                min_det_threshold=cls.MIN_DET_THRESHOLD,
                det_threshold_invalid=cls.DET_THRESHOLD_INVALID,
                det_threshold_low=cls.DET_THRESHOLD_LOW,
                det_threshold_high=cls.DET_THRESHOLD_HIGH,
                confidence_large_det=cls.CONFIDENCE_LARGE_DET,
                cond_threshold_degenerate=cls.COND_THRESHOLD_DEGENERATE,
                cond_threshold_unstable=cls.COND_THRESHOLD_UNSTABLE,
                cond_threshold_marginal=cls.COND_THRESHOLD_MARGINAL,
                confidence_penalty_unstable=cls.CONFIDENCE_PENALTY_UNSTABLE,
                confidence_penalty_marginal=cls.CONFIDENCE_PENALTY_MARGINAL,
                confidence_penalty_bad_height=cls.CONFIDENCE_PENALTY_BAD_HEIGHT,
                confidence_penalty_bad_tilt=cls.CONFIDENCE_PENALTY_BAD_TILT,
                gimbal_lock_threshold_deg=cls.GIMBAL_LOCK_THRESHOLD_DEG,
                confidence_penalty_gimbal_lock=cls.CONFIDENCE_PENALTY_GIMBAL_LOCK,
            )

        return IntrinsicExtrinsicResult.create(
            homography_matrix=H,
            inverse_homography_matrix=H_inv,
            confidence=confidence,
            camera_position=camera_position,
            pan_deg=Degrees(pan_deg),
            tilt_deg=Degrees(tilt_deg),
            roll_deg=Degrees(roll_deg),
            map_width=Pixels(map_width),
            map_height=Pixels(map_height),
            determinant=det_H,
            validation_messages=validation_messages,
        )

    # =========================================================================
    # HomographyProvider Interface Implementation (kept for interface compliance)
    # =========================================================================

    def project_point(self, image_point: PixelPoint, point_id: str = "") -> MapPoint:
        """
        Project image point to map pixel coordinates.

        Note: This method requires that a result has been stored from a previous
        compute_from_config call. For pure functional usage, use the static
        project_point_with_result method instead.

        Args:
            image_point: Pixel coordinates in camera image
            point_id: Optional ID for the generated MapPoint

        Returns:
            MapPoint with pixel coordinates on the map

        Raises:
            RuntimeError: If no valid homography computed
        """
        raise NotImplementedError(
            "Use compute_from_config() to get a result, then use "
            "IntrinsicExtrinsicHomography.project_point_static(result, image_point, ...) instead."
        )

    def project_points(
        self, image_points: list[PixelPoint], point_id_prefix: str = "proj"
    ) -> list[MapPoint]:
        """
        Project multiple image points to map pixel coordinates.

        Note: This method requires that a result has been stored from a previous
        compute_from_config call. For pure functional usage, use the static methods instead.
        """
        raise NotImplementedError(
            "Use compute_from_config() to get a result, then use static projection methods."
        )

    def get_confidence(self) -> float:
        """
        Return confidence score of current homography.

        Note: For immutable usage, get confidence from IntrinsicExtrinsicResult.
        """
        raise NotImplementedError(
            "Use compute_from_config() to get a result, then access result.confidence."
        )

    def is_valid(self) -> bool:
        """
        Check if homography is valid and ready for projection.

        Note: For immutable usage, check validity from IntrinsicExtrinsicResult.
        """
        raise NotImplementedError(
            "Use compute_from_config() to get a result, then check result.confidence > 0."
        )

    # =========================================================================
    # Static utility methods for working with results
    # =========================================================================

    @staticmethod
    def project_image_point_to_world(
        result: IntrinsicExtrinsicResult, image_point: PixelPoint
    ) -> tuple[Meters, Meters]:
        """
        Project image point to world ground plane coordinates (meters).

        Args:
            result: The computed IntrinsicExtrinsicResult
            image_point: Pixel coordinates

        Returns:
            (x_world, y_world): Coordinates in meters (East, North)

        Raises:
            ValueError: If point projects to infinity (on horizon line)
        """
        u, v = image_point.x, image_point.y
        pt_homogeneous = np.array([u, v, 1.0])
        world_homogeneous = result.inverse_homography_matrix @ pt_homogeneous

        if abs(world_homogeneous[2]) < 1e-10:
            raise ValueError("Point projects to infinity (on horizon line)")

        x_world = world_homogeneous[0] / world_homogeneous[2]
        y_world = world_homogeneous[1] / world_homogeneous[2]

        return Meters(x_world), Meters(y_world)

    @staticmethod
    def world_to_map_pixels(
        x_world: Meters,
        y_world: Meters,
        map_width: Pixels,
        map_height: Pixels,
        pixels_per_meter: Unitless,
    ) -> tuple[int, int]:
        """
        Convert world coordinates (meters) to map pixel coordinates.

        Args:
            x_world: X coordinate in meters (East)
            y_world: Y coordinate in meters (North)
            map_width: Map width in pixels
            map_height: Map height in pixels
            pixels_per_meter: Scale factor

        Returns:
            (x_px, y_px): Pixel coordinates in map image
        """
        map_center_x = map_width // 2
        map_bottom_y = map_height

        x_px = int((x_world * pixels_per_meter) + map_center_x)
        y_px = int(map_bottom_y - (y_world * pixels_per_meter))

        return x_px, y_px

    @staticmethod
    def project_point_static(
        result: IntrinsicExtrinsicResult,
        image_point: PixelPoint,
        pixels_per_meter: Unitless,
    ) -> MapPoint:
        """
        Project image point to MapPoint using result (pure function).

        Args:
            result: The computed IntrinsicExtrinsicResult
            image_point: Pixel coordinates in camera image
            pixels_per_meter: Scale factor for world-to-map conversion

        Returns:
            MapPoint with pixel coordinates on the map

        Raises:
            ValueError: If point projects to infinity
        """
        x_world, y_world = IntrinsicExtrinsicHomography.project_image_point_to_world(
            result, image_point
        )

        map_pixel_x, map_pixel_y = IntrinsicExtrinsicHomography.world_to_map_pixels(
            x_world, y_world, result.map_width, result.map_height, pixels_per_meter
        )

        return MapPoint(
            pixel_x=float(map_pixel_x),
            pixel_y=float(map_pixel_y),
        )

    @staticmethod
    def project_image_to_map(
        result: IntrinsicExtrinsicResult,
        pts: list[tuple[int, int]],
        sw: int,
        sh: int,
        pixels_per_meter: Unitless,
    ) -> list[tuple[int, int]]:
        """
        Project image coordinates to map visualization pixel coordinates.

        Args:
            result: The computed IntrinsicExtrinsicResult
            pts: List of (u, v) image pixel coordinates
            sw: Side-panel (map) width in pixels
            sh: Side-panel (map) height in pixels
            pixels_per_meter: Scale factor

        Returns:
            List of (x, y) pixel coordinates in map visualization
        """
        if result.confidence < 0.3:
            return [(int(x / 2), int(y / 2)) for x, y in pts]

        pts_homogeneous = np.array(pts, dtype=np.float64).T
        pts_homogeneous = np.vstack([pts_homogeneous, np.ones(pts_homogeneous.shape[1])])

        pts_world_homogeneous = result.inverse_homography_matrix @ pts_homogeneous

        w_coords = pts_world_homogeneous[2, :]
        horizon_mask = np.abs(w_coords) < 1e-10
        w_safe = np.where(horizon_mask, np.sign(w_coords + 1e-20) * 1e-10, w_coords)

        Xw = pts_world_homogeneous[0, :] / w_safe
        Yw = pts_world_homogeneous[1, :] / w_safe

        max_coord = 1e6
        Xw = np.clip(Xw, -max_coord, max_coord)
        Yw = np.clip(Yw, -max_coord, max_coord)

        map_center_x = sw // 2
        map_bottom_y = sh

        pts_map_x = (Xw * pixels_per_meter) + map_center_x
        pts_map_y = map_bottom_y - (Yw * pixels_per_meter)

        pts_map = [(int(x), int(y)) for x, y in zip(pts_map_x, pts_map_y)]

        return pts_map

    @staticmethod
    def result_to_homography_result(result: IntrinsicExtrinsicResult) -> HomographyResult:
        """Convert IntrinsicExtrinsicResult to HomographyResult for interface compatibility."""
        return HomographyResult(
            homography_matrix=result.homography_matrix.copy(),
            confidence=result.confidence,
            metadata=result.to_metadata_dict(),
        )
