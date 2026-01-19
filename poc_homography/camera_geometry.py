from __future__ import annotations

# camera_geometry.py
import logging
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np

from poc_homography.domain.vo.image_dimensions import ImageDimensions
from poc_homography.types import Degrees, Millimeters, Pixels, Unitless, degrees_to_radians

if TYPE_CHECKING:
    from poc_homography.domain.vo import (
        CameraIntrinsics,
        Homography,
        LensDistortion,
        Matrix3x3,
        Orientation,
        Vector3,
    )

logger = logging.getLogger(__name__)


class CameraGeometry:
    """
    Handles all spatial and projection calculations for a PTZ camera.
    Calculates the homography matrix H to map image points to the world ground plane.

    This class uses an IMMUTABLE pattern - all computation is done through the
    `compute_from_vo()` classmethod which takes domain VOs and returns a Homography VO.

    COORDINATE SYSTEM CONVENTIONS:
    ===============================
    World Frame (Right-Handed):
      - Origin: Arbitrary reference point (typically camera location or scene center)
      - X-axis: East (positive = East, negative = West)
      - Y-axis: North (positive = North, negative = South)
      - Z-axis: Up (positive = Up, height above ground)
      - Ground plane: Z = 0

    Camera Frame (Right-Handed, standard computer vision):
      - Origin: Camera optical center
      - X-axis: Right (in image)
      - Y-axis: Down (in image)
      - Z-axis: Forward (along optical axis, into the scene)

    Image Frame:
      - Origin: Top-left corner
      - u-axis: Right (width)
      - v-axis: Down (height)
      - Units: Pixels

    HOMOGRAPHY:
    ===========
    The homography H maps world ground plane points (Z=0) to image pixels:
      [u]       [X_world]
      [v]  = H  [Y_world]
      [1]       [1      ]

    For inverse (image to world):
      [X_world]           [u]
      [Y_world]  = H^-1  [v]
      [1      ]           [1]

    USAGE:
    ======
    ```python
    from poc_homography.domain.vo import CameraIntrinsics, Vector3, Orientation
    from poc_homography.types import Millimeters, Pixels, Degrees

    intrinsics = CameraIntrinsics.create(
        sensor_width=Millimeters(6.78),
        base_focal_length=Millimeters(5.9),
        image_width=Pixels(1920),
        image_height=Pixels(1080),
        zoom_factor=1.0,
    )
    position = Vector3.create(0.0, 0.0, 10.0)
    orientation = Orientation.create(
        yaw=Degrees(45.0), pitch=Degrees(30.0), roll=Degrees(0.0)
    )
    homography = CameraGeometry.compute_from_vo(intrinsics, position, orientation)
    u, v = homography.world_to_image(5.0, 10.0)
    Xw, Yw = homography.image_to_world(u, v)
    ```
    """

    # Validation constants
    ZOOM_MIN = 1.0
    ZOOM_MAX = 25.0
    ZOOM_WARN_HIGH = 20.0

    TILT_MIN = 0.0  # Must be positive (pointing down)
    TILT_MAX = 90.0
    TILT_WARN_LOW = 10.0
    TILT_WARN_HIGH = 80.0

    HEIGHT_MIN = 1.0  # meters
    HEIGHT_MAX = 50.0
    HEIGHT_WARN_LOW = 2.0
    HEIGHT_WARN_HIGH = 30.0

    FOV_MIN_DEG = 2.0
    FOV_MAX_DEG = 120.0
    FOV_WARN_MIN_DEG = 10.0
    FOV_WARN_MAX_DEG = 90.0

    # Roll validation thresholds
    ROLL_WARN_THRESHOLD = 5.0  # Warning when |roll_deg| > 5.0
    ROLL_ERROR_THRESHOLD = 15.0  # Error when |roll_deg| > 15.0

    def __init__(self, w: Pixels, h: Pixels):
        """
        Initializes geometry class with image dimensions.

        Args:
            w: Image width (pixels).
            h: Image height (pixels).
        """
        self.dimensions = ImageDimensions.create(width=w, height=h)

    @property
    def w(self) -> Pixels:
        """Image width in pixels (backward-compatible property)."""
        return self.dimensions.width

    @property
    def h(self) -> Pixels:
        """Image height in pixels (backward-compatible property)."""
        return self.dimensions.height

    @staticmethod
    def get_intrinsics(
        zoom_factor: Unitless,
        W_px: Pixels = Pixels(1920),
        H_px: Pixels = Pixels(1080),
        sensor_width_mm: Millimeters = Millimeters(6.78),
    ) -> np.ndarray:
        """
        Calculates the 3x3 Intrinsic Matrix K based on camera specifications and zoom factor.

        Args:
            zoom_factor: Digital or optical zoom multiplier (e.g., 1.0 for no zoom).
            W_px: Image width in pixels.
            H_px: Image height in pixels.
            sensor_width_mm: Sensor width in millimeters.

        Returns:
            K (3x3): Intrinsic camera matrix.

        Raises:
            ValueError: If zoom_factor is outside valid range [ZOOM_MIN, ZOOM_MAX].
        """
        if zoom_factor < CameraGeometry.ZOOM_MIN or zoom_factor > CameraGeometry.ZOOM_MAX:
            raise ValueError(
                f"Zoom factor {zoom_factor} is out of valid range "
                f"[{CameraGeometry.ZOOM_MIN}, {CameraGeometry.ZOOM_MAX}]"
            )

        if zoom_factor > CameraGeometry.ZOOM_WARN_HIGH:
            logger.warning(
                f"Zoom factor {zoom_factor} is very high (>{CameraGeometry.ZOOM_WARN_HIGH}). "
                f"Results may be less accurate at extreme zoom levels."
            )

        # Calculate focal length in mm based on Hikvision DS-2DF8425IX-AELW datasheet:
        # Focal length: 5.9mm (wide) to 147.5mm (tele), 25x optical zoom
        f_mm = 5.9 * zoom_factor
        f_px = f_mm * (W_px / sensor_width_mm)

        cx, cy = W_px / 2.0, H_px / 2.0
        K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]])
        return K

    @staticmethod
    def _validate_parameters_static(
        intrinsics: CameraIntrinsics,
        camera_position: Vector3,
        tilt_deg: Degrees,
        roll_deg: Degrees = Degrees(0.0),
    ) -> None:
        """Validate camera parameters for homography computation.

        Note: Shape and NaN/Infinity checks are already enforced by VO factories
        (Matrix3x3.create, Vector3.create), so we only validate domain constraints.

        Args:
            intrinsics: Camera intrinsic parameters.
            camera_position: Camera position in world coordinates.
            tilt_deg: Tilt angle in degrees (positive = down, Hikvision convention).
            roll_deg: Roll angle in degrees (positive = clockwise, default = 0.0).

        Raises:
            ValueError: If any parameter is invalid or out of acceptable range.
        """
        cls = CameraGeometry

        # Camera height validation
        height = camera_position.z
        if height < cls.HEIGHT_MIN or height > cls.HEIGHT_MAX:
            raise ValueError(
                f"Camera height {height:.2f}m is out of valid range "
                f"[{cls.HEIGHT_MIN}, {cls.HEIGHT_MAX}]m"
            )

        if height < cls.HEIGHT_WARN_LOW:
            logger.warning(
                f"Camera height {height:.2f}m is very low (<{cls.HEIGHT_WARN_LOW}m). "
                f"Ground projection accuracy may be reduced."
            )
        elif height > cls.HEIGHT_WARN_HIGH:
            logger.warning(
                f"Camera height {height:.2f}m is very high (>{cls.HEIGHT_WARN_HIGH}m). "
                f"Ground projection accuracy may be reduced at extreme distances."
            )

        # Tilt angle validation
        if tilt_deg <= cls.TILT_MIN or tilt_deg > cls.TILT_MAX:
            raise ValueError(
                f"Tilt angle {tilt_deg:.1f} is out of valid range "
                f"({cls.TILT_MIN}, {cls.TILT_MAX}]. "
                f"Camera must point downward (positive tilt) for ground plane projection."
            )

        if tilt_deg < cls.TILT_WARN_LOW:
            logger.debug(
                f"Tilt angle {tilt_deg:.1f} is near horizontal (<{cls.TILT_WARN_LOW}). "
                f"Ground projection may be unstable or extend to very large distances."
            )
        elif tilt_deg > cls.TILT_WARN_HIGH:
            logger.debug(
                f"Tilt angle {tilt_deg:.1f} is very steep (>{cls.TILT_WARN_HIGH}). "
                f"Ground coverage area will be very limited."
            )

        # FOV validation using intrinsics properties
        focal_length = intrinsics.focal_length_px
        sensor_width_px = 2.0 * intrinsics.cx
        fov_rad = 2.0 * math.atan(sensor_width_px / (2.0 * focal_length))
        fov_deg = math.degrees(fov_rad)

        if fov_deg < cls.FOV_MIN_DEG or fov_deg > cls.FOV_MAX_DEG:
            raise ValueError(
                f"Calculated FOV {fov_deg:.1f} is out of reasonable range "
                f"[{cls.FOV_MIN_DEG}, {cls.FOV_MAX_DEG}]. "
                f"Check intrinsic matrix K and zoom factor."
            )

        if fov_deg < cls.FOV_WARN_MIN_DEG or fov_deg > cls.FOV_WARN_MAX_DEG:
            logger.warning(
                f"FOV {fov_deg:.1f} is unusual. Typical PTZ cameras have FOV between "
                f"{cls.FOV_WARN_MIN_DEG} and {cls.FOV_WARN_MAX_DEG}."
            )

        # Roll angle validation
        if abs(roll_deg) > cls.ROLL_ERROR_THRESHOLD:
            raise ValueError(
                f"Roll angle {roll_deg:.1f} is outside valid range "
                f"[-{cls.ROLL_ERROR_THRESHOLD}, {cls.ROLL_ERROR_THRESHOLD}]. "
                f"Check camera mount alignment."
            )

        if abs(roll_deg) > cls.ROLL_WARN_THRESHOLD:
            warnings.warn(
                f"Roll angle {roll_deg:.1f} is unusually large (>{cls.ROLL_WARN_THRESHOLD}). "
                f"Typical camera mount roll is +/-2. Verify configuration.",
                UserWarning,
            )

    @staticmethod
    def _get_rotation_matrix_static(
        pan_deg: Degrees, tilt_deg: Degrees, roll_deg: Degrees
    ) -> Matrix3x3:
        """Static version of rotation matrix calculation for use in compute().

        Calculates the 3x3 rotation matrix R from world to camera coordinates
        based on pan (Yaw), tilt (Pitch), and roll.

        Args:
            pan_deg: Pan angle in degrees
            tilt_deg: Tilt angle in degrees
            roll_deg: Roll angle in degrees

        Returns:
            R: 3x3 rotation matrix transforming world coordinates to camera frame
        """
        from poc_homography.domain.vo import Matrix3x3

        pan_rad = degrees_to_radians(pan_deg)
        tilt_rad = degrees_to_radians(tilt_deg)
        roll_rad = degrees_to_radians(roll_deg)

        # Base transformation from World to Camera when pan=0, tilt=0
        R_base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        # Pan rotation around world Z-axis (yaw)
        Rz_pan = np.array(
            [
                [math.cos(pan_rad), -math.sin(pan_rad), 0],
                [math.sin(pan_rad), math.cos(pan_rad), 0],
                [0, 0, 1],
            ]
        )

        # Roll rotation around camera Z-axis (optical axis)
        Rz_roll = np.array(
            [
                [math.cos(roll_rad), -math.sin(roll_rad), 0],
                [math.sin(roll_rad), math.cos(roll_rad), 0],
                [0, 0, 1],
            ]
        )

        # Tilt rotation around camera X-axis (pitch)
        Rx_tilt = np.array(
            [
                [1, 0, 0],
                [0, math.cos(tilt_rad), -math.sin(tilt_rad)],
                [0, math.sin(tilt_rad), math.cos(tilt_rad)],
            ]
        )

        # Full rotation: R = R_tilt @ R_roll @ R_base @ R_pan
        R = Rx_tilt @ Rz_roll @ R_base @ Rz_pan
        return Matrix3x3.create(R)

    @staticmethod
    def _calculate_ground_homography_static(
        K: Matrix3x3, w_pos: Vector3, R: Matrix3x3
    ) -> np.ndarray:
        """Static version of ground homography calculation for use in compute().

        Calculates the Homography matrix H that maps world ground plane (Z=0)
        to image pixels.

        Args:
            K: 3x3 intrinsic matrix
            w_pos: Camera position in world coordinates
            R: 3x3 rotation matrix from world to camera frame

        Returns:
            H: 3x3 normalized homography matrix
        """
        # Extract numpy arrays for computation
        K_arr = K._to_array()
        w_pos_arr = w_pos._to_array()
        R_arr = R._to_array()

        # Translation from camera to world origin: t = -R @ C
        t = -R_arr @ w_pos_arr

        # Build homography: H = K @ [r1, r2, t]
        r1 = R_arr[:, 0]  # Column 0: world X-axis in camera frame
        r2 = R_arr[:, 1]  # Column 1: world Y-axis in camera frame

        # Construct 3x3 extrinsic homography matrix: [r1, r2, t]
        H_extrinsic = np.column_stack([r1, r2, t])

        H = K_arr @ H_extrinsic

        # Normalize so H[2, 2] = 1 for consistent scale
        if abs(H[2, 2]) < 1e-10:
            logger.warning(
                "Homography normalization failed (H[2,2] near zero). Returning identity."
            )
            return np.eye(3)

        H_normalized: np.ndarray = H / H[2, 2]
        return H_normalized

    @classmethod
    def compute_from_vo(
        cls,
        intrinsics: CameraIntrinsics,
        camera_position: Vector3,
        orientation: Orientation,
        distortion: LensDistortion | None = None,  # noqa: ARG003
    ) -> Homography:
        """Compute homography from domain value objects.

        This is the primary API for computing homography matrices using the
        immutable VO pattern. It takes domain VOs and returns a Homography VO.

        Args:
            intrinsics: Camera intrinsic parameters (K matrix, dimensions).
            camera_position: Camera position in world coordinates [X, Y, Z] (meters).
            orientation: Camera orientation (yaw, pitch, roll).
            distortion: Optional lens distortion coefficients (currently not applied
                to homography, but reserved for future use).

        Returns:
            Homography VO with projection methods (world_to_image, image_to_world).

        Raises:
            ValueError: If parameters are invalid or produce degenerate homography.
        """
        # Import here to avoid circular imports
        from poc_homography.domain.vo import Homography

        # Validate parameters using VOs directly
        cls._validate_parameters_static(
            intrinsics, camera_position, orientation.pitch, orientation.roll
        )

        # Compute rotation matrix
        R = cls._get_rotation_matrix_static(orientation.yaw, orientation.pitch, orientation.roll)

        # Compute homography matrix
        H = cls._calculate_ground_homography_static(intrinsics.to_K(), camera_position, R)

        # Create and return Homography VO
        return Homography.create(H)
