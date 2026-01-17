from __future__ import annotations

# camera_geometry.py
import logging
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np

from poc_homography.domain.vo.image_dimensions import ImageDimensions
from poc_homography.types import Degrees, Meters, Millimeters, Pixels, Unitless, degrees_to_radians

if TYPE_CHECKING:
    from poc_homography.domain.vo import (
        CameraIntrinsics,
        Homography,
        LensDistortion,
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

    CONDITION_WARN = 1e6
    CONDITION_ERROR = 1e10

    # Maximum reasonable distance ratio for ground projection validation
    MAX_DISTANCE_HEIGHT_RATIO = 20.0

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
        K: np.ndarray,
        w_pos: np.ndarray,
        _pan_deg: Degrees,  # Currently unused but kept for API consistency
        tilt_deg: Degrees,
        roll_deg: Degrees = Degrees(0.0),
    ) -> None:
        """Static version of parameter validation for use in compute().

        This is a static method to enable pure function computation without
        requiring an instance.

        Args:
            K: 3x3 intrinsic matrix
            w_pos: Camera position in world coordinates [X, Y, Z] (meters)
            _pan_deg: Pan angle in degrees (currently unused, kept for API consistency)
            tilt_deg: Tilt angle in degrees (positive = down, Hikvision convention)
            roll_deg: Roll angle in degrees (positive = clockwise, default = 0.0)

        Raises:
            ValueError: If any parameter is invalid or out of acceptable range.
        """
        cls = CameraGeometry

        # Validate K matrix shape
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got shape {K.shape}")

        # Validate K matrix for NaN/Infinity
        if not np.all(np.isfinite(K)):
            raise ValueError("K matrix contains NaN or Infinity values")

        # Validate w_pos structure
        if len(w_pos) != 3:
            raise ValueError(f"w_pos must have 3 elements [X, Y, Z], got {len(w_pos)}")

        # Validate w_pos for NaN/Infinity
        if not np.all(np.isfinite(w_pos)):
            raise ValueError("w_pos contains NaN or Infinity values")

        # Camera height validation (w_pos[2])
        height = w_pos[2]
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

        # FOV validation (calculated from K)
        focal_length = K[0, 0]  # Assuming square pixels (fx = fy)
        sensor_width_px = 2.0 * K[0, 2]  # Principal point is at center
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
    ) -> np.ndarray:
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
        R: np.ndarray = Rx_tilt @ Rz_roll @ R_base @ Rz_pan
        return R

    @staticmethod
    def _calculate_ground_homography_static(
        K: np.ndarray, w_pos: np.ndarray, R: np.ndarray
    ) -> np.ndarray:
        """Static version of ground homography calculation for use in compute().

        Calculates the Homography matrix H that maps world ground plane (Z=0)
        to image pixels.

        Args:
            K: 3x3 intrinsic matrix
            w_pos: Camera position [X, Y, Z] in world coordinates
            R: 3x3 rotation matrix from world to camera frame

        Returns:
            H: 3x3 normalized homography matrix
        """
        # Translation from camera to world origin: t = -R @ C
        t = -R @ w_pos

        # Build homography: H = K @ [r1, r2, t]
        r1 = R[:, 0]  # Column 0: world X-axis in camera frame
        r2 = R[:, 1]  # Column 1: world Y-axis in camera frame

        # Construct 3x3 extrinsic homography matrix
        H_extrinsic = np.column_stack([r1, r2, t])
        if H_extrinsic.shape != (3, 3):
            raise ValueError(f"H_extrinsic must be 3x3, got {H_extrinsic.shape}")

        H = K @ H_extrinsic
        if H.shape != (3, 3):
            raise ValueError(f"Homography must be 3x3, got {H.shape}")

        # Normalize so H[2, 2] = 1 for consistent scale
        if abs(H[2, 2]) < 1e-10:
            logger.warning(
                "Homography normalization failed (H[2,2] near zero). Returning identity."
            )
            return np.eye(3)

        H_normalized: np.ndarray = H / H[2, 2]
        return H_normalized

    @staticmethod
    def _compute_center_projection_distance(
        H_inv: np.ndarray,
        w_pos: np.ndarray,
        image_width: int,
        image_height: int,
        validation_messages: list[str],
    ) -> float | None:
        """Compute distance from camera to where image center projects on ground.

        Args:
            H_inv: 3x3 inverse homography matrix
            w_pos: Camera position [X, Y, Z] in world coordinates
            image_width: Image width in pixels
            image_height: Image height in pixels
            validation_messages: List to append validation messages to

        Returns:
            Distance in meters, or None if projection is invalid
        """
        # Project image center to ground plane
        image_center = np.array([image_width / 2.0, image_height / 2.0, 1.0])
        world_point = H_inv @ image_center

        # Normalize homogeneous coordinates
        if abs(world_point[2]) < 1e-10:
            validation_messages.append(
                "Projection of image center yields invalid homogeneous coordinate (w near 0). "
                "Check tilt angle and camera height."
            )
            return None

        Xw = world_point[0] / world_point[2]
        Yw = world_point[1] / world_point[2]

        # Calculate distance from camera position to projected point
        distance = math.sqrt((Xw - w_pos[0]) ** 2 + (Yw - w_pos[1]) ** 2)

        # Validate distance
        if not math.isfinite(distance):
            validation_messages.append(
                "Projected ground distance is infinite or NaN. "
                "Check tilt angle - camera may be pointing too close to horizontal."
            )
            return None

        if distance < 0:
            validation_messages.append(
                f"Projected ground distance is negative ({distance:.2f}m). "
                f"This should not occur. Check homography calculation."
            )
            return None

        return distance

    @staticmethod
    def world_to_map(
        Xw: Meters,
        Yw: Meters,
        map_width: Pixels,
        map_height: Pixels,
        pixels_per_meter: Unitless,
    ) -> tuple[Pixels, Pixels]:
        """
        Map a world coordinate (meters) to map pixel coordinates.

        Args:
            Xw: World x-coordinate in meters (East)
            Yw: World y-coordinate in meters (North)
            map_width: Width of map in pixels
            map_height: Height of map in pixels
            pixels_per_meter: Scale factor

        Returns:
            (x_px, y_px) tuple in pixels relative to the top-left of the map image.
        """
        map_center_x = map_width // 2
        map_bottom_y = map_height

        x_px = Pixels(int((Xw * pixels_per_meter) + map_center_x))
        y_px = Pixels(int(map_bottom_y - (Yw * pixels_per_meter)))

        return x_px, y_px

    @staticmethod
    def undistort_point(
        u: float,
        v: float,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
        iterations: int = 10,
    ) -> tuple[float, float]:
        """
        Undistort a single image point to remove lens distortion.

        Converts a distorted pixel coordinate to the corresponding undistorted
        coordinate that would be observed with an ideal pinhole camera.

        Args:
            u: Distorted x pixel coordinate
            v: Distorted y pixel coordinate
            K: 3x3 intrinsic camera matrix
            dist_coeffs: Distortion coefficients array [k1, k2, p1, p2, k3]
            iterations: Number of iterations for undistortion (default: 10)

        Returns:
            Tuple (u_undistorted, v_undistorted) in pixel coordinates
        """
        if np.allclose(dist_coeffs, 0.0):
            return (u, v)

        # Get intrinsic parameters
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        k1, k2, p1, p2, k3 = dist_coeffs

        # Convert to normalized camera coordinates
        x_d = (u - cx) / fx
        y_d = (v - cy) / fy

        # Iterative undistortion (Newton-Raphson style)
        x = x_d
        y = y_d

        for _ in range(iterations):
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2

            # Radial distortion factor
            radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

            # Tangential distortion
            dx_tangential = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            dy_tangential = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

            # Update estimate (inverse iteration)
            x = (x_d - dx_tangential) / radial
            y = (y_d - dy_tangential) / radial

        # Convert back to pixel coordinates
        u_undist = x * fx + cx
        v_undist = y * fy + cy

        return (u_undist, v_undist)

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

        # Extract numpy arrays from VOs
        K = intrinsics.to_K()._to_array()
        w_pos = camera_position.to_array()
        pan_deg = Degrees(float(orientation.yaw))
        tilt_deg = Degrees(float(orientation.pitch))
        roll_deg = Degrees(float(orientation.roll))

        # Validate parameters
        cls._validate_parameters_static(K, w_pos, pan_deg, tilt_deg, roll_deg)

        # Compute rotation matrix
        R = cls._get_rotation_matrix_static(pan_deg, tilt_deg, roll_deg)

        # Compute homography matrix
        H = cls._calculate_ground_homography_static(K, w_pos, R)

        # Create and return Homography VO
        return Homography.create(H)
