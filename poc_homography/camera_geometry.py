# camera_geometry.py

import numpy as np
import math
import logging
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)

class CameraGeometry:
    """
    Handles all spatial and projection calculations for a PTZ camera.
    Calculates the homography matrix H to map image points to the world ground plane.

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
      [v]  ∝ H  [Y_world]
      [1]       [1      ]

    For inverse (image to world):
      [X_world]           [u]
      [Y_world]  ∝ H^-1  [v]
      [1      ]           [1]

    CAMERA PARAMETERS:
    ==================
    - K: 3x3 intrinsic matrix (focal length, principal point)
    - w_pos: Camera position [X, Y, Z] in world coordinates (meters)
    - pan_deg: Horizontal rotation (degrees, positive = right/clockwise from above)
    - tilt_deg: Vertical rotation (degrees, positive = down, Hikvision convention)
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
    # Set to 20x camera height as a heuristic for typical PTZ camera scenarios
    # Beyond this, projections likely indicate near-horizontal viewing angles
    MAX_DISTANCE_HEIGHT_RATIO = 20.0

    def __init__(self, w: int, h: int):
        """
        Initializes geometry class with image dimensions.

        Args:
            w: Image width (pixels).
            h: Image height (pixels).
        """
        self.w, self.h = w, h
        self.H = np.eye(3)  # Homography matrix (Image -> Map)
        self.H_inv = np.eye(3)  # Inverse Homography (Map -> Image)
        self.map_width = 640  # Default scale for the map view (pixels)
        self.map_height = 640  # Default scale for the map view (pixels)

        # Pixels per meter for mapping world coords to the side-panel map
        self.PPM = 100

        # Placeholder for real-world geometry parameters (to be set by user)
        self.K = None  # Intrinsic Matrix
        self.w_pos = None  # Camera World Position [Xw, Yw, Zw]
        self.height_m = 5.0  # Camera Height (m)
        self.pan_deg = 0.0  # Camera Pan (Yaw)
        self.tilt_deg = 0.0  # Camera Tilt (Pitch)

    @staticmethod
    def get_intrinsics(zoom_factor: float, W_px: int = 2560, H_px: int = 1440, sensor_width_mm: float = 7.18) -> np.ndarray:
        """
        Calculates the 3x3 Intrinsic Matrix K based on camera specifications and zoom factor.

        Args:
            zoom_factor (float): Digital or optical zoom multiplier (e.g., 1.0 for no zoom).
            W_px (int): Image width in pixels.
            H_px (int): Image height in pixels.
            sensor_width_mm (float): Sensor width in millimeters.

        Returns:
            K (3x3): Intrinsic camera matrix.

        Raises:
            ValueError: If zoom_factor is outside valid range [ZOOM_MIN, ZOOM_MAX].
        """
        # Validate zoom factor
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

        # 1. Calculate focal length in mm (Linear mapping based on datasheet: 1x=5.9mm)
        f_mm = 5.9 * zoom_factor

        # 2. Convert focal length to pixels (f_px = f_mm * (W_px / sensor_width_mm))
        f_px = f_mm * (W_px / sensor_width_mm)

        # 3. Construct K
        cx, cy = W_px / 2.0, H_px / 2.0
        K = np.array([
            [f_px, 0, cx],
            [0, f_px, cy],
            [0, 0, 1]
        ])
        return K

    def _validate_parameters(self, K: np.ndarray, w_pos: np.ndarray,
                            pan_deg: float, tilt_deg: float) -> None:
        """
        Validates all camera parameters before setting them.

        Args:
            K: 3x3 intrinsic matrix
            w_pos: Camera position in world coordinates [X, Y, Z] (meters)
            pan_deg: Pan angle in degrees (positive = right)
            tilt_deg: Tilt angle in degrees (positive = down, Hikvision convention)

        Raises:
            ValueError: If any parameter is invalid or out of acceptable range.
        """
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
        if height < self.HEIGHT_MIN or height > self.HEIGHT_MAX:
            raise ValueError(
                f"Camera height {height:.2f}m is out of valid range "
                f"[{self.HEIGHT_MIN}, {self.HEIGHT_MAX}]m"
            )

        if height < self.HEIGHT_WARN_LOW:
            logger.warning(
                f"Camera height {height:.2f}m is very low (<{self.HEIGHT_WARN_LOW}m). "
                f"Ground projection accuracy may be reduced."
            )
        elif height > self.HEIGHT_WARN_HIGH:
            logger.warning(
                f"Camera height {height:.2f}m is very high (>{self.HEIGHT_WARN_HIGH}m). "
                f"Ground projection accuracy may be reduced at extreme distances."
            )

        # Tilt angle validation
        if tilt_deg <= self.TILT_MIN or tilt_deg > self.TILT_MAX:
            raise ValueError(
                f"Tilt angle {tilt_deg:.1f}° is out of valid range "
                f"({self.TILT_MIN}, {self.TILT_MAX}]°. "
                f"Camera must point downward (positive tilt) for ground plane projection."
            )

        if tilt_deg < self.TILT_WARN_LOW:
            logger.warning(
                f"Tilt angle {tilt_deg:.1f}° is near horizontal (<{self.TILT_WARN_LOW}°). "
                f"Ground projection may be unstable or extend to very large distances."
            )
        elif tilt_deg > self.TILT_WARN_HIGH:
            logger.warning(
                f"Tilt angle {tilt_deg:.1f}° is very steep (>{self.TILT_WARN_HIGH}°). "
                f"Ground coverage area will be very limited."
            )

        # FOV validation (calculated from K)
        focal_length = K[0, 0]  # Assuming square pixels (fx = fy)
        sensor_width_px = 2.0 * K[0, 2]  # Principal point is at center
        fov_rad = 2.0 * math.atan(sensor_width_px / (2.0 * focal_length))
        fov_deg = math.degrees(fov_rad)

        if fov_deg < self.FOV_MIN_DEG or fov_deg > self.FOV_MAX_DEG:
            raise ValueError(
                f"Calculated FOV {fov_deg:.1f}° is out of reasonable range "
                f"[{self.FOV_MIN_DEG}, {self.FOV_MAX_DEG}]°. "
                f"Check intrinsic matrix K and zoom factor."
            )

        if fov_deg < self.FOV_WARN_MIN_DEG or fov_deg > self.FOV_WARN_MAX_DEG:
            logger.warning(
                f"FOV {fov_deg:.1f}° is unusual. Typical PTZ cameras have FOV between "
                f"{self.FOV_WARN_MIN_DEG}° and {self.FOV_WARN_MAX_DEG}°."
            )

    def set_camera_parameters(self, K: np.ndarray, w_pos: np.ndarray,
                              pan_deg: float, tilt_deg: float,
                              map_width: int, map_height: int):
        """
        Sets all required parameters and calculates the Homography matrix H.

        Args:
            K: 3x3 intrinsic matrix
            w_pos: Camera position in world coordinates [X, Y, Z] (meters)
            pan_deg: Pan angle in degrees (positive = right)
            tilt_deg: Tilt angle in degrees (positive = down, Hikvision convention)
            map_width: Width of output map in pixels
            map_height: Height of output map in pixels

        Raises:
            ValueError: If any parameter is invalid or out of acceptable range.
        """
        # Validate all parameters before proceeding
        self._validate_parameters(K, w_pos, pan_deg, tilt_deg)

        # Log all parameters at INFO level
        logger.info("Setting camera parameters:")
        logger.info(f"  Camera position: [{w_pos[0]:.2f}, {w_pos[1]:.2f}, {w_pos[2]:.2f}] meters")
        logger.info(f"  Pan: {pan_deg:.1f}°, Tilt: {tilt_deg:.1f}°")
        logger.info(f"  Intrinsic matrix K:\n{K}")
        logger.info(f"  Map dimensions: {map_width}x{map_height} pixels")

        self.K = K
        self.w_pos = w_pos
        self.height_m = w_pos[2]  # Z component is height
        self.pan_deg = pan_deg
        self.tilt_deg = tilt_deg
        self.map_width = map_width
        self.map_height = map_height

        self.H = self._calculate_ground_homography()

        # Validate and invert homography with condition number checks
        det_H = np.linalg.det(self.H)
        if abs(det_H) < 1e-10:
            raise ValueError(
                f"Homography matrix is singular (det={det_H:.2e}). "
                f"Cannot compute inverse. Check camera parameters."
            )

        # Compute condition number
        cond_H = np.linalg.cond(self.H)
        if cond_H > self.CONDITION_ERROR:
            raise ValueError(
                f"Homography condition number {cond_H:.2e} is too high (>{self.CONDITION_ERROR:.2e}). "
                f"Matrix is ill-conditioned. Check camera parameters, especially tilt angle."
            )
        elif cond_H > self.CONDITION_WARN:
            logger.warning(
                f"Homography condition number {cond_H:.2e} is high (>{self.CONDITION_WARN:.2e}). "
                f"Inverse may be numerically unstable."
            )

        self.H_inv = np.linalg.inv(self.H)

        # Validate projected distance from image center
        self._validate_projection()

        logger.info("Geometry setup complete. Homography matrix H calculated.")
        logger.info(f"  Homography det(H): {det_H:.2e}")
        logger.info(f"  Homography condition number: {cond_H:.2e}")

        # Also print to console for backward compatibility
        print("Geometry setup complete. Homography matrix H calculated.")
        print(f"  Camera position: [{w_pos[0]:.2f}, {w_pos[1]:.2f}, {w_pos[2]:.2f}] meters")
        print(f"  Pan: {pan_deg:.1f}°, Tilt: {tilt_deg:.1f}°")
        print(f"  Homography det(H): {det_H:.2e}")

    def _validate_projection(self) -> None:
        """
        Validates that the homography produces reasonable ground projections.

        Raises:
            ValueError: If projection yields invalid results (negative or infinite distances).
        """
        # Project image center to ground plane
        image_center = np.array([self.w / 2.0, self.h / 2.0, 1.0])
        world_point = self.H_inv @ image_center

        # Normalize homogeneous coordinates
        if abs(world_point[2]) < 1e-10:
            raise ValueError(
                "Projection of image center yields invalid homogeneous coordinate (w ≈ 0). "
                "Check tilt angle and camera height."
            )

        Xw = world_point[0] / world_point[2]
        Yw = world_point[1] / world_point[2]

        # Calculate distance from camera position to projected point
        distance = math.sqrt((Xw - self.w_pos[0])**2 + (Yw - self.w_pos[1])**2)

        # Validate distance
        if not math.isfinite(distance):
            raise ValueError(
                "Projected ground distance is infinite or NaN. "
                "Check tilt angle - camera may be pointing too close to horizontal."
            )

        if distance < 0:
            raise ValueError(
                f"Projected ground distance is negative ({distance:.2f}m). "
                f"This should not occur. Check homography calculation."
            )

        # Log the projected distance for information
        logger.info(f"  Image center projects to ground at distance: {distance:.2f}m")

        # Warn if distance seems unreasonable
        max_reasonable_distance = self.height_m * self.MAX_DISTANCE_HEIGHT_RATIO
        if distance > max_reasonable_distance:
            logger.warning(
                f"Projected ground distance {distance:.2f}m is very large (>{max_reasonable_distance:.2f}m). "
                f"This may indicate a near-horizontal tilt angle."
            )

    # --- Core Geometry Methods (Unchanged from previous output) ---

    def _get_rotation_matrix(self) -> np.ndarray:
        """
        Calculates the 3x3 rotation matrix R from world to camera coordinates
        based on pan (Yaw) and tilt (Pitch). Assumes zero roll.

        Coordinate System Convention:
        - World: X=East, Y=North, Z=Up
        - Camera: X=Right, Y=Down, Z=Forward (optical axis)

        Rotation order: Pan first (around Z-axis), then Tilt (around rotated X-axis).
        This matches standard PTZ camera behavior.

        Tilt Convention:
        - Positive tilt_deg = camera pointing downward (Hikvision convention)
        - Negative tilt_deg = camera pointing upward
        - The sign is negated when converting to radians because the standard
          Rx rotation matrix rotates in the opposite direction to achieve
          downward tilt in a Y=Down camera coordinate system.
        """
        pan_rad = math.radians(self.pan_deg)
        # Negate tilt: positive tilt_deg means "point down", but standard Rx
        # rotation needs negative angle to rotate the optical axis downward
        tilt_rad = math.radians(-self.tilt_deg)

        # 1. Yaw (Rotation around World Z-axis - Pan)
        # Positive pan rotates camera to the right (clockwise from above)
        Rz = np.array([
            [math.cos(pan_rad), -math.sin(pan_rad), 0],
            [math.sin(pan_rad), math.cos(pan_rad), 0],
            [0, 0, 1]
        ])

        # 2. Pitch (Rotation around X-axis - Tilt)
        # Standard rotation matrix (Camera Y=Down requires careful angle interpretation)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(tilt_rad), -math.sin(tilt_rad)],
            [0, math.sin(tilt_rad), math.cos(tilt_rad)]
        ])

        # Apply Pan first, then Tilt
        R = Rz @ Rx
        return R

    def _calculate_ground_homography(self) -> np.ndarray:
        """
        Calculates the Homography matrix H that maps world ground plane (Z=0) to image pixels.

        Mathematical derivation:
        - Camera position in world: C = w_pos = [Xw, Yw, Zw]
        - Rotation from world to camera: R (3x3)
        - Translation: t = -R @ C (position of world origin in camera frame)
        - For ground plane (Z=0): P_world = [X, Y, 0]
        - Projection: p_image = K @ [R @ P_world + t]
        - Since Z=0, homography becomes: H = K @ [r1, r2, t]
          where r1, r2 are first two columns of R

        Returns:
            H (3x3): Homography matrix mapping [X_world, Y_world, 1] -> [u, v, 1] (image coords)
        """

        if self.K is None:
            print("Warning: K not set. Returning Identity Homography.")
            return np.eye(3)

        R = self._get_rotation_matrix()

        # Camera position C in world coordinates
        C = self.w_pos  # [Xw, Yw, Zw]

        # Translation from camera to world origin: t = -R @ C
        t = -R @ C

        # Build homography: H = K @ [r1, r2, t]
        # r1, r2 are the first two columns of R (corresponding to X and Y axes)
        r1 = R[:, 0]
        r2 = R[:, 1]

        H_extrinsic = np.column_stack([r1, r2, t])

        H = self.K @ H_extrinsic

        # Normalize so H[2, 2] = 1
        if abs(H[2, 2]) < 1e-10:
            print("Warning: Homography normalization failed (H[2,2] near zero). Returning identity.")
            return np.eye(3)

        H /= H[2, 2]

        return H

    def project_image_to_map(self, pts: List[Tuple[int, int]], sw: int, sh: int) -> List[Tuple[int, int]]:
        """
        Projects image coordinates (pixels) to the world ground plane (meters).
        """
        if self.H_inv is None or np.all(self.H_inv == np.eye(3)):
            return [(int(x / 2), int(y / 2)) for x, y in pts]

        pts_homogeneous = np.array(pts, dtype=np.float64).T
        pts_homogeneous = np.vstack([pts_homogeneous, np.ones(pts_homogeneous.shape[1])])

        # Project from Image to World Ground Plane (Xw, Yw, 1)
        pts_world_homogeneous = self.H_inv @ pts_homogeneous

        # Normalize world coordinates (convert [Xw, Yw, W] to [Xw/W, Yw/W, 1])
        Xw = pts_world_homogeneous[0, :] / pts_world_homogeneous[2, :]
        Yw = pts_world_homogeneous[1, :] / pts_world_homogeneous[2, :]

        # --- Mapping Policy (uses self.PPM pixels per meter) ---
        map_center_x = sw // 2
        map_bottom_y = sh

        pts_map_x = (Xw * self.PPM) + map_center_x
        pts_map_y = map_bottom_y - (Yw * self.PPM)

        pts_map = [(int(x), int(y)) for x, y in zip(pts_map_x, pts_map_y)]

        return pts_map

    def world_to_map(self, Xw: float, Yw: float, sw: int = None, sh: int = None) -> Tuple[int, int]:
        """
        Map a world coordinate (meters) to side-panel pixel coordinates.

        Args:
            Xw, Yw: world coordinates in meters
            sw, sh: side-panel width/height in pixels. If omitted, uses self.map_width/self.map_height.

        Returns:
            (x_px, y_px) tuple in pixels relative to the top-left of the side panel image.
        """
        if sw is None:
            sw = self.map_width
        if sh is None:
            sh = self.map_height

        map_center_x = sw // 2
        map_bottom_y = sh

        x_px = int((Xw * self.PPM) + map_center_x)
        y_px = int(map_bottom_y - (Yw * self.PPM))

        return x_px, y_px