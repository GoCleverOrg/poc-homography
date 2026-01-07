# camera_geometry.py

import numpy as np
import math
import logging
from typing import List, Tuple, Union, Dict, Any, Optional

from poc_homography.types import (
    Degrees, Meters, Pixels, Millimeters, Unitless
)

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

    def __init__(self, w: Pixels, h: Pixels):
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

        # Height uncertainty for propagation (set via set_height_uncertainty)
        self.height_uncertainty_lower = None  # Lower bound of height confidence interval
        self.height_uncertainty_upper = None  # Upper bound of height confidence interval

        # Lens distortion coefficients (OpenCV model)
        # Radial: k1, k2, k3 (k3 often 0 for simple lenses)
        # Tangential: p1, p2
        self._dist_coeffs = None  # Will be np.array([k1, k2, p1, p2, k3]) when set
        self._use_distortion = False

    @staticmethod
    def get_intrinsics(
        zoom_factor: Unitless,
        W_px: Pixels = Pixels(1920),
        H_px: Pixels = Pixels(1080),
        sensor_width_mm: Millimeters = Millimeters(6.78)
    ) -> np.ndarray:
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

        # 1. Calculate focal length in mm
        # Based on Hikvision DS-2DF8425IX-AELW datasheet:
        #   - Focal length: 5.9mm (wide) to 147.5mm (tele)
        #   - Optical zoom: 25x (linear: f = 5.9mm × zoom_factor)
        f_mm = 5.9 * zoom_factor

        # 2. Convert focal length to pixels (f_px = f_mm * (W_px / sensor_width_mm))
        # Default sensor_width_mm (6.78mm) is calculated from:
        #   - Horizontal FOV = 59.8° at 5.9mm focal length
        #   - sensor_width = 2 × 5.9 × tan(59.8°/2) = 6.78mm
        f_px = f_mm * (W_px / sensor_width_mm)

        # 3. Construct K
        cx, cy = W_px / 2.0, H_px / 2.0
        K = np.array([
            [f_px, 0, cx],
            [0, f_px, cy],
            [0, 0, 1]
        ])
        return K

    def set_distortion_coefficients(
        self,
        k1: Unitless = Unitless(0.0),
        k2: Unitless = Unitless(0.0),
        p1: Unitless = Unitless(0.0),
        p2: Unitless = Unitless(0.0),
        k3: Unitless = Unitless(0.0)
    ) -> None:
        """
        Set lens distortion coefficients using the OpenCV distortion model.

        The distortion model corrects for radial and tangential lens distortion:
        - Radial distortion (barrel/pincushion): controlled by k1, k2, k3
        - Tangential distortion (decentering): controlled by p1, p2

        For most PTZ cameras, only k1 (and sometimes k2) are significant.
        Positive k1 = barrel distortion (edges curve outward)
        Negative k1 = pincushion distortion (edges curve inward)

        Args:
            k1: First radial distortion coefficient (most significant)
            k2: Second radial distortion coefficient
            p1: First tangential distortion coefficient
            p2: Second tangential distortion coefficient
            k3: Third radial distortion coefficient (usually 0)

        Example:
            >>> geo = CameraGeometry(1920, 1080)
            >>> # Typical barrel distortion for wide-angle PTZ
            >>> geo.set_distortion_coefficients(k1=-0.1, k2=0.01)
        """
        self._dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        self._use_distortion = not np.allclose(self._dist_coeffs, 0.0)

        if self._use_distortion:
            logger.info(f"Distortion coefficients set: k1={k1:.6f}, k2={k2:.6f}, "
                       f"p1={p1:.6f}, p2={p2:.6f}, k3={k3:.6f}")
        else:
            logger.info("Distortion coefficients cleared (all zero)")

    def get_distortion_coefficients(self) -> Optional[np.ndarray]:
        """
        Get current distortion coefficients.

        Returns:
            numpy array [k1, k2, p1, p2, k3] or None if not set
        """
        return self._dist_coeffs.copy() if self._dist_coeffs is not None else None

    def undistort_point(
        self,
        u: float,
        v: float,
        iterations: int = 10
    ) -> Tuple[float, float]:
        """
        Undistort a single image point to remove lens distortion.

        Converts a distorted pixel coordinate to the corresponding undistorted
        coordinate that would be observed with an ideal pinhole camera.

        This uses an iterative method to invert the distortion model.

        Args:
            u: Distorted x pixel coordinate
            v: Distorted y pixel coordinate
            iterations: Number of iterations for undistortion (default: 10)

        Returns:
            Tuple (u_undistorted, v_undistorted) in pixel coordinates

        Note:
            If no distortion coefficients are set, returns the input unchanged.
        """
        if not self._use_distortion or self.K is None:
            return (u, v)

        # Get intrinsic parameters
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        k1, k2, p1, p2, k3 = self._dist_coeffs

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

    def undistort_points(
        self,
        points: List[Tuple[float, float]],
        iterations: int = 10
    ) -> List[Tuple[float, float]]:
        """
        Undistort multiple image points.

        Args:
            points: List of (u, v) distorted pixel coordinates
            iterations: Number of iterations for undistortion

        Returns:
            List of (u, v) undistorted pixel coordinates
        """
        return [self.undistort_point(u, v, iterations) for u, v in points]

    def distort_point(self, u: float, v: float) -> Tuple[float, float]:
        """
        Apply lens distortion to an undistorted point.

        Converts an ideal pinhole camera coordinate to the corresponding
        distorted coordinate as observed through the actual lens.

        Args:
            u: Undistorted x pixel coordinate
            v: Undistorted y pixel coordinate

        Returns:
            Tuple (u_distorted, v_distorted) in pixel coordinates

        Note:
            If no distortion coefficients are set, returns the input unchanged.
        """
        if not self._use_distortion or self.K is None:
            return (u, v)

        # Get intrinsic parameters
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        k1, k2, p1, p2, k3 = self._dist_coeffs

        # Convert to normalized camera coordinates
        x = (u - cx) / fx
        y = (v - cy) / fy

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        # Radial distortion
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

        # Tangential distortion
        dx_tangential = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy_tangential = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

        # Apply distortion
        x_d = x * radial + dx_tangential
        y_d = y * radial + dy_tangential

        # Convert back to pixel coordinates
        u_dist = x_d * fx + cx
        v_dist = y_d * fy + cy

        return (u_dist, v_dist)

    def _validate_parameters(
        self,
        K: np.ndarray,
        w_pos: np.ndarray,
        pan_deg: Degrees,
        tilt_deg: Degrees
    ) -> None:
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
            logger.debug(
                f"Tilt angle {tilt_deg:.1f}° is near horizontal (<{self.TILT_WARN_LOW}°). "
                f"Ground projection may be unstable or extend to very large distances."
            )
        elif tilt_deg > self.TILT_WARN_HIGH:
            logger.debug(
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

    def set_camera_parameters(
        self,
        K: np.ndarray,
        w_pos: np.ndarray,
        pan_deg: Degrees,
        tilt_deg: Degrees,
        map_width: Pixels,
        map_height: Pixels
    ):
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
        logger.info(f"  Camera position: [{w_pos[0]:.2f}, {w_pos[1]:.2f}, {w_pos[2]:.2f}] meters")
        logger.info(f"  Pan: {pan_deg:.1f}°, Tilt: {tilt_deg:.1f}°")
        logger.info(f"  Homography det(H): {det_H:.2e}")
        logger.info(f"  Homography condition number: {cond_H:.2e}")

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

        The transformation consists of:
        1. Base transform: Convert world coords to camera coords when pan=0, tilt=0
           (camera looking North, horizontal)
        2. Pan rotation: Rotate around world Z-axis (yaw)
        3. Tilt rotation: Rotate around camera X-axis (pitch)

        Tilt Convention (Hikvision):
        - Positive tilt_deg = camera pointing downward
        - Negative tilt_deg = camera pointing upward
        """
        pan_rad = math.radians(self.pan_deg)
        tilt_rad = math.radians(self.tilt_deg)

        # Base transformation from World to Camera when pan=0, tilt=0
        # (camera looking North, horizontal):
        # - World X (East)  -> Camera X (Right)
        # - World Y (North) -> Camera Z (Forward)
        # - World Z (Up)    -> Camera -Y (camera Y points Down)
        #
        # This matrix transforms [Xw, Yw, Zw] to [Xc, Yc, Zc]:
        #   Xc = Xw
        #   Yc = -Zw
        #   Zc = Yw
        R_base = np.array([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0]
        ])

        # Pan rotation around world Z-axis (yaw)
        # Positive pan = clockwise from above = camera looks right (towards East when starting North)
        # This is applied in world coordinates before the base transform
        Rz_pan = np.array([
            [math.cos(pan_rad), -math.sin(pan_rad), 0],
            [math.sin(pan_rad),  math.cos(pan_rad), 0],
            [0,                  0,                 1]
        ])

        # Tilt rotation around camera X-axis (pitch)
        # Positive tilt = camera looks down
        # This is applied in camera coordinates after pan and base transform
        Rx_tilt = np.array([
            [1,  0,                   0],
            [0,  math.cos(tilt_rad), -math.sin(tilt_rad)],
            [0,  math.sin(tilt_rad),  math.cos(tilt_rad)]
        ])

        # Full rotation: first pan in world, then base transform, then tilt in camera
        # R_world_to_cam = R_tilt @ R_base @ R_pan
        R = Rx_tilt @ R_base @ Rz_pan
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

        If distortion coefficients are set, points are undistorted before projection.
        """
        if self.H_inv is None or np.all(self.H_inv == np.eye(3)):
            return [(int(x / 2), int(y / 2)) for x, y in pts]

        # Undistort points if distortion is enabled
        if self._use_distortion:
            pts = self.undistort_points(pts)

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

    def world_to_map(
        self,
        Xw: Meters,
        Yw: Meters,
        sw: Optional[Pixels] = None,
        sh: Optional[Pixels] = None
    ) -> Tuple[Pixels, Pixels]:
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

        x_px = Pixels(int((Xw * self.PPM) + map_center_x))
        y_px = Pixels(int(map_bottom_y - (Yw * self.PPM)))

        return x_px, y_px

    def set_height_uncertainty(self, confidence_interval: Tuple[Meters, Meters]) -> None:
        """
        Set height uncertainty for propagation to future projections.

        This stores the confidence interval bounds that will be used by
        project_to_world_with_uncertainty() to compute uncertainty in
        projected world coordinates.

        Args:
            confidence_interval: Tuple of (lower_bound, upper_bound) for camera
                                height in meters. Typically a 95% confidence interval
                                from height calibration.

        Raises:
            ValueError: If confidence_interval is not a 2-tuple or bounds are invalid

        Example:
            >>> # After height calibration
            >>> result = calibrator.optimize_height_least_squares()
            >>> geo.set_height_uncertainty(result.confidence_interval)
        """
        if not isinstance(confidence_interval, tuple) or len(confidence_interval) != 2:
            raise ValueError(
                f"confidence_interval must be a 2-tuple, got {type(confidence_interval).__name__}"
            )

        lower, upper = confidence_interval

        if lower <= 0 or upper <= 0:
            raise ValueError(
                f"Height bounds must be positive, got lower={lower}, upper={upper}"
            )

        if lower > upper:
            raise ValueError(
                f"Lower bound ({lower}) cannot exceed upper bound ({upper})"
            )

        self.height_uncertainty_lower = lower
        self.height_uncertainty_upper = upper

    def project_to_world_with_uncertainty(
        self,
        pixel_x: float,
        pixel_y: float
    ) -> Dict[str, Any]:
        """
        Project pixel to world coordinates with uncertainty propagation.

        This method projects an image pixel to world coordinates and computes
        uncertainty bounds based on the height uncertainty set via
        set_height_uncertainty(). The uncertainty is computed by projecting
        at the lower and upper height bounds and measuring the variation in
        the resulting world coordinates.

        Physical principle:
            When camera height changes, world coordinates scale proportionally:
            - new_world_x = original_world_x * (new_height / original_height)
            - new_world_y = original_world_y * (new_height / original_height)

        Args:
            pixel_x: Horizontal pixel coordinate in image
            pixel_y: Vertical pixel coordinate in image

        Returns:
            Dictionary with the following keys:
                - world_x (float): Mean X coordinate in meters (East-West)
                - world_y (float): Mean Y coordinate in meters (North-South)
                - distance (float): Mean distance from camera in meters
                - x_uncertainty (float): Half-width of X uncertainty (± value in meters)
                - y_uncertainty (float): Half-width of Y uncertainty (± value in meters)
                - distance_uncertainty (float): Half-width of distance uncertainty (± value in meters)
                - confidence_level (float): The confidence level (0.95)

        Raises:
            ValueError: If pixel is near horizon and cannot be projected

        Note:
            - If set_height_uncertainty() has not been called, all uncertainty
              values will be 0.0
            - The mean coordinates are computed from the current height
            - Uncertainty represents the range due to height uncertainty only
            - Does not account for other error sources (camera calibration,
              lens distortion, etc.)

        Example:
            >>> geo.set_height_uncertainty((4.8, 5.2))
            >>> result = geo.project_to_world_with_uncertainty(1280, 720)
            >>> print(f"Position: ({result['world_x']:.2f} ± {result['x_uncertainty']:.2f}, "
            ...       f"{result['world_y']:.2f} ± {result['y_uncertainty']:.2f}) meters")
        """
        # Project at current height to get mean coordinates
        pt_img = np.array([[pixel_x], [pixel_y], [1.0]])
        pt_world = self.H_inv @ pt_img

        # Check for valid projection (not near horizon)
        if abs(pt_world[2, 0]) < 1e-6:
            raise ValueError(
                f"Invalid point at pixel ({pixel_x}, {pixel_y}): "
                "Point is too close to horizon and cannot be projected"
            )

        # Normalize to get world coordinates at current height
        world_x_mean = pt_world[0, 0] / pt_world[2, 0]
        world_y_mean = pt_world[1, 0] / pt_world[2, 0]
        distance_mean = np.sqrt(world_x_mean**2 + world_y_mean**2)

        # Initialize uncertainty to zero
        x_uncertainty = 0.0
        y_uncertainty = 0.0
        distance_uncertainty = 0.0

        # If height uncertainty has been set, compute uncertainty bounds
        if (self.height_uncertainty_lower is not None and
            self.height_uncertainty_upper is not None):

            # World coordinates scale proportionally with height
            # new_coord = original_coord * (new_height / original_height)

            # Project at lower height bound
            height_ratio_lower = self.height_uncertainty_lower / self.height_m
            world_x_lower = world_x_mean * height_ratio_lower
            world_y_lower = world_y_mean * height_ratio_lower
            distance_lower = distance_mean * height_ratio_lower

            # Project at upper height bound
            height_ratio_upper = self.height_uncertainty_upper / self.height_m
            world_x_upper = world_x_mean * height_ratio_upper
            world_y_upper = world_y_mean * height_ratio_upper
            distance_upper = distance_mean * height_ratio_upper

            # Compute half-width of uncertainty range (± value)
            x_uncertainty = abs(world_x_upper - world_x_lower) / 2.0
            y_uncertainty = abs(world_y_upper - world_y_lower) / 2.0
            distance_uncertainty = abs(distance_upper - distance_lower) / 2.0

        return {
            'world_x': world_x_mean,
            'world_y': world_y_mean,
            'distance': distance_mean,
            'x_uncertainty': x_uncertainty,
            'y_uncertainty': y_uncertainty,
            'distance_uncertainty': distance_uncertainty,
            'confidence_level': 0.95
        }
