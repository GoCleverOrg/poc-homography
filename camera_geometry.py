# camera_geometry.py

import numpy as np
import math
from typing import List, Tuple, Union

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
    - tilt_deg: Vertical rotation (degrees, negative = down)
    """

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
        """
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

    def set_camera_parameters(self, K: np.ndarray, w_pos: np.ndarray,
                              pan_deg: float, tilt_deg: float,
                              map_width: int, map_height: int):
        """
        Sets all required parameters and calculates the Homography matrix H.

        Args:
            K: 3x3 intrinsic matrix
            w_pos: Camera position in world coordinates [X, Y, Z] (meters)
            pan_deg: Pan angle in degrees (positive = right)
            tilt_deg: Tilt angle in degrees (negative = down)
            map_width: Width of output map in pixels
            map_height: Height of output map in pixels
        """
        # Validation
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got shape {K.shape}")
        if len(w_pos) != 3:
            raise ValueError(f"w_pos must have 3 elements [X, Y, Z], got {len(w_pos)}")
        if w_pos[2] <= 0:
            print(f"Warning: Camera height (Z={w_pos[2]}) should be positive for ground plane homography.")

        self.K = K
        self.w_pos = w_pos
        self.height_m = w_pos[2]  # Z component is height
        self.pan_deg = pan_deg
        self.tilt_deg = tilt_deg
        self.map_width = map_width
        self.map_height = map_height

        self.H = self._calculate_ground_homography()

        # Validate and invert homography
        det_H = np.linalg.det(self.H)
        if abs(det_H) < 1e-10:
            print(f"Warning: Homography is singular (det={det_H:.2e}). Inverse may be unstable.")
            self.H_inv = np.eye(3)
        else:
            self.H_inv = np.linalg.inv(self.H)

        print("Geometry setup complete. Homography matrix H calculated.")
        print(f"  Camera position: [{w_pos[0]:.2f}, {w_pos[1]:.2f}, {w_pos[2]:.2f}] meters")
        print(f"  Pan: {pan_deg:.1f}°, Tilt: {tilt_deg:.1f}°")
        print(f"  Homography det(H): {det_H:.2e}")

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
        """
        pan_rad = math.radians(self.pan_deg)
        tilt_rad = math.radians(self.tilt_deg)

        # 1. Yaw (Rotation around World Z-axis - Pan)
        # Positive pan rotates camera to the right (clockwise from above)
        Rz = np.array([
            [math.cos(pan_rad), -math.sin(pan_rad), 0],
            [math.sin(pan_rad), math.cos(pan_rad), 0],
            [0, 0, 1]
        ])

        # 2. Pitch (Rotation around X-axis - Tilt)
        # Negative tilt points camera downward
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