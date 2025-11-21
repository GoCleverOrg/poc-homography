# camera_geometry.py

import numpy as np
import math
from typing import List, Tuple, Union

class CameraGeometry:
    """
    Handles all spatial and projection calculations for a fixed camera setup.
    Calculates the homography matrix H to map image points to the world ground plane.
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
        self.H_inv = np.eye(3) # Inverse Homography (Map -> Image)
        self.map_width = 640 # Default scale for the map view (pixels)
        self.map_height = 640 # Default scale for the map view (pixels)

        # Placeholder for real-world geometry parameters (to be set by user)
        self.K = None          # Intrinsic Matrix
        self.w_pos = None      # Camera World Position [Xw, Yw, Zw]
        self.height_m = 5.0    # Camera Height (m)
        self.pan_deg = 0.0     # Camera Pan (Yaw)
        self.tilt_deg = 0.0    # Camera Tilt (Pitch)

    @staticmethod
    def get_intrinsics(zoom_factor: float, W_px: int=2560, H_px: int=1440, sensor_width_mm: float=7.18) -> np.ndarray:
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
            [f_px, 0,    cx],
            [0,    f_px, cy],
            [0,    0,    1 ]
        ])
        return K

    def set_camera_parameters(self, K: np.ndarray, w_pos: np.ndarray, 
                              pan_deg: float, tilt_deg: float, 
                              map_width: int, map_height: int):
        """
        Sets all required parameters and calculates the Homography matrix H.
        """
        self.K = K
        self.w_pos = w_pos
        self.height_m = w_pos[2] # Assumes height is the Z component
        self.pan_deg = pan_deg
        self.tilt_deg = tilt_deg
        self.map_width = map_width
        self.map_height = map_height

        self.H = self._calculate_ground_homography()
        self.H_inv = np.linalg.inv(self.H)
        
        print("Geometry setup complete. Homography matrix H calculated.")

    # --- Core Geometry Methods (Unchanged from previous output) ---

    def _get_rotation_matrix(self) -> np.ndarray:
        """
        Calculates the 3x3 rotation matrix R from world to camera coordinates
        based on pan (Yaw) and tilt (Pitch). Assumes zero roll.
        Rotation order: Yaw (Z-axis, Pan) then Pitch (X-axis, Tilt).
        """
        pan_rad = math.radians(self.pan_deg)
        tilt_rad = math.radians(self.tilt_deg)
        
        # 1. Yaw (Rotation around World Z-axis - Pan)
        Rz = np.array([
            [math.cos(pan_rad), -math.sin(pan_rad), 0],
            [math.sin(pan_rad), math.cos(pan_rad), 0],
            [0, 0, 1]
        ])
        
        # 2. Pitch (Rotation around World X-axis - Tilt)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(tilt_rad), -math.sin(tilt_rad)],
            [0, math.sin(tilt_rad), math.cos(tilt_rad)]
        ])
        
        R = Rx @ Rz
        return R

    def _calculate_ground_homography(self) -> np.ndarray:
        """
        Calculates the Homography matrix H = K * [r1, r2, t].
        """
        
        if self.K is None:
            print("Warning: K not set. Returning Identity Homography.")
            return np.eye(3)
        
        R = self._get_rotation_matrix()
        
        # t is the position of the World origin in the Camera frame: t = -R @ w_pos
        t = -R @ self.w_pos
        
        # H = K * [r1, r2, t]
        r1 = R[:, 0]
        r2 = R[:, 1]
        
        H_extrinsic = np.hstack([r1[:, np.newaxis], r2[:, np.newaxis], t[:, np.newaxis]])
        
        H = self.K @ H_extrinsic
        
        H /= H[2, 2]
        
        return H

    def project_image_to_map(self, pts: List[Tuple[int, int]], sw: int, sh: int) -> List[Tuple[int, int]]:
        """
        Projects image coordinates (pixels) to the world ground plane (meters).
        """
        if self.H_inv is None or np.all(self.H_inv == np.eye(3)):
             return [(int(x/2), int(y/2)) for x, y in pts] 
             
        pts_homogeneous = np.array(pts, dtype=np.float64).T
        pts_homogeneous = np.vstack([pts_homogeneous, np.ones(pts_homogeneous.shape[1])])
        
        # Project from Image to World Ground Plane (Xw, Yw, 1)
        pts_world_homogeneous = self.H_inv @ pts_homogeneous
        
        # Normalize world coordinates (convert [Xw, Yw, W] to [Xw/W, Yw/W, 1])
        Xw = pts_world_homogeneous[0, :] / pts_world_homogeneous[2, :]
        Yw = pts_world_homogeneous[1, :] / pts_world_homogeneous[2, :]
        
        # --- Mapping Policy (Assumes 100 pixels per meter) ---
        PPM = 100 
        
        map_center_x = sw // 2
        map_bottom_y = sh
        
        pts_map_x = (Xw * PPM) + map_center_x
        pts_map_y = map_bottom_y - (Yw * PPM) 

        pts_map = [(int(x), int(y)) for x, y in zip(pts_map_x, pts_map_y)]
        
        return pts_map