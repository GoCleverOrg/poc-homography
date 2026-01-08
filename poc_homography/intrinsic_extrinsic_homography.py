"""
Intrinsic/Extrinsic Homography Provider Implementation.

This module implements the HomographyProviderExtended interface using camera
intrinsic parameters (focal length, principal point) and extrinsic parameters
(rotation, translation) to compute homography transformations.

The homography maps image coordinates to ground plane coordinates using the
pinhole camera model with known camera calibration and pose.

Coordinate Systems:
    - World Frame: X=East, Y=North, Z=Up (right-handed)
    - Camera Frame: X=Right, Y=Down, Z=Forward (standard CV, right-handed)
    - Image Frame: origin top-left, u=right, v=down (pixels)
"""

import numpy as np
import math
import logging
from typing import List, Tuple, Dict, Any, Optional

from poc_homography.homography_interface import (
    HomographyProviderExtended,
    HomographyResult,
    WorldPoint,
    MapCoordinate,
    HomographyApproach,
    GPSPositionMixin
)
from poc_homography.coordinate_converter import local_xy_to_gps
from poc_homography.types import Degrees, Meters, Pixels, Millimeters, Unitless

logger = logging.getLogger(__name__)


class IntrinsicExtrinsicHomography(GPSPositionMixin, HomographyProviderExtended):
    """
    Homography provider using camera intrinsic/extrinsic parameters.

    This implementation computes homography for a ground plane (Z=0) using:
    - Camera intrinsic matrix K (focal length, principal point)
    - Camera extrinsic parameters (position, rotation via pan/tilt/roll)

    The homography H maps world ground plane points to image pixels:
        [u]       [X_world]
        [v]  ∝ H  [Y_world]
        [1]       [1      ]

    For inverse projection (image to world):
        [X_world]           [u]
        [Y_world]  ∝ H^-1  [v]
        [1      ]           [1]

    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        H: Current homography matrix (3x3) mapping world to image
        H_inv: Inverse homography matrix mapping image to world
        confidence: Current homography confidence score [0.0, 1.0]
        map_width: Width of map visualization in pixels
        map_height: Height of map visualization in pixels
        pixels_per_meter: Scale factor for map visualization (default: 100)
        _camera_gps_lat: Camera GPS latitude for WorldPoint conversion
        _camera_gps_lon: Camera GPS longitude for WorldPoint conversion

    Note:
        GPS coordinates (_camera_gps_lat, _camera_gps_lon) default to None.
        Call set_camera_gps_position(lat, lon) before using project_point()
        to get WorldPoint results with GPS coordinates.
    """

    # Minimum determinant threshold for valid homography
    MIN_DET_THRESHOLD = 1e-10

    # Minimum confidence threshold for validity
    MIN_CONFIDENCE_THRESHOLD = 0.3

    # Confidence thresholds based on homography matrix determinant
    # These values are empirically determined from typical camera configurations
    DET_THRESHOLD_INVALID = 1e-6    # Below this: homography is degenerate
    DET_THRESHOLD_LOW = 1e-3        # Below this: low confidence
    DET_THRESHOLD_HIGH = 1e3        # Above this: possible numerical issues

    # Condition number thresholds for numerical stability assessment
    # High condition numbers indicate the matrix is sensitive to small input changes
    COND_THRESHOLD_DEGENERATE = 1e10  # Matrix is numerically degenerate
    COND_THRESHOLD_UNSTABLE = 1e6     # Matrix is numerically unstable
    COND_THRESHOLD_MARGINAL = 1e3     # Matrix is marginally stable

    # Confidence penalty multipliers for various conditions
    CONFIDENCE_PENALTY_UNSTABLE = 0.5    # Applied when condition number > COND_THRESHOLD_UNSTABLE
    CONFIDENCE_PENALTY_MARGINAL = 0.9    # Applied when condition number > COND_THRESHOLD_MARGINAL
    CONFIDENCE_PENALTY_BAD_HEIGHT = 0.5  # Applied when camera height <= 0
    CONFIDENCE_PENALTY_BAD_TILT = 0.8    # Applied when tilt outside [-90, 90]
    CONFIDENCE_LARGE_DET = 0.7           # Confidence for very large determinant

    # Gimbal lock threshold - angles within this range of ±90° cause numerical instability
    GIMBAL_LOCK_THRESHOLD_DEG = 0.1      # Degrees from ±90° considered gimbal lock zone
    CONFIDENCE_PENALTY_GIMBAL_LOCK = 0.3 # Severe penalty for near-gimbal-lock configuration

    # Edge factor constants for point confidence calculation
    EDGE_FACTOR_CENTER = 1.0             # Confidence factor at image center
    EDGE_FACTOR_EDGE = 0.7               # Confidence factor at image edges
    EDGE_FACTOR_CORNER_DECAY = 0.2       # Decay rate beyond edge radius
    EDGE_FACTOR_MIN = 0.3                # Minimum edge factor

    # Roll validation thresholds (consistent with CameraGeometry)
    ROLL_WARN_THRESHOLD = 5.0   # Warning when |roll_deg| > 5.0
    ROLL_ERROR_THRESHOLD = 15.0  # Error when |roll_deg| > 15.0

    def __init__(
        self,
        width: Pixels,
        height: Pixels,
        pixels_per_meter: Unitless = Unitless(100.0),
        sensor_width_mm: Millimeters = Millimeters(7.18),
        base_focal_length_mm: Millimeters = Millimeters(5.9),
        **kwargs  # Accept and ignore other kwargs for forward compatibility
    ):
        """
        Initialize intrinsic/extrinsic homography provider.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            pixels_per_meter: Scale factor for map visualization (default: 100)
            sensor_width_mm: Physical sensor width in millimeters (default: 7.18)
            base_focal_length_mm: Base focal length at 1x zoom in mm (default: 5.9).
                This value is used by get_intrinsics() to calculate the camera
                intrinsic matrix. It represents the physical focal length when
                zoom_factor=1.0, and scales linearly with zoom.
            **kwargs: Additional parameters (ignored, for forward compatibility)
        """
        # Validate image dimensions
        if width <= 0 or height <= 0:
            raise ValueError(f"Image dimensions must be positive, got width={width}, height={height}")

        self.width = width
        self.height = height
        self.pixels_per_meter = pixels_per_meter
        self.sensor_width_mm = sensor_width_mm
        self.base_focal_length_mm = base_focal_length_mm

        # Homography state
        self.H = np.eye(3)
        self.H_inv = np.eye(3)
        self.confidence = 0.0

        # Map dimensions (set during compute_homography)
        self.map_width = 640
        self.map_height = 640

        # GPS reference point for WorldPoint conversion
        self._camera_gps_lat: Optional[float] = None
        self._camera_gps_lon: Optional[float] = None

        # Current roll angle (degrees, default 0.0 for backward compatibility)
        self.roll_deg: float = 0.0

        # Last used camera parameters (for metadata)
        self._last_camera_matrix: Optional[np.ndarray] = None
        self._last_camera_position: Optional[np.ndarray] = None
        self._last_pan_deg: Optional[float] = None
        self._last_tilt_deg: Optional[float] = None
        self._last_roll_deg: Optional[float] = None

    def get_intrinsics(
        self,
        zoom_factor: Unitless,
        width_px: Optional[Pixels] = None,
        height_px: Optional[Pixels] = None,
        sensor_width_mm: Optional[Millimeters] = None
    ) -> np.ndarray:
        """
        Calculate camera intrinsic matrix K from zoom factor and sensor specs.

        This is a convenience method for computing the camera matrix based on
        physical camera specifications. The focal length is computed as a linear
        function of zoom factor based on the base focal length configured for
        this instance.

        Args:
            zoom_factor: Digital or optical zoom multiplier (1.0 = no zoom)
            width_px: Image width in pixels (default: uses instance's width)
            height_px: Image height in pixels (default: uses instance's height)
            sensor_width_mm: Physical sensor width in millimeters
                (default: uses instance's sensor_width_mm)

        Returns:
            K: 3x3 camera intrinsic matrix with:
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
                where fx=fy is the focal length in pixels, and (cx, cy) is
                the principal point (typically image center).

        Example:
            >>> homography = IntrinsicExtrinsicHomography(1920, 1080)
            >>> K = homography.get_intrinsics(zoom_factor=5.0)
            >>> print(K)
            [[2106.27...    0.     1280.  ]
             [   0.     2106.27...  720.  ]
             [   0.        0.        1.  ]]
        """
        # Use instance dimensions if not provided
        if width_px is None:
            width_px = self.width
        if height_px is None:
            height_px = self.height

        # Validate inputs
        if zoom_factor <= 0:
            raise ValueError(f"zoom_factor must be positive, got {zoom_factor}")
        if width_px <= 0 or height_px <= 0:
            raise ValueError("Image dimensions must be positive")

        # Use instance sensor_width_mm if not provided
        if sensor_width_mm is None:
            sensor_width_mm = self.sensor_width_mm

        if sensor_width_mm <= 0:
            raise ValueError("sensor_width_mm must be positive")

        # Linear mapping based on instance's base focal length
        # Uses base_focal_length_mm at 1x zoom, scales linearly with zoom_factor
        f_mm = self.base_focal_length_mm * zoom_factor

        # Convert focal length from millimeters to pixels
        f_px = f_mm * (width_px / sensor_width_mm)

        # Principal point at image center
        cx, cy = width_px / 2.0, height_px / 2.0

        # Construct intrinsic matrix
        K = np.array([
            [f_px, 0.0, cx],
            [0.0, f_px, cy],
            [0.0, 0.0, 1.0]
        ])
        return K

    def _get_rotation_matrix(
        self,
        pan_deg: Degrees,
        tilt_deg: Degrees,
        roll_deg: Degrees = Degrees(0.0)
    ) -> np.ndarray:
        """
        Calculate rotation matrix from world to camera coordinates.

        Computes the 3x3 rotation matrix based on pan (yaw), tilt (pitch), and roll.
        The transformation consists of:
        1. Pan rotation around world Z-axis (yaw)
        2. Base transform from world to camera coordinates
        3. Roll rotation around camera Z-axis (optical axis)
        4. Tilt rotation around camera X-axis (pitch)

        Rotation order: R = R_tilt @ R_roll @ R_base @ R_pan

        Coordinate System Convention:
            - World: X=East, Y=North, Z=Up
            - Camera: X=Right, Y=Down, Z=Forward (optical axis)

        At pan=0, tilt=0, roll=0, the camera looks North (world +Y direction).

        Tilt Convention (Hikvision):
            - Positive tilt_deg = camera pointing downward
            - Negative tilt_deg = camera pointing upward

        Roll Convention:
            - Positive roll_deg = clockwise rotation when looking from behind camera
              (along +Z axis, into the scene)
            - Roll is applied in camera frame after base transformation but before tilt

        Args:
            pan_deg: Pan angle in degrees (positive = right/clockwise from above)
            tilt_deg: Tilt angle in degrees (positive = down, Hikvision convention)
            roll_deg: Roll angle in degrees (positive = clockwise, default = 0.0)

        Returns:
            R: 3x3 rotation matrix transforming world coordinates to camera frame
        """
        pan_rad = math.radians(pan_deg)
        tilt_rad = math.radians(tilt_deg)
        roll_rad = math.radians(roll_deg)

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
            [1.0,  0.0,  0.0],
            [0.0,  0.0, -1.0],
            [0.0,  1.0,  0.0]
        ])

        # Pan rotation around world Z-axis (yaw)
        # Positive pan = clockwise from above = camera looks right
        Rz_pan = np.array([
            [math.cos(pan_rad), -math.sin(pan_rad), 0.0],
            [math.sin(pan_rad),  math.cos(pan_rad), 0.0],
            [0.0,                0.0,               1.0]
        ])

        # Roll rotation around camera Z-axis (optical axis)
        # Positive roll = clockwise when looking from behind camera (along +Z axis)
        # This rotates the image plane around the optical axis
        Rz_roll = np.array([
            [math.cos(roll_rad), -math.sin(roll_rad), 0.0],
            [math.sin(roll_rad),  math.cos(roll_rad), 0.0],
            [0.0,                 0.0,                1.0]
        ])

        # Tilt rotation around camera X-axis (pitch)
        # Positive tilt = camera looks down
        Rx_tilt = np.array([
            [1.0,  0.0,                0.0],
            [0.0,  math.cos(tilt_rad), -math.sin(tilt_rad)],
            [0.0,  math.sin(tilt_rad),  math.cos(tilt_rad)]
        ])

        # Full rotation: first pan in world, then base transform, then roll in camera, then tilt in camera
        # R_world_to_cam = R_tilt @ R_roll @ R_base @ R_pan
        R = Rx_tilt @ Rz_roll @ R_base @ Rz_pan
        return R

    def _calculate_ground_homography(
        self,
        K: np.ndarray,
        camera_position: np.ndarray,
        pan_deg: Degrees,
        tilt_deg: Degrees,
        roll_deg: Degrees = Degrees(0.0)
    ) -> np.ndarray:
        """
        Calculate homography matrix mapping world ground plane (Z=0) to image.

        Coordinate Frame Conventions:
            World Frame:
                - Origin: Arbitrary reference point (camera position or scene center)
                - Axes: X=East (meters), Y=North (meters), Z=Up (meters)
                - Ground plane: Z=0
                - Right-handed coordinate system

            Camera Frame:
                - Origin: Camera optical center
                - Axes: X=Right (meters), Y=Down (meters), Z=Forward (meters, optical axis)
                - Transformation: P_camera = R @ P_world + t, where t = -R @ C
                - Right-handed coordinate system (standard computer vision)

            Image Frame:
                - Origin: Top-left corner of image
                - Axes: u=horizontal (pixels, right), v=vertical (pixels, down)
                - Projection: p_image = K @ P_camera

        Mathematical Derivation (Planar Homography for Z=0 Ground Plane):
            Step 1 - General 3D projection:
                P_camera = R @ P_world + t
                p_image ~ K @ P_camera = K @ (R @ P_world + t)

            Step 2 - For ground plane Z=0, P_world = [X, Y, 0]^T:
                Rotation matrix R = [r1, r2, r3] (columns)
                R @ [X, Y, 0]^T = r1*X + r2*Y + r3*0 = r1*X + r2*Y

            Step 3 - Substitute into projection:
                p_image ~ K @ (r1*X + r2*Y + t)
                        = K @ [r1, r2, t] @ [X, Y, 1]^T

            Step 4 - Define homography:
                H = K @ [r1, r2, t]
                This is a 3x3 matrix (NOT 3x4 projection matrix)
                Maps 2D ground plane [X, Y, 1] to 2D image [u, v, 1]

            Normalization: H is normalized so H[2,2] = 1 for numerical stability.

        Args:
            K: 3x3 camera intrinsic matrix
            camera_position: Camera position [X, Y, Z] in world coordinates (meters)
            pan_deg: Pan angle in degrees
            tilt_deg: Tilt angle in degrees
            roll_deg: Roll angle in degrees (default = 0.0)

        Returns:
            H (np.ndarray): 3x3 homography matrix mapping [X_world, Y_world, 1] -> [u, v, 1]
        """
        # Get rotation matrix (includes roll if specified)
        R = self._get_rotation_matrix(pan_deg, tilt_deg, roll_deg)

        # Camera position C in world coordinates
        C = camera_position

        # Translation from camera to world origin: t = -R @ C
        # This gives the position of the world origin in camera frame
        t = -R @ C

        # Build homography: H = K @ [r1, r2, t]
        # r1, r2 are the first two COLUMNS of R (corresponding to world X and Y axes)
        # This is correct because for Z=0: R @ [X, Y, 0]^T = r1*X + r2*Y
        r1 = R[:, 0]  # Column 0: world X-axis in camera frame
        r2 = R[:, 1]  # Column 1: world Y-axis in camera frame

        # Construct 3x3 extrinsic homography matrix (NOT 3x4 projection matrix)
        H_extrinsic = np.column_stack([r1, r2, t])
        if H_extrinsic.shape != (3, 3):
            raise ValueError(f"H_extrinsic must be 3x3, got {H_extrinsic.shape}")

        H = K @ H_extrinsic
        if H.shape != (3, 3):
            raise ValueError(f"Homography must be 3x3, got {H.shape}")

        # Normalize so H[2, 2] = 1 for consistent scale
        if abs(H[2, 2]) < self.MIN_DET_THRESHOLD:
            logger.warning(
                f"Homography normalization failed (H[2,2]={H[2,2]:.2e}). Returning identity."
            )
            return np.eye(3)

        H = H / H[2, 2]

        return H

    def _calculate_confidence(
        self,
        H: np.ndarray,
        camera_position: Optional[np.ndarray] = None,
        tilt_deg: Optional[float] = None
    ) -> float:
        """
        Calculate confidence score for the homography matrix.

        Confidence is computed using class constants for thresholds and penalties:

        1. Determinant-based checks (DET_THRESHOLD_* constants):
           - Degenerate, low quality, good, or poorly scaled based on |det|

        2. Condition number checks (COND_THRESHOLD_* constants):
           - Penalties applied via CONFIDENCE_PENALTY_* multipliers

        3. Camera parameter validity:
           - Height and tilt range violations apply penalties

        See class constants for specific threshold and penalty values.

        Args:
            H: 3x3 homography matrix
            camera_position: Camera position [X, Y, Z] in meters (optional)
            tilt_deg: Tilt angle in degrees (optional)

        Returns:
            float: Confidence score in range [0.0, 1.0]
        """
        det_H = np.linalg.det(H)

        # Check if homography is singular
        if abs(det_H) < self.MIN_DET_THRESHOLD:
            return 0.0

        # Base confidence on determinant magnitude
        # Normalized determinant should be around 1.0 for well-conditioned matrices
        # We use threshold-based scoring to map determinant to [0, 1]
        det_abs = abs(det_H)

        # Threshold-based heuristic: good homographies have |det| in reasonable range
        # Too small -> singular, too large -> poorly scaled
        if det_abs < self.DET_THRESHOLD_INVALID:
            confidence = 0.0
        elif det_abs < self.DET_THRESHOLD_LOW:
            confidence = 0.5
        elif det_abs < self.DET_THRESHOLD_HIGH:
            confidence = 1.0
        else:
            confidence = self.CONFIDENCE_LARGE_DET

        # Apply condition number checks (measures numerical stability)
        cond_H = np.linalg.cond(H)
        if cond_H > self.COND_THRESHOLD_DEGENERATE:
            confidence = 0.0
        elif cond_H > self.COND_THRESHOLD_UNSTABLE:
            confidence *= self.CONFIDENCE_PENALTY_UNSTABLE
        elif cond_H > self.COND_THRESHOLD_MARGINAL:
            confidence *= self.CONFIDENCE_PENALTY_MARGINAL

        # Factor in camera parameter validity
        # Check camera height (should be positive)
        if camera_position is not None:
            camera_height = camera_position[2]
            if camera_height <= 0:
                confidence *= self.CONFIDENCE_PENALTY_BAD_HEIGHT

        # Check tilt angle (should be in [-90, 90] range)
        if tilt_deg is not None:
            if tilt_deg < -90.0 or tilt_deg > 90.0:
                confidence *= self.CONFIDENCE_PENALTY_BAD_TILT
            # Check for gimbal lock near ±90° (cos(90°) = 0 causes singularity)
            elif abs(abs(tilt_deg) - 90.0) < self.GIMBAL_LOCK_THRESHOLD_DEG:
                confidence *= self.CONFIDENCE_PENALTY_GIMBAL_LOCK

        return confidence

    def _calculate_point_confidence(
        self,
        image_point: Tuple[float, float],
        base_confidence: float
    ) -> float:
        """
        Calculate per-point confidence based on distance from image center.

        Points near the image edges are less reliable due to lens distortion
        and perspective effects.

        Args:
            image_point: (u, v) pixel coordinates
            base_confidence: Base confidence from homography quality

        Returns:
            float: Adjusted confidence score in range [0.0, 1.0]
        """
        u, v = image_point

        # Calculate distance from image center (normalized)
        # Protect against division by zero
        if self.width <= 0 or self.height <= 0:
            return base_confidence

        center_u = self.width / 2.0
        center_v = self.height / 2.0

        dx = (u - center_u) / (self.width / 2.0)
        dy = (v - center_v) / (self.height / 2.0)

        dist_from_center = math.sqrt(dx * dx + dy * dy)

        # Reduce confidence for points far from center
        # Linear falloff: CENTER at center, EDGE at edges, decays toward corners
        if dist_from_center < 1.0:
            # Interpolate from CENTER (1.0) to EDGE (0.7) within the image boundary
            edge_factor = self.EDGE_FACTOR_CENTER - (self.EDGE_FACTOR_CENTER - self.EDGE_FACTOR_EDGE) * dist_from_center
        else:
            # Beyond edge, decay further toward minimum
            edge_factor = self.EDGE_FACTOR_EDGE - self.EDGE_FACTOR_CORNER_DECAY * (dist_from_center - 1.0)

        edge_factor = max(self.EDGE_FACTOR_MIN, min(self.EDGE_FACTOR_CENTER, edge_factor))

        return base_confidence * edge_factor

    def _local_to_gps(self, x_meters: Meters, y_meters: Meters) -> Tuple[Degrees, Degrees]:
        """
        Convert local metric coordinates to GPS coordinates.

        Uses the shared coordinate_converter module for consistency across
        the codebase.

        Args:
            x_meters: X coordinate in meters (East)
            y_meters: Y coordinate in meters (North)

        Returns:
            (latitude, longitude): GPS coordinates in decimal degrees

        Raises:
            RuntimeError: If camera GPS position not set
        """
        if self._camera_gps_lat is None or self._camera_gps_lon is None:
            raise RuntimeError(
                "Camera GPS position not set. Call set_camera_gps_position() first."
            )

        return local_xy_to_gps(
            self._camera_gps_lat,
            self._camera_gps_lon,
            x_meters,
            y_meters
        )

    def _project_image_point_to_world(
        self,
        image_point: Tuple[float, float]
    ) -> Tuple[Meters, Meters]:
        """
        Project image point to world ground plane coordinates (meters).

        Args:
            image_point: (u, v) pixel coordinates

        Returns:
            (x_world, y_world): Coordinates in meters (East, North)
        """
        u, v = image_point

        # Convert to homogeneous coordinates
        pt_homogeneous = np.array([u, v, 1.0])

        # Project to world using inverse homography
        world_homogeneous = self.H_inv @ pt_homogeneous

        # Check for division by zero (point at infinity/horizon)
        if abs(world_homogeneous[2]) < 1e-10:
            raise ValueError("Point projects to infinity (on horizon line)")

        # Normalize
        x_world = world_homogeneous[0] / world_homogeneous[2]
        y_world = world_homogeneous[1] / world_homogeneous[2]

        return x_world, y_world

    def _world_to_map_pixels(
        self,
        x_world: Meters,
        y_world: Meters,
        map_width: Pixels,
        map_height: Pixels
    ) -> Tuple[int, int]:
        """
        Convert world coordinates (meters) to map pixel coordinates.

        Map convention:
            - Center horizontally at map_width / 2
            - Bottom at map_height (Y increases upward in world)
            - Scale: pixels_per_meter

        Args:
            x_world: X coordinate in meters (East)
            y_world: Y coordinate in meters (North)
            map_width: Map width in pixels
            map_height: Map height in pixels

        Returns:
            (x_px, y_px): Pixel coordinates in map image
        """
        map_center_x = map_width // 2
        map_bottom_y = map_height

        x_px = int((x_world * self.pixels_per_meter) + map_center_x)
        y_px = int(map_bottom_y - (y_world * self.pixels_per_meter))

        return x_px, y_px

    # =========================================================================
    # HomographyProvider Interface Implementation
    # =========================================================================

    def compute_homography(
        self,
        frame: np.ndarray,
        reference: Dict[str, Any]
    ) -> HomographyResult:
        """
        Compute homography from camera parameters.

        For intrinsic/extrinsic approach, the frame is not used directly.
        The homography is computed from camera calibration and pose.

        Args:
            frame: Image frame (not used for this approach, but required by interface)
            reference: Dictionary with required keys:
                - 'camera_matrix': 3x3 intrinsic camera matrix K
                - 'camera_position': Camera position [X, Y, Z] in meters
                - 'pan_deg': Pan angle in degrees
                - 'tilt_deg': Tilt angle in degrees
                - 'roll_deg': Roll angle in degrees (optional, defaults to 0.0)
                - 'map_width': Output map width in pixels
                - 'map_height': Output map height in pixels

        Returns:
            HomographyResult with computed homography matrix and confidence

        Raises:
            ValueError: If required reference data is missing or invalid
        """
        # Validate reference data
        required_keys = [
            'camera_matrix', 'camera_position', 'pan_deg',
            'tilt_deg', 'map_width', 'map_height'
        ]
        for key in required_keys:
            if key not in reference:
                raise ValueError(f"Missing required reference key: '{key}'")

        K = reference['camera_matrix']
        camera_position = reference['camera_position']
        pan_deg = reference['pan_deg']
        tilt_deg = reference['tilt_deg']
        roll_deg = reference.get('roll_deg', 0.0)  # Default to 0.0 for backward compatibility
        map_width = reference['map_width']
        map_height = reference['map_height']

        # Validate inputs
        if not isinstance(K, np.ndarray) or K.shape != (3, 3):
            raise ValueError(f"camera_matrix must be 3x3 numpy array, got shape {K.shape}")

        if not isinstance(camera_position, np.ndarray) or len(camera_position) != 3:
            raise ValueError(
                f"camera_position must be array of 3 elements [X, Y, Z], "
                f"got {len(camera_position)}"
            )

        if camera_position[2] <= 0:
            logger.warning(
                "Camera height (Z=%s) should be positive for ground plane homography.",
                camera_position[2]
            )

        # Validate and clamp tilt angle to valid range [-90, 90] degrees
        if tilt_deg < -90.0 or tilt_deg > 90.0:
            logger.warning(
                "Tilt angle (%.2f degrees) is outside valid range [-90, 90]. "
                "Clamping to valid range.",
                tilt_deg
            )
            tilt_deg = max(-90.0, min(90.0, tilt_deg))

        # Warn about gimbal lock near ±90° (cos(90°) = 0 causes degenerate homography)
        if abs(abs(tilt_deg) - 90.0) < self.GIMBAL_LOCK_THRESHOLD_DEG:
            logger.warning(
                "Tilt angle (%.2f degrees) is near ±90° gimbal lock zone. "
                "Homography may be numerically unstable.",
                tilt_deg
            )

        # Validate roll angle (consistent with CameraGeometry)
        if abs(roll_deg) > self.ROLL_ERROR_THRESHOLD:
            raise ValueError(
                f"Roll angle {roll_deg:.1f}° is outside valid range "
                f"[-{self.ROLL_ERROR_THRESHOLD}°, {self.ROLL_ERROR_THRESHOLD}°]. "
                f"Check camera mount alignment."
            )

        if abs(roll_deg) > self.ROLL_WARN_THRESHOLD:
            logger.warning(
                "Roll angle (%.2f degrees) is unusually large (>%.1f degrees). "
                "Typical camera mount roll is ±2 degrees.",
                roll_deg, self.ROLL_WARN_THRESHOLD
            )

        # Store map dimensions
        self.map_width = map_width
        self.map_height = map_height

        # Calculate homography (includes roll rotation)
        self.H = self._calculate_ground_homography(K, camera_position, pan_deg, tilt_deg, roll_deg)

        # Calculate inverse homography
        det_H = np.linalg.det(self.H)
        if abs(det_H) < self.MIN_DET_THRESHOLD:
            logger.warning(
                "Homography is singular (det=%.2e). Inverse may be unstable.",
                det_H
            )
            self.H_inv = np.eye(3)
            self.confidence = 0.0
        else:
            self.H_inv = np.linalg.inv(self.H)
            self.confidence = self._calculate_confidence(self.H, camera_position, tilt_deg)

        # Store parameters for metadata
        self._last_camera_matrix = K.copy()
        self._last_camera_position = camera_position.copy()
        self._last_pan_deg = pan_deg
        self._last_tilt_deg = tilt_deg
        self._last_roll_deg = roll_deg
        self.roll_deg = roll_deg

        # Build metadata
        metadata = {
            'approach': HomographyApproach.INTRINSIC_EXTRINSIC.value,
            'camera_position': camera_position.tolist(),
            'pan_deg': pan_deg,
            'tilt_deg': tilt_deg,
            'roll_deg': roll_deg,
            'determinant': det_H,
            'map_dimensions': (map_width, map_height)
        }

        return HomographyResult(
            homography_matrix=self.H.copy(),
            confidence=self.confidence,
            metadata=metadata
        )

    def project_point(self, image_point: Tuple[float, float]) -> WorldPoint:
        """
        Project image point to GPS world coordinates.

        Args:
            image_point: (u, v) pixel coordinates

        Returns:
            WorldPoint with GPS latitude/longitude and confidence

        Raises:
            RuntimeError: If no valid homography computed or GPS position not set
            ValueError: If image_point is outside valid bounds
        """
        if self._camera_gps_lat is None or self._camera_gps_lon is None:
            raise RuntimeError(
                "Camera GPS position must be set before projecting to WorldPoint. "
                "Call set_camera_gps_position(lat, lon) first."
            )

        if not self.is_valid():
            raise RuntimeError("No valid homography available. Call compute_homography() first.")

        u, v = image_point
        if not (0 <= u < self.width) or not (0 <= v < self.height):
            raise ValueError(
                f"Image point ({u}, {v}) outside valid bounds "
                f"[0, {self.width}) x [0, {self.height})"
            )

        # Project to world coordinates (meters)
        x_world, y_world = self._project_image_point_to_world(image_point)

        # Convert to GPS
        latitude, longitude = self._local_to_gps(x_world, y_world)

        # Calculate point-specific confidence
        point_confidence = self._calculate_point_confidence(image_point, self.confidence)

        return WorldPoint(
            latitude=latitude,
            longitude=longitude,
            confidence=point_confidence
        )

    def project_points(
        self,
        image_points: List[Tuple[float, float]]
    ) -> List[WorldPoint]:
        """
        Project multiple image points to GPS world coordinates.

        Args:
            image_points: List of (u, v) pixel coordinates

        Returns:
            List of WorldPoint objects with GPS coordinates

        Raises:
            RuntimeError: If no valid homography computed or GPS position not set
            ValueError: If any image_point is outside valid bounds
        """
        if not self.is_valid():
            raise RuntimeError("No valid homography available. Call compute_homography() first.")

        # Iterate over points
        world_points = []

        for image_point in image_points:
            world_point = self.project_point(image_point)
            world_points.append(world_point)

        return world_points

    def get_confidence(self) -> float:
        """
        Return confidence score of current homography.

        Returns:
            float: Confidence in range [0.0, 1.0]
        """
        return self.confidence

    def is_valid(self) -> bool:
        """
        Check if homography is valid and ready for projection.

        A homography is valid if:
            - Confidence meets minimum threshold
            - Homography matrix is not identity (has been computed)

        Returns:
            bool: True if homography is valid for projection
        """
        # Check if homography has been computed (not identity)
        if np.allclose(self.H, np.eye(3)):
            return False

        # Check confidence threshold
        if self.confidence < self.MIN_CONFIDENCE_THRESHOLD:
            return False

        return True

    # =========================================================================
    # HomographyProviderExtended Interface Implementation
    # =========================================================================

    def project_point_to_map(
        self,
        image_point: Tuple[float, float]
    ) -> MapCoordinate:
        """
        Project image point to local map coordinates.

        Args:
            image_point: (u, v) pixel coordinates

        Returns:
            MapCoordinate with x, y in meters from camera position

        Raises:
            RuntimeError: If no valid homography computed
            ValueError: If image_point is outside valid bounds
        """
        if not self.is_valid():
            raise RuntimeError("No valid homography available. Call compute_homography() first.")

        u, v = image_point
        if not (0 <= u < self.width) or not (0 <= v < self.height):
            raise ValueError(
                f"Image point ({u}, {v}) outside valid bounds "
                f"[0, {self.width}) x [0, {self.height})"
            )

        # Project to world coordinates (meters)
        x_world, y_world = self._project_image_point_to_world(image_point)

        # Calculate point-specific confidence
        point_confidence = self._calculate_point_confidence(image_point, self.confidence)

        return MapCoordinate(
            x=x_world,
            y=y_world,
            confidence=point_confidence,
            elevation=0.0  # Ground plane assumption
        )

    def project_points_to_map(
        self,
        image_points: List[Tuple[float, float]]
    ) -> List[MapCoordinate]:
        """
        Project multiple image points to local map coordinates.

        Args:
            image_points: List of (u, v) pixel coordinates

        Returns:
            List of MapCoordinate objects with x, y in meters

        Raises:
            RuntimeError: If no valid homography computed
            ValueError: If any image_point is outside valid bounds
        """
        if not self.is_valid():
            raise RuntimeError("No valid homography available. Call compute_homography() first.")

        # Iterate over points
        map_coords = []

        for image_point in image_points:
            map_coord = self.project_point_to_map(image_point)
            map_coords.append(map_coord)

        return map_coords

    # =========================================================================
    # Additional Utility Methods (from camera_geometry.py)
    # =========================================================================

    def project_image_to_map(
        self,
        pts: List[Tuple[int, int]],
        sw: int,
        sh: int
    ) -> List[Tuple[int, int]]:
        """
        Project image coordinates to map visualization pixel coordinates.

        This is a legacy method from camera_geometry.py for compatibility.
        It projects image points to world coordinates, then to map pixels.

        Args:
            pts: List of (u, v) image pixel coordinates
            sw: Side-panel (map) width in pixels
            sh: Side-panel (map) height in pixels

        Returns:
            List of (x, y) pixel coordinates in map visualization
        """
        if not self.is_valid():
            # Fallback: simple downscaling if no homography
            return [(int(x / 2), int(y / 2)) for x, y in pts]

        # Convert to numpy array for vectorized operations
        pts_homogeneous = np.array(pts, dtype=np.float64).T
        pts_homogeneous = np.vstack([pts_homogeneous, np.ones(pts_homogeneous.shape[1])])

        # Project from image to world ground plane
        pts_world_homogeneous = self.H_inv @ pts_homogeneous

        # Normalize with protection against division by zero (points on/near horizon)
        w_coords = pts_world_homogeneous[2, :]
        # Replace near-zero values with a small epsilon to avoid division by zero
        # Points with |w| < threshold are on or near the horizon line
        horizon_mask = np.abs(w_coords) < 1e-10
        w_safe = np.where(horizon_mask, np.sign(w_coords + 1e-20) * 1e-10, w_coords)

        Xw = pts_world_homogeneous[0, :] / w_safe
        Yw = pts_world_homogeneous[1, :] / w_safe

        # Clamp extreme values for horizon points to avoid overflow in pixel conversion
        max_coord = 1e6  # Maximum reasonable world coordinate in meters
        Xw = np.clip(Xw, -max_coord, max_coord)
        Yw = np.clip(Yw, -max_coord, max_coord)

        # Convert world coordinates to map pixels
        map_center_x = sw // 2
        map_bottom_y = sh

        pts_map_x = (Xw * self.pixels_per_meter) + map_center_x
        pts_map_y = map_bottom_y - (Yw * self.pixels_per_meter)

        pts_map = [(int(x), int(y)) for x, y in zip(pts_map_x, pts_map_y)]

        return pts_map

    def world_to_map(
        self,
        Xw: float,
        Yw: float,
        sw: Optional[int] = None,
        sh: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Convert world coordinates (meters) to map pixel coordinates.

        This is a legacy method from camera_geometry.py for compatibility.

        Args:
            Xw: X coordinate in world frame (meters, East)
            Yw: Y coordinate in world frame (meters, North)
            sw: Map width in pixels (default: self.map_width)
            sh: Map height in pixels (default: self.map_height)

        Returns:
            (x_px, y_px): Pixel coordinates in map visualization
        """
        if sw is None:
            sw = self.map_width
        if sh is None:
            sh = self.map_height

        return self._world_to_map_pixels(Xw, Yw, sw, sh)

    def get_camera_position(self) -> Optional[np.ndarray]:
        """
        Get the last used camera position in world coordinates.

        Returns:
            Camera position as [X, Y, Z] in meters, or None if not set.
        """
        return self._last_camera_position
