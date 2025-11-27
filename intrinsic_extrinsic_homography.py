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
from typing import List, Tuple, Dict, Any, Optional

from homography_interface import (
    HomographyProviderExtended,
    HomographyResult,
    WorldPoint,
    MapCoordinate,
    HomographyApproach
)


class IntrinsicExtrinsicHomography(HomographyProviderExtended):
    """
    Homography provider using camera intrinsic/extrinsic parameters.

    This implementation computes homography for a ground plane (Z=0) using:
    - Camera intrinsic matrix K (focal length, principal point)
    - Camera extrinsic parameters (position, rotation via pan/tilt)

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
        camera_gps_lat: Camera GPS latitude for WorldPoint conversion
        camera_gps_lon: Camera GPS longitude for WorldPoint conversion
    """

    # Earth radius for GPS conversion (meters) - approximate for equirectangular projection
    EARTH_RADIUS = 6371000.0

    # Minimum determinant threshold for valid homography
    MIN_DET_THRESHOLD = 1e-10

    # Minimum confidence threshold for validity
    MIN_CONFIDENCE_THRESHOLD = 0.3

    def __init__(
        self,
        width: int,
        height: int,
        pixels_per_meter: float = 100.0,
        sensor_width_mm: float = 7.18,
        base_focal_length_mm: float = 5.9,
        **kwargs  # Accept and ignore other kwargs for forward compatibility
    ):
        """
        Initialize intrinsic/extrinsic homography provider.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            pixels_per_meter: Scale factor for map visualization (default: 100)
            sensor_width_mm: Physical sensor width in millimeters (default: 7.18)
            base_focal_length_mm: Base focal length at 1x zoom in mm (default: 5.9)
            **kwargs: Additional parameters (ignored, for forward compatibility)
        """
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
        self.camera_gps_lat: Optional[float] = None
        self.camera_gps_lon: Optional[float] = None

        # Last used camera parameters (for metadata)
        self._last_camera_matrix: Optional[np.ndarray] = None
        self._last_camera_position: Optional[np.ndarray] = None
        self._last_pan_deg: Optional[float] = None
        self._last_tilt_deg: Optional[float] = None

    @staticmethod
    def get_intrinsics(
        zoom_factor: float,
        width_px: int = 2560,
        height_px: int = 1440,
        sensor_width_mm: float = 7.18
    ) -> np.ndarray:
        """
        Calculate camera intrinsic matrix K from zoom factor and sensor specs.

        This is a convenience method for computing the camera matrix based on
        physical camera specifications. The focal length is computed as a linear
        function of zoom factor based on typical PTZ camera characteristics.

        Args:
            zoom_factor: Digital or optical zoom multiplier (1.0 = no zoom)
            width_px: Image width in pixels (default: 2560)
            height_px: Image height in pixels (default: 1440)
            sensor_width_mm: Physical sensor width in millimeters (default: 7.18)

        Returns:
            K: 3x3 camera intrinsic matrix with:
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
                where fx=fy is the focal length in pixels, and (cx, cy) is
                the principal point (typically image center).

        Example:
            >>> K = IntrinsicExtrinsicHomography.get_intrinsics(zoom_factor=5.0)
            >>> print(K)
            [[2106.27...    0.     1280.  ]
             [   0.     2106.27...  720.  ]
             [   0.        0.        1.  ]]
        """
        # Linear mapping based on camera datasheet: 1x zoom = 5.9mm focal length
        f_mm = 5.9 * zoom_factor

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

    def set_camera_gps_position(self, lat: float, lon: float) -> None:
        """
        Set camera GPS position for WorldPoint conversion.

        This establishes the reference point for converting local metric
        coordinates (X, Y in meters) to GPS coordinates (latitude, longitude).

        Args:
            lat: Camera latitude in decimal degrees [-90, 90]
            lon: Camera longitude in decimal degrees [-180, 180]

        Raises:
            ValueError: If latitude or longitude out of valid range
        """
        if not -90 <= lat <= 90:
            raise ValueError(f"Latitude must be in range [-90, 90], got {lat}")
        if not -180 <= lon <= 180:
            raise ValueError(f"Longitude must be in range [-180, 180], got {lon}")

        self.camera_gps_lat = lat
        self.camera_gps_lon = lon

    def _get_rotation_matrix(self, pan_deg: float, tilt_deg: float) -> np.ndarray:
        """
        Calculate rotation matrix from world to camera coordinates.

        Computes the 3x3 rotation matrix based on pan (yaw) and tilt (pitch).
        Assumes zero roll. Rotation order: Pan first (around Z-axis), then
        Tilt (around rotated X-axis). This matches standard PTZ camera behavior.

        Coordinate System Convention:
            - World: X=East, Y=North, Z=Up
            - Camera: X=Right, Y=Down, Z=Forward (optical axis)

        Args:
            pan_deg: Pan angle in degrees (positive = right/clockwise from above)
            tilt_deg: Tilt angle in degrees (negative = down)

        Returns:
            R: 3x3 rotation matrix transforming world coordinates to camera frame
        """
        pan_rad = math.radians(pan_deg)
        tilt_rad = math.radians(tilt_deg)

        # Yaw rotation around World Z-axis (Pan)
        # Positive pan rotates camera to the right (clockwise from above)
        Rz = np.array([
            [math.cos(pan_rad), -math.sin(pan_rad), 0.0],
            [math.sin(pan_rad), math.cos(pan_rad), 0.0],
            [0.0, 0.0, 1.0]
        ])

        # Pitch rotation around X-axis (Tilt)
        # Negative tilt points camera downward
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(tilt_rad), -math.sin(tilt_rad)],
            [0.0, math.sin(tilt_rad), math.cos(tilt_rad)]
        ])

        # Apply Pan first, then Tilt
        R = Rz @ Rx
        return R

    def _calculate_ground_homography(
        self,
        K: np.ndarray,
        camera_position: np.ndarray,
        pan_deg: float,
        tilt_deg: float
    ) -> np.ndarray:
        """
        Calculate homography matrix mapping world ground plane (Z=0) to image.

        Mathematical derivation:
            - Camera position in world: C = [Xw, Yw, Zw]
            - Rotation from world to camera: R (3x3)
            - Translation: t = -R @ C (world origin position in camera frame)
            - For ground plane (Z=0): P_world = [X, Y, 0]
            - Projection: p_image = K @ [R @ P_world + t]
            - Since Z=0, homography becomes: H = K @ [r1, r2, t]
              where r1, r2 are first two columns of R

        Args:
            K: 3x3 camera intrinsic matrix
            camera_position: Camera position [X, Y, Z] in world coordinates (meters)
            pan_deg: Pan angle in degrees
            tilt_deg: Tilt angle in degrees

        Returns:
            H: 3x3 homography matrix mapping [X_world, Y_world, 1] -> [u, v, 1]
        """
        # Get rotation matrix
        R = self._get_rotation_matrix(pan_deg, tilt_deg)

        # Camera position C in world coordinates
        C = camera_position

        # Translation from camera to world origin: t = -R @ C
        t = -R @ C

        # Build homography: H = K @ [r1, r2, t]
        # r1, r2 are the first two columns of R (corresponding to X and Y axes)
        r1 = R[:, 0]
        r2 = R[:, 1]

        H_extrinsic = np.column_stack([r1, r2, t])

        H = K @ H_extrinsic

        # Normalize so H[2, 2] = 1
        if abs(H[2, 2]) < self.MIN_DET_THRESHOLD:
            # Return identity if normalization fails
            return np.eye(3)

        H = H / H[2, 2]

        return H

    def _calculate_confidence(self, H: np.ndarray) -> float:
        """
        Calculate confidence score for the homography matrix.

        Confidence is based on the determinant of the homography matrix.
        A well-conditioned homography has a non-zero determinant.

        Args:
            H: 3x3 homography matrix

        Returns:
            float: Confidence score in range [0.0, 1.0]
        """
        det_H = np.linalg.det(H)

        # Check if homography is singular
        if abs(det_H) < self.MIN_DET_THRESHOLD:
            return 0.0

        # Base confidence on determinant magnitude
        # Normalized determinant should be around 1.0 for well-conditioned matrices
        # We use a sigmoid-like function to map determinant to [0, 1]
        det_abs = abs(det_H)

        # Very rough heuristic: good homographies have |det| in reasonable range
        # Too small -> singular, too large -> poorly scaled
        if det_abs < 1e-6:
            confidence = 0.0
        elif det_abs < 1e-3:
            confidence = 0.5
        elif det_abs < 1e3:
            confidence = 1.0
        else:
            confidence = 0.7  # Very large determinant, might be poorly scaled

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
        center_u = self.width / 2.0
        center_v = self.height / 2.0

        dx = (u - center_u) / (self.width / 2.0)
        dy = (v - center_v) / (self.height / 2.0)

        dist_from_center = math.sqrt(dx * dx + dy * dy)

        # Reduce confidence for points far from center
        # Linear falloff: 1.0 at center, 0.7 at edges, 0.5 at corners
        if dist_from_center < 1.0:
            edge_factor = 1.0 - 0.3 * dist_from_center
        else:
            edge_factor = 0.7 - 0.2 * (dist_from_center - 1.0)

        edge_factor = max(0.3, min(1.0, edge_factor))

        return base_confidence * edge_factor

    def _local_to_gps(self, x_meters: float, y_meters: float) -> Tuple[float, float]:
        """
        Convert local metric coordinates to GPS coordinates.

        Uses equirectangular projection approximation, which is accurate enough
        for small areas (< 10 km). For larger areas, a more sophisticated
        projection should be used.

        Args:
            x_meters: X coordinate in meters (East)
            y_meters: Y coordinate in meters (North)

        Returns:
            (latitude, longitude): GPS coordinates in decimal degrees

        Raises:
            RuntimeError: If camera GPS position not set
        """
        if self.camera_gps_lat is None or self.camera_gps_lon is None:
            raise RuntimeError(
                "Camera GPS position not set. Call set_camera_gps_position() first."
            )

        # Convert meters to degrees
        # Latitude: 1 degree ≈ 111,111 meters
        # Longitude: 1 degree ≈ 111,111 * cos(latitude) meters
        lat_deg = y_meters / 111111.0

        # Adjust longitude for latitude
        lat_rad = math.radians(self.camera_gps_lat)
        lon_deg = x_meters / (111111.0 * math.cos(lat_rad))

        # Add to camera position
        latitude = self.camera_gps_lat + lat_deg
        longitude = self.camera_gps_lon + lon_deg

        # Clamp to valid ranges
        latitude = max(-90.0, min(90.0, latitude))
        longitude = max(-180.0, min(180.0, longitude))

        return latitude, longitude

    def _project_image_point_to_world(
        self,
        image_point: Tuple[float, float]
    ) -> Tuple[float, float]:
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

        # Normalize
        x_world = world_homogeneous[0] / world_homogeneous[2]
        y_world = world_homogeneous[1] / world_homogeneous[2]

        return x_world, y_world

    def _world_to_map_pixels(
        self,
        x_world: float,
        y_world: float,
        map_width: int,
        map_height: int
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
            print(
                f"Warning: Camera height (Z={camera_position[2]}) should be positive "
                f"for ground plane homography."
            )

        # Store map dimensions
        self.map_width = map_width
        self.map_height = map_height

        # Calculate homography
        self.H = self._calculate_ground_homography(K, camera_position, pan_deg, tilt_deg)

        # Calculate inverse homography
        det_H = np.linalg.det(self.H)
        if abs(det_H) < self.MIN_DET_THRESHOLD:
            print(
                f"Warning: Homography is singular (det={det_H:.2e}). "
                f"Inverse may be unstable."
            )
            self.H_inv = np.eye(3)
            self.confidence = 0.0
        else:
            self.H_inv = np.linalg.inv(self.H)
            self.confidence = self._calculate_confidence(self.H)

        # Store parameters for metadata
        self._last_camera_matrix = K.copy()
        self._last_camera_position = camera_position.copy()
        self._last_pan_deg = pan_deg
        self._last_tilt_deg = tilt_deg

        # Build metadata
        metadata = {
            'approach': HomographyApproach.INTRINSIC_EXTRINSIC.value,
            'camera_position': camera_position.tolist(),
            'pan_deg': pan_deg,
            'tilt_deg': tilt_deg,
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
        if not self.is_valid():
            raise RuntimeError("No valid homography available. Call compute_homography() first.")

        u, v = image_point
        if not (0 <= u <= self.width) or not (0 <= v <= self.height):
            raise ValueError(
                f"Image point ({u}, {v}) outside valid bounds "
                f"[0, {self.width}] x [0, {self.height}]"
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

        # Vectorized implementation for efficiency
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
        if not (0 <= u <= self.width) or not (0 <= v <= self.height):
            raise ValueError(
                f"Image point ({u}, {v}) outside valid bounds "
                f"[0, {self.width}] x [0, {self.height}]"
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

        # Vectorized implementation for efficiency
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

        # Normalize
        Xw = pts_world_homogeneous[0, :] / pts_world_homogeneous[2, :]
        Yw = pts_world_homogeneous[1, :] / pts_world_homogeneous[2, :]

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
