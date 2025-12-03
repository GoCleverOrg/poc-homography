"""
Camera Position Deriver using PnP (Perspective-n-Point) solver.

This module implements camera position derivation from ground control points (GCPs)
using OpenCV's solvePnP/solvePnPRansac. Given known 3D world coordinates and their
corresponding 2D pixel projections, the PnP solver determines camera rotation and
translation, from which camera position and orientation are computed.

Mathematical Background:
    The PnP solver finds R (rotation) and t (translation) such that:
        s * p_image = K @ [R | t] @ P_world

    Where:
        - p_image: 2D pixel coordinates [u, v, 1]
        - P_world: 3D world coordinates [X, Y, Z, 1]
        - K: Camera intrinsic matrix (3x3)
        - R: Rotation matrix (3x3) from world to camera frame
        - t: Translation vector (3x1) from world origin to camera origin in camera frame
        - s: Scale factor (homogeneous coordinates normalization)

    Camera position in world frame: C = -R.T @ t

Coordinate Systems:
    - World Frame: X=East, Y=North, Z=Up (right-handed)
    - Camera Frame: X=Right, Y=Down, Z=Forward (standard CV, right-handed)
    - Image Frame: origin top-left, u=right, v=down (pixels)
    - GCP GPS: latitude/longitude in decimal degrees (WGS84)

Reference:
    ptz_intrinsics_and_pose.md Section 2 Method B (PnP Solver)
"""

import numpy as np
import cv2
import math
import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union

from poc_homography.gps_distance_calculator import gps_to_local_xy, local_xy_to_gps

logger = logging.getLogger(__name__)


class AccuracyLevel(Enum):
    """
    Accuracy levels for PnP solver configuration.

    Higher accuracy levels use more RANSAC iterations and tighter thresholds,
    providing better outlier rejection at the cost of computation time.

    Attributes:
        LOW: Fast computation, suitable for continuous updates.
             6-10 GCPs, 100 RANSAC iterations, 5.0px threshold.
        MEDIUM: Balanced accuracy and speed.
                10-15 GCPs, 500 RANSAC iterations, 3.0px threshold.
        HIGH: Maximum accuracy for initial calibration.
              15+ GCPs, 1000 RANSAC iterations, 1.0px threshold.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class GroundControlPoint:
    """
    A ground control point with GPS world coordinates and image pixel coordinates.

    Attributes:
        latitude: GPS latitude in decimal degrees (-90 to 90)
        longitude: GPS longitude in decimal degrees (-180 to 180)
        u: Pixel x-coordinate (column, 0 at left)
        v: Pixel y-coordinate (row, 0 at top)
    """
    latitude: float
    longitude: float
    u: float
    v: float

    def __post_init__(self):
        """Validate GCP coordinate ranges."""
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"Latitude {self.latitude} outside valid range [-90, 90]")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"Longitude {self.longitude} outside valid range [-180, 180]")
        if self.u < 0:
            raise ValueError(f"Pixel u-coordinate {self.u} must be non-negative")
        if self.v < 0:
            raise ValueError(f"Pixel v-coordinate {self.v} must be non-negative")


@dataclass
class PnPResult:
    """
    Result of PnP camera position derivation.

    Attributes:
        success: Whether derivation succeeded
        position: Camera position [X, Y, Z] in meters (world frame)
        rotation_matrix: 3x3 rotation matrix (world to camera)
        rotation_vector: Rodrigues rotation vector (3,)
        pan_deg: Pan angle in degrees (positive = right/clockwise from above)
        tilt_deg: Tilt angle in degrees (positive = down, Hikvision convention)
        reprojection_error_mean: Mean reprojection error in pixels (inliers only)
        reprojection_error_max: Maximum reprojection error in pixels (inliers only)
        inlier_ratio: Ratio of inliers to total GCPs
        num_inliers: Number of inlier GCPs after RANSAC
        inliers_mask: Boolean mask indicating which GCPs are inliers
    """
    success: bool
    position: Optional[np.ndarray] = None
    rotation_matrix: Optional[np.ndarray] = None
    rotation_vector: Optional[np.ndarray] = None
    pan_deg: Optional[float] = None
    tilt_deg: Optional[float] = None
    reprojection_error_mean: Optional[float] = None
    reprojection_error_max: Optional[float] = None
    inlier_ratio: Optional[float] = None
    num_inliers: Optional[int] = None
    inliers_mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'success': self.success,
            'position': self.position.tolist() if self.position is not None else None,
            'rotation_matrix': self.rotation_matrix.tolist() if self.rotation_matrix is not None else None,
            'rotation_vector': self.rotation_vector.tolist() if self.rotation_vector is not None else None,
            'pan_deg': self.pan_deg,
            'tilt_deg': self.tilt_deg,
            'reprojection_error_mean': self.reprojection_error_mean,
            'reprojection_error_max': self.reprojection_error_max,
            'inlier_ratio': self.inlier_ratio,
            'num_inliers': self.num_inliers,
            'inliers_mask': self.inliers_mask.tolist() if self.inliers_mask is not None else None,
        }


class CameraPositionDeriver:
    """
    Derives camera position and orientation from ground control points using PnP.

    This class implements Method B from ptz_intrinsics_and_pose.md: using cv2.solvePnPRansac
    to derive camera position from known 3D world points and their 2D image projections.

    The class supports configurable accuracy levels that trade off computation time
    for robustness, making it suitable for both initial calibration (HIGH accuracy)
    and continuous drift correction (LOW accuracy).

    Coordinate Frame Conventions:
        - World Frame: X=East, Y=North, Z=Up (meters from reference GPS point)
        - Camera Frame: X=Right, Y=Down, Z=Forward
        - GPS coordinates are converted to local metric coordinates using equirectangular
          approximation (accurate for distances < 10km)

    Example:
        >>> # Initialize with camera intrinsics
        >>> K = homography_provider.get_intrinsics(zoom_factor=5.0)
        >>> deriver = CameraPositionDeriver(
        ...     K=K,
        ...     reference_lat=39.640583,
        ...     reference_lon=-0.230194,
        ...     accuracy=AccuracyLevel.HIGH
        ... )
        >>>
        >>> # Provide ground control points
        >>> gcps = [
        ...     GroundControlPoint(lat=39.6406, lon=-0.2302, u=1280, v=720),
        ...     # ... at least 6 GCPs total
        ... ]
        >>>
        >>> # Derive camera position
        >>> result = deriver.derive_position(gcps)
        >>> if result.success:
        ...     print(f"Camera position: {result.position}")
        ...     print(f"Pan: {result.pan_deg}°, Tilt: {result.tilt_deg}°")
    """

    # Minimum number of GCPs required for RANSAC robustness
    MIN_GCP_COUNT = 6

    # Default RANSAC configuration per accuracy level
    RANSAC_CONFIG = {
        AccuracyLevel.LOW: {
            'iterations': 100,
            'reprojection_threshold': 5.0,
            'confidence': 0.95,
        },
        AccuracyLevel.MEDIUM: {
            'iterations': 500,
            'reprojection_threshold': 3.0,
            'confidence': 0.99,
        },
        AccuracyLevel.HIGH: {
            'iterations': 1000,
            'reprojection_threshold': 1.0,
            'confidence': 0.999,
        },
    }

    # Validation thresholds for result quality
    MAX_REPROJECTION_ERROR = 5.0  # pixels
    MIN_INLIER_RATIO = 0.6  # 60% of GCPs must be inliers

    # Gimbal lock detection threshold: cos(tilt) < this value means tilt ≈ ±90°
    # Value of 1e-6 corresponds to |tilt| > 89.9999° which provides numerical stability
    GIMBAL_LOCK_COS_THRESHOLD = 1e-6

    def __init__(
        self,
        K: np.ndarray,
        reference_lat: float,
        reference_lon: float,
        accuracy: AccuracyLevel = AccuracyLevel.MEDIUM,
        ransac_iterations: Optional[int] = None,
        ransac_reprojection_threshold: Optional[float] = None,
        ransac_confidence: Optional[float] = None,
        solver_method: int = cv2.SOLVEPNP_ITERATIVE,
        max_reprojection_error: Optional[float] = None,
        min_inlier_ratio: Optional[float] = None,
    ):
        """
        Initialize camera position deriver.

        Args:
            K: Camera intrinsic matrix (3x3 numpy array)
            reference_lat: Reference GPS latitude for coordinate conversion
            reference_lon: Reference GPS longitude for coordinate conversion
            accuracy: Accuracy level (LOW, MEDIUM, HIGH) controlling RANSAC parameters
            ransac_iterations: Override RANSAC iteration count (None = use accuracy default)
            ransac_reprojection_threshold: Override RANSAC reprojection threshold in pixels
            ransac_confidence: Override RANSAC confidence level (0.0-1.0)
            solver_method: OpenCV PnP solver method (default: cv2.SOLVEPNP_ITERATIVE)
            max_reprojection_error: Override max reprojection error threshold for validation (pixels)
            min_inlier_ratio: Override minimum inlier ratio threshold for validation (0.0-1.0)

        Raises:
            ValueError: If K matrix is invalid or reference coordinates are out of range
        """
        # Validate intrinsic matrix
        if not isinstance(K, np.ndarray) or K.shape != (3, 3):
            raise ValueError(f"K must be a 3x3 numpy array, got shape {K.shape if isinstance(K, np.ndarray) else type(K)}")

        # Validate reference GPS coordinates
        if not -90.0 <= reference_lat <= 90.0:
            raise ValueError(f"Reference latitude {reference_lat} outside valid range [-90, 90]")
        if not -180.0 <= reference_lon <= 180.0:
            raise ValueError(f"Reference longitude {reference_lon} outside valid range [-180, 180]")

        self.K = K.astype(np.float64)
        self.reference_lat = reference_lat
        self.reference_lon = reference_lon
        self.accuracy = accuracy
        self.solver_method = solver_method

        # Get default RANSAC config for accuracy level
        default_config = self.RANSAC_CONFIG[accuracy]

        # Apply overrides or use defaults for RANSAC parameters
        self.ransac_iterations = ransac_iterations if ransac_iterations is not None else default_config['iterations']
        self.ransac_reprojection_threshold = ransac_reprojection_threshold if ransac_reprojection_threshold is not None else default_config['reprojection_threshold']
        self.ransac_confidence = ransac_confidence if ransac_confidence is not None else default_config['confidence']

        # Apply overrides or use defaults for validation thresholds
        self.max_reprojection_error = max_reprojection_error if max_reprojection_error is not None else self.MAX_REPROJECTION_ERROR
        self.min_inlier_ratio = min_inlier_ratio if min_inlier_ratio is not None else self.MIN_INLIER_RATIO

        logger.debug(
            f"CameraPositionDeriver initialized: accuracy={accuracy.value}, "
            f"iterations={self.ransac_iterations}, threshold={self.ransac_reprojection_threshold}px, "
            f"confidence={self.ransac_confidence}, "
            f"max_reproj_error={self.max_reprojection_error}px, min_inlier_ratio={self.min_inlier_ratio:.0%}"
        )

    def _convert_gcps_to_world_coords(
        self,
        gcps: List[GroundControlPoint]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert GCPs from GPS coordinates to local metric 3D coordinates.

        Uses equirectangular approximation to convert GPS coordinates to local
        metric coordinates relative to the reference point. All GCPs are assumed
        to be on the ground plane (Z=0).

        Args:
            gcps: List of ground control points with GPS coordinates

        Returns:
            object_points: Nx3 array of world coordinates [X, Y, Z] in meters
            image_points: Nx2 array of pixel coordinates [u, v]
        """
        object_points = []
        image_points = []

        for gcp in gcps:
            # Convert GPS to local X, Y (meters)
            x, y = gps_to_local_xy(
                self.reference_lat, self.reference_lon,
                gcp.latitude, gcp.longitude
            )
            # Ground plane assumption: Z = 0
            object_points.append([x, y, 0.0])
            image_points.append([gcp.u, gcp.v])

        return (
            np.array(object_points, dtype=np.float64),
            np.array(image_points, dtype=np.float64)
        )

    def _extract_pan_tilt_from_rotation(
        self,
        R: np.ndarray
    ) -> Tuple[float, float]:
        """
        Extract pan and tilt angles from rotation matrix.

        Decomposes the rotation matrix R (world to camera) into pan (yaw around Z)
        and tilt (pitch around X) angles. This is the inverse operation of
        IntrinsicExtrinsicHomography._get_rotation_matrix().

        The decomposition assumes ZXY Euler angle order:
            R = Rz(pan) @ Rx(tilt)

        where Rz is rotation around world Z-axis (vertical) and Rx is rotation
        around the rotated X-axis.

        Args:
            R: 3x3 rotation matrix (world to camera frame)

        Returns:
            pan_deg: Pan angle in degrees (positive = right/clockwise from above)
            tilt_deg: Tilt angle in degrees (positive = down, Hikvision convention)

        Note:
            - Pan range: -180 to 180 degrees
            - Tilt range: -90 to 90 degrees
            - Near gimbal lock (tilt ≈ ±90°), pan becomes ambiguous
        """
        # For R = Rz(pan) @ Rx(tilt), the matrix is:
        # R = [cos(pan), -sin(pan)*cos(tilt), sin(pan)*sin(tilt)]
        #     [sin(pan),  cos(pan)*cos(tilt), -cos(pan)*sin(tilt)]
        #     [0,         sin(tilt),          cos(tilt)]
        #
        # However, due to the Y-down camera convention, we use -tilt internally
        # (see _get_rotation_matrix which uses tilt_rad = math.radians(-tilt_deg))
        #
        # So the actual matrix is R = Rz(pan) @ Rx(-tilt_internal):
        # R[2,1] = sin(-tilt_internal) = -sin(tilt_internal)
        # R[2,2] = cos(-tilt_internal) = cos(tilt_internal)
        #
        # And we want tilt_deg (Hikvision convention, positive = down)

        # Extract tilt from R[2,1] and R[2,2]
        # Note: tilt_internal = -tilt_deg (due to Y-down coordinate system)
        # R[2,1] = sin(-tilt_deg) = -sin(tilt_deg)
        # R[2,2] = cos(-tilt_deg) = cos(tilt_deg)
        tilt_rad_internal = math.atan2(R[2, 1], R[2, 2])
        tilt_deg = -math.degrees(tilt_rad_internal)  # Invert to get Hikvision convention

        # Check for gimbal lock (tilt near ±90°)
        cos_tilt = math.cos(tilt_rad_internal)
        if abs(cos_tilt) < self.GIMBAL_LOCK_COS_THRESHOLD:
            # Near gimbal lock - pan is mathematically ambiguous
            # At gimbal lock, pan and roll are coupled (only their sum/difference is defined)
            # We set pan to 0 and let the rotation be absorbed into the ambiguous state
            logger.warning(
                f"Near gimbal lock (tilt={tilt_deg:.1f}°), pan angle is mathematically "
                f"ambiguous. Setting pan=0.0° (pan and roll are coupled at gimbal lock)."
            )
            # At tilt = ±90°, R[0,0] = cos(pan±roll), R[1,0] = sin(pan±roll)
            # Without additional constraints, we cannot separate pan from roll
            # Convention: set pan to 0 at gimbal lock
            pan_rad = 0.0
        else:
            # Normal case: extract pan from R[0,0], R[1,0]
            # R[0,0] = cos(pan), R[1,0] = sin(pan)
            pan_rad = math.atan2(R[1, 0], R[0, 0])

        pan_deg = math.degrees(pan_rad)

        return pan_deg, tilt_deg

    def _compute_reprojection_errors(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        inliers_mask: Optional[np.ndarray] = None
    ) -> Tuple[float, float, np.ndarray]:
        """
        Compute reprojection errors for all points or inliers only.

        Projects 3D object points back to image using the estimated pose and
        computes pixel distance from observed image points.

        Args:
            object_points: Nx3 array of world coordinates
            image_points: Nx2 array of observed pixel coordinates
            rvec: Rotation vector from solvePnP
            tvec: Translation vector from solvePnP
            inliers_mask: Optional boolean mask for inliers

        Returns:
            mean_error: Mean reprojection error in pixels
            max_error: Maximum reprojection error in pixels
            errors: Array of per-point reprojection errors
        """
        # Project object points to image
        projected_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, self.K, distCoeffs=None
        )
        projected_points = projected_points.reshape(-1, 2)

        # Compute per-point errors (Euclidean distance in pixels)
        errors = np.linalg.norm(projected_points - image_points, axis=1)

        # Filter to inliers if mask provided
        if inliers_mask is not None:
            inlier_errors = errors[inliers_mask.flatten()]
            if len(inlier_errors) > 0:
                mean_error = float(np.mean(inlier_errors))
                max_error = float(np.max(inlier_errors))
            else:
                mean_error = float('inf')
                max_error = float('inf')
        else:
            mean_error = float(np.mean(errors))
            max_error = float(np.max(errors))

        return mean_error, max_error, errors

    def derive_position(
        self,
        gcps: Union[List[GroundControlPoint], List[Dict[str, Any]], List[Tuple[float, float, float, float]]]
    ) -> PnPResult:
        """
        Derive camera position from ground control points using PnP solver.

        Uses OpenCV's solvePnPRansac for robust estimation with outlier rejection.
        Returns camera position in world coordinates, rotation matrix, and
        pan/tilt angles compatible with the PTZ camera convention.

        Args:
            gcps: Ground control points in one of these formats:
                - List[GroundControlPoint]: Dataclass objects
                - List[Dict]: Dicts with 'lat'/'latitude', 'lon'/'longitude', 'u', 'v'
                - List[Tuple]: Tuples of (latitude, longitude, u, v)

        Returns:
            PnPResult: Result containing position, rotation, angles, and quality metrics

        Raises:
            ValueError: If fewer than 6 GCPs provided

        Example:
            >>> result = deriver.derive_position(gcps)
            >>> if result.success:
            ...     print(f"Position: {result.position}")
            ...     print(f"Reprojection error: {result.reprojection_error_mean:.2f}px")
        """
        # Normalize GCP input format
        normalized_gcps = self._normalize_gcps(gcps)

        # Validate minimum GCP count
        if len(normalized_gcps) < self.MIN_GCP_COUNT:
            raise ValueError(
                f"At least {self.MIN_GCP_COUNT} GCPs required for RANSAC robustness, "
                f"got {len(normalized_gcps)}"
            )

        # Convert GCPs to world and image coordinates
        object_points, image_points = self._convert_gcps_to_world_coords(normalized_gcps)

        logger.debug(
            f"Solving PnP with {len(normalized_gcps)} GCPs, "
            f"object points range: X=[{object_points[:,0].min():.1f}, {object_points[:,0].max():.1f}], "
            f"Y=[{object_points[:,1].min():.1f}, {object_points[:,1].max():.1f}]"
        )

        # Call solvePnPRansac
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=self.K,
            distCoeffs=None,
            iterationsCount=self.ransac_iterations,
            reprojectionError=self.ransac_reprojection_threshold,
            confidence=self.ransac_confidence,
            flags=self.solver_method
        )

        # Handle RANSAC failure
        if not success or inliers is None or len(inliers) == 0:
            logger.warning("solvePnPRansac failed to find a valid solution")
            return PnPResult(success=False)

        # Convert inliers to boolean mask
        inliers_mask = np.zeros(len(object_points), dtype=bool)
        inliers_mask[inliers.flatten()] = True
        num_inliers = int(np.sum(inliers_mask))
        inlier_ratio = num_inliers / len(object_points)

        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)

        # Compute camera position in world frame: C = -R.T @ t
        # Reference: ptz_intrinsics_and_pose.md line 92
        camera_position = -R.T @ tvec.flatten()

        # Handle planar PnP ambiguity: when all GCPs are on the ground plane (Z=0),
        # solvePnP may find an equivalent solution with camera below the plane.
        # For PTZ cameras, we know the camera is always above ground, so ensure Z > 0.
        if camera_position[2] < 0:
            logger.debug(
                f"PnP returned camera below ground (Z={camera_position[2]:.2f}m), "
                f"flipping to above-ground solution"
            )
            # Flip the solution: negate Z and adjust rotation accordingly
            camera_position[2] = -camera_position[2]
            # The rotation matrix needs to be adjusted for the flipped coordinate
            # This is equivalent to applying a reflection across the XY plane
            R = R @ np.diag([1.0, 1.0, -1.0])
            # Note: rvec/tvec are NOT updated here. The original OpenCV solution
            # is mathematically correct for reprojection - we only flip the
            # interpretation of camera_position and R for the output values.
            # The planar PnP ambiguity means both solutions have identical
            # reprojection errors.

        # Extract pan/tilt angles from rotation matrix
        pan_deg, tilt_deg = self._extract_pan_tilt_from_rotation(R)

        # Compute reprojection errors for quality assessment
        mean_error, max_error, _ = self._compute_reprojection_errors(
            object_points, image_points, rvec, tvec, inliers_mask
        )

        # Validate result quality using instance thresholds (configurable)
        is_valid = (
            mean_error <= self.max_reprojection_error and
            inlier_ratio >= self.min_inlier_ratio
        )

        if not is_valid:
            logger.warning(
                f"PnP result quality below threshold: "
                f"reprojection_error={mean_error:.2f}px (max {self.max_reprojection_error}), "
                f"inlier_ratio={inlier_ratio:.2%} (min {self.min_inlier_ratio:.0%})"
            )

        logger.info(
            f"PnP derivation {'succeeded' if is_valid else 'failed quality check'}: "
            f"position=[{camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f}]m, "
            f"pan={pan_deg:.1f}°, tilt={tilt_deg:.1f}°, "
            f"inliers={num_inliers}/{len(object_points)}, error={mean_error:.2f}px"
        )

        return PnPResult(
            success=is_valid,
            position=camera_position,
            rotation_matrix=R,
            rotation_vector=rvec.flatten(),
            pan_deg=pan_deg,
            tilt_deg=tilt_deg,
            reprojection_error_mean=mean_error,
            reprojection_error_max=max_error,
            inlier_ratio=inlier_ratio,
            num_inliers=num_inliers,
            inliers_mask=inliers_mask,
        )

    def _normalize_gcps(
        self,
        gcps: Union[List[GroundControlPoint], List[Dict[str, Any]], List[Tuple[float, float, float, float]]]
    ) -> List[GroundControlPoint]:
        """
        Normalize GCP input to list of GroundControlPoint objects.

        Supports multiple input formats for flexibility:
            - GroundControlPoint dataclass objects
            - Dicts with 'lat'/'latitude', 'lon'/'longitude', 'u', 'v' keys
            - Tuples of (latitude, longitude, u, v)

        Args:
            gcps: GCPs in any supported format

        Returns:
            List of GroundControlPoint objects

        Raises:
            ValueError: If GCP format is invalid or required fields are missing
        """
        normalized = []

        for i, gcp in enumerate(gcps):
            if isinstance(gcp, GroundControlPoint):
                normalized.append(gcp)
            elif isinstance(gcp, dict):
                # Extract latitude (support both 'lat' and 'latitude' keys)
                lat = gcp.get('lat', gcp.get('latitude'))
                if lat is None:
                    raise ValueError(f"GCP {i}: missing 'lat' or 'latitude' field")

                # Extract longitude (support both 'lon' and 'longitude' keys)
                lon = gcp.get('lon', gcp.get('longitude'))
                if lon is None:
                    raise ValueError(f"GCP {i}: missing 'lon' or 'longitude' field")

                # Extract pixel coordinates
                u = gcp.get('u')
                v = gcp.get('v')
                if u is None or v is None:
                    raise ValueError(f"GCP {i}: missing 'u' or 'v' field")

                normalized.append(GroundControlPoint(
                    latitude=float(lat),
                    longitude=float(lon),
                    u=float(u),
                    v=float(v)
                ))
            elif isinstance(gcp, (tuple, list)) and len(gcp) == 4:
                lat, lon, u, v = gcp
                normalized.append(GroundControlPoint(
                    latitude=float(lat),
                    longitude=float(lon),
                    u=float(u),
                    v=float(v)
                ))
            else:
                raise ValueError(
                    f"GCP {i}: invalid format. Expected GroundControlPoint, "
                    f"dict with lat/lon/u/v, or tuple (lat, lon, u, v), got {type(gcp)}"
                )

        return normalized
