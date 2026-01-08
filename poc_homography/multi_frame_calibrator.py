#!/usr/bin/env python3
"""
Multi-frame PTZ calibration using shared camera parameters across multiple frames.

This module extends the single-frame GCP calibrator to simultaneously optimize
camera parameters across multiple PTZ positions. Each frame has known pan_i and
tilt_i values from the PTZ API, while optimizing SHARED camera parameter deltas
(Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ) that apply to all frames.

Mathematical Model:
    For each frame i at PTZ position (pan_i, tilt_i), the rotation matrix is:

    R_i = R_mount × R_pan(pan_i + Δpan) × R_tilt(tilt_i + Δtilt) × R_roll(Δroll)

    Where:
    - pan_i, tilt_i: KNOWN PTZ encoder readings for frame i
    - Δpan, Δtilt, Δroll: SHARED calibration offsets (unknowns)
    - ΔX, ΔY, ΔZ: SHARED position offsets (unknowns)
    - R_mount: Fixed camera mount orientation (incorporated into R_pan/R_tilt)

    Global objective function minimizes reprojection error across all frames:

    E(p) = Σ_i Σ_j ρ(||e_i,j||²)

    Where:
    - p = [Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ] is the shared parameter vector
    - e_i,j = x_observed,i,j - x_predicted,i,j is the residual for GCP j in frame i
    - ρ() is a robust loss function (Huber or Cauchy)

Key Concepts:
    - Multi-frame calibration: Uses multiple PTZ positions to improve accuracy
    - Shared parameters: Same 6 parameter deltas apply to ALL frames
    - PTZ encoder readings: Known pan_i, tilt_i from camera API for each frame
    - Robust optimization: Handles outlier GCPs across frames
    - Regularization: Optional priors on parameter deviations

Regularization (Prior-Based Constraints):
    Extends single-frame regularization to multi-frame case:

    E_total(p) = E_reproj(p) + E_prior(p)

    Where:
    - E_reproj(p) = Σ_i Σ_j ρ(||e_i,j||²) is the multi-frame reprojection error
    - E_prior(p) = λ * ||p - p0||²_Σ is the prior penalty
    - p0 = [0, 0, 0, 0, 0, 0] is the initial estimate (no adjustment)
    - λ is the regularization weight

    The prior term prevents implausible parameter deviations when GCP data
    is sparse or noisy across frames.

Usage Example:
    >>> from poc_homography.camera_geometry import CameraGeometry
    >>> from poc_homography.multi_frame_calibrator import (
    ...     MultiFrameCalibrator, FrameObservation, MultiFrameGCP,
    ...     MultiFrameCalibrationData, PTZPosition
    ... )
    >>>
    >>> # Initial camera setup
    >>> geo = CameraGeometry(1920, 1080)
    >>> K = CameraGeometry.get_intrinsics(zoom_factor=10.0)
    >>> geo.set_camera_parameters(
    ...     K=K, w_pos=[0, 0, 5.0], pan_deg=0, tilt_deg=45,
    ...     map_width=640, map_height=640
    ... )
    >>>
    >>> # Define frames captured at different PTZ positions
    >>> frames = [
    ...     FrameObservation(
    ...         frame_id="frame_001",
    ...         timestamp=datetime.now(timezone.utc),
    ...         ptz_position=PTZPosition(pan=10.5, tilt=45.2, zoom=10.0),
    ...         image_path="/path/to/frame_001.jpg"
    ...     ),
    ...     FrameObservation(
    ...         frame_id="frame_002",
    ...         timestamp=datetime.now(timezone.utc),
    ...         ptz_position=PTZPosition(pan=25.3, tilt=50.1, zoom=10.0),
    ...         image_path="/path/to/frame_002.jpg"
    ...     ),
    ...     # ... more frames ...
    >>> ]
    >>>
    >>> # Define GCPs with observations across multiple frames
    >>> gcps = [
    ...     MultiFrameGCP(
    ...         gcp_id="gcp_001",
    ...         gps_lat=39.640444,
    ...         gps_lon=-0.230111,
    ...         frame_observations={
    ...             "frame_001": {"u": 960, "v": 540},
    ...             "frame_002": {"u": 1200, "v": 600}
    ...         }
    ...     ),
    ...     # ... more GCPs ...
    >>> ]
    >>>
    >>> # Create calibration data
    >>> calib_data = MultiFrameCalibrationData(
    ...     frames=frames,
    ...     gcps=gcps,
    ...     camera_config={"K": K, "w_pos": [0, 0, 5.0]}
    ... )
    >>>
    >>> # Calibrate shared parameters across all frames
    >>> calibrator = MultiFrameCalibrator(geo, calib_data, loss_function='huber')
    >>> result = calibrator.calibrate()
    >>>
    >>> print(f"Optimized parameters: {result.optimized_params}")
    >>> print(f"Initial error: {result.initial_error:.2f}px")
    >>> print(f"Final error: {result.final_error:.2f}px")
    >>> print(f"Per-frame RMS errors:")
    >>> for frame_id, rms_error in result.per_frame_errors.items():
    ...     print(f"  {frame_id}: {rms_error:.2f}px")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from poc_homography.types import Degrees, Meters, Unitless
import numpy as np
import logging
import copy

# Import scipy for optimization
from scipy.optimize import least_squares

# Import base calibration result for extension
from poc_homography.gcp_calibrator import CalibrationResult

# Import coordinate conversion for GPS to local metric
try:
    from poc_homography.coordinate_converter import gps_to_local_xy, UTMConverter
    UTM_CONVERTER_AVAILABLE = True
except ImportError:
    UTM_CONVERTER_AVAILABLE = False
    # Fallback if coordinate_converter is not available
    def gps_to_local_xy(ref_lat, ref_lon, lat, lon):
        """Simple equirectangular approximation for testing."""
        import math
        R_EARTH = 6371000  # meters
        ref_lat_rad = math.radians(ref_lat)
        x = math.radians(lon - ref_lon) * math.cos(ref_lat_rad) * R_EARTH
        y = math.radians(lat - ref_lat) * R_EARTH
        return x, y

logger = logging.getLogger(__name__)


@dataclass
class PTZPosition:
    """
    PTZ camera position for a single frame.

    Attributes:
        pan: Pan angle in degrees (from PTZ encoder)
        tilt: Tilt angle in degrees (from PTZ encoder)
        zoom: Zoom factor (unitless, typically 1.0-25.0)
    """
    pan: Degrees
    tilt: Degrees
    zoom: Unitless


@dataclass
class FrameObservation:
    """
    Observation data for a single frame captured at a specific PTZ position.

    Each frame represents a camera view at a known PTZ position with observed
    GCP pixel locations. Multiple frames at different PTZ positions enable
    multi-frame calibration.

    Attributes:
        frame_id: Unique identifier for this frame (e.g., "frame_001")
        timestamp: When the frame was captured (UTC)
        ptz_position: PTZ encoder readings (pan, tilt, zoom)
        image_path: Path to the captured image file
    """
    frame_id: str
    timestamp: datetime
    ptz_position: PTZPosition
    image_path: str


@dataclass
class MultiFrameGCP:
    """
    Ground Control Point with observations across multiple frames.

    A single GCP (fixed world location) may be visible in multiple camera frames
    at different PTZ positions. This class stores the GPS coordinates and the
    pixel observations in each frame where the GCP is visible.

    Attributes:
        gcp_id: Unique identifier for this GCP (e.g., "gcp_001")
        gps_lat: GPS latitude in degrees
        gps_lon: GPS longitude in degrees
        frame_observations: Dictionary mapping frame_id to pixel coordinates
                           Format: {"frame_id": {"u": float, "v": float}}
        utm_easting: Optional UTM easting coordinate (meters)
        utm_northing: Optional UTM northing coordinate (meters)
    """
    gcp_id: str
    gps_lat: Degrees
    gps_lon: Degrees
    frame_observations: Dict[str, Dict[str, float]]
    utm_easting: Optional[Meters] = None
    utm_northing: Optional[Meters] = None


@dataclass
class MultiFrameCalibrationData:
    """
    Complete dataset for multi-frame calibration.

    Contains all frames, GCPs, and camera configuration needed for multi-frame
    parameter optimization.

    Attributes:
        frames: List of frame observations at different PTZ positions
        gcps: List of GCPs with observations across frames
        camera_config: Dictionary with initial camera configuration
                      Keys: 'K', 'w_pos', and optional 'reference_lat', 'reference_lon'
    """
    frames: List[FrameObservation]
    gcps: List[MultiFrameGCP]
    camera_config: Dict[str, Any]


@dataclass
class MultiFrameCalibrationResult(CalibrationResult):
    """
    Results from multi-frame GCP-based calibration.

    Extends CalibrationResult with per-frame error metrics to track how well
    each frame's GCPs fit the optimized shared parameters.

    Additional Attributes:
        per_frame_errors: Dictionary mapping frame_id to RMS reprojection error (pixels)
        per_frame_inliers: Dictionary mapping frame_id to number of inlier GCPs
        per_frame_outliers: Dictionary mapping frame_id to number of outlier GCPs
        total_observations: Total number of GCP observations across all frames
    """
    per_frame_errors: Dict[str, float] = field(default_factory=dict)
    per_frame_inliers: Dict[str, int] = field(default_factory=dict)
    per_frame_outliers: Dict[str, int] = field(default_factory=dict)
    total_observations: int = 0


class MultiFrameCalibrator:
    """
    Multi-frame GCP-based reprojection error minimization calibrator.

    This class extends the single-frame GCPCalibrator to simultaneously optimize
    camera parameters across multiple frames captured at different PTZ positions.
    The optimization finds shared parameter adjustments (Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ)
    that minimize reprojection error for all GCPs across all frames.

    The key difference from single-frame calibration:
    - Each frame has KNOWN pan_i, tilt_i from PTZ API
    - Optimization solves for SHARED parameter deltas that apply to all frames
    - Residuals are computed across ALL frame-GCP pairs simultaneously

    Attributes:
        camera_geometry: Initial CameraGeometry instance (defines camera model)
        calibration_data: MultiFrameCalibrationData with frames and GCPs
        loss_function: Robust loss function name ('huber' or 'cauchy')
        loss_scale: Scale parameter for robust loss (pixels)
        prior_sigmas: Dictionary of prior standard deviations for regularization
        regularization_weight: Lambda parameter balancing reprojection vs prior penalty
    """

    # Supported loss functions
    LOSS_HUBER = 'huber'
    LOSS_CAUCHY = 'cauchy'
    VALID_LOSS_FUNCTIONS = [LOSS_HUBER, LOSS_CAUCHY]

    # Large residual value for points projecting to infinity (horizon)
    INFINITY_RESIDUAL = 1e6

    # Default parameter bounds (conservative ranges, same as GCPCalibrator)
    DEFAULT_BOUNDS = {
        'pan': (-10.0, 10.0),      # ±10 degrees
        'tilt': (-10.0, 10.0),     # ±10 degrees
        'roll': (-10.0, 10.0),     # ±10 degrees
        'X': (-5.0, 5.0),          # ±5 meters
        'Y': (-5.0, 5.0),          # ±5 meters
        'Z': (-5.0, 5.0),          # ±5 meters
    }

    # Default prior standard deviations for regularization
    DEFAULT_PRIOR_SIGMAS: Dict[str, float] = {
        'gps_position_m': 10.0,    # X, Y position uncertainty (meters)
        'height_m': 2.0,           # Z position uncertainty (meters)
        'pan_deg': 3.0,            # Pan angle uncertainty (degrees)
        'tilt_deg': 3.0,           # Tilt angle uncertainty (degrees)
        'roll_deg': 1.0,           # Roll angle uncertainty (degrees)
    }

    def __init__(
        self,
        camera_geometry: 'CameraGeometry',
        calibration_data: MultiFrameCalibrationData,
        loss_function: str = 'huber',
        loss_scale: float = 1.0,
        reference_lat: Optional[Degrees] = None,
        reference_lon: Optional[Degrees] = None,
        utm_crs: Optional[str] = None,
        prior_sigmas: Optional[Dict[str, float]] = None,
        regularization_weight: float = 1.0
    ):
        """
        Initialize multi-frame GCP-based calibrator.

        Args:
            camera_geometry: Initial CameraGeometry instance with camera parameters
            calibration_data: MultiFrameCalibrationData with frames, GCPs, and config
            loss_function: Robust loss function to use ('huber' or 'cauchy')
            loss_scale: Scale parameter for robust loss (pixels)
            reference_lat: Reference latitude for GPS-to-local conversion (camera position).
                          If None, uses camera config or GCP centroid.
            reference_lon: Reference longitude for GPS-to-local conversion (camera position).
                          If None, uses camera config or GCP centroid.
            utm_crs: UTM coordinate reference system (e.g., "EPSG:25830")
            prior_sigmas: Dictionary of prior standard deviations for regularization.
                         Same keys as GCPCalibrator.DEFAULT_PRIOR_SIGMAS.
            regularization_weight: Lambda parameter (>= 0.0) balancing reprojection error
                                  vs prior penalty. Default 1.0.

        Raises:
            ValueError: If parameters are invalid (empty frames/GCPs, invalid loss function, etc.)
        """
        # Validate loss function
        if loss_function.lower() not in self.VALID_LOSS_FUNCTIONS:
            raise ValueError(
                f"Invalid loss_function '{loss_function}'. "
                f"Must be one of {self.VALID_LOSS_FUNCTIONS}"
            )

        # Validate frames
        if not calibration_data.frames:
            raise ValueError("calibration_data.frames list cannot be empty")

        # Validate GCPs
        if not calibration_data.gcps:
            raise ValueError("calibration_data.gcps list cannot be empty")

        # Validate frame structure
        frame_ids = set()
        for i, frame in enumerate(calibration_data.frames):
            if not isinstance(frame, FrameObservation):
                raise ValueError(f"Frame at index {i} must be a FrameObservation instance")
            if frame.frame_id in frame_ids:
                raise ValueError(f"Duplicate frame_id '{frame.frame_id}' at index {i}")
            frame_ids.add(frame.frame_id)

        # Validate GCP structure
        for i, gcp in enumerate(calibration_data.gcps):
            if not isinstance(gcp, MultiFrameGCP):
                raise ValueError(f"GCP at index {i} must be a MultiFrameGCP instance")
            if not gcp.frame_observations:
                raise ValueError(f"GCP '{gcp.gcp_id}' has no frame observations")

            # Check that frame_ids in observations exist in frames list
            for frame_id in gcp.frame_observations.keys():
                if frame_id not in frame_ids:
                    raise ValueError(
                        f"GCP '{gcp.gcp_id}' references unknown frame_id '{frame_id}'"
                    )

        # Validate loss_scale
        if loss_scale <= 0:
            raise ValueError(f"loss_scale must be positive, got {loss_scale}")

        # Validate regularization_weight
        if not np.isfinite(regularization_weight) or regularization_weight < 0.0:
            raise ValueError(
                f"regularization_weight must be >= 0.0 and finite, got {regularization_weight}"
            )

        # Process prior_sigmas: merge with defaults
        if prior_sigmas is None:
            merged_sigmas = self.DEFAULT_PRIOR_SIGMAS.copy()
        else:
            merged_sigmas = self.DEFAULT_PRIOR_SIGMAS.copy()
            merged_sigmas.update(prior_sigmas)

        # Validate prior_sigmas values
        for key, value in merged_sigmas.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"prior_sigmas['{key}'] must be positive and finite, got {value}"
                )

        # Store configuration
        self.camera_geometry = camera_geometry
        self.calibration_data = calibration_data
        self.loss_function = loss_function.lower()
        self.loss_scale = loss_scale
        self._prior_sigmas = merged_sigmas
        self._regularization_weight = regularization_weight

        # Convert prior_sigmas to internal numpy array format
        self._sigma_vector = np.array([
            merged_sigmas['pan_deg'],
            merged_sigmas['tilt_deg'],
            merged_sigmas['roll_deg'],
            merged_sigmas['gps_position_m'],
            merged_sigmas['gps_position_m'],
            merged_sigmas['height_m'],
        ], dtype=np.float64)

        # Set reference GPS coordinates
        if reference_lat is not None and reference_lon is not None:
            self._reference_lat = reference_lat
            self._reference_lon = reference_lon
            logger.info(f"Using camera position as reference: ({reference_lat:.6f}, {reference_lon:.6f})")
        elif 'reference_lat' in calibration_data.camera_config and \
             'reference_lon' in calibration_data.camera_config:
            self._reference_lat = calibration_data.camera_config['reference_lat']
            self._reference_lon = calibration_data.camera_config['reference_lon']
            logger.info(f"Using config reference: ({self._reference_lat:.6f}, {self._reference_lon:.6f})")
        else:
            # Fallback to GCP centroid
            gps_lats = [gcp.gps_lat for gcp in calibration_data.gcps]
            gps_lons = [gcp.gps_lon for gcp in calibration_data.gcps]
            self._reference_lat = np.mean(gps_lats)
            self._reference_lon = np.mean(gps_lons)
            logger.warning(
                f"No reference coordinates provided, using GCP centroid ({self._reference_lat:.6f}, {self._reference_lon:.6f}). "
                "Pass camera GPS coordinates for better accuracy."
            )

        # Set up UTM converter if available
        self._utm_converter = None
        if utm_crs and UTM_CONVERTER_AVAILABLE:
            try:
                self._utm_converter = UTMConverter(utm_crs)
                self._utm_converter.set_reference(self._reference_lat, self._reference_lon)
                logger.info(f"Using UTM converter with CRS {utm_crs}")
            except Exception as e:
                logger.warning(f"Failed to initialize UTM converter: {e}")
                self._utm_converter = None

        # Convert GCPs to world coordinates (cached for efficiency)
        self._world_coords = {}  # Dict[gcp_id, (x, y)]
        utm_count = 0
        gps_count = 0

        for gcp in calibration_data.gcps:
            # Try UTM coordinates first
            if self._utm_converter and gcp.utm_easting is not None and gcp.utm_northing is not None:
                x, y = self._utm_converter.utm_to_local_xy(gcp.utm_easting, gcp.utm_northing)
                utm_count += 1
            else:
                # Fall back to GPS conversion
                if self._utm_converter:
                    x, y = self._utm_converter.gps_to_local_xy(gcp.gps_lat, gcp.gps_lon)
                else:
                    x, y = gps_to_local_xy(self._reference_lat, self._reference_lon,
                                          gcp.gps_lat, gcp.gps_lon)
                gps_count += 1

            self._world_coords[gcp.gcp_id] = (x, y)

        if utm_count > 0:
            logger.info(f"MultiFrameCalibrator: Using {utm_count} UTM coordinates, {gps_count} GPS coordinates")

        # Count total observations
        self._total_observations = sum(
            len(gcp.frame_observations) for gcp in calibration_data.gcps
        )

        logger.info(
            f"MultiFrameCalibrator initialized with {len(calibration_data.frames)} frames, "
            f"{len(calibration_data.gcps)} GCPs, {self._total_observations} total observations, "
            f"loss={self.loss_function}, scale={self.loss_scale}, "
            f"regularization_weight={self._regularization_weight}"
        )

    def _compute_predicted_homography_for_frame(
        self,
        params: np.ndarray,
        frame: FrameObservation
    ) -> np.ndarray:
        """
        Compute predicted homography for a specific frame with parameter adjustments.

        For multi-frame calibration, each frame has known PTZ encoder readings
        (pan_i, tilt_i) from the camera API. The homography for frame i is computed
        using the frame's PTZ position plus the shared parameter deltas being optimized.

        Args:
            params: Parameter vector [Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ]
                   Units: [degrees, degrees, degrees, meters, meters, meters]
            frame: FrameObservation with PTZ position for this frame

        Returns:
            H_pred: 3x3 homography matrix mapping world coordinates to image pixels
        """
        # Extract parameter increments
        delta_pan, delta_tilt, delta_roll, delta_x, delta_y, delta_z = params

        # Compute updated camera parameters for this frame:
        # - Use frame's PTZ encoder readings as base
        # - Add shared parameter deltas being optimized
        updated_pan = frame.ptz_position.pan + delta_pan
        updated_tilt = frame.ptz_position.tilt + delta_tilt
        # Roll is not typically available from PTZ API, so it's purely a calibration offset
        # updated_roll = 0.0 + delta_roll (roll not currently used in CameraGeometry)

        # Position adjustment (same for all frames)
        updated_pos = self.camera_geometry.w_pos + np.array([delta_x, delta_y, delta_z])

        # Create temporary CameraGeometry with updated parameters for this frame
        temp_geo = copy.copy(self.camera_geometry)
        temp_geo.set_camera_parameters(
            K=self.camera_geometry.K,
            w_pos=updated_pos,
            pan_deg=updated_pan,
            tilt_deg=updated_tilt,
            map_width=self.camera_geometry.map_width,
            map_height=self.camera_geometry.map_height,
            roll_deg=delta_roll  # Roll offset applied directly
        )

        return temp_geo.H

    def _compute_residuals(self, params: np.ndarray) -> np.ndarray:
        """
        Compute reprojection residuals across all frames and GCPs.

        For multi-frame calibration, residuals are computed for every GCP observation
        in every frame where that GCP is visible. The residuals are concatenated into
        a single flat array.

        Residual ordering:
            For frame_1: [Δu_gcp1, Δv_gcp1, Δu_gcp2, Δv_gcp2, ...]
            For frame_2: [Δu_gcp1, Δv_gcp1, Δu_gcp3, Δv_gcp3, ...]
            ...
            All frames concatenated sequentially

        Args:
            params: Parameter vector [Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ]

        Returns:
            residuals: Flattened array of length 2N for N total observations
                      Format: [Δu_1, Δv_1, Δu_2, Δv_2, ..., Δu_N, Δv_N]
        """
        residuals_list = []

        # Process each frame
        for frame in self.calibration_data.frames:
            # Compute predicted homography for this frame
            H_pred = self._compute_predicted_homography_for_frame(params, frame)

            # Process each GCP that has an observation in this frame
            for gcp in self.calibration_data.gcps:
                if frame.frame_id not in gcp.frame_observations:
                    continue  # GCP not visible in this frame

                # Get observed pixel coordinates for this GCP in this frame
                obs = gcp.frame_observations[frame.frame_id]
                u_observed = obs['u']
                v_observed = obs['v']

                # Get world coordinates for this GCP
                x_world, y_world = self._world_coords[gcp.gcp_id]

                # Project world point to camera pixel using predicted homography
                world_point_homogeneous = np.array([x_world, y_world, 1.0])
                predicted_homogeneous = H_pred @ world_point_homogeneous

                # Normalize homogeneous coordinates
                if abs(predicted_homogeneous[2]) < 1e-10:
                    # Point at infinity (horizon)
                    residuals_list.extend([self.INFINITY_RESIDUAL, self.INFINITY_RESIDUAL])
                    logger.warning(
                        f"GCP '{gcp.gcp_id}' in frame '{frame.frame_id}' projects to infinity"
                    )
                    continue

                u_predicted = predicted_homogeneous[0] / predicted_homogeneous[2]
                v_predicted = predicted_homogeneous[1] / predicted_homogeneous[2]

                # Compute residuals (observed - predicted)
                residuals_list.append(u_observed - u_predicted)
                residuals_list.append(v_observed - v_predicted)

        return np.array(residuals_list, dtype=np.float64)

    def _compute_regularization_residuals(self, params: np.ndarray) -> np.ndarray:
        """
        Compute regularization residuals for Tikhonov regularization.

        Same formulation as GCPCalibrator:
            r_prior[j] = sqrt(lambda) * params[j] / sigma[j]

        Args:
            params: Parameter vector [Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ]

        Returns:
            Regularization residuals array of shape (6,)
        """
        if self._regularization_weight == 0.0:
            return np.zeros(6, dtype=np.float64)

        sqrt_lambda = np.sqrt(self._regularization_weight)
        regularization_residuals = sqrt_lambda * params / self._sigma_vector

        return regularization_residuals

    def _compute_rms_error(self, residuals: np.ndarray) -> float:
        """
        Compute RMS reprojection error from residuals.

        Args:
            residuals: Flattened residual array [Δu_1, Δv_1, Δu_2, Δv_2, ...]

        Returns:
            RMS error in pixels
        """
        residuals_2d = residuals.reshape(-1, 2)
        per_point_errors = np.linalg.norm(residuals_2d, axis=1)
        return np.sqrt(np.mean(per_point_errors**2))

    def _compute_per_frame_errors(self, params: np.ndarray) -> Dict[str, float]:
        """
        Compute RMS error for each frame separately.

        Args:
            params: Parameter vector [Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ]

        Returns:
            Dictionary mapping frame_id to RMS error (pixels)
        """
        per_frame_errors = {}

        for frame in self.calibration_data.frames:
            # Compute predicted homography for this frame
            H_pred = self._compute_predicted_homography_for_frame(params, frame)

            frame_residuals = []

            # Process each GCP visible in this frame
            for gcp in self.calibration_data.gcps:
                if frame.frame_id not in gcp.frame_observations:
                    continue

                obs = gcp.frame_observations[frame.frame_id]
                u_observed = obs['u']
                v_observed = obs['v']

                x_world, y_world = self._world_coords[gcp.gcp_id]

                world_point_homogeneous = np.array([x_world, y_world, 1.0])
                predicted_homogeneous = H_pred @ world_point_homogeneous

                if abs(predicted_homogeneous[2]) < 1e-10:
                    frame_residuals.extend([self.INFINITY_RESIDUAL, self.INFINITY_RESIDUAL])
                    continue

                u_predicted = predicted_homogeneous[0] / predicted_homogeneous[2]
                v_predicted = predicted_homogeneous[1] / predicted_homogeneous[2]

                frame_residuals.append(u_observed - u_predicted)
                frame_residuals.append(v_observed - v_predicted)

            # Compute RMS for this frame
            if frame_residuals:
                frame_residuals_array = np.array(frame_residuals, dtype=np.float64)
                per_frame_errors[frame.frame_id] = self._compute_rms_error(frame_residuals_array)
            else:
                per_frame_errors[frame.frame_id] = 0.0

        return per_frame_errors

    def _compute_per_frame_inliers_outliers(
        self,
        params: np.ndarray,
        inlier_threshold: float
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Compute number of inliers and outliers for each frame.

        Args:
            params: Parameter vector
            inlier_threshold: Error threshold for inlier classification (pixels)

        Returns:
            Tuple of (per_frame_inliers, per_frame_outliers) dictionaries
        """
        per_frame_inliers = {}
        per_frame_outliers = {}

        for frame in self.calibration_data.frames:
            H_pred = self._compute_predicted_homography_for_frame(params, frame)

            frame_errors = []

            for gcp in self.calibration_data.gcps:
                if frame.frame_id not in gcp.frame_observations:
                    continue

                obs = gcp.frame_observations[frame.frame_id]
                u_observed = obs['u']
                v_observed = obs['v']

                x_world, y_world = self._world_coords[gcp.gcp_id]

                world_point_homogeneous = np.array([x_world, y_world, 1.0])
                predicted_homogeneous = H_pred @ world_point_homogeneous

                if abs(predicted_homogeneous[2]) < 1e-10:
                    frame_errors.append(self.INFINITY_RESIDUAL)
                    continue

                u_predicted = predicted_homogeneous[0] / predicted_homogeneous[2]
                v_predicted = predicted_homogeneous[1] / predicted_homogeneous[2]

                error = np.sqrt((u_observed - u_predicted)**2 + (v_observed - v_predicted)**2)
                frame_errors.append(error)

            # Classify inliers/outliers
            frame_errors_array = np.array(frame_errors)
            inlier_mask = frame_errors_array <= inlier_threshold
            per_frame_inliers[frame.frame_id] = int(np.sum(inlier_mask))
            per_frame_outliers[frame.frame_id] = int(np.sum(~inlier_mask))

        return per_frame_inliers, per_frame_outliers

    def calibrate(
        self,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> MultiFrameCalibrationResult:
        """
        Calibrate shared camera parameters using multi-frame GCP observations.

        This method performs robust least-squares optimization to find the shared
        parameter adjustments (Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ) that minimize
        reprojection error across all GCP observations in all frames.

        The optimization simultaneously processes all frames, using each frame's
        known PTZ position (pan_i, tilt_i) and solving for shared parameter deltas
        that best explain all observations.

        Args:
            bounds: Optional parameter bounds dictionary. Keys are parameter names
                   ('pan', 'tilt', 'roll', 'X', 'Y', 'Z'), values are (lower, upper)
                   tuples. If None, uses DEFAULT_BOUNDS.

        Returns:
            MultiFrameCalibrationResult with optimized parameters and per-frame statistics

        Raises:
            RuntimeError: If optimization fails to converge

        Example:
            >>> calibrator = MultiFrameCalibrator(camera_geo, calib_data)
            >>> result = calibrator.calibrate(bounds={'pan': (-5, 5), 'tilt': (-5, 5)})
            >>> print(f"Pan adjustment: {result.optimized_params[0]:.2f}°")
            >>> for frame_id, rms_error in result.per_frame_errors.items():
            ...     print(f"{frame_id}: {rms_error:.2f}px")
        """
        # Use default bounds if not provided
        if bounds is None:
            bounds = self.DEFAULT_BOUNDS.copy()
        else:
            merged_bounds = self.DEFAULT_BOUNDS.copy()
            merged_bounds.update(bounds)
            bounds = merged_bounds

        # Extract bounds in parameter vector order
        lower_bounds = np.array([
            bounds['pan'][0],
            bounds['tilt'][0],
            bounds['roll'][0],
            bounds['X'][0],
            bounds['Y'][0],
            bounds['Z'][0]
        ])
        upper_bounds = np.array([
            bounds['pan'][1],
            bounds['tilt'][1],
            bounds['roll'][1],
            bounds['X'][1],
            bounds['Y'][1],
            bounds['Z'][1]
        ])

        # Initial parameter guess: all zeros (no adjustment)
        params_initial = np.zeros(6)

        # Compute initial error
        residuals_initial = self._compute_residuals(params_initial)
        initial_error = self._compute_rms_error(residuals_initial)

        logger.info(
            f"Starting multi-frame calibration: {len(self.calibration_data.frames)} frames, "
            f"{self._total_observations} observations, initial RMS error = {initial_error:.3f}px"
        )

        # Map loss function to scipy
        scipy_loss_map = {
            'huber': 'soft_l1',
            'cauchy': 'cauchy'
        }
        scipy_loss = scipy_loss_map.get(self.loss_function, 'linear')

        # Define residual function for optimization
        def residual_fn(params):
            reproj_residuals = self._compute_residuals(params)

            # Append regularization residuals if enabled
            if self._regularization_weight > 0.0:
                reg_residuals = self._compute_regularization_residuals(params)
                return np.concatenate([reproj_residuals, reg_residuals])

            return reproj_residuals

        # Run least-squares optimization
        try:
            result = least_squares(
                fun=residual_fn,
                x0=params_initial,
                bounds=(lower_bounds, upper_bounds),
                loss=scipy_loss,
                f_scale=self.loss_scale,
                method='trf',
                verbose=0,
                max_nfev=1000
            )
        except Exception as e:
            logger.error(f"Multi-frame optimization failed: {e}")
            raise RuntimeError(f"Multi-frame calibration optimization failed: {e}")

        # Extract optimized parameters
        optimized_params = result.x

        # Compute final error
        residuals_final = self._compute_residuals(optimized_params)
        final_error = self._compute_rms_error(residuals_final)

        # Compute per-frame errors
        per_frame_errors = self._compute_per_frame_errors(optimized_params)

        # Compute per-GCP errors across all frames (for compatibility with base class)
        # Create a flat list of errors for all observations
        residuals_2d = residuals_final.reshape(-1, 2)
        per_observation_errors = np.linalg.norm(residuals_2d, axis=1).tolist()

        # Classify inliers/outliers
        inlier_threshold = self.loss_scale * 2.0
        inlier_mask = np.array(per_observation_errors) <= inlier_threshold
        num_inliers = int(np.sum(inlier_mask))
        num_outliers = self._total_observations - num_inliers
        inlier_ratio = num_inliers / self._total_observations

        # Compute per-frame inliers/outliers
        per_frame_inliers, per_frame_outliers = self._compute_per_frame_inliers_outliers(
            optimized_params, inlier_threshold
        )

        # Build convergence info
        convergence_info = {
            'success': result.success,
            'message': result.message,
            'iterations': result.nfev,
            'function_evals': result.nfev,
            'optimality': result.optimality,
            'status': result.status
        }

        if hasattr(result, 'njev'):
            convergence_info['jacobian_evals'] = result.njev

        logger.info(
            f"Multi-frame calibration complete: final RMS error = {final_error:.3f}px "
            f"({num_inliers}/{self._total_observations} inliers, {result.nfev} iterations)"
        )

        # Log per-frame errors
        for frame_id, rms_error in per_frame_errors.items():
            logger.info(f"  {frame_id}: {rms_error:.3f}px")

        if not result.success:
            logger.warning(f"Optimization did not fully converge: {result.message}")

        # Calculate regularization penalty if enabled
        if self._regularization_weight > 0.0:
            reg_residuals = self._compute_regularization_residuals(optimized_params)
            regularization_penalty = float(np.sum(reg_residuals ** 2))
        else:
            regularization_penalty = None

        return MultiFrameCalibrationResult(
            optimized_params=optimized_params,
            initial_error=initial_error,
            final_error=final_error,
            num_inliers=num_inliers,
            num_outliers=num_outliers,
            inlier_ratio=inlier_ratio,
            per_gcp_errors=per_observation_errors,
            convergence_info=convergence_info,
            regularization_penalty=regularization_penalty,
            per_frame_errors=per_frame_errors,
            per_frame_inliers=per_frame_inliers,
            per_frame_outliers=per_frame_outliers,
            total_observations=self._total_observations
        )
