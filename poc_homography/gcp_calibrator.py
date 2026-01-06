#!/usr/bin/env python3
"""
GCP-based reprojection error minimization calibration.

This module implements automatic camera parameter calibration by minimizing
reprojection error between observed Ground Control Points (GCPs) and predicted
pixel locations. The calibration optimizes camera orientation (pan, tilt, roll)
and position (X, Y, Z) to find the best fit to observed GCP data.

Mathematical Model:
    For parameter vector p = {Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ}, minimize:

    E(p) = Σᵢ ρ(||eᵢ||²)

    where:
    - eᵢ = x_observed,i - x_predicted,i is the residual for GCP i
    - x_observed,i is the observed camera pixel location [u_cam, v_cam]ᵀ
    - x_predicted,i = π(H_pred(p) × x_world,i) is the predicted camera pixel
    - x_world,i is the world coordinate [X_world, Y_world]ᵀ from GPS
    - π() is the perspective projection (homogeneous coordinate normalization)
    - ρ() is a robust loss function (Huber or Cauchy) to handle outliers
    - H_pred(p) is the predicted homography from updated camera parameters

Key Concepts:
    - GCP (Ground Control Point): Known correspondence between GPS and image pixel
    - Reprojection error: Distance between observed and predicted pixel locations
    - Robust loss: Reduces influence of outlier GCPs on optimization
    - Parameter increments: Optimization adjusts camera parameters incrementally

Usage Example:
    >>> from poc_homography.camera_geometry import CameraGeometry
    >>>
    >>> # Initial camera setup
    >>> geo = CameraGeometry(1920, 1080)
    >>> K = CameraGeometry.get_intrinsics(zoom_factor=10.0)
    >>> geo.set_camera_parameters(
    ...     K=K, w_pos=[0, 0, 5.0], pan_deg=0, tilt_deg=45,
    ...     map_width=640, map_height=640
    ... )
    >>>
    >>> # GCPs with observed camera pixels and GPS coordinates
    >>> gcps = [
    ...     {'gps': {'latitude': 39.640444, 'longitude': -0.230111},
    ...      'image': {'u': 960, 'v': 540}},
    ...     # ... more GCPs ...
    ... ]
    >>>
    >>> # Calibrate camera parameters
    >>> calibrator = GCPCalibrator(geo, gcps, loss_function='huber')
    >>> result = calibrator.calibrate()
    >>>
    >>> print(f"Optimized parameters: {result.optimized_params}")
    >>> print(f"Initial error: {result.initial_error:.2f}px")
    >>> print(f"Final error: {result.final_error:.2f}px")
    >>> print(f"Inliers: {result.num_inliers}/{result.num_inliers + result.num_outliers}")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
import copy

# Import scipy for optimization
from scipy.optimize import least_squares

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
        # Convert degree differences to radians, then scale by Earth radius
        x = math.radians(lon - ref_lon) * math.cos(ref_lat_rad) * R_EARTH
        y = math.radians(lat - ref_lat) * R_EARTH
        return x, y

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """
    Results from GCP-based reprojection error calibration.

    Contains the optimized camera parameters along with statistical information
    about the calibration quality, convergence, and per-GCP error analysis.

    Attributes:
        optimized_params: Final parameter vector [Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ]
                         in units [deg, deg, deg, m, m, m]
        initial_error: RMS reprojection error before optimization (pixels)
        final_error: RMS reprojection error after optimization (pixels)
        num_inliers: Number of inlier GCPs (within robust loss threshold)
        num_outliers: Number of outlier GCPs (rejected by robust loss)
        inlier_ratio: Ratio of inliers to total GCPs [0.0, 1.0]
        per_gcp_errors: Individual reprojection error for each GCP (pixels)
        convergence_info: Dictionary with optimizer details:
            - 'success': bool, whether optimization converged
            - 'message': str, optimizer status message
            - 'iterations': int, number of iterations
            - 'function_evals': int, number of function evaluations
            - 'jacobian_evals': int, number of Jacobian evaluations (if available)
            - 'optimality': float, first-order optimality measure
        timestamp: When the calibration was performed (UTC)
    """
    optimized_params: np.ndarray
    initial_error: float
    final_error: float
    num_inliers: int
    num_outliers: int
    inlier_ratio: float
    per_gcp_errors: List[float]
    convergence_info: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class GCPCalibrator:
    """
    GCP-based reprojection error minimization calibrator.

    This class optimizes camera parameters (pan, tilt, roll, position) by
    minimizing the reprojection error between observed GCP pixel locations
    and predicted pixel locations from the camera model.

    The optimization uses scipy.optimize.least_squares with robust loss functions
    to handle outlier GCPs gracefully.

    Attributes:
        camera_geometry: Initial CameraGeometry instance (defines camera model)
        gcps: List of Ground Control Points with 'gps' and 'image' keys
        loss_function: Robust loss function name ('huber' or 'cauchy')
        loss_scale: Scale parameter for robust loss (pixels)
    """

    # Supported loss functions
    LOSS_HUBER = 'huber'
    LOSS_CAUCHY = 'cauchy'
    VALID_LOSS_FUNCTIONS = [LOSS_HUBER, LOSS_CAUCHY]

    # Large residual value for points projecting to infinity (horizon)
    # This value is intentionally large to penalize such configurations during optimization
    INFINITY_RESIDUAL = 1e6

    # Default parameter bounds (conservative ranges)
    DEFAULT_BOUNDS = {
        'pan': (-10.0, 10.0),      # ±10 degrees
        'tilt': (-10.0, 10.0),     # ±10 degrees
        'roll': (-10.0, 10.0),     # ±10 degrees (reserved for future CameraGeometry support)
        'X': (-5.0, 5.0),          # ±5 meters
        'Y': (-5.0, 5.0),          # ±5 meters
        'Z': (-5.0, 5.0),          # ±5 meters
    }

    def __init__(
        self,
        camera_geometry: 'CameraGeometry',
        gcps: List[Dict[str, Any]],
        loss_function: str = 'huber',
        loss_scale: float = 1.0,
        reference_lat: Optional[float] = None,
        reference_lon: Optional[float] = None,
        utm_crs: Optional[str] = None
    ):
        """
        Initialize GCP-based calibrator.

        Args:
            camera_geometry: Initial CameraGeometry instance with camera parameters
            gcps: List of Ground Control Point dictionaries, each with:
                - 'gps': {'latitude': float, 'longitude': float}
                - 'image': {'u': float, 'v': float}  # Camera pixel coordinates
            loss_function: Robust loss function to use ('huber' or 'cauchy')
            loss_scale: Scale parameter for robust loss (pixels). Determines the
                       threshold at which errors are considered outliers.
                       - For 'huber': transition from quadratic to linear at loss_scale
                       - For 'cauchy': scale of the Cauchy distribution
            reference_lat: Reference latitude for GPS-to-local conversion (should be camera lat).
                          If None, uses GCP centroid (not recommended - causes coordinate mismatch).
            reference_lon: Reference longitude for GPS-to-local conversion (should be camera lon).
                          If None, uses GCP centroid (not recommended - causes coordinate mismatch).
            utm_crs: UTM coordinate reference system (e.g., "EPSG:25830"). If provided and GCPs
                    have UTM coordinates, uses UTM for more accurate local XY conversion.

        Raises:
            ValueError: If parameters are invalid (empty GCPs, invalid loss function, etc.)
        """
        # Validate loss function
        if loss_function.lower() not in self.VALID_LOSS_FUNCTIONS:
            raise ValueError(
                f"Invalid loss_function '{loss_function}'. "
                f"Must be one of {self.VALID_LOSS_FUNCTIONS}"
            )

        # Validate GCPs
        if not gcps:
            raise ValueError("gcps list cannot be empty")

        # Validate GCP structure (basic check, detailed validation in gcp_validation.py)
        for i, gcp in enumerate(gcps):
            if not isinstance(gcp, dict):
                raise ValueError(f"GCP at index {i} must be a dictionary")
            if 'gps' not in gcp:
                raise ValueError(f"GCP at index {i} missing required 'gps' key")
            if 'image' not in gcp:
                raise ValueError(f"GCP at index {i} missing required 'image' key")

            # Check GPS keys
            gps = gcp['gps']
            if 'latitude' not in gps or 'longitude' not in gps:
                raise ValueError(
                    f"GCP at index {i}: 'gps' must have 'latitude' and 'longitude' keys"
                )

            # Check image keys
            image = gcp['image']
            if 'u' not in image or 'v' not in image:
                raise ValueError(
                    f"GCP at index {i}: 'image' must have 'u' and 'v' keys"
                )

        # Validate loss_scale
        if loss_scale <= 0:
            raise ValueError(f"loss_scale must be positive, got {loss_scale}")

        # Store configuration
        self.camera_geometry = camera_geometry
        self.gcps = gcps
        self.loss_function = loss_function.lower()
        self.loss_scale = loss_scale

        # Set reference GPS coordinates for local coordinate conversion
        # IMPORTANT: This should be the camera position so that world coordinates
        # are relative to the camera (camera at origin), matching CameraGeometry expectations
        if reference_lat is not None and reference_lon is not None:
            self._reference_lat = reference_lat
            self._reference_lon = reference_lon
            logger.info(f"Using camera position as reference: ({reference_lat:.6f}, {reference_lon:.6f})")
        else:
            # Fallback to GCP centroid (NOT recommended - will cause coordinate mismatch)
            gps_lats = [gcp['gps']['latitude'] for gcp in gcps]
            gps_lons = [gcp['gps']['longitude'] for gcp in gcps]
            self._reference_lat = np.mean(gps_lats)
            self._reference_lon = np.mean(gps_lons)
            logger.warning(
                f"No reference coordinates provided, using GCP centroid ({self._reference_lat:.6f}, {self._reference_lon:.6f}). "
                "This may cause large errors - pass camera GPS coordinates as reference_lat/reference_lon."
            )

        # Set up UTM converter if available and CRS provided
        self._utm_converter = None
        if utm_crs and UTM_CONVERTER_AVAILABLE:
            try:
                self._utm_converter = UTMConverter(utm_crs)
                self._utm_converter.set_reference(self._reference_lat, self._reference_lon)
                logger.info(f"Using UTM converter with CRS {utm_crs} for accurate coordinate conversion")
            except Exception as e:
                logger.warning(f"Failed to initialize UTM converter: {e}, falling back to GPS conversion")
                self._utm_converter = None

        # Convert GCPs to world coordinates (cached for efficiency)
        # PRIORITY: UTM coordinates > GPS coordinates
        self._world_coords = []
        utm_count = 0
        gps_count = 0
        for gcp in gcps:
            # Try UTM coordinates first (more accurate)
            if self._utm_converter and 'utm' in gcp and gcp['utm']:
                utm = gcp['utm']
                if utm.get('easting') is not None and utm.get('northing') is not None:
                    x, y = self._utm_converter.utm_to_local_xy(utm['easting'], utm['northing'])
                    self._world_coords.append([x, y])
                    utm_count += 1
                    continue

            # Fall back to GPS conversion
            lat = gcp['gps']['latitude']
            lon = gcp['gps']['longitude']
            if self._utm_converter:
                x, y = self._utm_converter.gps_to_local_xy(lat, lon)
            else:
                x, y = gps_to_local_xy(self._reference_lat, self._reference_lon, lat, lon)
            self._world_coords.append([x, y])
            gps_count += 1
        self._world_coords = np.array(self._world_coords, dtype=np.float64)

        # Log coordinate source breakdown
        if utm_count > 0:
            print(f"GCPCalibrator: Using {utm_count} UTM coordinates, {gps_count} GPS coordinates")

        # DEBUG: Print first GCP's local XY for verification
        if len(self._world_coords) > 0:
            print(f"GCPCalibrator DEBUG: First GCP local XY = ({self._world_coords[0][0]:.2f}, {self._world_coords[0][1]:.2f}) meters")

        logger.info(
            f"GCPCalibrator initialized with {len(gcps)} GCPs, "
            f"loss={self.loss_function}, scale={self.loss_scale}"
        )

    def _compute_predicted_homography(self, params: np.ndarray) -> np.ndarray:
        """
        Compute predicted homography H_pred from parameter vector.

        Creates a temporary CameraGeometry instance with updated camera parameters
        (initial parameters + deltas from params vector) and returns the computed
        homography matrix.

        Args:
            params: Parameter vector [Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ]
                   Units: [degrees, degrees, degrees, meters, meters, meters]

        Returns:
            H_pred: 3x3 homography matrix mapping world coordinates to image pixels

        Note:
            Roll parameter (params[2]) is currently unused since CameraGeometry
            assumes zero roll. It is included for future extensibility.
        """
        # Extract parameter increments
        delta_pan, delta_tilt, delta_roll, delta_x, delta_y, delta_z = params

        # Compute updated camera parameters (initial + deltas)
        updated_pan = self.camera_geometry.pan_deg + delta_pan
        updated_tilt = self.camera_geometry.tilt_deg + delta_tilt
        # Note: roll is currently not supported in CameraGeometry, so delta_roll is unused
        updated_pos = self.camera_geometry.w_pos + np.array([delta_x, delta_y, delta_z])

        # Create temporary CameraGeometry with updated parameters
        # We need to create a fresh instance to avoid modifying the original
        temp_geo = copy.copy(self.camera_geometry)
        temp_geo.set_camera_parameters(
            K=self.camera_geometry.K,
            w_pos=updated_pos,
            pan_deg=updated_pan,
            tilt_deg=updated_tilt,
            map_width=self.camera_geometry.map_width,
            map_height=self.camera_geometry.map_height
        )

        return temp_geo.H

    def _compute_residuals(self, params: np.ndarray) -> np.ndarray:
        """
        Compute reprojection residuals for all GCPs.

        For each GCP:
        1. Get world coordinates from GPS (cached in self._world_coords)
        2. Compute predicted camera pixel via H_pred(params) × [x_world, y_world, 1]
        3. Compare with observed camera pixel from GCP
        4. Return flattened residuals: [u_err_1, v_err_1, u_err_2, v_err_2, ...]

        Args:
            params: Parameter vector [Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ]

        Returns:
            residuals: Flattened array of length 2N for N GCPs
                      Format: [Δu_1, Δv_1, Δu_2, Δv_2, ..., Δu_N, Δv_N]
                      where Δu_i = u_observed - u_predicted
                            Δv_i = v_observed - v_predicted
        """
        # Compute predicted homography for these parameters
        H_pred = self._compute_predicted_homography(params)

        # Initialize residuals array
        residuals = np.zeros(2 * len(self.gcps), dtype=np.float64)

        # For each GCP, compute reprojection error
        for i, gcp in enumerate(self.gcps):
            # Get observed camera pixel coordinates
            u_observed = gcp['image']['u']
            v_observed = gcp['image']['v']

            # Get world coordinates (pre-computed from GPS)
            x_world, y_world = self._world_coords[i]

            # Project world point to camera pixel using predicted homography
            # [u, v, w] = H_pred @ [x_world, y_world, 1]
            world_point_homogeneous = np.array([x_world, y_world, 1.0])
            predicted_homogeneous = H_pred @ world_point_homogeneous

            # Normalize homogeneous coordinates (perspective division)
            if abs(predicted_homogeneous[2]) < 1e-10:
                # Point is at infinity (horizon), use large residual to penalize
                residuals[2*i] = self.INFINITY_RESIDUAL
                residuals[2*i + 1] = self.INFINITY_RESIDUAL
                logger.warning(f"GCP {i} projects to infinity with params {params}")
                continue

            u_predicted = predicted_homogeneous[0] / predicted_homogeneous[2]
            v_predicted = predicted_homogeneous[1] / predicted_homogeneous[2]

            # Compute residuals (observed - predicted)
            residuals[2*i] = u_observed - u_predicted
            residuals[2*i + 1] = v_observed - v_predicted

        return residuals

    def _apply_robust_loss(self, residuals: np.ndarray) -> np.ndarray:
        """
        Apply robust loss function to residuals (reference implementation).

        This method provides an explicit implementation of the Huber and Cauchy
        loss functions for reference and testing purposes. The actual optimization
        uses scipy's built-in loss functions via the `loss` parameter.

        Transforms squared residuals using selected robust loss function to reduce
        the influence of outlier GCPs on the optimization.

        Huber loss:
            ρ(r) = r²/2                    if |r| ≤ scale
            ρ(r) = scale·|r| - scale²/2    if |r| > scale

        Cauchy loss:
            ρ(r) = (scale²/2) · log(1 + (r/scale)²)

        Args:
            residuals: Array of residuals (any shape)

        Returns:
            loss_values: Array of same shape with robust loss applied

        Note:
            This method is NOT called during calibration (scipy handles the
            robust loss internally). It is provided for:
            - Understanding the loss function behavior
            - Unit testing the loss function implementation
            - Manual loss computation if needed
        """
        # Compute absolute values (robust losses are symmetric)
        abs_residuals = np.abs(residuals)

        if self.loss_function == self.LOSS_HUBER:
            # Huber loss: quadratic for small errors, linear for large errors
            # Transition at self.loss_scale
            loss_values = np.where(
                abs_residuals <= self.loss_scale,
                # Quadratic region: r²/2
                0.5 * residuals**2,
                # Linear region: scale·|r| - scale²/2
                self.loss_scale * abs_residuals - 0.5 * self.loss_scale**2
            )
        elif self.loss_function == self.LOSS_CAUCHY:
            # Cauchy loss: logarithmic growth for large errors
            # ρ(r) = (scale²/2) · log(1 + (r/scale)²)
            loss_values = (self.loss_scale**2 / 2.0) * np.log1p(
                (residuals / self.loss_scale)**2
            )
        else:
            # Fallback to L2 loss (should not reach here due to validation)
            loss_values = 0.5 * residuals**2

        return loss_values

    def _compute_rms_error(self, residuals: np.ndarray) -> float:
        """
        Compute RMS reprojection error from residuals.

        Args:
            residuals: Flattened residual array [Δu_1, Δv_1, Δu_2, Δv_2, ...]

        Returns:
            RMS error in pixels
        """
        # Reshape to (N, 2) for per-point errors
        residuals_2d = residuals.reshape(-1, 2)
        # Compute Euclidean distance for each point
        per_point_errors = np.linalg.norm(residuals_2d, axis=1)
        # Return RMS
        return np.sqrt(np.mean(per_point_errors**2))

    def calibrate(
        self,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> CalibrationResult:
        """
        Calibrate camera parameters using GCP-based reprojection error minimization.

        This method performs robust least-squares optimization to find the camera
        parameter adjustments (Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ) that minimize
        reprojection error between observed and predicted GCP pixel locations.

        The optimization uses scipy.optimize.least_squares with the selected
        robust loss function to handle outlier GCPs gracefully.

        Args:
            bounds: Optional parameter bounds dictionary. Keys are parameter names
                   ('pan', 'tilt', 'roll', 'X', 'Y', 'Z'), values are (lower, upper)
                   tuples. If None, uses DEFAULT_BOUNDS.

        Returns:
            CalibrationResult with optimized parameters and statistics

        Raises:
            RuntimeError: If optimization fails to converge

        Example:
            >>> calibrator = GCPCalibrator(camera_geo, gcps)
            >>> result = calibrator.calibrate(bounds={'pan': (-5, 5), 'tilt': (-5, 5)})
            >>> print(f"Pan adjustment: {result.optimized_params[0]:.2f}°")
            >>> print(f"RMS error reduced from {result.initial_error:.2f} to {result.final_error:.2f} pixels")
        """
        # Use default bounds if not provided
        if bounds is None:
            bounds = self.DEFAULT_BOUNDS.copy()
        else:
            # Merge with defaults (user bounds override defaults)
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

        logger.info(f"Starting calibration optimization: initial RMS error = {initial_error:.3f}px")

        # Map loss function names to scipy loss strings
        # Note: 'soft_l1' is scipy's pseudo-Huber loss: 2*((1+r²)^0.5 - 1)
        # It behaves similarly to Huber (quadratic for small r, linear for large r)
        scipy_loss_map = {
            'huber': 'soft_l1',  # pseudo-Huber loss (smooth approximation)
            'cauchy': 'cauchy'
        }
        scipy_loss = scipy_loss_map.get(self.loss_function, 'linear')

        # Run least-squares optimization
        try:
            result = least_squares(
                fun=self._compute_residuals,
                x0=params_initial,
                bounds=(lower_bounds, upper_bounds),
                loss=scipy_loss,
                f_scale=self.loss_scale,  # Scale parameter for robust loss
                method='trf',  # Trust Region Reflective (handles bounds)
                verbose=0,
                max_nfev=1000  # Maximum function evaluations
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise RuntimeError(f"Calibration optimization failed: {e}")

        # Extract optimized parameters
        optimized_params = result.x

        # Compute final error
        residuals_final = self._compute_residuals(optimized_params)
        final_error = self._compute_rms_error(residuals_final)

        # Compute per-GCP errors
        residuals_2d = residuals_final.reshape(-1, 2)
        per_gcp_errors = np.linalg.norm(residuals_2d, axis=1).tolist()

        # Classify inliers/outliers based on robust loss threshold
        # Inliers: error <= loss_scale * 2 (conservative threshold)
        inlier_threshold = self.loss_scale * 2.0
        inlier_mask = np.array(per_gcp_errors) <= inlier_threshold
        num_inliers = int(np.sum(inlier_mask))
        num_outliers = len(self.gcps) - num_inliers
        inlier_ratio = num_inliers / len(self.gcps)

        # Build convergence info
        convergence_info = {
            'success': result.success,
            'message': result.message,
            'iterations': result.nfev,  # Number of function evaluations
            'function_evals': result.nfev,
            'optimality': result.optimality,
            'status': result.status
        }

        # Add Jacobian evaluations if available
        if hasattr(result, 'njev'):
            convergence_info['jacobian_evals'] = result.njev

        logger.info(
            f"Calibration complete: final RMS error = {final_error:.3f}px "
            f"({num_inliers}/{len(self.gcps)} inliers, {result.nfev} iterations)"
        )

        if not result.success:
            logger.warning(f"Optimization did not fully converge: {result.message}")

        return CalibrationResult(
            optimized_params=optimized_params,
            initial_error=initial_error,
            final_error=final_error,
            num_inliers=num_inliers,
            num_outliers=num_outliers,
            inlier_ratio=inlier_ratio,
            per_gcp_errors=per_gcp_errors,
            convergence_info=convergence_info
        )
