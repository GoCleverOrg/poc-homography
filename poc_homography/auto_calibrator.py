"""
Auto-calibration via mask matching with OpenCV ECC.

This module implements sequential parameter optimization to automatically calibrate
camera parameters (pan, tilt, height, GPS) by maximizing correlation between a
cartography mask (satellite/map imagery) and a camera frame mask.

The optimization uses a greedy sequential approach:
    1. Pan (±10° search range)
    2. Tilt (±5° search range)
    3. Height (±5m search range)
    4. GPS Latitude (±0.0001° ≈ 11m)
    5. GPS Longitude (±0.0001° ≈ 11m)

Each parameter is optimized independently using scipy's minimize_scalar with Brent's
method, which is gradient-free and designed for univariate optimization. After each
parameter update, the homography is regenerated and the map mask is re-projected to
compute the new correlation score.

Algorithm Details:
    - Method: Brent's method (scipy.optimize.minimize_scalar)
    - Convergence: Stop if parameter change < 0.001 or max 20 iterations per parameter
    - Global timeout: 15 seconds wall-clock time for entire sequence
    - Bounds: Relative to current parameter value (e.g., pan=45° → bounds [35°, 55°])

Usage Example:
    from poc_homography.auto_calibrator import AutoCalibrator
    from poc_homography.mask_matcher import ECCMaskMatcher
    from poc_homography.camera_geometry import CameraGeometry

    # Initialize components
    matcher = ECCMaskMatcher()
    calibrator = AutoCalibrator(
        camera_geometry=geo,
        map_mask=cartography_mask,
        camera_mask=camera_frame_mask,
        matcher=matcher,
        callback=lambda msg: print(msg)
    )

    # Run optimization
    result = calibrator.run()
    print(f"Correlation improved from {result.initial_correlation:.3f} to {result.final_correlation:.3f}")
    print(f"Pan: {result.original_params['pan_deg']:.2f}° → {result.optimized_params['pan_deg']:.2f}°")

References:
    - Brent's method: https://en.wikipedia.org/wiki/Brent%27s_method
    - ECC algorithm: Evangelidis & Psarakis (2008), IEEE TPAMI
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, Tuple
import cv2
import numpy as np
from scipy.optimize import minimize_scalar

from poc_homography.mask_matcher import MaskMatcher, ECCMaskMatcher
from poc_homography.camera_geometry import CameraGeometry

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """
    Result object containing all auto-calibration outputs.

    Attributes:
        initial_correlation: Correlation score before optimization [0.0, 1.0]
        final_correlation: Correlation score after optimization [0.0, 1.0]
        original_params: Dictionary of original camera parameters
        optimized_params: Dictionary of optimized camera parameters
        elapsed_time: Total wall-clock time in seconds
        steps_completed: Number of parameter optimization steps completed (0-5)
        success: Whether optimization improved correlation
        improvement: Absolute improvement in correlation (final - initial)
        relative_improvement: Relative improvement as percentage
        timeout_reached: Whether global timeout was reached before completion
        message: Human-readable status message
    """
    initial_correlation: float
    final_correlation: float
    original_params: Dict[str, Any]
    optimized_params: Dict[str, Any]
    elapsed_time: float
    steps_completed: int
    success: bool
    improvement: float
    relative_improvement: float
    timeout_reached: bool
    message: str


class AutoCalibrator:
    """
    Sequential parameter optimization for camera calibration via mask matching.

    This class implements the auto-calibration algorithm described in issue #82.
    It optimizes camera parameters sequentially (pan → tilt → height → GPS) by
    maximizing the ECC correlation between a cartography mask and camera frame mask.

    The optimization is greedy: each parameter is optimized while holding others
    constant. This is not guaranteed to find the global optimum, but is fast and
    works well in practice when starting from reasonable initial parameters.

    Parameter Bounds (relative to current value):
        - Pan: ±10° (e.g., current 45° → search [35°, 55°])
        - Tilt: ±5° (e.g., current 30° → search [25°, 35°])
        - Height: ±5m (e.g., current 10m → search [5m, 15m])
        - GPS Lat: ±0.0001° ≈ ±11m at equator (e.g., 40.4168° → [40.4167°, 40.4169°])
        - GPS Lon: ±0.0001° ≈ ±11m at equator (e.g., -3.7038° → [-3.7039°, -3.7037°])

    Note:
        Current implementation optimizes Pan, Tilt, and Height only. GPS optimization
        (Lat/Lon) requires a coordinate converter to translate GPS coordinates to
        world coordinates (w_pos). To enable GPS optimization, use the extended
        AutoCalibratorWithGPS class (to be implemented) or manually update w_pos
        after optimization.

    Convergence Criteria (per parameter):
        - Max iterations: 20
        - Tolerance: 0.001 (parameter-specific units)

    Global Timeout:
        - 15 seconds wall-clock time for entire sequence
        - If exceeded, optimization stops and returns current best parameters

    Attributes:
        camera_geometry: CameraGeometry instance with current camera parameters
        map_mask: Cartography mask (binary uint8 array, 0=background, 255=features)
        camera_mask: Camera frame mask (binary uint8 array, 0=background, 255=features)
        matcher: MaskMatcher instance (typically ECCMaskMatcher)
        callback: Optional callback function(message: str) for status updates
        timeout: Global timeout in seconds (default: 15.0)
        pan_bounds: Relative pan bounds in degrees (default: ±10°)
        tilt_bounds: Relative tilt bounds in degrees (default: ±5°)
        height_bounds: Relative height bounds in meters (default: ±5m)
        gps_bounds: Relative GPS bounds in degrees (default: ±0.0001°)
    """

    # Default parameter bounds (relative to current value)
    DEFAULT_PAN_BOUNDS_DEG = 10.0
    DEFAULT_TILT_BOUNDS_DEG = 5.0
    DEFAULT_HEIGHT_BOUNDS_M = 5.0
    DEFAULT_GPS_BOUNDS_DEG = 0.0001  # ~11m at equator

    # Convergence criteria
    DEFAULT_MAX_ITERATIONS = 20
    DEFAULT_TOLERANCE = 0.001

    # Global timeout
    DEFAULT_TIMEOUT_SEC = 15.0

    def __init__(
        self,
        camera_geometry: CameraGeometry,
        map_mask: np.ndarray,
        camera_mask: np.ndarray,
        matcher: Optional[MaskMatcher] = None,
        callback: Optional[Callable[[str], None]] = None,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        pan_bounds: float = DEFAULT_PAN_BOUNDS_DEG,
        tilt_bounds: float = DEFAULT_TILT_BOUNDS_DEG,
        height_bounds: float = DEFAULT_HEIGHT_BOUNDS_M,
        gps_bounds: float = DEFAULT_GPS_BOUNDS_DEG,
        camera_gps: Optional[tuple] = None,
        coordinate_converter: Optional[Any] = None,
        geotiff_params: Optional[Dict[str, float]] = None,
        camera_utm_position: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize AutoCalibrator.

        Args:
            camera_geometry: CameraGeometry instance with initial camera parameters
            map_mask: Cartography mask (binary uint8, 0=background, 255=features)
            camera_mask: Camera frame mask (binary uint8, 0=background, 255=features)
            matcher: MaskMatcher instance (defaults to ECCMaskMatcher if None)
            callback: Optional callback function for status updates
            timeout: Global timeout in seconds (default: 15.0)
            pan_bounds: Relative pan bounds in degrees (default: ±10°)
            tilt_bounds: Relative tilt bounds in degrees (default: ±5°)
            height_bounds: Relative height bounds in meters (default: ±5m)
            gps_bounds: Relative GPS bounds in degrees (default: ±0.0001°)
            camera_gps: Optional tuple (lat, lon) for GPS optimization
            coordinate_converter: Optional coordinate converter with gps_to_local_xy() method
            geotiff_params: Optional dict with GeoTIFF georeferencing parameters.
                Required keys: 'pixel_size_x', 'pixel_size_y', 'origin_easting', 'origin_northing'.
                If provided, enables complete cartography-pixel-to-camera transformation.
                If None, falls back to using camera homography only (reduced accuracy).
            camera_utm_position: Optional tuple (easting, northing) of camera position in UTM.
                Required if geotiff_params is provided. Used to compute relative coordinates
                for T_pixel_to_localXY transformation.

        Raises:
            ValueError: If masks are invalid or camera_geometry is not initialized

        Note:
            GPS optimization (lat/lon) is only enabled if both camera_gps and
            coordinate_converter are provided. The converter must have a
            gps_to_local_xy(lat, lon) method that returns (x_meters, y_meters).
        """
        # Validate inputs
        if camera_geometry.K is None or camera_geometry.w_pos is None:
            raise ValueError(
                "CameraGeometry must be initialized with set_camera_parameters() "
                "before auto-calibration"
            )

        if not isinstance(map_mask, np.ndarray) or map_mask.size == 0:
            raise ValueError("map_mask must be a non-empty numpy array")

        if not isinstance(camera_mask, np.ndarray) or camera_mask.size == 0:
            raise ValueError("camera_mask must be a non-empty numpy array")

        self.camera_geometry = camera_geometry
        self.map_mask = map_mask
        self.camera_mask = camera_mask
        self.matcher = matcher if matcher is not None else ECCMaskMatcher()
        self.callback = callback
        self.timeout = timeout

        # Store bounds
        self.pan_bounds = pan_bounds
        self.tilt_bounds = tilt_bounds
        self.height_bounds = height_bounds
        self.gps_bounds = gps_bounds

        # GPS optimization support
        self.camera_gps = camera_gps
        self.coordinate_converter = coordinate_converter
        self.gps_enabled = (camera_gps is not None and coordinate_converter is not None)

        # Geotiff transformation support
        self.geotiff_params = geotiff_params
        self.camera_utm_position = camera_utm_position
        self._partial_geotiff_warned = False  # Track if warning already logged

        # State tracking
        self.start_time = None
        self.steps_completed = 0

        # Store original parameters
        self.original_params = self._get_current_parameters()

        logger.info("AutoCalibrator initialized")
        logger.info(f"  Timeout: {timeout}s")
        logger.info(f"  Bounds: Pan ±{pan_bounds}°, Tilt ±{tilt_bounds}°, "
                   f"Height ±{height_bounds}m, GPS ±{gps_bounds}°")

    def _get_current_parameters(self) -> Dict[str, Any]:
        """
        Extract current camera parameters from CameraGeometry instance.

        Returns:
            Dictionary with keys: pan_deg, tilt_deg, height_m, w_pos, K,
            map_width, map_height, and optionally gps_lat, gps_lon
        """
        params = {
            'pan_deg': self.camera_geometry.pan_deg,
            'tilt_deg': self.camera_geometry.tilt_deg,
            'height_m': self.camera_geometry.height_m,
            'w_pos': self.camera_geometry.w_pos.copy(),
            'K': self.camera_geometry.K.copy(),
            'map_width': self.camera_geometry.map_width,
            'map_height': self.camera_geometry.map_height
        }

        # Add GPS coordinates if available
        if self.gps_enabled and self.camera_gps is not None:
            params['gps_lat'] = self.camera_gps[0]
            params['gps_lon'] = self.camera_gps[1]

        return params

    def _set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update camera parameters in CameraGeometry instance.

        This calls set_camera_parameters() which regenerates the homography matrix.

        Args:
            params: Dictionary with camera parameters (same format as _get_current_parameters)
        """
        self.camera_geometry.set_camera_parameters(
            K=params['K'],
            w_pos=params['w_pos'],
            pan_deg=params['pan_deg'],
            tilt_deg=params['tilt_deg'],
            map_width=params['map_width'],
            map_height=params['map_height']
        )

    def _project_map_mask(self) -> np.ndarray:
        """
        Project the cartography mask to camera frame coordinates.

        This uses the complete transformation chain:
        1. T_pixel_to_localXY: Cartography pixels → Local XY meters (from geotiff_params)
        2. H_camera: Local XY meters → Camera pixels (from camera geometry)
        3. H_total = H_camera @ T_pixel_to_localXY: Cartography pixels → Camera pixels

        If geotiff_params is not provided, falls back to using H_camera only
        (backwards compatibility, but reduced accuracy for georeferenced masks).

        Returns:
            Projected mask as binary uint8 array with same dimensions as camera_mask
        """
        # Get current camera homography (regenerated when parameters change during optimization)
        H_camera = self.camera_geometry.H

        # Compute complete homography if geotiff parameters available
        if self.geotiff_params is not None and self.camera_utm_position is not None:
            # Build T_pixel_to_localXY: transforms cartography pixels to local XY
            # Cartography pixel (px, py) → Local XY (x, y):
            #   easting = origin_easting + px * pixel_size_x
            #   northing = origin_northing + py * pixel_size_y
            #   x = easting - camera_easting
            #   y = northing - camera_northing
            #
            # In homogeneous coordinates:
            #   [x]   [pixel_size_x  0            dx] [px]
            #   [y] = [0             pixel_size_y dy] [py]
            #   [1]   [0             0            1 ] [1 ]
            # where dx = origin_easting - camera_easting
            #       dy = origin_northing - camera_northing

            pixel_size_x = self.geotiff_params['pixel_size_x']
            pixel_size_y = self.geotiff_params['pixel_size_y']
            origin_easting = self.geotiff_params['origin_easting']
            origin_northing = self.geotiff_params['origin_northing']

            camera_easting, camera_northing = self.camera_utm_position

            dx = origin_easting - camera_easting
            dy = origin_northing - camera_northing

            T_pixel_to_localXY = np.array([
                [pixel_size_x, 0,            dx],
                [0,            pixel_size_y, dy],
                [0,            0,            1 ]
            ], dtype=np.float64)

            # Compose complete homography
            H_total = H_camera @ T_pixel_to_localXY
        else:
            # Fallback: use camera homography only (backwards compatibility)
            # Log warning once if partial configuration detected
            if not self._partial_geotiff_warned and (
                self.geotiff_params is not None or self.camera_utm_position is not None
            ):
                logger.warning(
                    "Partial geotiff configuration: geotiff_params=%s, camera_utm_position=%s. "
                    "Both parameters are required for complete transformation. "
                    "Falling back to camera homography only.",
                    self.geotiff_params is not None,
                    self.camera_utm_position is not None
                )
                self._partial_geotiff_warned = True
            H_total = H_camera

        # Warp map mask to camera frame coordinates
        camera_height, camera_width = self.camera_mask.shape[:2]
        projected_mask = cv2.warpPerspective(
            self.map_mask,
            H_total,
            (camera_width, camera_height),
            flags=cv2.INTER_NEAREST
        )

        return projected_mask

    def _compute_correlation(self) -> float:
        """
        Compute correlation between projected map mask and camera mask.

        Returns:
            Correlation score in [0.0, 1.0]
        """
        # Project map mask with current parameters
        projected_mask = self._project_map_mask()

        # Compute correlation using the matcher
        correlation = self.matcher.compute_correlation(projected_mask, self.camera_mask)

        return correlation

    def _log_status(self, message: str) -> None:
        """Log status message and call callback if provided."""
        logger.info(message)
        if self.callback is not None:
            self.callback(message)

    def _check_timeout(self) -> bool:
        """Check if global timeout has been exceeded."""
        if self.start_time is None:
            return False
        elapsed = time.time() - self.start_time
        return elapsed > self.timeout

    def _optimize_parameter(
        self,
        param_name: str,
        current_value: float,
        bounds: tuple,
        step_number: int,
        total_steps: int
    ) -> float:
        """
        Optimize a single parameter using Brent's method.

        Args:
            param_name: Parameter name ('pan_deg', 'tilt_deg', 'height_m', 'gps_lat', 'gps_lon')
            current_value: Current parameter value
            bounds: Absolute bounds as (lower, upper) tuple
            step_number: Current step number (1-indexed)
            total_steps: Total number of optimization steps

        Returns:
            Optimal parameter value
        """
        self._log_status(f"Optimizing {param_name} ({step_number}/{total_steps})...")

        # Define objective function (negative correlation for minimization)
        def objective(value: float) -> float:
            # Get current parameters
            params = self._get_current_parameters()

            # Update the parameter being optimized
            if param_name in ['pan_deg', 'tilt_deg', 'height_m']:
                params[param_name] = value
                # Height also updates w_pos[2]
                if param_name == 'height_m':
                    params['w_pos'][2] = value
            elif param_name == 'gps_lat' and self.gps_enabled:
                # Update GPS latitude and convert to world coordinates
                lat = value
                lon = params['gps_lon']
                x, y = self.coordinate_converter.gps_to_local_xy(lat, lon)
                params['w_pos'][0] = x  # X = East
                params['w_pos'][1] = y  # Y = North
                params['gps_lat'] = lat
                # Update internal GPS state for subsequent iterations
                self.camera_gps = (lat, lon)
            elif param_name == 'gps_lon' and self.gps_enabled:
                # Update GPS longitude and convert to world coordinates
                lat = params['gps_lat']
                lon = value
                x, y = self.coordinate_converter.gps_to_local_xy(lat, lon)
                params['w_pos'][0] = x  # X = East
                params['w_pos'][1] = y  # Y = North
                params['gps_lon'] = lon
                # Update internal GPS state for subsequent iterations
                self.camera_gps = (lat, lon)

            # Set updated parameters (regenerates homography)
            self._set_parameters(params)

            # Compute correlation
            correlation = self._compute_correlation()

            # Return negative (minimize = maximize correlation)
            return -correlation

        # Run optimization with Brent's method
        result = minimize_scalar(
            objective,
            bounds=bounds,
            method='bounded',
            options={
                'maxiter': self.DEFAULT_MAX_ITERATIONS,
                'xatol': self.DEFAULT_TOLERANCE
            }
        )

        optimal_value = result.x
        optimal_correlation = -result.fun

        self._log_status(
            f"  {param_name}: {current_value:.4f} → {optimal_value:.4f} "
            f"(correlation: {optimal_correlation:.4f})"
        )

        return optimal_value

    def run(self) -> CalibrationResult:
        """
        Run sequential auto-calibration optimization.

        Optimizes parameters in order: Pan → Tilt → Height → GPS Lat → GPS Lon

        Returns:
            CalibrationResult object with optimization results
        """
        self.start_time = time.time()
        self.steps_completed = 0

        # Compute initial correlation
        initial_correlation = self._compute_correlation()
        self._log_status(f"Initial correlation: {initial_correlation:.4f}")

        # Define parameter sequence
        # Each entry: (param_name, bounds_offset)
        param_sequence = [
            ('pan_deg', self.pan_bounds),
            ('tilt_deg', self.tilt_bounds),
            ('height_m', self.height_bounds),
        ]

        # Add GPS optimization if enabled
        if self.gps_enabled:
            param_sequence.extend([
                ('gps_lat', self.gps_bounds),
                ('gps_lon', self.gps_bounds)
            ])

        total_steps = len(param_sequence)
        timeout_reached = False

        # Sequential optimization
        for i, (param_name, bounds_offset) in enumerate(param_sequence, start=1):
            # Check timeout
            if self._check_timeout():
                self._log_status("Global timeout reached, stopping optimization")
                timeout_reached = True
                break

            # Get current parameter value
            params = self._get_current_parameters()
            if param_name in ['pan_deg', 'tilt_deg', 'height_m', 'gps_lat', 'gps_lon']:
                current_value = params.get(param_name)
                if current_value is None:
                    # GPS parameter not available
                    logger.warning(f"Skipping {param_name}: not available in parameters")
                    continue
            else:
                logger.warning(f"Unknown parameter: {param_name}")
                continue

            # Compute absolute bounds (relative to current value)
            bounds = (
                current_value - bounds_offset,
                current_value + bounds_offset
            )

            # Validate bounds (enforce camera geometry constraints)
            if param_name == 'tilt_deg':
                # Tilt must be positive (camera pointing down)
                bounds = (
                    max(bounds[0], CameraGeometry.TILT_MIN + 0.1),
                    min(bounds[1], CameraGeometry.TILT_MAX)
                )
            elif param_name == 'height_m':
                # Height must be positive
                bounds = (
                    max(bounds[0], CameraGeometry.HEIGHT_MIN),
                    min(bounds[1], CameraGeometry.HEIGHT_MAX)
                )

            # Optimize parameter
            optimal_value = self._optimize_parameter(
                param_name,
                current_value,
                bounds,
                i,
                total_steps
            )

            # Update parameter in geometry
            params = self._get_current_parameters()
            params[param_name] = optimal_value

            # Update dependent parameters
            if param_name == 'height_m':
                params['w_pos'][2] = optimal_value
            elif param_name == 'gps_lat' and self.gps_enabled:
                lat = optimal_value
                lon = params['gps_lon']
                x, y = self.coordinate_converter.gps_to_local_xy(lat, lon)
                params['w_pos'][0] = x
                params['w_pos'][1] = y
                self.camera_gps = (lat, lon)
            elif param_name == 'gps_lon' and self.gps_enabled:
                lat = params['gps_lat']
                lon = optimal_value
                x, y = self.coordinate_converter.gps_to_local_xy(lat, lon)
                params['w_pos'][0] = x
                params['w_pos'][1] = y
                self.camera_gps = (lat, lon)

            self._set_parameters(params)

            self.steps_completed += 1

        # Compute final correlation
        final_correlation = self._compute_correlation()
        elapsed_time = time.time() - self.start_time

        # Get final parameters
        optimized_params = self._get_current_parameters()

        # Compute improvement
        improvement = final_correlation - initial_correlation
        relative_improvement = (improvement / initial_correlation * 100) if initial_correlation > 0 else 0.0
        success = improvement > 0.0

        # Generate status message
        if success:
            message = (
                f"Auto-calibration successful! "
                f"Correlation improved from {initial_correlation:.4f} to {final_correlation:.4f} "
                f"({relative_improvement:+.2f}%)"
            )
        elif timeout_reached:
            message = (
                f"Auto-calibration stopped due to timeout. "
                f"Completed {self.steps_completed}/{total_steps} steps. "
                f"Correlation: {initial_correlation:.4f} → {final_correlation:.4f}"
            )
        else:
            message = (
                f"Auto-calibration did not improve correlation. "
                f"Final correlation: {final_correlation:.4f} (initial: {initial_correlation:.4f})"
            )

        self._log_status(message)
        self._log_status(f"Elapsed time: {elapsed_time:.2f}s")

        # Create result object
        result = CalibrationResult(
            initial_correlation=initial_correlation,
            final_correlation=final_correlation,
            original_params=self.original_params,
            optimized_params=optimized_params,
            elapsed_time=elapsed_time,
            steps_completed=self.steps_completed,
            success=success,
            improvement=improvement,
            relative_improvement=relative_improvement,
            timeout_reached=timeout_reached,
            message=message
        )

        return result
