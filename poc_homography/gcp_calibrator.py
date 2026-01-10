#!/usr/bin/env python3
"""
GCP-based reprojection error minimization calibration.

This module implements automatic camera parameter calibration by minimizing
reprojection error between observed Ground Control Points (GCPs) and predicted
pixel locations. The calibration optimizes camera orientation (pan, tilt, roll)
and position (X, Y, Z) to find the best fit to observed GCP data.

GCPs use map pixel coordinates (not GPS). Each GCP contains:
- map_id: Identifier for the map
- map_pixel_x, map_pixel_y: Position on the map in pixels
- image_u, image_v: Position in the camera image in pixels

Mathematical Model:
    For parameter vector p = {Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ}, minimize:

    E(p) = Σᵢ ρ(||eᵢ||²)

    where:
    - eᵢ = x_observed,i - x_predicted,i is the residual for GCP i
    - x_observed,i is the observed camera pixel location [u_cam, v_cam]ᵀ
    - x_predicted,i = π(H_pred(p) × x_map,i) is the predicted camera pixel
    - x_map,i is the map pixel coordinate [map_pixel_x, map_pixel_y]ᵀ
    - π() is the perspective projection (homogeneous coordinate normalization)
    - ρ() is a robust loss function (Huber or Cauchy) to handle outliers
    - H_pred(p) is the predicted homography from updated camera parameters

Key Concepts:
    - GCP (Ground Control Point): Known correspondence between map pixel and camera pixel
    - Reprojection error: Distance between observed and predicted pixel locations
    - Robust loss: Reduces influence of outlier GCPs on optimization
    - Parameter increments: Optimization adjusts camera parameters incrementally

Regularization (Prior-Based Constraints):
    When GCP data is noisy, sparse, or poorly distributed, the optimization may
    produce implausible camera parameters. Regularization adds soft constraints
    that pull parameters toward their initial sensor-based estimates.

    Extended Objective Function:
        E_total(p) = E_reproj(p) + E_prior(p)

        where:
        - E_reproj(p) = sum_i rho(||x_observed,i - pi(H_pred(p) * x_ref,i)||^2)
          Measures geometric accuracy: how well predicted pixels match observations.

        - E_prior(p) = lambda * ||p - p0||^2_Sigma
                     = lambda * sum_j ((p_j - p0_j) / sigma_j)^2
          Measures plausibility: how far parameters deviate from sensor readings.

        - p0 = [0, 0, 0, 0, 0, 0] is the initial estimate (no adjustment)
        - sigma_j is the prior uncertainty for parameter j
        - lambda is the regularization weight balancing the two terms

    Physical Interpretation:
        - E_reproj: Captures how well the camera model explains the observed GCPs.
          Lower values mean better geometric fit to the data.

        - E_prior: Captures how plausible the parameters are given sensor accuracy.
          Lower values mean parameters stay close to initial sensor readings.

        - lambda (regularization_weight): Controls the trade-off between terms.
          - lambda -> 0: Ignores priors, pure data fitting (may overfit noise)
          - lambda -> inf: Forces parameters to stay at initial values (underfits)

        - sigma_j: Sets the "trust" in sensor j. Smaller sigma means higher trust
          (stronger pull toward initial value). Larger sigma means lower trust
          (parameter can deviate more freely).

    Prior Configuration (prior_sigmas dict):
        Default values represent typical uncertainties:

        | Key              | Default | Unit    | Description                    |
        |------------------|---------|---------|--------------------------------|
        | map_position_px  | 10.0    | pixels  | X, Y position on map           |
        | height_m         | 2.0     | meters  | Z position (camera height)     |
        | pan_deg          | 3.0     | degrees | Compass heading accuracy       |
        | tilt_deg         | 3.0     | degrees | IMU pitch accuracy             |
        | roll_deg         | 1.0     | degrees | IMU roll accuracy              |

        Adjusting sigmas based on GCP quality:
        - High-precision GCPs:   map_position_px = 1.0 - 5.0
        - Standard GCPs:         map_position_px = 10.0 - 20.0
        - Survey-grade IMU:      pan_deg = 0.5, tilt_deg = 0.5
        - Phone magnetometer:    pan_deg = 5.0 - 10.0 (high interference)

    Regularization Weight Guidelines (regularization_weight / lambda):
        | Value | Effect           | When to Use                          |
        |-------|------------------|--------------------------------------|
        | 0.0   | No regularization| Many high-quality GCPs (20+)         |
        | 0.1   | Weak             | Good GCPs, trust data more           |
        | 1.0   | Balanced         | Default, moderate GCP quality        |
        | 10.0  | Strong           | Few/noisy GCPs, trust sensors more   |

        Rule of thumb:
        - With 20+ well-distributed GCPs: lambda = 0.0 to 0.1
        - With 10-20 GCPs of moderate quality: lambda = 1.0 (default)
        - With 5-10 sparse or noisy GCPs: lambda = 5.0 to 10.0
        - With very few GCPs (< 5): Consider lambda = 10.0+ or collect more data

    Example with Regularization:
        >>> calibrator = GCPCalibrator(
        ...     camera_geometry=geo,
        ...     gcps=gcps,
        ...     prior_sigmas={
        ...         'map_position_px': 5.0,  # High precision GCPs
        ...         'pan_deg': 5.0,          # Consumer compass, less trusted
        ...     },
        ...     regularization_weight=1.0    # Balanced trade-off
        ... )
        >>> result = calibrator.calibrate()
        >>> print(f"Regularization penalty: {result.regularization_penalty:.2f}")

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
    >>> # GCPs with map pixel and camera pixel coordinates
    >>> gcps = [
    ...     {'map_id': 'map1', 'map_pixel_x': 320.0, 'map_pixel_y': 240.0,
    ...      'image_u': 960.0, 'image_v': 540.0},
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

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from poc_homography.camera_geometry import CameraGeometry

# Import scipy for optimization
from scipy.optimize import least_squares

from poc_homography.types import Pixels

# Import matplotlib for visualization (optional, lazy import)
try:
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


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
        validation_metrics: Optional validation metrics dictionary with train/test splits:
            {
                'train': {'rms': float, 'p90': float, 'max': float, 'count': int},
                'test': {'rms': float, 'p90': float, 'max': float, 'count': int}
            }
        train_indices: Optional list of GCP indices used for training (optimization)
        test_indices: Optional list of GCP indices used for testing (validation)
    """

    optimized_params: np.ndarray
    initial_error: float
    final_error: float
    num_inliers: int
    num_outliers: int
    inlier_ratio: float
    per_gcp_errors: list[float]
    convergence_info: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validation_metrics: dict[str, dict[str, float]] | None = None
    train_indices: list[int] | None = None
    test_indices: list[int] | None = None
    regularization_penalty: float | None = None  # Weighted squared deviation from priors


class GCPCalibrator:
    """
    GCP-based reprojection error minimization calibrator.

    This class optimizes camera parameters (pan, tilt, roll, position) by
    minimizing the reprojection error between observed GCP pixel locations
    and predicted pixel locations from the camera model.

    The optimization uses scipy.optimize.least_squares with robust loss functions
    to handle outlier GCPs gracefully.

    GCPs use map pixel coordinates (flat structure):
        {'map_id': str, 'map_pixel_x': float, 'map_pixel_y': float,
         'image_u': float, 'image_v': float}

    Attributes:
        camera_geometry: Initial CameraGeometry instance (defines camera model)
        gcps: List of Ground Control Points with map pixel and image pixel coordinates
        loss_function: Robust loss function name ('huber' or 'cauchy')
        loss_scale: Scale parameter for robust loss (pixels)
        validation_split: Fraction of GCPs to reserve for validation (0.0-0.5)
        random_seed: Random seed for reproducible train/test splitting
        prior_sigmas: Dictionary of prior standard deviations for regularization
        regularization_weight: Lambda parameter balancing reprojection vs prior penalty
    """

    # Supported loss functions
    LOSS_HUBER = "huber"
    LOSS_CAUCHY = "cauchy"
    VALID_LOSS_FUNCTIONS = [LOSS_HUBER, LOSS_CAUCHY]

    # Large residual value for points projecting to infinity (horizon)
    # This value is intentionally large to penalize such configurations during optimization
    INFINITY_RESIDUAL = 1e6

    # Default parameter bounds (conservative ranges)
    DEFAULT_BOUNDS = {
        "pan": (-10.0, 10.0),  # ±10 degrees
        "tilt": (-10.0, 10.0),  # ±10 degrees
        "roll": (-10.0, 10.0),  # ±10 degrees (reserved for future CameraGeometry support)
        "X": (-5.0, 5.0),  # ±5 meters
        "Y": (-5.0, 5.0),  # ±5 meters
        "Z": (-5.0, 5.0),  # ±5 meters
    }

    # Default prior standard deviations for regularization
    # These represent typical uncertainties in camera pose estimates
    DEFAULT_PRIOR_SIGMAS: dict[str, float] = {
        "map_position_px": 10.0,  # X, Y position uncertainty (map pixels)
        "height_m": 2.0,  # Z position uncertainty (meters)
        "pan_deg": 3.0,  # Pan angle uncertainty (degrees)
        "tilt_deg": 3.0,  # Tilt angle uncertainty (degrees)
        "roll_deg": 1.0,  # Roll angle uncertainty (degrees)
    }

    def __init__(
        self,
        camera_geometry: CameraGeometry,
        gcps: list[dict[str, Any]],
        loss_function: str = "huber",
        loss_scale: float = 1.0,
        validation_split: float = 0.0,
        random_seed: int | None = None,
        prior_sigmas: dict[str, float] | None = None,
        regularization_weight: float = 1.0,
    ):
        """
        Initialize GCP-based calibrator.

        Args:
            camera_geometry: Initial CameraGeometry instance with camera parameters
            gcps: List of Ground Control Point dictionaries with flat structure:
                - 'map_id': str - Identifier for the map
                - 'map_pixel_x': float - X coordinate on map in pixels
                - 'map_pixel_y': float - Y coordinate on map in pixels
                - 'image_u': float - U coordinate in camera image in pixels
                - 'image_v': float - V coordinate in camera image in pixels
            loss_function: Robust loss function to use ('huber' or 'cauchy')
            loss_scale: Scale parameter for robust loss (pixels). Determines the
                       threshold at which errors are considered outliers.
                       - For 'huber': transition from quadratic to linear at loss_scale
                       - For 'cauchy': scale of the Cauchy distribution
            validation_split: Fraction of GCPs to reserve for validation (0.0-0.5).
                             Default 0.0 (no split, all GCPs used for training).
                             If > 0, randomly splits GCPs into train and test sets.
            random_seed: Random seed for reproducible train/test splitting.
                        If None, uses non-deterministic splitting.
            prior_sigmas: Dictionary of prior standard deviations for regularization.
                         Keys and default values:
                         - 'map_position_px': X, Y position uncertainty (default: 10.0 pixels)
                         - 'height_m': Z position uncertainty (default: 2.0 meters)
                         - 'pan_deg': Pan angle uncertainty (default: 3.0 degrees)
                         - 'tilt_deg': Tilt angle uncertainty (default: 3.0 degrees)
                         - 'roll_deg': Roll angle uncertainty (default: 1.0 degree)
                         If None, uses DEFAULT_PRIOR_SIGMAS. If provided, merges with defaults
                         (user values override defaults).
            regularization_weight: Lambda parameter (>= 0.0) balancing reprojection error
                                  vs prior penalty. Default 1.0.
                                  - 0.0: No regularization (pure reprojection minimization)
                                  - Higher values: Stronger pull towards prior estimates

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

        # Validate GCP structure (flat map-pixel format)
        required_keys = ["map_id", "map_pixel_x", "map_pixel_y", "image_u", "image_v"]
        for i, gcp in enumerate(gcps):
            if not isinstance(gcp, dict):
                raise ValueError(f"GCP at index {i} must be a dictionary")

            # Check required keys for flat map-pixel format
            for key in required_keys:
                if key not in gcp:
                    raise ValueError(f"GCP at index {i} missing required '{key}' key")

        # Validate loss_scale
        if loss_scale <= 0:
            raise ValueError(f"loss_scale must be positive, got {loss_scale}")

        # Validate validation_split
        if not (0.0 <= validation_split <= 0.5):
            raise ValueError(
                f"validation_split must be between 0.0 and 0.5, got {validation_split}"
            )

        # Validate regularization_weight
        if not np.isfinite(regularization_weight) or regularization_weight < 0.0:
            raise ValueError(
                f"regularization_weight must be >= 0.0 and finite, got {regularization_weight}"
            )

        # Process prior_sigmas: merge with defaults (user values override defaults)
        if prior_sigmas is None:
            merged_sigmas = self.DEFAULT_PRIOR_SIGMAS.copy()
        else:
            merged_sigmas = self.DEFAULT_PRIOR_SIGMAS.copy()
            merged_sigmas.update(prior_sigmas)

        # Validate prior_sigmas values (must be positive and finite)
        for key, value in merged_sigmas.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"prior_sigmas['{key}'] must be positive and finite, got {value}")

        # Store configuration
        self.camera_geometry = camera_geometry
        self.gcps = gcps
        self.loss_function = loss_function.lower()
        self.loss_scale = loss_scale
        self.validation_split = validation_split
        self.random_seed = random_seed
        self._prior_sigmas = merged_sigmas
        self._regularization_weight = regularization_weight

        # Convert prior_sigmas to internal numpy array format for efficient computation
        # Order: [sigma_pan, sigma_tilt, sigma_roll, sigma_X, sigma_Y, sigma_Z]
        self._sigma_vector = np.array(
            [
                merged_sigmas["pan_deg"],  # sigma_pan (degrees)
                merged_sigmas["tilt_deg"],  # sigma_tilt (degrees)
                merged_sigmas["roll_deg"],  # sigma_roll (degrees)
                merged_sigmas["map_position_px"],  # sigma_X (map pixels)
                merged_sigmas["map_position_px"],  # sigma_Y (map pixels)
                merged_sigmas["height_m"],  # sigma_Z (meters)
            ],
            dtype=np.float64,
        )

        # Extract map pixel coordinates from GCPs (cached for efficiency)
        # Map pixel coordinates are used directly - no GPS/UTM conversion needed
        map_coords_list: list[list[float]] = []
        for gcp in gcps:
            x = float(gcp["map_pixel_x"])
            y = float(gcp["map_pixel_y"])
            map_coords_list.append([x, y])
        self._map_coords: np.ndarray = np.array(map_coords_list, dtype=np.float64)

        # DEBUG: Print first GCP's map pixel coordinates for verification
        if len(self._map_coords) > 0:
            logger.debug(
                f"First GCP map pixel = ({self._map_coords[0][0]:.2f}, {self._map_coords[0][1]:.2f})"
            )

        # Initialize train/test split indices (will be set in calibrate() if validation_split > 0)
        self._train_indices: list[int] | None = None
        self._test_indices: list[int] | None = None

        logger.info(
            f"GCPCalibrator initialized with {len(gcps)} GCPs, "
            f"loss={self.loss_function}, scale={self.loss_scale}, "
            f"validation_split={self.validation_split}, "
            f"regularization_weight={self._regularization_weight}"
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
        K = self.camera_geometry.K
        if K is None:
            raise ValueError("Camera intrinsic matrix K is not set")
        temp_geo.set_camera_parameters(
            K=K,
            w_pos=updated_pos,
            pan_deg=updated_pan,
            tilt_deg=updated_tilt,
            map_width=Pixels(self.camera_geometry.map_width),
            map_height=Pixels(self.camera_geometry.map_height),
        )

        return np.asarray(temp_geo.H)

    def _compute_residuals(
        self, params: np.ndarray, gcp_indices: list[int] | None = None
    ) -> np.ndarray:
        """
        Compute reprojection residuals for specified GCPs.

        For each GCP:
        1. Get map pixel coordinates (cached in self._map_coords)
        2. Compute predicted camera pixel via H_pred(params) × [map_x, map_y, 1]
        3. Compare with observed camera pixel from GCP
        4. Return flattened residuals: [u_err_1, v_err_1, u_err_2, v_err_2, ...]

        Args:
            params: Parameter vector [Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ]
            gcp_indices: Optional list of GCP indices to use. If None, uses all GCPs.

        Returns:
            residuals: Flattened array of length 2N for N GCPs
                      Format: [Δu_1, Δv_1, Δu_2, Δv_2, ..., Δu_N, Δv_N]
                      where Δu_i = u_observed - u_predicted
                            Δv_i = v_observed - v_predicted
        """
        # Compute predicted homography for these parameters
        H_pred = self._compute_predicted_homography(params)

        # Determine which GCPs to use
        if gcp_indices is None:
            gcp_indices = list(range(len(self.gcps)))

        # Initialize residuals array
        residuals = np.zeros(2 * len(gcp_indices), dtype=np.float64)

        # For each GCP, compute reprojection error
        for res_idx, gcp_idx in enumerate(gcp_indices):
            gcp = self.gcps[gcp_idx]

            # Get observed camera pixel coordinates (flat structure)
            u_observed = gcp["image_u"]
            v_observed = gcp["image_v"]

            # Get map pixel coordinates (pre-computed)
            map_x, map_y = self._map_coords[gcp_idx]

            # Project map point to camera pixel using predicted homography
            # [u, v, w] = H_pred @ [map_x, map_y, 1]
            map_point_homogeneous = np.array([map_x, map_y, 1.0])
            predicted_homogeneous = H_pred @ map_point_homogeneous

            # Normalize homogeneous coordinates (perspective division)
            if abs(predicted_homogeneous[2]) < 1e-10:
                # Point is at infinity (horizon), use large residual to penalize
                residuals[2 * res_idx] = self.INFINITY_RESIDUAL
                residuals[2 * res_idx + 1] = self.INFINITY_RESIDUAL
                logger.warning(f"GCP {gcp_idx} projects to infinity with params {params}")
                continue

            u_predicted = predicted_homogeneous[0] / predicted_homogeneous[2]
            v_predicted = predicted_homogeneous[1] / predicted_homogeneous[2]

            # Compute residuals (observed - predicted)
            residuals[2 * res_idx] = u_observed - u_predicted
            residuals[2 * res_idx + 1] = v_observed - v_predicted

        return residuals

    def _compute_regularization_residuals(self, params: np.ndarray) -> np.ndarray:
        """
        Compute regularization residuals for Tikhonov regularization.

        The regularization penalizes parameter deviations from initial values (zeros)
        weighted by prior uncertainties. This stabilizes optimization when GCP
        data is noisy.

        Mathematical formulation:
            r_prior[j] = sqrt(lambda) * (params[j] - 0) / sigma[j]

        When squared and summed by least_squares, this produces:
            E_prior = lambda * sum_j((params[j] / sigma[j])^2)

        Args:
            params: Parameter vector [Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ]
                    Units: [deg, deg, deg, m, m, m]

        Returns:
            Regularization residuals array of shape (6,)
            Returns zeros if regularization_weight is 0.0
        """
        # Handle case where regularization is disabled
        if self._regularization_weight == 0.0:
            return np.zeros(6, dtype=np.float64)

        # Compute regularization residuals:
        # r_prior[j] = sqrt(lambda) * (params[j] - 0) / sigma[j]
        # Since p₀ = 0 (initial parameters are zeros), this simplifies to:
        # r_prior[j] = sqrt(lambda) * params[j] / sigma[j]
        sqrt_lambda = np.sqrt(self._regularization_weight)
        regularization_residuals: np.ndarray = sqrt_lambda * params / self._sigma_vector

        return regularization_residuals

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
                self.loss_scale * abs_residuals - 0.5 * self.loss_scale**2,
            )
        elif self.loss_function == self.LOSS_CAUCHY:
            # Cauchy loss: logarithmic growth for large errors
            # ρ(r) = (scale²/2) · log(1 + (r/scale)²)
            loss_values = (self.loss_scale**2 / 2.0) * np.log1p((residuals / self.loss_scale) ** 2)
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
        return float(np.sqrt(np.mean(per_point_errors**2)))

    def _split_train_test(self) -> tuple[list[int], list[int]]:
        """
        Split GCPs into train and test sets based on validation_split.

        Uses stratified random sampling to ensure:
        - Train set has at least 6 GCPs (minimum for 6-DOF optimization)
        - Test set has at least 3 GCPs (minimum for meaningful validation)
        - No overlap between train and test sets
        - Reproducible with random_seed

        Returns:
            Tuple of (train_indices, test_indices)

        If insufficient GCPs, returns (all_indices, []) and logs warning.
        """
        n_gcps = len(self.gcps)
        all_indices = list(range(n_gcps))

        # If validation_split is 0, use all for training
        if self.validation_split == 0.0:
            return all_indices, []

        # Calculate split sizes
        n_test = max(3, int(n_gcps * self.validation_split))
        n_train = n_gcps - n_test

        # Check minimum requirements
        if n_train < 6 or n_test < 3:
            logger.warning(
                f"Insufficient GCPs for train/test split with validation_split={self.validation_split}. "
                f"Need at least 6 train + 3 test = 9 total, got {n_gcps}. "
                f"Using all GCPs for training."
            )
            return all_indices, []

        # Use local random generator for reproducibility without polluting global state
        rng = np.random.default_rng(self.random_seed)

        # Randomly shuffle indices
        shuffled_indices = all_indices.copy()
        rng.shuffle(shuffled_indices)

        # Split into train and test
        train_indices = sorted(shuffled_indices[:n_train])
        test_indices = sorted(shuffled_indices[n_train:])

        logger.info(
            f"Split {n_gcps} GCPs into {n_train} train and {n_test} test "
            f"(validation_split={self.validation_split}, seed={self.random_seed})"
        )

        return train_indices, test_indices

    def _compute_validation_metrics(
        self, params: np.ndarray, indices: list[int]
    ) -> dict[str, float]:
        """
        Compute validation metrics (RMS, P90, max) for specified GCP indices.

        Args:
            params: Parameter vector to evaluate
            indices: List of GCP indices to use

        Returns:
            Dictionary with 'rms', 'p90', 'max', 'count' metrics
        """
        if not indices:
            return {"rms": 0.0, "p90": 0.0, "max": 0.0, "count": 0}

        # Compute residuals for these GCPs
        residuals = self._compute_residuals(params, gcp_indices=indices)

        # Reshape to (N, 2) for per-point errors
        residuals_2d = residuals.reshape(-1, 2)
        per_point_errors = np.linalg.norm(residuals_2d, axis=1)

        # Compute metrics
        rms = np.sqrt(np.mean(per_point_errors**2))
        p90 = np.percentile(per_point_errors, 90)
        max_err = np.max(per_point_errors)

        return {"rms": float(rms), "p90": float(p90), "max": float(max_err), "count": len(indices)}

    def calibrate(self, bounds: dict[str, tuple[float, float]] | None = None) -> CalibrationResult:
        """
        Calibrate camera parameters using GCP-based reprojection error minimization.

        This method performs robust least-squares optimization to find the camera
        parameter adjustments (Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ) that minimize
        reprojection error between observed and predicted GCP pixel locations.

        The optimization uses scipy.optimize.least_squares with the selected
        robust loss function to handle outlier GCPs gracefully.

        If validation_split > 0, splits GCPs into train (optimization) and test
        (validation) sets, optimizes on train set, and evaluates on both sets.

        Args:
            bounds: Optional parameter bounds dictionary. Keys are parameter names
                   ('pan', 'tilt', 'roll', 'X', 'Y', 'Z'), values are (lower, upper)
                   tuples. If None, uses DEFAULT_BOUNDS.

        Returns:
            CalibrationResult with optimized parameters, statistics, and validation metrics

        Raises:
            RuntimeError: If optimization fails to converge

        Example:
            >>> calibrator = GCPCalibrator(camera_geo, gcps, validation_split=0.2)
            >>> result = calibrator.calibrate(bounds={'pan': (-5, 5), 'tilt': (-5, 5)})
            >>> print(f"Pan adjustment: {result.optimized_params[0]:.2f}°")
            >>> print(f"Train RMS: {result.validation_metrics['train']['rms']:.2f}px")
            >>> print(f"Test RMS: {result.validation_metrics['test']['rms']:.2f}px")
        """
        # Split train/test if validation_split > 0
        self._train_indices, self._test_indices = self._split_train_test()

        # Use default bounds if not provided
        if bounds is None:
            bounds = self.DEFAULT_BOUNDS.copy()
        else:
            # Merge with defaults (user bounds override defaults)
            merged_bounds = self.DEFAULT_BOUNDS.copy()
            merged_bounds.update(bounds)
            bounds = merged_bounds

        # Extract bounds in parameter vector order
        lower_bounds = np.array(
            [
                bounds["pan"][0],
                bounds["tilt"][0],
                bounds["roll"][0],
                bounds["X"][0],
                bounds["Y"][0],
                bounds["Z"][0],
            ]
        )
        upper_bounds = np.array(
            [
                bounds["pan"][1],
                bounds["tilt"][1],
                bounds["roll"][1],
                bounds["X"][1],
                bounds["Y"][1],
                bounds["Z"][1],
            ]
        )

        # Initial parameter guess: all zeros (no adjustment)
        params_initial = np.zeros(6)

        # Compute initial error (on train set if split, otherwise all GCPs)
        residuals_initial = self._compute_residuals(params_initial, gcp_indices=self._train_indices)
        initial_error = self._compute_rms_error(residuals_initial)

        logger.info(f"Starting calibration optimization: initial RMS error = {initial_error:.3f}px")

        # Map loss function names to scipy loss strings
        # Note: 'soft_l1' is scipy's pseudo-Huber loss: 2*((1+r²)^0.5 - 1)
        # It behaves similarly to Huber (quadratic for small r, linear for large r)
        scipy_loss_map = {
            "huber": "soft_l1",  # pseudo-Huber loss (smooth approximation)
            "cauchy": "cauchy",
        }
        scipy_loss = scipy_loss_map.get(self.loss_function, "linear")

        # Define residual function for optimization (uses train indices)
        # Returns reprojection residuals (2N elements) plus optional regularization
        # residuals (6 elements) for a total of (2N + 6) elements when regularization
        # is enabled
        def residual_fn(params):
            reproj_residuals = self._compute_residuals(params, gcp_indices=self._train_indices)

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
                f_scale=self.loss_scale,  # Scale parameter for robust loss
                method="trf",  # Trust Region Reflective (handles bounds)
                verbose=0,
                max_nfev=1000,  # Maximum function evaluations
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise RuntimeError(f"Calibration optimization failed: {e}")

        # Extract optimized parameters
        optimized_params = result.x

        # Compute final error on train set
        residuals_final = self._compute_residuals(optimized_params, gcp_indices=self._train_indices)
        final_error = self._compute_rms_error(residuals_final)

        # Compute per-GCP errors for ALL GCPs (for backward compatibility)
        residuals_all = self._compute_residuals(optimized_params, gcp_indices=None)
        residuals_2d = residuals_all.reshape(-1, 2)
        per_gcp_errors = np.linalg.norm(residuals_2d, axis=1).tolist()

        # Classify inliers/outliers based on robust loss threshold (on all GCPs)
        # Inliers: error <= loss_scale * 2 (conservative threshold)
        inlier_threshold = self.loss_scale * 2.0
        inlier_mask = np.array(per_gcp_errors) <= inlier_threshold
        num_inliers = int(np.sum(inlier_mask))
        num_outliers = len(self.gcps) - num_inliers
        inlier_ratio = num_inliers / len(self.gcps)

        # Build convergence info
        convergence_info = {
            "success": result.success,
            "message": result.message,
            "iterations": result.nfev,  # Number of function evaluations
            "function_evals": result.nfev,
            "optimality": result.optimality,
            "status": result.status,
        }

        # Add Jacobian evaluations if available
        if hasattr(result, "njev"):
            convergence_info["jacobian_evals"] = result.njev

        # Compute validation metrics if we have train/test split
        validation_metrics = None
        if self._train_indices and self._test_indices:
            train_metrics = self._compute_validation_metrics(optimized_params, self._train_indices)
            test_metrics = self._compute_validation_metrics(optimized_params, self._test_indices)
            validation_metrics = {"train": train_metrics, "test": test_metrics}

        logger.info(
            f"Calibration complete: final RMS error = {final_error:.3f}px "
            f"({num_inliers}/{len(self.gcps)} inliers, {result.nfev} iterations)"
        )

        if not result.success:
            logger.warning(f"Optimization did not fully converge: {result.message}")

        # Calculate regularization penalty if regularization is enabled
        if self._regularization_weight > 0.0:
            reg_residuals = self._compute_regularization_residuals(optimized_params)
            regularization_penalty = float(np.sum(reg_residuals**2))
        else:
            regularization_penalty = None

        return CalibrationResult(
            optimized_params=optimized_params,
            initial_error=initial_error,
            final_error=final_error,
            num_inliers=num_inliers,
            num_outliers=num_outliers,
            inlier_ratio=inlier_ratio,
            per_gcp_errors=per_gcp_errors,
            convergence_info=convergence_info,
            validation_metrics=validation_metrics,
            train_indices=self._train_indices,
            test_indices=self._test_indices,
            regularization_penalty=regularization_penalty,
        )


# ============================================================================
# Validation and Diagnostic Functions
# ============================================================================


def detect_systematic_errors(
    residuals_2d: np.ndarray,
    image_coords: np.ndarray,
    image_center: tuple[float, float] | None = None,
) -> list[str]:
    """
    Detect systematic error patterns in calibration residuals.

    Analyzes residual patterns to identify potential calibration issues:
    - Directional bias: Consistent residual direction across all GCPs
    - Radial growth: Errors increase with distance from image center

    Args:
        residuals_2d: Nx2 array of [Δu, Δv] residuals per GCP
        image_coords: Nx2 array of [u, v] observed pixel coordinates
        image_center: Optional (u_center, v_center) tuple. If None, uses
                     centroid of image_coords.

    Returns:
        List of warning strings describing detected systematic patterns.
        Empty list if no patterns detected.

    Example:
        >>> residuals = np.array([[1.0, 2.0], [-0.5, 1.5], [0.8, 2.2]])
        >>> image_coords = np.array([[100, 200], [500, 600], [900, 400]])
        >>> warnings = detect_systematic_errors(residuals, image_coords)
        >>> for w in warnings:
        ...     print(w)
        Directional bias detected: mean residual magnitude 2.1px
    """
    warnings: list[str] = []

    # Validate inputs
    if residuals_2d.shape[0] != image_coords.shape[0]:
        raise ValueError(
            f"residuals_2d and image_coords must have same length, "
            f"got {residuals_2d.shape[0]} vs {image_coords.shape[0]}"
        )

    if residuals_2d.shape[1] != 2 or image_coords.shape[1] != 2:
        raise ValueError("residuals_2d and image_coords must be Nx2 arrays")

    # Early return for insufficient points (need at least 2 for meaningful analysis)
    if residuals_2d.shape[0] < 2:
        return warnings

    # Compute image center if not provided
    if image_center is None:
        image_center = (float(np.mean(image_coords[:, 0])), float(np.mean(image_coords[:, 1])))

    # 1. Check for directional bias
    # Compute mean residual vector
    mean_residual = np.mean(residuals_2d, axis=0)
    mean_residual_magnitude = np.linalg.norm(mean_residual)

    # Threshold: flag if mean residual > 2 pixels
    DIRECTIONAL_BIAS_THRESHOLD = 2.0
    if mean_residual_magnitude > DIRECTIONAL_BIAS_THRESHOLD:
        warnings.append(
            f"Directional bias detected: mean residual vector ({mean_residual[0]:.2f}, "
            f"{mean_residual[1]:.2f}) with magnitude {mean_residual_magnitude:.2f}px "
            f"(threshold: {DIRECTIONAL_BIAS_THRESHOLD}px)"
        )

    # 2. Check for radial error growth
    # Compute distance from image center for each point
    distances = np.sqrt(
        (image_coords[:, 0] - image_center[0]) ** 2 + (image_coords[:, 1] - image_center[1]) ** 2
    )

    # Compute error magnitude for each point
    error_magnitudes = np.linalg.norm(residuals_2d, axis=1)

    # Compute correlation between distance and error magnitude
    # Need at least 5 points for statistically meaningful correlation
    MIN_POINTS_FOR_CORRELATION = 5
    if len(distances) >= MIN_POINTS_FOR_CORRELATION:
        # Check for zero variance (all distances or errors identical)
        if np.std(distances) > 1e-10 and np.std(error_magnitudes) > 1e-10:
            correlation = np.corrcoef(distances, error_magnitudes)[0, 1]

            # Threshold: flag if correlation > 0.5
            RADIAL_CORRELATION_THRESHOLD = 0.5
            if not np.isnan(correlation) and correlation > RADIAL_CORRELATION_THRESHOLD:
                warnings.append(
                    f"Radial error growth detected: correlation between distance from center "
                    f"and error magnitude is {correlation:.3f} (threshold: {RADIAL_CORRELATION_THRESHOLD})"
                )

    return warnings


def generate_residual_plot(
    calibration_result: CalibrationResult,
    gcps: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    output_path: str | None = None,
):
    """
    Generate scatter plot of calibration residuals on image plane.

    Creates visualization showing:
    - GCP locations on image plane
    - Error magnitude indicated by point color (viridis colormap)
    - Train/test split distinguished by marker shape (circles vs triangles)

    Args:
        calibration_result: CalibrationResult from calibration
        gcps: List of GCP dictionaries with flat map-pixel format:
              {'map_id': str, 'map_pixel_x': float, 'map_pixel_y': float,
               'image_u': float, 'image_v': float}
        image_width: Camera image width in pixels
        image_height: Camera image height in pixels
        output_path: Optional path to save plot. If None, returns figure without saving.

    Returns:
        matplotlib Figure object

    Raises:
        ImportError: If matplotlib is not available
        ValueError: If number of GCPs doesn't match per_gcp_errors length
    """
    if not MATPLOTLIB_AVAILABLE or plt is None:
        raise ImportError(
            "matplotlib is required for generate_residual_plot(). "
            "Install it with: pip install matplotlib"
        )

    # Validate inputs
    if len(gcps) != len(calibration_result.per_gcp_errors):
        raise ValueError(
            f"Number of GCPs ({len(gcps)}) doesn't match per_gcp_errors length "
            f"({len(calibration_result.per_gcp_errors)})"
        )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Determine train/test split
    train_indices = (
        calibration_result.train_indices
        if calibration_result.train_indices
        else list(range(len(gcps)))
    )
    test_indices = calibration_result.test_indices if calibration_result.test_indices else []

    # Extract image coordinates (flat format)
    u_coords = np.array([gcp["image_u"] for gcp in gcps])
    v_coords = np.array([gcp["image_v"] for gcp in gcps])

    # Compute unified color scale across all errors
    all_errors = np.array(calibration_result.per_gcp_errors)
    vmin, vmax = 0, np.max(all_errors) if len(all_errors) > 0 else 1

    # Plot train points (circles)
    train_u = u_coords[train_indices]
    train_v = v_coords[train_indices]
    train_errors = all_errors[train_indices]

    scatter_train = ax.scatter(
        train_u,
        train_v,
        c=train_errors,
        s=100,
        alpha=0.6,
        cmap="viridis",
        edgecolors="black",
        linewidths=1,
        label=f"Train ({len(train_indices)} GCPs)",
        vmin=vmin,
        vmax=vmax,
    )

    # Plot test points if available (triangles with same colormap)
    if test_indices:
        test_u = u_coords[test_indices]
        test_v = v_coords[test_indices]
        test_errors = all_errors[test_indices]

        ax.scatter(
            test_u,
            test_v,
            c=test_errors,
            s=100,
            alpha=0.6,
            cmap="viridis",
            marker="^",
            edgecolors="red",
            linewidths=2,
            label=f"Test ({len(test_indices)} GCPs)",
            vmin=vmin,
            vmax=vmax,
        )

    # Add colorbar using unified scale
    cbar = plt.colorbar(scatter_train, ax=ax, label="Reprojection Error (px)")

    # Set axis labels and title
    ax.set_xlabel("Image U (pixels)", fontsize=12)
    ax.set_ylabel("Image V (pixels)", fontsize=12)
    ax.set_title(
        f"Calibration Residuals\n"
        f"RMS Error: {calibration_result.final_error:.2f}px | "
        f"Inliers: {calibration_result.num_inliers}/{len(gcps)}",
        fontsize=14,
    )

    # Set axis limits
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)  # Invert Y axis (image coordinates)
    ax.set_aspect("equal")

    # Add legend
    ax.legend(loc="best", fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save if output path provided
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Residual plot saved to {output_path}")

    return fig


def generate_residual_histogram(
    calibration_result: CalibrationResult, bins: int = 20, output_path: str | None = None
):
    """
    Generate histogram of residual magnitudes.

    Creates visualization showing:
    - Distribution of per-GCP reprojection errors
    - Separate histograms for train/test sets (if available)
    - Statistical markers (mean, median, P90)

    Args:
        calibration_result: CalibrationResult from calibration
        bins: Number of histogram bins (default: 20)
        output_path: Optional path to save plot. If None, returns figure without saving.

    Returns:
        matplotlib Figure object

    Raises:
        ImportError: If matplotlib is not available
    """
    if not MATPLOTLIB_AVAILABLE or plt is None:
        raise ImportError(
            "matplotlib is required for generate_residual_histogram(). "
            "Install it with: pip install matplotlib"
        )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get all errors
    all_errors = np.array(calibration_result.per_gcp_errors)

    # Determine train/test split
    train_indices = (
        calibration_result.train_indices
        if calibration_result.train_indices
        else list(range(len(all_errors)))
    )
    test_indices = calibration_result.test_indices if calibration_result.test_indices else []

    # Plot train histogram
    train_errors = all_errors[train_indices]
    ax.hist(
        train_errors,
        bins=bins,
        alpha=0.6,
        label=f"Train ({len(train_indices)} GCPs)",
        color="blue",
        edgecolor="black",
    )

    # Plot test histogram if available
    if test_indices:
        test_errors = all_errors[test_indices]
        ax.hist(
            test_errors,
            bins=bins,
            alpha=0.6,
            label=f"Test ({len(test_indices)} GCPs)",
            color="red",
            edgecolor="black",
        )

    # Add statistical markers
    train_mean = float(np.mean(train_errors))
    train_median = float(np.median(train_errors))
    ax.axvline(
        train_mean,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Train Mean: {train_mean:.2f}px",
    )
    ax.axvline(
        train_median,
        color="blue",
        linestyle=":",
        linewidth=2,
        label=f"Train Median: {train_median:.2f}px",
    )

    if test_indices:
        test_mean = float(np.mean(test_errors))
        test_median = float(np.median(test_errors))
        ax.axvline(
            test_mean,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Test Mean: {test_mean:.2f}px",
        )
        ax.axvline(
            test_median,
            color="red",
            linestyle=":",
            linewidth=2,
            label=f"Test Median: {test_median:.2f}px",
        )

    # Set axis labels and title
    ax.set_xlabel("Reprojection Error (pixels)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Calibration Residuals", fontsize=14)

    # Add legend
    ax.legend(loc="best", fontsize=9)

    # Add grid
    ax.grid(True, alpha=0.3, axis="y")

    # Save if output path provided
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Residual histogram saved to {output_path}")

    return fig


def validate_calibration(
    calibration_result: CalibrationResult, thresholds: dict[str, float] | None = None
) -> tuple[bool, list[str]]:
    """
    Validate calibration result against quality thresholds.

    Checks calibration quality metrics against configurable thresholds:
    - Train RMS error
    - Test RMS error (if validation split used)
    - Train P90 error
    - Test P90 error (if validation split used)
    - Inlier ratio

    Args:
        calibration_result: CalibrationResult from calibration
        thresholds: Optional dictionary of threshold values:
            - 'train_rms_max': Maximum acceptable train RMS error (pixels)
            - 'test_rms_max': Maximum acceptable test RMS error (pixels)
            - 'train_p90_max': Maximum acceptable train 90th percentile error (pixels)
            - 'test_p90_max': Maximum acceptable test 90th percentile error (pixels)
            - 'min_inlier_ratio': Minimum acceptable inlier ratio [0.0, 1.0]
            If None, uses default thresholds.

    Returns:
        Tuple of (is_valid, failure_messages)
        - is_valid: True if all checks pass, False otherwise
        - failure_messages: List of failure descriptions (empty if all pass)

    Example:
        >>> thresholds = {'train_rms_max': 3.0, 'test_rms_max': 3.5, 'min_inlier_ratio': 0.8}
        >>> is_valid, failures = validate_calibration(result, thresholds)
        >>> if not is_valid:
        ...     for failure in failures:
        ...         print(f"FAIL: {failure}")
    """
    # Default thresholds (generous for typical calibration)
    default_thresholds = {
        "train_rms_max": 5.0,  # 5 pixels RMS
        "test_rms_max": 6.0,  # 6 pixels RMS (slightly higher for test)
        "train_p90_max": 8.0,  # 8 pixels P90
        "test_p90_max": 10.0,  # 10 pixels P90
        "min_inlier_ratio": 0.7,  # 70% inliers minimum
    }

    # Use provided thresholds or defaults
    if thresholds is None:
        thresholds = default_thresholds
    else:
        # Merge with defaults (provided thresholds override defaults)
        merged = default_thresholds.copy()
        merged.update(thresholds)
        thresholds = merged

    failures = []

    # Check inlier ratio (always available)
    if calibration_result.inlier_ratio < thresholds["min_inlier_ratio"]:
        failures.append(
            f"Inlier ratio {calibration_result.inlier_ratio:.2f} below threshold "
            f"{thresholds['min_inlier_ratio']:.2f}"
        )

    # Check train/test metrics if validation_metrics available
    if calibration_result.validation_metrics:
        train_metrics = calibration_result.validation_metrics["train"]
        test_metrics = calibration_result.validation_metrics["test"]

        # Check train RMS
        if train_metrics["rms"] > thresholds["train_rms_max"]:
            failures.append(
                f"Train RMS error {train_metrics['rms']:.2f}px exceeds threshold "
                f"{thresholds['train_rms_max']:.2f}px"
            )

        # Check test RMS (if test set exists)
        if test_metrics["count"] > 0 and test_metrics["rms"] > thresholds["test_rms_max"]:
            failures.append(
                f"Test RMS error {test_metrics['rms']:.2f}px exceeds threshold "
                f"{thresholds['test_rms_max']:.2f}px"
            )

        # Check train P90
        if train_metrics["p90"] > thresholds["train_p90_max"]:
            failures.append(
                f"Train P90 error {train_metrics['p90']:.2f}px exceeds threshold "
                f"{thresholds['train_p90_max']:.2f}px"
            )

        # Check test P90 (if test set exists)
        if test_metrics["count"] > 0 and test_metrics["p90"] > thresholds["test_p90_max"]:
            failures.append(
                f"Test P90 error {test_metrics['p90']:.2f}px exceeds threshold "
                f"{thresholds['test_p90_max']:.2f}px"
            )

    is_valid = len(failures) == 0
    return is_valid, failures


def generate_validation_report(
    calibration_result: CalibrationResult, validation_thresholds: dict[str, float] | None = None
) -> str:
    """
    Generate comprehensive text report of calibration validation.

    Creates detailed report including:
    - Optimized parameter values
    - Error metrics (RMS, P90, max) for train/test sets
    - Convergence information
    - Validation pass/fail status (if thresholds provided)

    Args:
        calibration_result: CalibrationResult from calibration
        validation_thresholds: Optional thresholds for pass/fail validation

    Returns:
        Formatted validation report as string

    Example:
        >>> report = generate_validation_report(result, thresholds={'train_rms_max': 3.0})
        >>> print(report)
        ========================================
        Calibration Validation Report
        ========================================
        ...
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Calibration Validation Report")
    lines.append("=" * 60)
    lines.append(f"Timestamp: {calibration_result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    # Optimized Parameters
    lines.append("Optimized Parameters:")
    lines.append("-" * 60)
    params = calibration_result.optimized_params
    lines.append(f"  Pan adjustment:   {params[0]:+.3f}°")
    lines.append(f"  Tilt adjustment:  {params[1]:+.3f}°")
    lines.append(f"  Roll adjustment:  {params[2]:+.3f}° (reserved)")
    lines.append(f"  X adjustment:     {params[3]:+.3f} m")
    lines.append(f"  Y adjustment:     {params[4]:+.3f} m")
    lines.append(f"  Z adjustment:     {params[5]:+.3f} m")
    lines.append("")

    # Error Metrics
    lines.append("Error Metrics:")
    lines.append("-" * 60)
    lines.append(f"  Initial RMS error: {calibration_result.initial_error:.2f} px")
    lines.append(f"  Final RMS error:   {calibration_result.final_error:.2f} px")
    improvement_px = calibration_result.initial_error - calibration_result.final_error
    if calibration_result.initial_error > 0:
        improvement_pct = (
            1 - calibration_result.final_error / calibration_result.initial_error
        ) * 100
        lines.append(f"  Improvement:       {improvement_px:.2f} px ({improvement_pct:.1f}%)")
    else:
        lines.append(f"  Improvement:       {improvement_px:.2f} px (N/A%)")
    lines.append("")
    lines.append(
        f"  Inliers:  {calibration_result.num_inliers} / {calibration_result.num_inliers + calibration_result.num_outliers} "
        f"({calibration_result.inlier_ratio * 100:.1f}%)"
    )
    lines.append(f"  Outliers: {calibration_result.num_outliers}")
    lines.append("")

    # Train/Test Metrics (if available)
    if calibration_result.validation_metrics:
        lines.append("Train/Test Split Metrics:")
        lines.append("-" * 60)

        train_metrics = calibration_result.validation_metrics["train"]
        lines.append(f"  Train Set ({train_metrics['count']} GCPs):")
        lines.append(f"    RMS error: {train_metrics['rms']:.2f} px")
        lines.append(f"    P90 error: {train_metrics['p90']:.2f} px")
        lines.append(f"    Max error: {train_metrics['max']:.2f} px")
        lines.append("")

        test_metrics = calibration_result.validation_metrics["test"]
        if test_metrics["count"] > 0:
            lines.append(f"  Test Set ({test_metrics['count']} GCPs):")
            lines.append(f"    RMS error: {test_metrics['rms']:.2f} px")
            lines.append(f"    P90 error: {test_metrics['p90']:.2f} px")
            lines.append(f"    Max error: {test_metrics['max']:.2f} px")
            lines.append("")

    # Convergence Information
    lines.append("Convergence:")
    lines.append("-" * 60)
    conv = calibration_result.convergence_info
    lines.append(f"  Status:           {'SUCCESS' if conv.get('success', False) else 'FAILED'}")
    if "message" in conv:
        lines.append(f"  Message:          {conv['message']}")
    if "iterations" in conv:
        lines.append(f"  Iterations:       {conv['iterations']}")
    if "function_evals" in conv:
        lines.append(f"  Function evals:   {conv['function_evals']}")
    if "jacobian_evals" in conv:
        lines.append(f"  Jacobian evals:   {conv['jacobian_evals']}")
    if "optimality" in conv:
        lines.append(f"  Optimality:       {conv['optimality']:.2e}")
    lines.append("")

    # Validation Status (if thresholds provided)
    if validation_thresholds:
        is_valid, failures = validate_calibration(calibration_result, validation_thresholds)
        lines.append("Validation Status:")
        lines.append("-" * 60)

        if is_valid:
            lines.append("  Status: PASS")
            lines.append("  All quality thresholds met.")
        else:
            lines.append("  Status: FAIL")
            lines.append("  Failures:")
            for failure in failures:
                lines.append(f"    - {failure}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
