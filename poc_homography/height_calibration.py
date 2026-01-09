#!/usr/bin/env python3
"""
Automatic height calibration from multiple GPS reference points.

This module provides data classes and a calibrator for estimating camera height
by comparing homography-projected distances with actual GPS distances. The height
estimation uses scale factors derived from multiple reference points.

Key Concepts:
    - Homography distance: Distance calculated from image projection to world plane
    - GPS distance: Actual ground truth distance from haversine formula
    - Scale factor: Ratio of GPS distance to homography distance
    - Height calibration: Adjusting camera height to minimize distance errors

The calibration process:
    1. Collect multiple reference points (pixel + GPS coordinates)
    2. For each point, calculate homography distance and GPS distance
    3. Compute scale factor = gps_distance / homography_distance
    4. Estimate height = current_height * scale_factor
    5. Aggregate estimates to find optimal height

Usage Example:
    >>> from poc_homography.camera_geometry import CameraGeometry
    >>>
    >>> camera_gps = {"lat": "39°38'25.7\"N", "lon": "0°13'48.7\"W"}
    >>> calibrator = HeightCalibrator(camera_gps, min_points=5)
    >>>
    >>> # Add calibration points (from user clicking on image)
    >>> calibrator.add_point(
    ...     pixel_x=100, pixel_y=200,
    ...     gps_lat=39.640444, gps_lon=-0.230111,
    ...     current_height=5.0, geo=camera_geometry
    ... )
    >>>
    >>> # Check if ready for calibration
    >>> if calibrator.is_ready():
    ...     estimates = calibrator.get_all_height_estimates()
    ...     print(f"Height estimates: {estimates}")
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as t_dist

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.gps_distance_calculator import dms_to_dd, haversine_distance

# Numerical threshold for detecting near-zero values (used for horizon detection
# and invalid homography distances). Values below this are considered effectively zero.
NEAR_ZERO_THRESHOLD = 1e-6


@dataclass
class CalibrationPoint:
    """
    Represents a single calibration point with pixel, GPS, and distance data.

    A calibration point pairs an image pixel location with its corresponding
    GPS coordinates, allowing comparison between homography-projected distances
    and actual GPS distances.

    Attributes:
        pixel_x: Horizontal pixel coordinate in image
        pixel_y: Vertical pixel coordinate in image
        gps_lat: GPS latitude in decimal degrees
        gps_lon: GPS longitude in decimal degrees
        world_x: World X coordinate (East-West) in meters from homography
        world_y: World Y coordinate (North-South) in meters from homography
        gps_distance: Actual distance from camera to point via GPS (meters)
        homography_distance: Distance from camera to point via homography (meters)
        current_height: Camera height setting when this point was collected (meters)
    """

    pixel_x: float
    pixel_y: float
    gps_lat: float
    gps_lon: float
    world_x: float
    world_y: float
    gps_distance: float
    homography_distance: float
    current_height: float


@dataclass
class CalibrationResult:
    """
    Results from a height calibration estimation.

    Contains the estimated camera height along with statistical information
    about the calibration quality and the underlying data points.

    Attributes:
        estimated_height: Estimated camera height in meters
        confidence_interval: Tuple of (lower_bound, upper_bound) for 95% confidence
        inlier_count: Number of points used in final estimation
        outlier_count: Number of points rejected as outliers
        timestamp: When the calibration was performed
        calibration_points: List of calibration points used in estimation
    """

    estimated_height: float
    confidence_interval: tuple[float, float]
    inlier_count: int
    outlier_count: int
    timestamp: datetime
    calibration_points: list[CalibrationPoint] = field(default_factory=list)


class HeightCalibrator:
    """
    Calibrates camera height using multiple GPS reference points.

    The calibrator collects pairs of (pixel coordinates, GPS coordinates) and
    uses the discrepancy between homography-projected distances and actual GPS
    distances to estimate the correct camera height.

    The height estimation is based on the principle that if homography distances
    are incorrect, it's often due to incorrect camera height. The scale factor
    between GPS and homography distances can be used to correct the height.

    Attributes:
        camera_gps: Camera GPS coordinates (can be dict with DMS strings or already converted)
        camera_lat_dd: Camera latitude in decimal degrees
        camera_lon_dd: Camera longitude in decimal degrees
        min_points: Minimum number of calibration points required
        calibration_points: List of collected calibration points
    """

    def __init__(self, camera_gps: dict, min_points: int = 5):
        """
        Initialize the height calibrator.

        Args:
            camera_gps: Camera GPS coordinates. Can be either:
                       - {"lat": "39°38'25.7\"N", "lon": "0°13'48.7\"W"} (DMS format)
                       - {"lat": 39.640472, "lon": -0.230194} (decimal degrees)
            min_points: Minimum number of calibration points needed (default: 5)

        Raises:
            ValueError: If camera_gps is missing required keys or min_points is invalid
        """
        if "lat" not in camera_gps or "lon" not in camera_gps:
            raise ValueError("camera_gps must contain 'lat' and 'lon' keys")

        if min_points < 1:
            raise ValueError(f"min_points must be at least 1, got {min_points}")

        self.camera_gps = camera_gps
        self.min_points = min_points
        self.calibration_points: list[CalibrationPoint] = []

        # Convert camera GPS to decimal degrees if needed
        if isinstance(camera_gps["lat"], str):
            self.camera_lat_dd = dms_to_dd(camera_gps["lat"])
            self.camera_lon_dd = dms_to_dd(camera_gps["lon"])
        else:
            self.camera_lat_dd = float(camera_gps["lat"])
            self.camera_lon_dd = float(camera_gps["lon"])

    def add_point(
        self,
        pixel_x: float,
        pixel_y: float,
        gps_lat: float,
        gps_lon: float,
        current_height: float,
        geo: CameraGeometry,
    ) -> CalibrationPoint:
        """
        Add a calibration point by clicking on image and providing GPS coordinates.

        This method:
        1. Projects pixel coordinates to world plane using homography
        2. Calculates homography distance from camera to point
        3. Calculates actual GPS distance using haversine formula
        4. Stores the point for later height estimation

        Args:
            pixel_x: Horizontal pixel coordinate in image
            pixel_y: Vertical pixel coordinate in image
            gps_lat: GPS latitude of the point in decimal degrees
            gps_lon: GPS longitude of the point in decimal degrees
            current_height: Current camera height setting in meters
            geo: CameraGeometry instance with current homography matrix

        Returns:
            CalibrationPoint object that was added

        Raises:
            ValueError: If pixel coordinates cannot be projected (e.g., near horizon)
            ValueError: If current_height is zero or negative
        """
        # Validate current_height to prevent division by zero in calibration
        if current_height <= 0:
            raise ValueError(f"current_height must be positive, got {current_height}m")

        # Project image point to world coordinates using homography
        pt_img = np.array([[pixel_x], [pixel_y], [1.0]])
        pt_world = geo.H_inv @ pt_img

        # Check for valid projection (not near horizon)
        if abs(pt_world[2, 0]) < NEAR_ZERO_THRESHOLD:
            raise ValueError(
                f"Invalid point at pixel ({pixel_x}, {pixel_y}): "
                "Point is too close to horizon and cannot be projected"
            )

        # Normalize homogeneous coordinates
        world_x = pt_world[0, 0] / pt_world[2, 0]
        world_y = pt_world[1, 0] / pt_world[2, 0]

        # Calculate homography distance from camera to point
        homography_distance = np.sqrt(world_x**2 + world_y**2)

        # Calculate actual GPS distance using haversine formula
        gps_distance = haversine_distance(self.camera_lat_dd, self.camera_lon_dd, gps_lat, gps_lon)

        # Create and store calibration point
        point = CalibrationPoint(
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            world_x=world_x,
            world_y=world_y,
            gps_distance=gps_distance,
            homography_distance=homography_distance,
            current_height=current_height,
        )

        self.calibration_points.append(point)
        return point

    def clear_points(self) -> None:
        """
        Clear all collected calibration points.

        This resets the calibrator to its initial state, removing all collected
        data points. Useful when starting a new calibration session or when
        recalibrating with a new height.
        """
        self.calibration_points.clear()

    def get_point_count(self) -> int:
        """
        Get the number of calibration points collected.

        Returns:
            Number of calibration points currently stored
        """
        return len(self.calibration_points)

    def estimate_height_from_point(self, point: CalibrationPoint) -> float:
        """
        Estimate camera height from a single calibration point.

        The estimation uses the scale factor between GPS and homography distances:
            scale_factor = gps_distance / homography_distance
            estimated_height = current_height * scale_factor

        This works because homography distances are proportional to camera height.
        If the homography underestimates distance, the camera is likely too low,
        and vice versa.

        Args:
            point: CalibrationPoint to use for estimation (includes current_height)

        Returns:
            Estimated camera height in meters

        Raises:
            ValueError: If homography_distance is zero or near-zero
        """
        if abs(point.homography_distance) < NEAR_ZERO_THRESHOLD:
            raise ValueError(
                f"Cannot estimate height from point with near-zero homography distance: "
                f"{point.homography_distance}m"
            )

        # Calculate scale factor: how much to scale distances
        scale_factor = point.gps_distance / point.homography_distance

        # Height scales proportionally with distance
        estimated_height = point.current_height * scale_factor

        return estimated_height

    def get_all_height_estimates(self) -> list[float]:
        """
        Get height estimates from all calibration points.

        For each calibration point, computes the height estimate using the
        scale factor method. This returns a list of estimates that can be
        used for statistical analysis or averaging.

        Each point is estimated independently using the height setting that
        was active when the point was collected (stored in point.current_height).

        Returns:
            List of estimated heights in meters, one per calibration point

        Raises:
            ValueError: If no calibration points have been collected

        Note:
            Points with near-zero homography distance are skipped as they
            cannot produce reliable height estimates (e.g., points near horizon).
        """
        if not self.calibration_points:
            raise ValueError("No calibration points available for height estimation")

        estimates = []
        for point in self.calibration_points:
            # Skip points with invalid homography distance
            if abs(point.homography_distance) < NEAR_ZERO_THRESHOLD:
                continue

            try:
                height_estimate = self.estimate_height_from_point(point)
                estimates.append(height_estimate)
            except ValueError:
                # Skip points that can't be estimated
                continue

        return estimates

    def is_ready(self) -> bool:
        """
        Check if calibrator has enough points for height estimation.

        Returns:
            True if number of collected points >= min_points, False otherwise
        """
        return len(self.calibration_points) >= self.min_points

    def _compute_residual(self, height: float | np.ndarray) -> np.ndarray:
        """
        Compute residuals for all calibration points at a given height.

        For each calibration point, the residual is the difference between:
        - GPS distance (ground truth)
        - Expected homography distance if we were to use the given height

        The relationship used is:
            new_homography_distance = original_homography_distance * (height / original_height)

        This works because homography distances scale linearly with camera height.
        The residual measures how well a proposed height explains all the data.

        Args:
            height: Proposed camera height in meters (scalar or 1D array)

        Returns:
            Array of residuals, one per calibration point (in meters)
        """
        # Handle array input from scipy.optimize.least_squares
        if isinstance(height, np.ndarray):
            height = height[0]

        residuals = []
        for point in self.calibration_points:
            # Skip points with invalid homography distance
            if abs(point.homography_distance) < NEAR_ZERO_THRESHOLD:
                continue

            # Scale the homography distance based on the proposed height
            # If height increases, homography distances increase proportionally
            expected_homography_distance = point.homography_distance * (
                height / point.current_height
            )

            # Residual = GPS distance - expected homography distance
            # Positive residual: GPS distance is larger (height is too low)
            # Negative residual: GPS distance is smaller (height is too high)
            residual = point.gps_distance - expected_homography_distance
            residuals.append(residual)

        return np.array(residuals)

    def _compute_confidence_interval(
        self, height_estimates: list[float], estimated_height: float, confidence_level: float = 0.95
    ) -> tuple[float, float]:
        """
        Compute confidence interval using t-distribution for small samples.

        For a given set of height estimates and a final estimated height,
        this method computes a confidence interval using the Student's t-distribution.
        This is appropriate for small sample sizes (typical in calibration) and
        accounts for uncertainty in estimating the population standard deviation.

        The confidence interval is calculated as:
            CI = estimated_height ± t * (std / sqrt(n))

        Where:
            - t is the t-critical value for n-1 degrees of freedom at confidence_level
            - std is the sample standard deviation of height_estimates
            - n is the number of height estimates

        Args:
            height_estimates: List of individual height estimates from calibration points
            estimated_height: Final estimated height (e.g., from optimization)
            confidence_level: Desired confidence level (default: 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound) defining the confidence interval

        Note:
            If n < 2, we cannot compute a meaningful confidence interval, so we
            return (estimated_height, estimated_height) indicating zero interval width.
        """
        n = len(height_estimates)

        # Edge case: cannot compute confidence interval with fewer than 2 samples
        if n < 2:
            return (estimated_height, estimated_height)

        # Calculate sample standard deviation
        std = np.std(height_estimates, ddof=1)  # ddof=1 for sample std

        # Calculate standard error
        std_error = std / np.sqrt(n)

        # Get t-critical value for two-tailed test
        # degrees of freedom = n - 1
        # alpha/2 for two-tailed test
        alpha = 1 - confidence_level
        t_critical = t_dist.ppf(1 - alpha / 2, df=n - 1)

        # Calculate margin of error
        margin_of_error = t_critical * std_error

        # Compute confidence interval bounds
        lower_bound = estimated_height - margin_of_error
        upper_bound = estimated_height + margin_of_error

        return (lower_bound, upper_bound)

    def optimize_height_least_squares(self) -> CalibrationResult:
        """
        Estimate optimal camera height using least-squares optimization.

        This method finds the camera height that minimizes the sum of squared
        residuals across all calibration points. The optimization uses scipy's
        least_squares function with the relationship:

            homography_distance(h) = original_homography_distance * (h / original_height)

        The optimal height minimizes:
            sum((gps_distance - homography_distance(h))^2) for all points

        Returns:
            CalibrationResult with:
                - estimated_height: Optimized camera height in meters
                - confidence_interval: 95% confidence interval using t-distribution
                - inlier_count: Number of calibration points used
                - outlier_count: 0 (no outlier detection yet)
                - timestamp: Current time
                - calibration_points: All points used in optimization

        Raises:
            ValueError: If is_ready() returns False (not enough points)
            ValueError: If all points have invalid homography distances

        Note:
            The confidence interval is computed using the t-distribution, which is
            appropriate for small sample sizes typical in calibration scenarios.
        """
        if not self.is_ready():
            raise ValueError(
                f"Not enough calibration points for optimization. "
                f"Have {len(self.calibration_points)}, need {self.min_points}"
            )

        # Filter out points with invalid homography distance
        valid_points = [
            p for p in self.calibration_points if abs(p.homography_distance) >= NEAR_ZERO_THRESHOLD
        ]

        if not valid_points:
            raise ValueError(
                "All calibration points have invalid homography distances. "
                "Cannot perform height optimization."
            )

        # Use median of individual height estimates as initial guess
        individual_estimates = []
        for point in valid_points:
            scale_factor = point.gps_distance / point.homography_distance
            estimated_height = point.current_height * scale_factor
            individual_estimates.append(estimated_height)

        initial_guess = np.median(individual_estimates)

        # Run least-squares optimization
        result = least_squares(
            fun=self._compute_residual,
            x0=[initial_guess],
            method="lm",  # Levenberg-Marquardt algorithm
        )

        # Extract optimized height
        optimized_height = result.x[0]

        # Compute confidence interval using t-distribution
        # Use the individual height estimates from valid points
        confidence_interval = self._compute_confidence_interval(
            height_estimates=individual_estimates,
            estimated_height=optimized_height,
            confidence_level=0.95,
        )

        return CalibrationResult(
            estimated_height=optimized_height,
            confidence_interval=confidence_interval,
            inlier_count=len(valid_points),
            outlier_count=0,  # No outlier detection yet
            timestamp=datetime.now(),
            calibration_points=self.calibration_points.copy(),
        )

    def _detect_outliers_mad(
        self, height_estimates: list[float], threshold: float = 2.5
    ) -> tuple[list[float], list[int]]:
        """
        Detect outliers using MAD (Median Absolute Deviation) method.

        MAD is a robust statistical measure of variability that is less sensitive
        to extreme values than standard deviation. This method identifies outliers
        by measuring how far each estimate deviates from the median.

        The MAD is defined as:
            MAD = median(|x_i - median(x)|)

        A point is considered an outlier if:
            |x_i - median| > threshold * MAD * 1.4826

        The factor 1.4826 is a scale constant that makes MAD comparable to standard
        deviation for normally distributed data.

        Args:
            height_estimates: List of height estimates to analyze
            threshold: Number of MAD units for outlier threshold (default: 2.5)
                      Higher values are more permissive (fewer outliers)

        Returns:
            Tuple of (inlier_estimates, inlier_indices):
                - inlier_estimates: List of height estimates that are not outliers
                - inlier_indices: Indices of inliers in original height_estimates list

        Raises:
            ValueError: If height_estimates is empty or has fewer than 3 elements
        """
        if not height_estimates:
            raise ValueError("Cannot detect outliers: height_estimates is empty")

        if len(height_estimates) < 3:
            raise ValueError(
                f"Need at least 3 height estimates for MAD outlier detection, "
                f"got {len(height_estimates)}"
            )

        estimates_array = np.array(height_estimates)

        # Compute median
        median = np.median(estimates_array)

        # Compute MAD (Median Absolute Deviation)
        absolute_deviations = np.abs(estimates_array - median)
        mad = np.median(absolute_deviations)

        # Handle case where MAD is zero (all estimates are identical)
        if mad < 1e-10:
            # All points are inliers if they're all the same
            return (height_estimates.copy(), list(range(len(height_estimates))))

        # Scale factor for normal distribution comparison
        scale_factor = 1.4826

        # Identify inliers: points within threshold * MAD * scale_factor
        threshold_value = threshold * mad * scale_factor
        is_inlier = absolute_deviations <= threshold_value

        # Extract inliers and their indices
        inlier_indices = [i for i, inlier in enumerate(is_inlier) if inlier]
        inlier_estimates = [height_estimates[i] for i in inlier_indices]

        return inlier_estimates, inlier_indices

    def _detect_outliers_ransac(
        self,
        height_estimates: list[float],
        min_samples: int = 3,
        max_trials: int = 100,
        threshold_ratio: float = 0.1,
    ) -> tuple[list[float], list[int]]:
        """
        Detect outliers using RANSAC (Random Sample Consensus) method.

        RANSAC is an iterative method to estimate parameters of a mathematical model
        from a set of observed data that contains outliers. For height estimation,
        we use the median of random samples as the model.

        Algorithm:
            1. For each trial:
               a. Randomly sample min_samples estimates
               b. Compute their median (the model)
               c. Count how many estimates are within threshold_ratio of this median
            2. Select the model (median) with the most inliers
            3. Return all estimates that are inliers to the best model

        Args:
            height_estimates: List of height estimates to analyze
            min_samples: Minimum number of samples to draw per trial (default: 3)
            max_trials: Maximum number of RANSAC iterations (default: 100)
            threshold_ratio: Inlier threshold as ratio of median value (default: 0.1)
                           e.g., 0.1 means within 10% of median

        Returns:
            Tuple of (inlier_estimates, inlier_indices):
                - inlier_estimates: List of height estimates that are inliers
                - inlier_indices: Indices of inliers in original height_estimates list

        Raises:
            ValueError: If height_estimates has fewer elements than min_samples
            ValueError: If min_samples < 1 or max_trials < 1 or threshold_ratio <= 0
        """
        if not height_estimates:
            raise ValueError("Cannot detect outliers: height_estimates is empty")

        if len(height_estimates) < min_samples:
            raise ValueError(
                f"Need at least {min_samples} height estimates for RANSAC, "
                f"got {len(height_estimates)}"
            )

        if min_samples < 1:
            raise ValueError(f"min_samples must be at least 1, got {min_samples}")

        if max_trials < 1:
            raise ValueError(f"max_trials must be at least 1, got {max_trials}")

        if threshold_ratio <= 0:
            raise ValueError(f"threshold_ratio must be positive, got {threshold_ratio}")

        estimates_array = np.array(height_estimates)
        n_estimates = len(estimates_array)

        best_inliers = []
        best_inlier_indices = []
        best_inlier_count = 0

        # RANSAC iterations
        for _ in range(max_trials):
            # Randomly sample min_samples indices
            sample_indices = np.random.choice(n_estimates, min_samples, replace=False)
            sample = estimates_array[sample_indices]

            # Compute median of the sample (the model)
            model_median = np.median(sample)

            # Determine inliers: estimates within threshold_ratio of model_median
            threshold_value = threshold_ratio * abs(model_median)
            is_inlier = np.abs(estimates_array - model_median) <= threshold_value

            # Count inliers
            inlier_count = np.sum(is_inlier)

            # Update best model if this one has more inliers
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inlier_indices = [i for i, inlier in enumerate(is_inlier) if inlier]
                best_inliers = [height_estimates[i] for i in best_inlier_indices]

        # If no inliers were found (shouldn't happen with reasonable parameters),
        # return all estimates
        if not best_inliers:
            return (height_estimates.copy(), list(range(len(height_estimates))))

        return best_inliers, best_inlier_indices

    def optimize_height_with_outliers(self, method: str = "mad", **kwargs) -> CalibrationResult:
        """
        Estimate optimal camera height with outlier detection and removal.

        This method extends optimize_height_least_squares by first detecting and
        removing outliers from the height estimates before performing least-squares
        optimization. This produces more robust height estimates when some calibration
        points are erroneous (e.g., due to GPS errors, incorrect pixel clicks, or
        unusual terrain).

        Workflow:
            1. Get height estimates from all calibration points
            2. Detect outliers using specified method ('mad' or 'ransac')
            3. Filter calibration points to keep only inliers
            4. Run least-squares optimization on inlier points only
            5. Return CalibrationResult with correct inlier/outlier counts

        Args:
            method: Outlier detection method to use. Options:
                   - 'mad': Median Absolute Deviation (default)
                   - 'ransac': Random Sample Consensus
                   Case-insensitive.
            **kwargs: Additional arguments passed to the outlier detection method:
                     For 'mad':
                         - threshold: float (default 2.5)
                     For 'ransac':
                         - min_samples: int (default 3)
                         - max_trials: int (default 100)
                         - threshold_ratio: float (default 0.1)

        Returns:
            CalibrationResult with:
                - estimated_height: Optimized camera height in meters
                - confidence_interval: 95% confidence interval using t-distribution
                - inlier_count: Number of calibration points after outlier removal
                - outlier_count: Number of calibration points rejected as outliers
                - timestamp: Current time
                - calibration_points: Only the inlier calibration points

        Raises:
            ValueError: If method is not 'mad' or 'ransac'
            ValueError: If not enough calibration points available
            ValueError: If too few inliers remain after outlier removal

        Example:
            >>> calibrator = HeightCalibrator(camera_gps, min_points=5)
            >>> # ... add calibration points ...
            >>> result = calibrator.optimize_height_with_outliers(method='mad', threshold=3.0)
            >>> print(f"Height: {result.estimated_height:.2f}m")
            >>> print(f"Inliers: {result.inlier_count}, Outliers: {result.outlier_count}")
        """
        # Validate method parameter
        method_lower = method.lower()
        if method_lower not in ["mad", "ransac"]:
            raise ValueError(
                f"Invalid outlier detection method: '{method}'. Must be 'mad' or 'ransac'"
            )

        # Check if we have enough calibration points
        if not self.is_ready():
            raise ValueError(
                f"Not enough calibration points for optimization. "
                f"Have {len(self.calibration_points)}, need {self.min_points}"
            )

        # Get all height estimates
        try:
            height_estimates = self.get_all_height_estimates()
        except ValueError as e:
            raise ValueError(f"Cannot get height estimates: {e}")

        if not height_estimates:
            raise ValueError(
                "No valid height estimates available. "
                "All calibration points may have invalid homography distances."
            )

        # Build mapping from estimate index to calibration point index
        # (some calibration points may be skipped if homography_distance is invalid)
        estimate_to_point_index = []
        for i, point in enumerate(self.calibration_points):
            if abs(point.homography_distance) >= NEAR_ZERO_THRESHOLD:
                try:
                    self.estimate_height_from_point(point)
                    estimate_to_point_index.append(i)
                except ValueError:
                    continue

        # Detect outliers using specified method
        if method_lower == "mad":
            inlier_estimates, inlier_indices = self._detect_outliers_mad(height_estimates, **kwargs)
        else:  # method_lower == 'ransac'
            inlier_estimates, inlier_indices = self._detect_outliers_ransac(
                height_estimates, **kwargs
            )

        # Check if we have enough inliers remaining
        if len(inlier_indices) < 3:
            raise ValueError(
                f"Too few inliers after outlier removal. "
                f"Found {len(inlier_indices)} inliers, need at least 3. "
                f"Try adjusting outlier detection parameters."
            )

        # Map inlier estimate indices back to calibration point indices
        inlier_point_indices = [estimate_to_point_index[i] for i in inlier_indices]

        # Store original calibration points
        original_points = self.calibration_points.copy()
        original_count = len(original_points)

        # Temporarily replace calibration points with only inliers
        self.calibration_points = [original_points[i] for i in inlier_point_indices]

        try:
            # Run least-squares optimization on inliers only
            # We need to bypass the is_ready() check since we've filtered points
            # So we'll call the internal optimization logic directly

            # Filter out points with invalid homography distance
            valid_points = [
                p
                for p in self.calibration_points
                if abs(p.homography_distance) >= NEAR_ZERO_THRESHOLD
            ]

            if not valid_points:
                raise ValueError(
                    "All inlier calibration points have invalid homography distances. "
                    "Cannot perform height optimization."
                )

            # Use median of individual height estimates as initial guess
            individual_estimates = []
            for point in valid_points:
                scale_factor = point.gps_distance / point.homography_distance
                estimated_height = point.current_height * scale_factor
                individual_estimates.append(estimated_height)

            initial_guess = np.median(individual_estimates)

            # Run least-squares optimization
            result = least_squares(
                fun=self._compute_residual,
                x0=[initial_guess],
                method="lm",  # Levenberg-Marquardt algorithm
            )

            # Extract optimized height
            optimized_height = result.x[0]

            # Compute confidence interval using t-distribution
            # Use the individual height estimates from inlier points
            confidence_interval = self._compute_confidence_interval(
                height_estimates=individual_estimates,
                estimated_height=optimized_height,
                confidence_level=0.95,
            )

            # Calculate outlier count
            outlier_count = original_count - len(inlier_point_indices)

            calibration_result = CalibrationResult(
                estimated_height=optimized_height,
                confidence_interval=confidence_interval,
                inlier_count=len(valid_points),
                outlier_count=outlier_count,
                timestamp=datetime.now(),
                calibration_points=self.calibration_points.copy(),
            )

        finally:
            # Restore original calibration points
            self.calibration_points = original_points

        return calibration_result
