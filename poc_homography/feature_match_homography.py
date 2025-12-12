"""
GCP-based Homography Provider Implementation.

This module implements the HomographyProviderExtended interface using Ground
Control Points (GCPs) - known correspondences between image coordinates and
GPS coordinates.

The homography is computed directly from these point correspondences using
cv2.findHomography with RANSAC for robust outlier rejection.

Coordinate Systems:
    - Image coordinates: (u, v) in pixels, origin at top-left
    - World coordinates: (latitude, longitude) in decimal degrees (WGS84)
    - Local metric coordinates: (x, y) in meters using equirectangular projection
    - Map coordinates: (x, y) in meters from reference point on ground plane
"""

import numpy as np
import cv2
import math
import logging
from typing import List, Tuple, Dict, Any, Optional

from poc_homography.homography_interface import (
    HomographyProviderExtended,
    HomographyResult,
    WorldPoint,
    MapCoordinate,
    HomographyApproach,
    validate_homography_matrix,
    GPSPositionMixin
)
from poc_homography.coordinate_converter import gps_to_local_xy, local_xy_to_gps

logger = logging.getLogger(__name__)


class FeatureMatchHomography(GPSPositionMixin, HomographyProviderExtended):
    """
    GCP-based homography computation provider.

    This implementation computes homography using Ground Control Points (GCPs),
    which are known correspondences between image pixel coordinates and GPS
    world coordinates. The homography maps between image space and local metric
    space (derived from GPS coordinates).

    The computation process:
    1. Extract GCPs from reference data (pixel coords + GPS coords)
    2. Convert GPS coordinates to local metric coordinates using equirectangular projection
    3. Compute homography using cv2.findHomography with RANSAC
    4. Store homography matrix and inverse for bidirectional projection
    5. Calculate confidence based on inlier ratio, reprojection error, and spatial distribution

    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        detector: Feature detector type (kept for API compatibility, not used)
        min_matches: Minimum number of GCP matches required for valid homography
        ransac_threshold: Maximum reprojection error (pixels) for RANSAC inlier
        confidence_threshold: Minimum confidence score to consider homography valid
        H: Current homography matrix (3x3) mapping local metric to image
        H_inv: Inverse homography matrix mapping image to local metric
        _confidence: Current homography confidence score [0.0, 1.0]
        _camera_gps_lat: Reference GPS latitude for local metric conversion
        _camera_gps_lon: Reference GPS longitude for local metric conversion
    """

    # Minimum determinant threshold for valid homography
    MIN_DET_THRESHOLD = 1e-10

    # Fitting method options
    FITTING_METHOD_RANSAC = 'ransac'
    FITTING_METHOD_LMEDS = 'lmeds'
    FITTING_METHOD_AUTO = 'auto'  # Use LMEDS first, fall back to RANSAC

    # Confidence calculation parameters
    MIN_INLIER_RATIO = 0.5  # Minimum ratio of inliers to total points
    CONFIDENCE_PENALTY_LOW_INLIERS = 0.7  # Penalty for low inlier ratio

    # Spatial distribution thresholds
    MIN_COVERAGE_RATIO = 0.15  # Minimum convex hull area / image area ratio
    GOOD_COVERAGE_RATIO = 0.35  # Good coverage threshold
    MIN_QUADRANT_COVERAGE = 2  # Minimum number of quadrants with GCPs
    GOOD_QUADRANT_COVERAGE = 3  # Good quadrant coverage

    # Distribution confidence multipliers
    DIST_PENALTY_POOR_COVERAGE = 0.5  # Penalty for very clustered GCPs
    DIST_PENALTY_LOW_COVERAGE = 0.75  # Penalty for somewhat clustered GCPs
    DIST_BONUS_GOOD_COVERAGE = 1.1  # Bonus for well-distributed GCPs (capped at 1.0)

    # Edge factor constants for point confidence calculation
    EDGE_FACTOR_CENTER = 1.0
    EDGE_FACTOR_EDGE = 0.7
    EDGE_FACTOR_MIN = 0.3

    def __init__(
        self,
        width: int,
        height: int,
        detector: str = 'gcp',  # Not used, kept for API compatibility
        min_matches: int = 4,
        ransac_threshold: float = 3.0,
        confidence_threshold: float = 0.5,
        fitting_method: str = 'auto'
    ):
        """
        Initialize GCP-based homography provider.

        Args:
            width: Image width in pixels (e.g., 1920)
            height: Image height in pixels (e.g., 1080)
            detector: Detector type (kept for API compatibility, not used for GCP)
            min_matches: Minimum number of GCP matches required for computing
                homography. Must be at least 4 (minimum for homography estimation).
            ransac_threshold: RANSAC inlier threshold in pixels. Points with
                reprojection error below this are considered inliers.
                Typical values: 1.0-5.0 pixels.
            confidence_threshold: Minimum confidence score [0.0, 1.0] for
                homography to be considered valid. Based on inlier ratio and
                reprojection error.
            fitting_method: Method for robust fitting:
                - 'ransac': Use RANSAC (default for lower outlier rates)
                - 'lmeds': Use Least Median of Squares (more robust to high outlier rates)
                - 'auto': Try LMEDS first, use RANSAC if inlier ratio < 50%

        Raises:
            ValueError: If parameters are invalid (e.g., min_matches < 4)
        """
        if min_matches < 4:
            raise ValueError(
                f"min_matches must be at least 4 for homography estimation, "
                f"got {min_matches}"
            )

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in range [0.0, 1.0], "
                f"got {confidence_threshold}"
            )

        valid_methods = [self.FITTING_METHOD_RANSAC, self.FITTING_METHOD_LMEDS, self.FITTING_METHOD_AUTO]
        if fitting_method not in valid_methods:
            raise ValueError(
                f"fitting_method must be one of {valid_methods}, got {fitting_method}"
            )

        self.width = width
        self.height = height
        self.detector = detector
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.confidence_threshold = confidence_threshold
        self.fitting_method = fitting_method

        # Homography state
        self.H = np.eye(3)  # Maps local metric to image
        self.H_inv = np.eye(3)  # Maps image to local metric
        self._confidence: float = 0.0
        self._last_metadata: Dict[str, Any] = {}

        # GPS reference point for local metric to GPS conversion
        # This will be set from the first GCP or can be set explicitly
        self._camera_gps_lat: Optional[float] = None
        self._camera_gps_lon: Optional[float] = None
        self._reference_lat: Optional[float] = None
        self._reference_lon: Optional[float] = None

    def _gps_to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert GPS coordinates to local metric coordinates.

        Uses the shared coordinate_converter module for consistency.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees

        Returns:
            (x, y): Local metric coordinates in meters (East, North)

        Raises:
            RuntimeError: If reference GPS position not set
        """
        if self._reference_lat is None or self._reference_lon is None:
            raise RuntimeError(
                "Reference GPS position not set. This should be initialized "
                "during compute_homography() from GCPs."
            )

        return gps_to_local_xy(
            self._reference_lat,
            self._reference_lon,
            lat,
            lon
        )

    def _local_to_gps(self, x_meters: float, y_meters: float) -> Tuple[float, float]:
        """
        Convert local metric coordinates to GPS coordinates.

        Uses the shared coordinate_converter module for consistency.

        Args:
            x_meters: X coordinate in meters (East)
            y_meters: Y coordinate in meters (North)

        Returns:
            (latitude, longitude): GPS coordinates in decimal degrees

        Raises:
            RuntimeError: If reference GPS position not set
        """
        if self._reference_lat is None or self._reference_lon is None:
            raise RuntimeError(
                "Reference GPS position not set. Call compute_homography() first "
                "to initialize from GCPs."
            )

        return local_xy_to_gps(
            self._reference_lat,
            self._reference_lon,
            x_meters,
            y_meters
        )

    def _calculate_spatial_distribution(
        self,
        image_points: np.ndarray
    ) -> Dict[str, Any]:
        """
        Assess the spatial distribution quality of GCPs in image space.

        Good homography estimation requires GCPs distributed across the image,
        not clustered in one area. This method calculates:
        1. Convex hull coverage: Area covered by GCPs relative to image area
        2. Quadrant coverage: How many image quadrants contain GCPs
        3. Spread metrics: Standard deviation of point positions

        Args:
            image_points: Nx2 array of (u, v) pixel coordinates

        Returns:
            Dictionary with distribution metrics:
                - coverage_ratio: Convex hull area / image area [0.0, 1.0]
                - quadrants_covered: Number of quadrants with GCPs [0-4]
                - spread_x: Normalized std dev in X direction
                - spread_y: Normalized std dev in Y direction
                - distribution_score: Overall distribution quality [0.0, 1.0]
                - warnings: List of distribution issues
        """
        warnings = []
        n_points = len(image_points)

        if n_points < 3:
            return {
                'coverage_ratio': 0.0,
                'quadrants_covered': 0,
                'spread_x': 0.0,
                'spread_y': 0.0,
                'distribution_score': 0.0,
                'warnings': ['Too few points for distribution analysis']
            }

        # Calculate convex hull coverage
        try:
            hull = cv2.convexHull(image_points.astype(np.float32))
            hull_area = cv2.contourArea(hull)
            image_area = self.width * self.height
            coverage_ratio = hull_area / image_area if image_area > 0 else 0.0
        except Exception:
            coverage_ratio = 0.0
            warnings.append('Could not compute convex hull')

        # Calculate quadrant coverage
        center_u = self.width / 2.0
        center_v = self.height / 2.0
        quadrants = set()
        for u, v in image_points:
            q = 0
            if u >= center_u:
                q += 1
            if v >= center_v:
                q += 2
            quadrants.add(q)
        quadrants_covered = len(quadrants)

        # Calculate spread (normalized standard deviation)
        spread_x = np.std(image_points[:, 0]) / self.width if self.width > 0 else 0.0
        spread_y = np.std(image_points[:, 1]) / self.height if self.height > 0 else 0.0

        # Generate warnings
        if coverage_ratio < self.MIN_COVERAGE_RATIO:
            warnings.append(
                f'GCPs are clustered (coverage {coverage_ratio:.1%} < {self.MIN_COVERAGE_RATIO:.0%}). '
                'Add GCPs in different areas of the image.'
            )
        if quadrants_covered < self.MIN_QUADRANT_COVERAGE:
            warnings.append(
                f'GCPs only cover {quadrants_covered}/4 quadrants. '
                'Add GCPs to cover more of the image.'
            )
        if spread_x < 0.15 or spread_y < 0.15:
            warnings.append(
                'GCPs have low spatial variance. Spread points across the image.'
            )

        # Calculate overall distribution score
        # Weight: 40% coverage, 30% quadrant, 30% spread
        coverage_score = min(1.0, coverage_ratio / self.GOOD_COVERAGE_RATIO)
        quadrant_score = quadrants_covered / 4.0
        spread_score = min(1.0, (spread_x + spread_y) / 0.5)  # Normalize to ~1.0 for good spread

        distribution_score = (
            0.4 * coverage_score +
            0.3 * quadrant_score +
            0.3 * spread_score
        )

        return {
            'coverage_ratio': coverage_ratio,
            'quadrants_covered': quadrants_covered,
            'spread_x': spread_x,
            'spread_y': spread_y,
            'distribution_score': distribution_score,
            'warnings': warnings
        }

    def _calculate_confidence(
        self,
        num_inliers: int,
        total_points: int,
        reprojection_errors: np.ndarray,
        distribution_metrics: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate confidence score for the homography.

        Confidence is based on:
        1. Inlier ratio: Higher ratio = higher confidence
        2. Reprojection error: Lower mean error = higher confidence
        3. Spatial distribution: Well-distributed GCPs = higher confidence

        Args:
            num_inliers: Number of inlier points from RANSAC
            total_points: Total number of GCP correspondences
            reprojection_errors: Array of reprojection errors for inliers (pixels)
            distribution_metrics: Optional dict from _calculate_spatial_distribution

        Returns:
            float: Confidence score in range [0.0, 1.0]
        """
        if total_points == 0:
            return 0.0

        # Calculate inlier ratio
        inlier_ratio = num_inliers / total_points

        # Base confidence on inlier ratio
        if inlier_ratio < self.MIN_INLIER_RATIO:
            confidence = inlier_ratio * self.CONFIDENCE_PENALTY_LOW_INLIERS
        else:
            confidence = inlier_ratio

        # Factor in reprojection error
        if len(reprojection_errors) > 0:
            mean_error = np.mean(reprojection_errors)
            # Penalize based on error relative to RANSAC threshold
            # Error = 0 -> multiplier = 1.0
            # Error = ransac_threshold -> multiplier = 0.5
            # Error > ransac_threshold -> multiplier < 0.5
            error_factor = max(0.3, 1.0 - (mean_error / (2 * self.ransac_threshold)))
            confidence *= error_factor

        # Factor in spatial distribution
        if distribution_metrics:
            dist_score = distribution_metrics.get('distribution_score', 0.5)
            coverage = distribution_metrics.get('coverage_ratio', 0.0)

            if coverage < self.MIN_COVERAGE_RATIO:
                # Severe penalty for very clustered GCPs
                confidence *= self.DIST_PENALTY_POOR_COVERAGE
                logger.warning(
                    "GCPs are severely clustered (coverage=%.1f%%). "
                    "Homography may be unreliable outside the GCP cluster.",
                    coverage * 100
                )
            elif dist_score < 0.5:
                # Moderate penalty for somewhat clustered GCPs
                confidence *= self.DIST_PENALTY_LOW_COVERAGE
            elif dist_score > 0.7:
                # Small bonus for well-distributed GCPs (capped at 1.0)
                confidence *= self.DIST_BONUS_GOOD_COVERAGE

        return min(1.0, confidence)

    def _get_suggested_action(
        self,
        confidence_breakdown: Dict[str, Any],
        outlier_analysis: List[Dict[str, Any]]
    ) -> str:
        """Generate a suggested action based on confidence diagnostics.

        Args:
            confidence_breakdown: Confidence calculation breakdown
            outlier_analysis: Per-GCP outlier analysis

        Returns:
            Human-readable suggested action string
        """
        suggestions = []

        inlier_ratio = confidence_breakdown.get('inlier_ratio', 1.0)
        final_confidence = confidence_breakdown.get('final_confidence', 0.0)

        # Check inlier ratio
        if inlier_ratio < 0.5:
            num_outliers = sum(1 for o in outlier_analysis if not o['is_inlier'])
            high_error_outliers = [o for o in outlier_analysis if not o['is_inlier'] and o['error_px'] > 20]

            if len(high_error_outliers) > 0:
                worst = high_error_outliers[0]
                suggestions.append(
                    f"Remove or fix GCP '{worst['description']}' (error: {worst['error_px']:.1f}px)"
                )

            if num_outliers > 5:
                suggestions.append(
                    f"Consider increasing RANSAC threshold (current: {self.ransac_threshold}px) or "
                    "verifying GPS coordinate accuracy"
                )

        # Check if confidence is below threshold
        if final_confidence < self.confidence_threshold:
            if inlier_ratio >= 0.5:
                suggestions.append(
                    "Inlier ratio is acceptable but confidence is low. "
                    "Check reprojection errors and GCP spatial distribution."
                )

        # Distribution warnings
        dist_penalty = confidence_breakdown.get('distribution_penalty', '')
        if 'POOR_COVERAGE' in dist_penalty:
            suggestions.append(
                "GCPs are too clustered. Add GCPs spread across the full image area."
            )

        if not suggestions:
            if final_confidence >= self.confidence_threshold:
                return "Homography quality is acceptable."
            else:
                return "Review GCP quality and distribution."

        return " | ".join(suggestions)

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
        if self.width <= 0 or self.height <= 0:
            return base_confidence

        center_u = self.width / 2.0
        center_v = self.height / 2.0

        dx = (u - center_u) / (self.width / 2.0)
        dy = (v - center_v) / (self.height / 2.0)

        dist_from_center = math.sqrt(dx * dx + dy * dy)

        # Reduce confidence for points far from center
        if dist_from_center < 1.0:
            edge_factor = self.EDGE_FACTOR_CENTER - (
                self.EDGE_FACTOR_CENTER - self.EDGE_FACTOR_EDGE
            ) * dist_from_center
        else:
            edge_factor = self.EDGE_FACTOR_EDGE * (2.0 - dist_from_center)

        edge_factor = max(self.EDGE_FACTOR_MIN, min(self.EDGE_FACTOR_CENTER, edge_factor))

        return base_confidence * edge_factor

    def _project_image_point_to_local(
        self,
        image_point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Project image point to local metric coordinates.

        Args:
            image_point: (u, v) pixel coordinates

        Returns:
            (x_local, y_local): Local metric coordinates in meters

        Raises:
            ValueError: If point projects to infinity (on horizon)
        """
        u, v = image_point

        # Convert to homogeneous coordinates
        pt_homogeneous = np.array([u, v, 1.0])

        # Project to local metric using inverse homography
        local_homogeneous = self.H_inv @ pt_homogeneous

        # Check for division by zero (point at infinity/horizon)
        if abs(local_homogeneous[2]) < 1e-10:
            raise ValueError("Point projects to infinity (on horizon line)")

        # Normalize
        x_local = local_homogeneous[0] / local_homogeneous[2]
        y_local = local_homogeneous[1] / local_homogeneous[2]

        return x_local, y_local

    def _compute_homography_with_method(
        self,
        local_points: np.ndarray,
        image_points: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Compute homography using the configured fitting method.

        Args:
            local_points: Nx2 array of local metric coordinates
            image_points: Nx2 array of image pixel coordinates

        Returns:
            Tuple of (H, mask, method_used):
                - H: 3x3 homography matrix or None if failed
                - mask: Nx1 inlier mask or None if failed
                - method_used: String indicating which method was used
        """
        if self.fitting_method == self.FITTING_METHOD_RANSAC:
            # Use RANSAC directly
            H, mask = cv2.findHomography(
                local_points,
                image_points,
                cv2.RANSAC,
                self.ransac_threshold
            )
            return H, mask, 'ransac'

        elif self.fitting_method == self.FITTING_METHOD_LMEDS:
            # Use LMEDS directly (no threshold parameter needed)
            H, mask = cv2.findHomography(
                local_points,
                image_points,
                cv2.LMEDS
            )
            return H, mask, 'lmeds'

        else:  # AUTO mode
            # Try LMEDS first (more robust to high outlier rates up to 50%)
            logger.debug("AUTO mode: trying LMEDS first")
            H_lmeds, mask_lmeds = cv2.findHomography(
                local_points,
                image_points,
                cv2.LMEDS
            )

            if H_lmeds is not None and mask_lmeds is not None:
                inlier_ratio_lmeds = np.sum(mask_lmeds) / len(mask_lmeds)
                logger.debug("LMEDS inlier ratio: %.2f%%", inlier_ratio_lmeds * 100)

                # If LMEDS gives good results (>50% inliers), use it
                if inlier_ratio_lmeds >= 0.5:
                    logger.info(
                        "AUTO mode: using LMEDS (inlier ratio %.1f%% >= 50%%)",
                        inlier_ratio_lmeds * 100
                    )
                    return H_lmeds, mask_lmeds, 'lmeds'

                # Otherwise, try RANSAC and compare
                logger.debug("LMEDS inlier ratio low, trying RANSAC")

            # Try RANSAC
            H_ransac, mask_ransac = cv2.findHomography(
                local_points,
                image_points,
                cv2.RANSAC,
                self.ransac_threshold
            )

            if H_ransac is not None and mask_ransac is not None:
                inlier_ratio_ransac = np.sum(mask_ransac) / len(mask_ransac)
                logger.debug("RANSAC inlier ratio: %.2f%%", inlier_ratio_ransac * 100)

                # Compare RANSAC vs LMEDS (if LMEDS succeeded)
                if H_lmeds is not None and mask_lmeds is not None:
                    inlier_ratio_lmeds = np.sum(mask_lmeds) / len(mask_lmeds)

                    # Use whichever has more inliers
                    if inlier_ratio_lmeds > inlier_ratio_ransac:
                        logger.info(
                            "AUTO mode: using LMEDS (%.1f%% > RANSAC %.1f%%)",
                            inlier_ratio_lmeds * 100,
                            inlier_ratio_ransac * 100
                        )
                        return H_lmeds, mask_lmeds, 'lmeds'

                logger.info(
                    "AUTO mode: using RANSAC (inlier ratio %.1f%%)",
                    inlier_ratio_ransac * 100
                )
                return H_ransac, mask_ransac, 'ransac'

            # If RANSAC failed but LMEDS succeeded, use LMEDS
            if H_lmeds is not None:
                logger.info("AUTO mode: RANSAC failed, falling back to LMEDS")
                return H_lmeds, mask_lmeds, 'lmeds'

            # Both failed
            logger.error("AUTO mode: both LMEDS and RANSAC failed")
            return None, None, 'none'

    # =========================================================================
    # HomographyProvider Interface Implementation
    # =========================================================================

    def compute_homography(
        self,
        frame: np.ndarray,
        reference: Dict[str, Any]
    ) -> HomographyResult:
        """
        Compute homography from Ground Control Points.

        Args:
            frame: Image frame (not used for GCP approach, but required by interface)
            reference: Dictionary with required key:
                - 'ground_control_points': List of GCP dictionaries, each with:
                    - 'gps': {'latitude': float, 'longitude': float}
                    - 'image': {'u': float, 'v': float}

        Returns:
            HomographyResult with computed homography matrix and confidence

        Raises:
            ValueError: If required reference data is missing or invalid
            RuntimeError: If homography computation fails
        """
        # Validate reference data
        if 'ground_control_points' not in reference:
            raise ValueError("Missing required reference key: 'ground_control_points'")

        gcps = reference['ground_control_points']

        if not isinstance(gcps, list) or len(gcps) < self.min_matches:
            raise ValueError(
                f"Need at least {self.min_matches} ground control points, got {len(gcps)}"
            )

        # Extract GCP data
        image_points = []
        gps_points = []

        for gcp in gcps:
            if 'gps' not in gcp or 'image' not in gcp:
                raise ValueError("Each GCP must have 'gps' and 'image' keys")

            gps = gcp['gps']
            img = gcp['image']

            if 'latitude' not in gps or 'longitude' not in gps:
                raise ValueError("GPS must have 'latitude' and 'longitude' keys")

            if 'u' not in img or 'v' not in img:
                raise ValueError("Image must have 'u' and 'v' keys")

            image_points.append([img['u'], img['v']])
            gps_points.append([gps['latitude'], gps['longitude']])

        # Convert to numpy arrays
        image_points = np.array(image_points, dtype=np.float32)
        gps_points = np.array(gps_points, dtype=np.float64)

        # Set reference point for local coordinate system
        # Priority: 1) camera_gps from reference, 2) GCP centroid (more stable than first GCP)
        camera_gps = reference.get('camera_gps')
        if camera_gps and 'latitude' in camera_gps and 'longitude' in camera_gps:
            # Use explicit camera position as reference
            self._reference_lat = camera_gps['latitude']
            self._reference_lon = camera_gps['longitude']
            self._camera_gps_lat = self._reference_lat
            self._camera_gps_lon = self._reference_lon
            logger.info(
                "Using camera GPS as reference: lat=%.6f, lon=%.6f",
                self._reference_lat,
                self._reference_lon
            )
        else:
            # Fall back to GCP centroid (more stable than arbitrary first point)
            self._reference_lat = float(np.mean(gps_points[:, 0]))
            self._reference_lon = float(np.mean(gps_points[:, 1]))
            self._camera_gps_lat = self._reference_lat
            self._camera_gps_lon = self._reference_lon
            logger.info(
                "Using GCP centroid as reference (no camera_gps provided): lat=%.6f, lon=%.6f",
                self._reference_lat,
                self._reference_lon
            )

        # Convert GPS points to local metric coordinates
        local_points = []
        for lat, lon in gps_points:
            x, y = self._gps_to_local(lat, lon)
            local_points.append([x, y])

        local_points = np.array(local_points, dtype=np.float32)

        logger.info(
            "Computing homography from %d GCPs (image -> local metric), method=%s",
            len(image_points),
            self.fitting_method
        )

        # Compute homography using specified fitting method
        # H maps local metric coordinates to image coordinates
        H, mask, method_used = self._compute_homography_with_method(
            local_points, image_points
        )

        if H is None:
            logger.error("Failed to compute homography from GCPs")
            self.H = np.eye(3)
            self.H_inv = np.eye(3)
            self._confidence = 0.0
            raise RuntimeError(
                "Failed to compute homography. Check GCP quality and distribution."
            )

        # Store homography (local metric -> image)
        self.H = H

        # Calculate spatial distribution of GCPs (for confidence calculation)
        distribution_metrics = self._calculate_spatial_distribution(image_points)

        # Log distribution warnings
        for warning in distribution_metrics.get('warnings', []):
            logger.warning(warning)

        # Initialize reprojection error variables
        mean_reproj_error = None
        max_reproj_error = None

        # Compute inverse (image -> local metric)
        det_H = np.linalg.det(self.H)
        if abs(det_H) < self.MIN_DET_THRESHOLD:
            logger.warning(
                "Homography is singular (det=%.2e). Inverse may be unstable.",
                det_H
            )
            self.H_inv = np.eye(3)
            self._confidence = 0.0
        else:
            self.H_inv = np.linalg.inv(self.H)

            # Calculate confidence based on inliers
            num_inliers = int(np.sum(mask))
            total_points = len(image_points)

            # Calculate reprojection errors for inliers
            if num_inliers > 0:
                inlier_local = local_points[mask.ravel() == 1]
                inlier_image = image_points[mask.ravel() == 1]

                # Project local points to image using homography
                projected = cv2.perspectiveTransform(
                    inlier_local.reshape(-1, 1, 2),
                    self.H
                ).reshape(-1, 2)

                # Calculate reprojection errors
                errors = np.linalg.norm(projected - inlier_image, axis=1)
                self._confidence = self._calculate_confidence(
                    num_inliers, total_points, errors, distribution_metrics
                )

                # Calculate mean reprojection error for metadata
                mean_reproj_error = float(np.mean(errors))
                max_reproj_error = float(np.max(errors))
            else:
                self._confidence = 0.0

        # Identify outliers with per-GCP diagnostics
        outlier_analysis = []
        if mask is not None:
            # Calculate errors for ALL points (not just inliers)
            all_projected = cv2.perspectiveTransform(
                local_points.reshape(-1, 1, 2), self.H
            ).reshape(-1, 2)
            all_errors = np.linalg.norm(all_projected - image_points, axis=1)

            for i in range(len(gcps)):
                gcp = gcps[i]
                is_inlier = bool(mask[i][0])
                error = float(all_errors[i])
                desc = gcp.get('metadata', {}).get('description', f'GCP {i+1}')

                outlier_analysis.append({
                    'index': i,
                    'description': desc,
                    'is_inlier': is_inlier,
                    'error_px': error,
                    'pixel': [float(image_points[i][0]), float(image_points[i][1])],
                    'gps': [float(gps_points[i][0]), float(gps_points[i][1])]
                })

        # Sort outlier analysis by error (highest first)
        outlier_analysis.sort(key=lambda x: x['error_px'], reverse=True)

        # Build confidence breakdown for diagnostics
        inlier_ratio = num_inliers / total_points if 'num_inliers' in dir() else 0
        confidence_breakdown = {
            'inlier_ratio': inlier_ratio,
            'inlier_penalty_applied': inlier_ratio < self.MIN_INLIER_RATIO,
            'base_confidence': inlier_ratio * self.CONFIDENCE_PENALTY_LOW_INLIERS if inlier_ratio < self.MIN_INLIER_RATIO else inlier_ratio,
            'error_factor': max(0.3, 1.0 - (mean_reproj_error / (2 * self.ransac_threshold))) if mean_reproj_error else 1.0,
            'distribution_penalty': None,
            'final_confidence': self._confidence
        }

        # Determine distribution penalty
        dist_score = distribution_metrics.get('distribution_score', 0.5)
        coverage = distribution_metrics.get('coverage_ratio', 0.0)
        if coverage < self.MIN_COVERAGE_RATIO:
            confidence_breakdown['distribution_penalty'] = f'POOR_COVERAGE ({self.DIST_PENALTY_POOR_COVERAGE}x)'
        elif dist_score < 0.5:
            confidence_breakdown['distribution_penalty'] = f'LOW_COVERAGE ({self.DIST_PENALTY_LOW_COVERAGE}x)'
        elif dist_score > 0.7:
            confidence_breakdown['distribution_penalty'] = f'GOOD_COVERAGE_BONUS ({self.DIST_BONUS_GOOD_COVERAGE}x)'
        else:
            confidence_breakdown['distribution_penalty'] = 'NONE (1.0x)'

        # Build metadata with distribution info
        metadata = {
            'approach': HomographyApproach.FEATURE_MATCH.value,
            'method': 'gcp_based',
            'fitting_method': method_used,
            'fitting_method_config': self.fitting_method,
            'num_gcps': len(image_points),
            'num_inliers': int(np.sum(mask)) if mask is not None else 0,
            'inlier_ratio': float(np.sum(mask)) / len(image_points) if mask is not None else 0.0,
            'inlier_mask': mask.flatten().astype(bool).tolist() if mask is not None else None,
            'determinant': det_H,
            'reference_gps': {
                'latitude': self._reference_lat,
                'longitude': self._reference_lon
            },
            # Distribution metrics
            'distribution': {
                'coverage_ratio': distribution_metrics['coverage_ratio'],
                'quadrants_covered': distribution_metrics['quadrants_covered'],
                'spread_x': distribution_metrics['spread_x'],
                'spread_y': distribution_metrics['spread_y'],
                'distribution_score': distribution_metrics['distribution_score'],
                'warnings': distribution_metrics['warnings']
            },
            # Reprojection error stats
            'reprojection_error': {
                'mean_px': mean_reproj_error,
                'max_px': max_reproj_error,
                'threshold_px': self.ransac_threshold
            },
            # NEW: Detailed diagnostics
            'confidence_breakdown': confidence_breakdown,
            'outlier_analysis': outlier_analysis,
            'top_outliers': [o for o in outlier_analysis if not o['is_inlier']][:5],
            'suggested_action': self._get_suggested_action(confidence_breakdown, outlier_analysis)
        }

        self._last_metadata = metadata

        logger.info(
            "Homography computed: %d/%d inliers, confidence=%.3f, distribution_score=%.2f, method=%s",
            metadata['num_inliers'],
            metadata['num_gcps'],
            self._confidence,
            distribution_metrics['distribution_score'],
            method_used
        )

        return HomographyResult(
            homography_matrix=self.H.copy(),
            confidence=self._confidence,
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
        if not (0 <= u < self.width) or not (0 <= v < self.height):
            raise ValueError(
                f"Image point ({u}, {v}) outside valid bounds "
                f"[0, {self.width}) x [0, {self.height})"
            )

        # Project to local metric coordinates
        x_local, y_local = self._project_image_point_to_local(image_point)

        # Convert to GPS
        latitude, longitude = self._local_to_gps(x_local, y_local)

        # Calculate point-specific confidence
        point_confidence = self._calculate_point_confidence(image_point, self._confidence)

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
            RuntimeError: If no valid homography computed
            ValueError: If any image_point is outside valid bounds
        """
        if not self.is_valid():
            raise RuntimeError("No valid homography available. Call compute_homography() first.")

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
        return self._confidence

    def is_valid(self) -> bool:
        """
        Check if homography is valid and ready for projection.

        Returns:
            bool: True if homography is valid for projection
        """
        return validate_homography_matrix(
            self.H_inv,  # Check inverse since that's what we use for projection
            self._confidence,
            self.confidence_threshold
        )

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
            MapCoordinate with x, y in meters from reference point

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

        # Project to local metric coordinates
        x_local, y_local = self._project_image_point_to_local(image_point)

        # Calculate point-specific confidence
        point_confidence = self._calculate_point_confidence(image_point, self._confidence)

        return MapCoordinate(
            x=x_local,
            y=y_local,
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

        map_coords = []
        for image_point in image_points:
            map_coord = self.project_point_to_map(image_point)
            map_coords.append(map_coord)

        return map_coords
