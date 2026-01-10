"""
GCP-based Homography Provider Implementation.

This module implements the HomographyProvider interface using Ground
Control Points (GCPs) - known correspondences between image coordinates and
map pixel coordinates.

The homography is computed directly from these point correspondences using
cv2.findHomography with RANSAC for robust outlier rejection.

Coordinate Systems:
    - Image coordinates: (u, v) in pixels, origin at top-left of camera image
    - Map coordinates: (pixel_x, pixel_y) in pixels on the reference map image
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Any

import cv2
import numpy as np

from poc_homography.homography_interface import (
    HomographyApproach,
    HomographyProvider,
    HomographyResult,
    validate_homography_matrix,
)
from poc_homography.map_points import MapPoint

logger = logging.getLogger(__name__)


class FeatureMatchHomography(HomographyProvider):
    """
    GCP-based homography computation provider.

    This implementation computes homography using Ground Control Points (GCPs),
    which are known correspondences between image pixel coordinates and map
    pixel coordinates. The homography maps between image space and map space.

    The computation process:
    1. Extract GCPs from reference data (image pixel coords + map pixel coords)
    2. Compute homography using cv2.findHomography with RANSAC
    3. Store homography matrix and inverse for bidirectional projection
    4. Calculate confidence based on inlier ratio, reprojection error, and spatial distribution

    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        map_id: Identifier of the reference map
        detector: Feature detector type (kept for API compatibility, not used)
        min_matches: Minimum number of GCP matches required for valid homography
        ransac_threshold: Maximum reprojection error (pixels) for RANSAC inlier
        confidence_threshold: Minimum confidence score to consider homography valid
        H: Current homography matrix (3x3) mapping map pixels to image
        H_inv: Inverse homography matrix mapping image to map pixels
        _confidence: Current homography confidence score [0.0, 1.0]
    """

    # Minimum determinant threshold for valid homography
    MIN_DET_THRESHOLD = 1e-10

    # Fitting method options
    FITTING_METHOD_RANSAC = "ransac"
    FITTING_METHOD_LMEDS = "lmeds"
    FITTING_METHOD_AUTO = "auto"  # Use LMEDS first, fall back to RANSAC

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
        map_id: str,
        detector: str = "gcp",  # Not used, kept for API compatibility
        min_matches: int = 4,
        ransac_threshold: float = 3.0,
        confidence_threshold: float = 0.5,
        fitting_method: str = "auto",
    ):
        """
        Initialize GCP-based homography provider.

        Args:
            width: Image width in pixels (e.g., 1920)
            height: Image height in pixels (e.g., 1080)
            map_id: Identifier of the reference map for projected points
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
                f"min_matches must be at least 4 for homography estimation, got {min_matches}"
            )

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in range [0.0, 1.0], got {confidence_threshold}"
            )

        valid_methods = [
            self.FITTING_METHOD_RANSAC,
            self.FITTING_METHOD_LMEDS,
            self.FITTING_METHOD_AUTO,
        ]
        if fitting_method not in valid_methods:
            raise ValueError(f"fitting_method must be one of {valid_methods}, got {fitting_method}")

        self.width = width
        self.height = height
        self.map_id = map_id
        self.detector = detector
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.confidence_threshold = confidence_threshold
        self.fitting_method = fitting_method

        # Homography state
        self.H = np.eye(3)  # Maps map pixels to image
        self.H_inv = np.eye(3)  # Maps image to map pixels
        self._confidence: float = 0.0
        self._last_metadata: dict[str, Any] = {}

    def _calculate_spatial_distribution(self, image_points: np.ndarray) -> dict[str, Any]:
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
                "coverage_ratio": 0.0,
                "quadrants_covered": 0,
                "spread_x": 0.0,
                "spread_y": 0.0,
                "distribution_score": 0.0,
                "warnings": ["Too few points for distribution analysis"],
            }

        # Calculate convex hull coverage
        try:
            hull = cv2.convexHull(image_points.astype(np.float32))
            hull_area = cv2.contourArea(hull)
            image_area = self.width * self.height
            coverage_ratio = hull_area / image_area if image_area > 0 else 0.0
        except Exception:
            coverage_ratio = 0.0
            warnings.append("Could not compute convex hull")

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
                f"GCPs are clustered (coverage {coverage_ratio:.1%} < {self.MIN_COVERAGE_RATIO:.0%}). "
                "Add GCPs in different areas of the image."
            )
        if quadrants_covered < self.MIN_QUADRANT_COVERAGE:
            warnings.append(
                f"GCPs only cover {quadrants_covered}/4 quadrants. "
                "Add GCPs to cover more of the image."
            )
        if spread_x < 0.15 or spread_y < 0.15:
            warnings.append("GCPs have low spatial variance. Spread points across the image.")

        # Calculate overall distribution score
        # Weight: 40% coverage, 30% quadrant, 30% spread
        coverage_score = min(1.0, coverage_ratio / self.GOOD_COVERAGE_RATIO)
        quadrant_score = quadrants_covered / 4.0
        spread_score = min(1.0, (spread_x + spread_y) / 0.5)  # Normalize to ~1.0 for good spread

        distribution_score = 0.4 * coverage_score + 0.3 * quadrant_score + 0.3 * spread_score

        return {
            "coverage_ratio": coverage_ratio,
            "quadrants_covered": quadrants_covered,
            "spread_x": spread_x,
            "spread_y": spread_y,
            "distribution_score": distribution_score,
            "warnings": warnings,
        }

    def _calculate_confidence(
        self,
        num_inliers: int,
        total_points: int,
        reprojection_errors: np.ndarray,
        distribution_metrics: dict[str, Any] | None = None,
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
            dist_score = distribution_metrics.get("distribution_score", 0.5)
            coverage = distribution_metrics.get("coverage_ratio", 0.0)

            if coverage < self.MIN_COVERAGE_RATIO:
                # Severe penalty for very clustered GCPs
                confidence *= self.DIST_PENALTY_POOR_COVERAGE
                logger.warning(
                    "GCPs are severely clustered (coverage=%.1f%%). "
                    "Homography may be unreliable outside the GCP cluster.",
                    coverage * 100,
                )
            elif dist_score < 0.5:
                # Moderate penalty for somewhat clustered GCPs
                confidence *= self.DIST_PENALTY_LOW_COVERAGE
            elif dist_score > 0.7:
                # Small bonus for well-distributed GCPs (capped at 1.0)
                confidence *= self.DIST_BONUS_GOOD_COVERAGE

        return float(min(1.0, confidence))

    def _get_suggested_action(
        self, confidence_breakdown: dict[str, Any], outlier_analysis: list[dict[str, Any]]
    ) -> str:
        """Generate a suggested action based on confidence diagnostics.

        Args:
            confidence_breakdown: Confidence calculation breakdown
            outlier_analysis: Per-GCP outlier analysis

        Returns:
            Human-readable suggested action string
        """
        suggestions = []

        inlier_ratio = confidence_breakdown.get("inlier_ratio", 1.0)
        final_confidence = confidence_breakdown.get("final_confidence", 0.0)

        # Check inlier ratio
        if inlier_ratio < 0.5:
            num_outliers = sum(1 for o in outlier_analysis if not o["is_inlier"])
            high_error_outliers = [
                o for o in outlier_analysis if not o["is_inlier"] and o["error_px"] > 20
            ]

            if len(high_error_outliers) > 0:
                worst = high_error_outliers[0]
                suggestions.append(
                    f"Remove or fix GCP '{worst['description']}' (error: {worst['error_px']:.1f}px)"
                )

            if num_outliers > 5:
                suggestions.append(
                    f"Consider increasing RANSAC threshold (current: {self.ransac_threshold}px) or "
                    "verifying map coordinate accuracy"
                )

        # Check if confidence is below threshold
        if final_confidence < self.confidence_threshold:
            if inlier_ratio >= 0.5:
                suggestions.append(
                    "Inlier ratio is acceptable but confidence is low. "
                    "Check reprojection errors and GCP spatial distribution."
                )

        # Distribution warnings
        dist_penalty = confidence_breakdown.get("distribution_penalty", "")
        if "POOR_COVERAGE" in dist_penalty:
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
        self, image_point: tuple[float, float], base_confidence: float
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
            edge_factor = (
                self.EDGE_FACTOR_CENTER
                - (self.EDGE_FACTOR_CENTER - self.EDGE_FACTOR_EDGE) * dist_from_center
            )
        else:
            edge_factor = self.EDGE_FACTOR_EDGE * (2.0 - dist_from_center)

        edge_factor = max(self.EDGE_FACTOR_MIN, min(self.EDGE_FACTOR_CENTER, edge_factor))

        return base_confidence * edge_factor

    def _project_image_point_to_map(self, image_point: tuple[float, float]) -> tuple[float, float]:
        """
        Project image point to map pixel coordinates.

        Args:
            image_point: (u, v) pixel coordinates in camera image

        Returns:
            (pixel_x, pixel_y): Pixel coordinates on the reference map

        Raises:
            ValueError: If point projects to infinity (on horizon)
        """
        u, v = image_point

        # Convert to homogeneous coordinates
        pt_homogeneous = np.array([u, v, 1.0])

        # Project to map pixels using inverse homography
        map_homogeneous = self.H_inv @ pt_homogeneous

        # Check for division by zero (point at infinity/horizon)
        if abs(map_homogeneous[2]) < 1e-10:
            raise ValueError("Point projects to infinity (on horizon line)")

        # Normalize
        pixel_x = map_homogeneous[0] / map_homogeneous[2]
        pixel_y = map_homogeneous[1] / map_homogeneous[2]

        return pixel_x, pixel_y

    def _compute_homography_with_method(
        self, local_points: np.ndarray, image_points: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, str]:
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
                local_points, image_points, cv2.RANSAC, self.ransac_threshold
            )
            return H, mask, "ransac"

        elif self.fitting_method == self.FITTING_METHOD_LMEDS:
            # Use LMEDS directly (no threshold parameter needed)
            H, mask = cv2.findHomography(local_points, image_points, cv2.LMEDS)
            return H, mask, "lmeds"

        else:  # AUTO mode
            # Try LMEDS first (more robust to high outlier rates up to 50%)
            logger.debug("AUTO mode: trying LMEDS first")
            H_lmeds, mask_lmeds = cv2.findHomography(local_points, image_points, cv2.LMEDS)

            if H_lmeds is not None and mask_lmeds is not None:
                inlier_ratio_lmeds = np.sum(mask_lmeds) / len(mask_lmeds)
                logger.debug("LMEDS inlier ratio: %.2f%%", inlier_ratio_lmeds * 100)

                # If LMEDS gives good results (>50% inliers), use it
                if inlier_ratio_lmeds >= 0.5:
                    logger.info(
                        "AUTO mode: using LMEDS (inlier ratio %.1f%% >= 50%%)",
                        inlier_ratio_lmeds * 100,
                    )
                    return H_lmeds, mask_lmeds, "lmeds"

                # Otherwise, try RANSAC and compare
                logger.debug("LMEDS inlier ratio low, trying RANSAC")

            # Try RANSAC
            H_ransac, mask_ransac = cv2.findHomography(
                local_points, image_points, cv2.RANSAC, self.ransac_threshold
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
                            inlier_ratio_ransac * 100,
                        )
                        return H_lmeds, mask_lmeds, "lmeds"

                logger.info(
                    "AUTO mode: using RANSAC (inlier ratio %.1f%%)", inlier_ratio_ransac * 100
                )
                return H_ransac, mask_ransac, "ransac"

            # If RANSAC failed but LMEDS succeeded, use LMEDS
            if H_lmeds is not None:
                logger.info("AUTO mode: RANSAC failed, falling back to LMEDS")
                return H_lmeds, mask_lmeds, "lmeds"

            # Both failed
            logger.error("AUTO mode: both LMEDS and RANSAC failed")
            return None, None, "none"

    # =========================================================================
    # HomographyProvider Interface Implementation
    # =========================================================================

    def compute_homography(self, frame: np.ndarray, reference: dict[str, Any]) -> HomographyResult:
        """
        Compute homography from Ground Control Points.

        Args:
            frame: Image frame (not used for GCP approach, but required by interface)
            reference: Dictionary with required key:
                - 'ground_control_points': List of GCP dictionaries, each with:
                    - 'map': {'pixel_x': float, 'pixel_y': float} OR MapPoint object
                    - 'image': {'u': float, 'v': float}

        Returns:
            HomographyResult with computed homography matrix and confidence

        Raises:
            ValueError: If required reference data is missing or invalid
            RuntimeError: If homography computation fails
        """
        # Validate reference data
        if "ground_control_points" not in reference:
            raise ValueError("Missing required reference key: 'ground_control_points'")

        gcps = reference["ground_control_points"]

        if not isinstance(gcps, list) or len(gcps) < self.min_matches:
            raise ValueError(
                f"Need at least {self.min_matches} ground control points, got {len(gcps)}"
            )

        # Extract GCP data
        image_pts_list: list[list[float]] = []
        map_pts_list: list[list[float]] = []

        for gcp in gcps:
            # Support both MapPoint objects and dict format
            if isinstance(gcp.get("map"), MapPoint):
                map_point = gcp["map"]
                map_pts_list.append([map_point.pixel_x, map_point.pixel_y])
            elif "map" in gcp:
                map_data = gcp["map"]
                if "pixel_x" not in map_data or "pixel_y" not in map_data:
                    raise ValueError("Map must have 'pixel_x' and 'pixel_y' keys")
                map_pts_list.append([map_data["pixel_x"], map_data["pixel_y"]])
            else:
                raise ValueError("Each GCP must have 'map' key with pixel coordinates")

            if "image" not in gcp:
                raise ValueError("Each GCP must have 'image' key")

            img = gcp["image"]
            if "u" not in img or "v" not in img:
                raise ValueError("Image must have 'u' and 'v' keys")

            image_pts_list.append([img["u"], img["v"]])

        # Convert to numpy arrays
        image_points: np.ndarray = np.array(image_pts_list, dtype=np.float32)
        map_points: np.ndarray = np.array(map_pts_list, dtype=np.float32)

        logger.info(
            "Computing homography from %d GCPs (image -> map pixels), method=%s",
            len(image_points),
            self.fitting_method,
        )

        # Compute homography using specified fitting method
        # H maps map pixel coordinates to image coordinates
        H, mask, method_used = self._compute_homography_with_method(map_points, image_points)

        if H is None:
            logger.error("Failed to compute homography from GCPs")
            self.H = np.eye(3)
            self.H_inv = np.eye(3)
            self._confidence = 0.0
            raise RuntimeError("Failed to compute homography. Check GCP quality and distribution.")

        # Store homography (map pixels -> image)
        self.H = H

        # Calculate spatial distribution of GCPs (for confidence calculation)
        distribution_metrics = self._calculate_spatial_distribution(image_points)

        # Log distribution warnings
        for warning in distribution_metrics.get("warnings", []):
            logger.warning(warning)

        # Initialize reprojection error variables
        mean_reproj_error = None
        max_reproj_error = None

        # Initialize inlier tracking (will be updated if homography is valid)
        num_inliers = 0
        total_points = len(image_points)

        # Compute inverse (image -> map pixels)
        det_H = np.linalg.det(self.H)
        if abs(det_H) < self.MIN_DET_THRESHOLD:
            logger.warning("Homography is singular (det=%.2e). Inverse may be unstable.", det_H)
            self.H_inv = np.eye(3)
            self._confidence = 0.0
        else:
            self.H_inv = np.asarray(np.linalg.inv(self.H))

            # Calculate confidence based on inliers
            num_inliers = int(np.sum(mask)) if mask is not None else 0
            total_points = len(image_points)

            # Calculate reprojection errors for inliers
            if mask is not None and num_inliers > 0:
                inlier_map = map_points[mask.ravel() == 1]
                inlier_image = image_points[mask.ravel() == 1]

                # Project map points to image using homography
                projected = cv2.perspectiveTransform(inlier_map.reshape(-1, 1, 2), self.H).reshape(
                    -1, 2
                )

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
            all_projected = cv2.perspectiveTransform(map_points.reshape(-1, 1, 2), self.H).reshape(
                -1, 2
            )
            all_errors = np.linalg.norm(all_projected - image_points, axis=1)

            for i in range(len(gcps)):
                gcp = gcps[i]
                is_inlier = bool(mask[i][0])
                error = float(all_errors[i])
                desc = gcp.get("metadata", {}).get("description", f"GCP {i + 1}")

                outlier_analysis.append(
                    {
                        "index": i,
                        "description": desc,
                        "is_inlier": is_inlier,
                        "error_px": error,
                        "image_pixel": [float(image_points[i][0]), float(image_points[i][1])],
                        "map_pixel": [float(map_points[i][0]), float(map_points[i][1])],
                    }
                )

        # Sort outlier analysis by error (highest first)
        outlier_analysis.sort(key=lambda x: x["error_px"], reverse=True)

        # Build confidence breakdown for diagnostics
        inlier_ratio = num_inliers / total_points if total_points > 0 else 0
        confidence_breakdown: dict[str, Any] = {
            "inlier_ratio": inlier_ratio,
            "inlier_penalty_applied": inlier_ratio < self.MIN_INLIER_RATIO,
            "base_confidence": inlier_ratio * self.CONFIDENCE_PENALTY_LOW_INLIERS
            if inlier_ratio < self.MIN_INLIER_RATIO
            else inlier_ratio,
            "error_factor": max(0.3, 1.0 - (mean_reproj_error / (2 * self.ransac_threshold)))
            if mean_reproj_error
            else 1.0,
            "distribution_penalty": None,
            "final_confidence": self._confidence,
        }

        # Determine distribution penalty
        dist_score = distribution_metrics.get("distribution_score", 0.5)
        coverage = distribution_metrics.get("coverage_ratio", 0.0)
        if coverage < self.MIN_COVERAGE_RATIO:
            confidence_breakdown["distribution_penalty"] = (
                f"POOR_COVERAGE ({self.DIST_PENALTY_POOR_COVERAGE}x)"
            )
        elif dist_score < 0.5:
            confidence_breakdown["distribution_penalty"] = (
                f"LOW_COVERAGE ({self.DIST_PENALTY_LOW_COVERAGE}x)"
            )
        elif dist_score > 0.7:
            confidence_breakdown["distribution_penalty"] = (
                f"GOOD_COVERAGE_BONUS ({self.DIST_BONUS_GOOD_COVERAGE}x)"
            )
        else:
            confidence_breakdown["distribution_penalty"] = "NONE (1.0x)"

        # Build metadata with distribution info
        metadata = {
            "approach": HomographyApproach.FEATURE_MATCH.value,
            "method": "gcp_based",
            "fitting_method": method_used,
            "fitting_method_config": self.fitting_method,
            "num_gcps": len(image_points),
            "num_inliers": int(np.sum(mask)) if mask is not None else 0,
            "inlier_ratio": float(np.sum(mask)) / len(image_points) if mask is not None else 0.0,
            "inlier_mask": mask.flatten().astype(bool).tolist() if mask is not None else None,
            "determinant": det_H,
            "map_id": self.map_id,
            # Distribution metrics
            "distribution": {
                "coverage_ratio": distribution_metrics["coverage_ratio"],
                "quadrants_covered": distribution_metrics["quadrants_covered"],
                "spread_x": distribution_metrics["spread_x"],
                "spread_y": distribution_metrics["spread_y"],
                "distribution_score": distribution_metrics["distribution_score"],
                "warnings": distribution_metrics["warnings"],
            },
            # Reprojection error stats
            "reprojection_error": {
                "mean_px": mean_reproj_error,
                "max_px": max_reproj_error,
                "threshold_px": self.ransac_threshold,
            },
            # Detailed diagnostics
            "confidence_breakdown": confidence_breakdown,
            "outlier_analysis": outlier_analysis,
            "top_outliers": [o for o in outlier_analysis if not o["is_inlier"]][:5],
            "suggested_action": self._get_suggested_action(confidence_breakdown, outlier_analysis),
        }

        self._last_metadata = metadata

        logger.info(
            "Homography computed: %d/%d inliers, confidence=%.3f, distribution_score=%.2f, method=%s",
            metadata["num_inliers"],
            metadata["num_gcps"],
            self._confidence,
            distribution_metrics["distribution_score"],
            method_used,
        )

        return HomographyResult(
            homography_matrix=self.H.copy(), confidence=self._confidence, metadata=metadata
        )

    def project_point(self, image_point: tuple[float, float], point_id: str = "") -> MapPoint:
        """
        Project image point to map pixel coordinates.

        Args:
            image_point: (u, v) pixel coordinates in camera image
            point_id: Optional ID for the generated MapPoint (auto-generated if empty)

        Returns:
            MapPoint with pixel coordinates on the reference map

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

        # Project to map pixel coordinates
        pixel_x, pixel_y = self._project_image_point_to_map(image_point)

        # Generate a unique ID for this projected point if not provided
        if not point_id:
            point_id = f"proj_{uuid.uuid4().hex[:8]}"

        return MapPoint(
            id=point_id,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            map_id=self.map_id,
        )

    def project_points(
        self, image_points: list[tuple[float, float]], point_id_prefix: str = "proj"
    ) -> list[MapPoint]:
        """
        Project multiple image points to map pixel coordinates.

        Args:
            image_points: List of (u, v) pixel coordinates in camera image
            point_id_prefix: Prefix for generated MapPoint IDs (default: "proj")

        Returns:
            List of MapPoint objects with pixel coordinates on the reference map

        Raises:
            RuntimeError: If no valid homography computed
            ValueError: If any image_point is outside valid bounds
        """
        if not self.is_valid():
            raise RuntimeError("No valid homography available. Call compute_homography() first.")

        map_points = []
        for i, image_point in enumerate(image_points):
            point_id = f"{point_id_prefix}_{i}"
            map_point = self.project_point(image_point, point_id=point_id)
            map_points.append(map_point)

        return map_points

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
            self.confidence_threshold,
        )
