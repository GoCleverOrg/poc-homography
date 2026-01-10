"""
Abstract interface for homography computation and coordinate projection.

This module defines the core abstractions for computing homography transformations
between image coordinates and map coordinates (pixel coordinates on a reference map).

Coordinate Systems:
    - Image coordinates: (u, v) in pixels, origin at top-left of camera image
    - Map coordinates: (pixel_x, pixel_y) in pixels on the reference map image

The interface supports multiple homography computation approaches:
    - INTRINSIC_EXTRINSIC: Camera calibration parameters + pose
    - FEATURE_MATCH: Feature matching with known ground control points
    - LEARNED: Machine learning-based homography estimation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from poc_homography.map_points import MapPoint


class HomographyApproach(Enum):
    """Enumeration of supported homography computation approaches."""

    INTRINSIC_EXTRINSIC = "intrinsic_extrinsic"
    """Uses camera intrinsic parameters (focal length, principal point)
    and extrinsic parameters (rotation, translation) to compute homography."""

    FEATURE_MATCH = "feature_match"
    """Uses feature detection and matching with known ground control points
    to compute homography via point correspondences."""

    LEARNED = "learned"
    """Uses machine learning models (e.g., neural networks) to estimate
    homography directly from image data."""


class CoordinateSystemMode(Enum):
    """Enumeration of coordinate system origin modes for camera positioning.

    This defines where the origin (0, 0, 0) of the world coordinate system is placed
    when computing homography transformations.
    """

    ORIGIN_AT_CAMERA = "origin_at_camera"
    """Origin at camera position (Mode B - current default).

    In this mode:
    - Camera position is set to [0, 0, height] where height is camera elevation
    - World coordinate system origin is directly below the camera on the ground plane
    - Projected points are measured as offsets from the camera position

    This is the default and recommended mode for single-camera applications.
    """

    MAP_BASED_ORIGIN = "map_based_origin"
    """Map-based pixel coordinates (default for MapPoint system).

    In this mode:
    - Projections return pixel coordinates on the reference map image
    - Origin is at top-left of the map image (0, 0)
    - Coordinates increase right (x) and down (y)
    - This is the standard mode for MapPoint-based homography
    """


@dataclass
class HomographyResult:
    """Result of homography computation including matrix and metadata.

    Attributes:
        homography_matrix: 3x3 homography transformation matrix mapping
            image coordinates to map plane coordinates. The matrix
            transforms homogeneous coordinates [u, v, 1]^T to [x, y, w]^T
            where the final coordinates are (x/w, y/w).
        confidence: Overall confidence score for this homography, range [0.0, 1.0]
            Interpretation depends on approach:
            - For feature matching: based on reprojection error and inlier ratio
            - For intrinsic/extrinsic: based on calibration quality
            - For learned: model-specific confidence score
        metadata: Additional approach-specific information such as:
            - 'num_inliers': Number of inlier points (feature matching)
            - 'reprojection_error': Mean reprojection error in pixels
            - 'approach': HomographyApproach used
            - 'timestamp': When homography was computed
            - 'camera_pose': Camera rotation and translation (intrinsic/extrinsic)
    """

    homography_matrix: np.ndarray
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate homography matrix and confidence."""
        if self.homography_matrix.shape != (3, 3):
            raise ValueError(
                f"Homography matrix must be 3x3, got shape {self.homography_matrix.shape}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in range [0.0, 1.0], got {self.confidence}")


class HomographyProvider(ABC):
    """Abstract base class for homography computation and coordinate projection.

    This interface defines the contract for all homography computation approaches.
    Implementers should subclass this and provide concrete implementations for
    each abstract method.

    Typical usage:
        1. Create provider instance with approach-specific configuration
        2. Call compute_homography() with frame and reference data
        3. Use project_point() or project_points() to transform coordinates
        4. Check is_valid() before projections to ensure homography is ready
        5. Use get_confidence() to assess quality of current homography

    Thread Safety:
        Implementations should document their thread-safety guarantees.
        If compute_homography() can be called from multiple threads,
        implementers should handle synchronization internally.

    State Management:
        The provider maintains internal state (current homography matrix).
        compute_homography() updates this state. Subsequent project_point()
        calls use the most recently computed homography until compute_homography()
        is called again.
    """

    @abstractmethod
    def compute_homography(self, frame: np.ndarray, reference: dict[str, Any]) -> HomographyResult:
        """Compute homography matrix from image frame to map plane.

        This method analyzes the input frame and reference data to compute
        a homography transformation. The transformation maps image coordinates
        to map plane coordinates.

        Args:
            frame: Image frame as numpy array with shape (height, width, channels)
                or (height, width) for grayscale. Typically uint8 BGR or RGB.
            reference: Reference data dictionary containing approach-specific
                information. Common keys:
                - 'camera_matrix': 3x3 intrinsic camera matrix (intrinsic/extrinsic)
                - 'dist_coeffs': Distortion coefficients (intrinsic/extrinsic)
                - 'ground_points': Known map coordinates (feature matching)
                - 'image_points': Corresponding image points (feature matching)
                - 'model_path': Path to trained model (learned approach)

        Returns:
            HomographyResult containing:
                - homography_matrix: 3x3 transformation matrix
                - confidence: Quality score [0.0, 1.0]
                - metadata: Approach-specific debug/diagnostic information

        Raises:
            ValueError: If inputs are invalid or malformed
            RuntimeError: If homography computation fails

        Note:
            This method updates the provider's internal state. Subsequent
            calls to project_point() will use this computed homography.
        """
        pass

    @abstractmethod
    def project_point(self, image_point: tuple[float, float]) -> MapPoint:
        """Project single image coordinate to map coordinate.

        Transforms a 2D image point to a map coordinate using the most
        recently computed homography matrix. The point is assumed to lie
        on the ground plane.

        Args:
            image_point: (u, v) pixel coordinates in image space
                u: horizontal pixel coordinate (0 = left edge)
                v: vertical pixel coordinate (0 = top edge)

        Returns:
            MapPoint with:
                - id: Identifier for the projected point
                - pixel_x: X coordinate on the map image
                - pixel_y: Y coordinate on the map image
                - map_id: Identifier of the reference map

        Raises:
            RuntimeError: If no valid homography has been computed yet
            ValueError: If image_point is outside valid image bounds

        Note:
            Call is_valid() first to ensure homography is ready for projection.
        """
        pass

    @abstractmethod
    def project_points(self, image_points: list[tuple[float, float]]) -> list[MapPoint]:
        """Project multiple image points to map coordinates.

        Batch version of project_point() for efficiency when projecting
        many points. Uses the same homography matrix for all points.

        Args:
            image_points: List of (u, v) pixel coordinates to project

        Returns:
            List of MapPoint objects, one per input point, in same order.

        Raises:
            RuntimeError: If no valid homography has been computed yet
            ValueError: If any image_point is outside valid image bounds

        Note:
            Implementations may optimize batch projection using vectorized
            operations for better performance than calling project_point()
            repeatedly.
        """
        pass

    @abstractmethod
    def get_confidence(self) -> float:
        """Return confidence score of current homography.

        Returns:
            float: Confidence score in range [0.0, 1.0] where:
                - 1.0 = highest confidence, excellent homography quality
                - 0.5 = moderate confidence, usable but with caution
                - 0.0 = no confidence, homography should not be used

        Note:
            Returns 0.0 if no homography has been computed yet.
            Confidence interpretation is approach-specific. Check metadata
            from compute_homography() for detailed quality metrics.
        """
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if homography is valid and ready for projection.

        A homography is considered valid if:
            - compute_homography() has been called successfully at least once
            - The computed matrix is well-conditioned (not singular/degenerate)
            - Confidence score meets minimum threshold for the approach

        Returns:
            bool: True if homography is valid and projections can be performed,
                False otherwise.

        Note:
            Always check this before calling project_point() or project_points()
            to avoid runtime errors.
        """
        pass


def validate_homography_matrix(
    matrix: np.ndarray | None,
    confidence: float,
    confidence_threshold: float,
    min_det_threshold: float = 1e-15,
) -> bool:
    """
    Validate a homography matrix for use in projections.

    This helper function provides common validation logic for homography matrices,
    checking for None, identity, singularity, and confidence threshold.

    Args:
        matrix: The homography matrix to validate, or None
        confidence: Current confidence score
        confidence_threshold: Minimum confidence for validity
        min_det_threshold: Minimum determinant magnitude for non-singular check.
            Note: Homographies mapping between pixel coords (0-2000) and metric
            coords (0-500m) naturally have very small determinants due to scale
            differences. Default of 1e-15 accommodates this.

    Returns:
        True if the matrix is valid for projections, False otherwise
    """
    # Check if matrix exists
    if matrix is None:
        return False

    # Check if it's just an identity matrix (not computed)
    if np.allclose(matrix, np.eye(3)):
        return False

    # Check if matrix is not singular
    # Note: For homographies with large scale differences (pixels to meters),
    # the determinant can be very small but the matrix is still valid.
    det_H = np.linalg.det(matrix)
    if abs(det_H) < min_det_threshold:
        return False

    # Check confidence threshold
    if confidence < confidence_threshold:
        return False

    return True
