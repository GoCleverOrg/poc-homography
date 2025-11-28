"""
Abstract interface for homography computation and coordinate projection.

This module defines the core abstractions for computing homography transformations
between image coordinates and world coordinates (GPS or local map coordinates).

Coordinate Systems:
    - Image coordinates: (u, v) in pixels, origin at top-left
    - World coordinates: (latitude, longitude) in decimal degrees (WGS84)
    - Map coordinates: (x, y) in meters from camera position on ground plane

The interface supports multiple homography computation approaches:
    - INTRINSIC_EXTRINSIC: Camera calibration parameters + pose
    - FEATURE_MATCH: Feature matching with known ground control points
    - LEARNED: Machine learning-based homography estimation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional, List, Dict, Any
import numpy as np


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


@dataclass
class WorldPoint:
    """Represents a point in world coordinates with confidence score.

    Attributes:
        latitude: Latitude in decimal degrees (WGS84), range [-90, 90]
        longitude: Longitude in decimal degrees (WGS84), range [-180, 180]
        confidence: Confidence score for this projection, range [0.0, 1.0]
            where 1.0 indicates highest confidence
    """
    latitude: float
    longitude: float
    confidence: float

    def __post_init__(self):
        """Validate coordinate ranges."""
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Latitude must be in range [-90, 90], got {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Longitude must be in range [-180, 180], got {self.longitude}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in range [0.0, 1.0], got {self.confidence}")


@dataclass
class MapCoordinate:
    """Represents a point in local map coordinates relative to camera.

    Local map coordinates use a metric coordinate system with the camera
    position as the origin, projected onto the ground plane.

    Attributes:
        x: Distance in meters along the x-axis (typically east-west)
        y: Distance in meters along the y-axis (typically north-south)
        confidence: Confidence score for this projection, range [0.0, 1.0]
        elevation: Optional elevation above ground plane in meters
    """
    x: float
    y: float
    confidence: float
    elevation: Optional[float] = None

    def __post_init__(self):
        """Validate confidence range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in range [0.0, 1.0], got {self.confidence}")


@dataclass
class HomographyResult:
    """Result of homography computation including matrix and metadata.

    Attributes:
        homography_matrix: 3x3 homography transformation matrix mapping
            image coordinates to ground plane coordinates. The matrix
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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate homography matrix and confidence."""
        if self.homography_matrix.shape != (3, 3):
            raise ValueError(
                f"Homography matrix must be 3x3, got shape {self.homography_matrix.shape}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in range [0.0, 1.0], got {self.confidence}")


class GPSPositionMixin:
    """
    Mixin class providing GPS position storage and validation.

    Classes using this mixin must initialize _camera_gps_lat and _camera_gps_lon
    attributes (typically to None) in their __init__.
    """
    _camera_gps_lat: Optional[float]
    _camera_gps_lon: Optional[float]

    def set_camera_gps_position(self, lat: float, lon: float) -> None:
        """
        Set camera GPS position for WorldPoint conversion.

        This establishes the reference point for converting local metric
        coordinates to GPS coordinates.

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

        self._camera_gps_lat = lat
        self._camera_gps_lon = lon


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
    def compute_homography(
        self,
        frame: np.ndarray,
        reference: Dict[str, Any]
    ) -> HomographyResult:
        """Compute homography matrix from image frame to ground plane.

        This method analyzes the input frame and reference data to compute
        a homography transformation. The transformation maps image coordinates
        to ground plane coordinates.

        Args:
            frame: Image frame as numpy array with shape (height, width, channels)
                or (height, width) for grayscale. Typically uint8 BGR or RGB.
            reference: Reference data dictionary containing approach-specific
                information. Common keys:
                - 'camera_matrix': 3x3 intrinsic camera matrix (intrinsic/extrinsic)
                - 'dist_coeffs': Distortion coefficients (intrinsic/extrinsic)
                - 'ground_points': Known world coordinates (feature matching)
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
    def project_point(self, image_point: Tuple[float, float]) -> WorldPoint:
        """Project single image coordinate to world coordinate (GPS).

        Transforms a 2D image point to a GPS coordinate using the most
        recently computed homography matrix. The point is assumed to lie
        on the ground plane.

        Args:
            image_point: (u, v) pixel coordinates in image space
                u: horizontal pixel coordinate (0 = left edge)
                v: vertical pixel coordinate (0 = top edge)

        Returns:
            WorldPoint with:
                - latitude: Projected latitude in decimal degrees
                - longitude: Projected longitude in decimal degrees
                - confidence: Point-specific confidence score [0.0, 1.0]
                    May be lower than overall homography confidence if point
                    is near image edges or in low-confidence regions

        Raises:
            RuntimeError: If no valid homography has been computed yet
            ValueError: If image_point is outside valid image bounds

        Note:
            Call is_valid() first to ensure homography is ready for projection.
        """
        pass

    @abstractmethod
    def project_points(
        self,
        image_points: List[Tuple[float, float]]
    ) -> List[WorldPoint]:
        """Project multiple image points to world coordinates (GPS).

        Batch version of project_point() for efficiency when projecting
        many points. Uses the same homography matrix for all points.

        Args:
            image_points: List of (u, v) pixel coordinates to project

        Returns:
            List of WorldPoint objects, one per input point, in same order.
            Each WorldPoint contains lat/lon and per-point confidence score.

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


class HomographyProviderExtended(HomographyProvider):
    """Extended interface with additional coordinate projection methods.

    This optional extension adds support for local map coordinate projections
    in addition to GPS coordinates. Useful for applications that work in
    metric local coordinate systems.

    Implementers can choose to implement this extended interface if their
    use case requires local map coordinates.
    """

    @abstractmethod
    def project_point_to_map(
        self,
        image_point: Tuple[float, float]
    ) -> MapCoordinate:
        """Project image coordinate to local map coordinate system.

        Args:
            image_point: (u, v) pixel coordinates in image space

        Returns:
            MapCoordinate with x, y in meters from camera position

        Raises:
            RuntimeError: If no valid homography has been computed yet
        """
        pass

    @abstractmethod
    def project_points_to_map(
        self,
        image_points: List[Tuple[float, float]]
    ) -> List[MapCoordinate]:
        """Project multiple image points to local map coordinates.

        Args:
            image_points: List of (u, v) pixel coordinates

        Returns:
            List of MapCoordinate objects with x, y in meters

        Raises:
            RuntimeError: If no valid homography has been computed yet
        """
        pass


def validate_homography_matrix(
    matrix: Optional[np.ndarray],
    confidence: float,
    confidence_threshold: float,
    min_det_threshold: float = 1e-10
) -> bool:
    """
    Validate a homography matrix for use in projections.

    This helper function provides common validation logic for homography matrices,
    checking for None, identity, singularity, and confidence threshold.

    Args:
        matrix: The homography matrix to validate, or None
        confidence: Current confidence score
        confidence_threshold: Minimum confidence for validity
        min_det_threshold: Minimum determinant magnitude for non-singular check

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
    det_H = np.linalg.det(matrix)
    if abs(det_H) < min_det_threshold:
        return False

    # Check confidence threshold
    if confidence < confidence_threshold:
        return False

    return True
