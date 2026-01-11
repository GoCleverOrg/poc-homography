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
import numpy.typing as npt

from poc_homography.pixel_point import PixelPoint

if TYPE_CHECKING:
    from poc_homography.map_points import MapPoint


class HomographyApproach(Enum):
    """Supported homography computation approaches."""

    INTRINSIC_EXTRINSIC = "intrinsic_extrinsic"
    FEATURE_MATCH = "feature_match"
    LEARNED = "learned"


class CoordinateSystemMode(Enum):
    """Coordinate system origin modes for camera positioning."""

    ORIGIN_AT_CAMERA = "origin_at_camera"
    MAP_BASED_ORIGIN = "map_based_origin"


@dataclass(frozen=True)
class HomographyMatrix:
    """Immutable wrapper for a 3x3 homography transformation matrix.

    A homography matrix maps homogeneous coordinates between two planes:
        [x']       [x]
        [y']  = H  [y]
        [w']       [1]

    The final coordinates are (x'/w', y'/w').

    Attributes:
        data: 3x3 numpy array containing the homography coefficients.
    """

    data: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate matrix shape and type."""
        if self.data.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got shape {self.data.shape}")
        # Ensure the data is float64 for numerical stability
        if self.data.dtype != np.float64:
            # Use object.__setattr__ since frozen=True
            object.__setattr__(self, "data", np.asarray(self.data, dtype=np.float64))

    @classmethod
    def from_array(cls, array: npt.ArrayLike) -> HomographyMatrix:
        """Create HomographyMatrix from a numpy array.

        Args:
            array: 3x3 array-like object.

        Returns:
            New HomographyMatrix instance.

        Raises:
            ValueError: If array is not 3x3.
        """
        return cls(data=np.asarray(array, dtype=np.float64))

    @classmethod
    def identity(cls) -> HomographyMatrix:
        """Create identity homography matrix.

        Returns:
            HomographyMatrix representing the identity transformation.
        """
        return cls(data=np.eye(3, dtype=np.float64))

    def inverse(self) -> HomographyMatrix:
        """Compute the inverse homography matrix.

        Returns:
            New HomographyMatrix representing the inverse transformation.

        Raises:
            ValueError: If matrix is singular (non-invertible).
        """
        det = np.linalg.det(self.data)
        if abs(det) < 1e-15:
            raise ValueError("Homography matrix is singular and cannot be inverted")
        return HomographyMatrix(data=np.asarray(np.linalg.inv(self.data), dtype=np.float64))

    @property
    def determinant(self) -> float:
        """Compute the determinant of the matrix.

        Returns:
            Determinant value. Non-zero indicates invertibility.
        """
        return float(np.linalg.det(self.data))

    @property
    def is_valid(self) -> bool:
        """Check if the homography is valid (non-singular, non-identity).

        Returns:
            True if the matrix is valid for transformations.
        """
        if np.allclose(self.data, np.eye(3)):
            return False
        if abs(self.determinant) < 1e-15:
            return False
        return True

    def transform(self, point: PixelPoint) -> PixelPoint:
        """Apply homography transformation to a point.

        Args:
            point: Input pixel coordinates.

        Returns:
            Transformed pixel coordinates.

        Raises:
            ValueError: If point transforms to infinity (w ≈ 0).
        """
        pt = np.array([point.x, point.y, 1.0])
        result = self.data @ pt
        if abs(result[2]) < 1e-10:
            raise ValueError("Point transforms to infinity (on horizon)")
        return PixelPoint(float(result[0] / result[2]), float(result[1] / result[2]))


@dataclass(frozen=True)
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

    This interface uses an IMMUTABLE pattern - implementations should use
    `compute_from_config()` classmethods that take immutable config objects
    and return immutable result objects.

    The legacy `compute_homography(frame, reference)` method has been removed.
    Use the config-based pattern instead:

    ```python
    config = IntrinsicExtrinsicConfig.create(...)
    result = IntrinsicExtrinsicHomography.compute_from_config(config)
    ```
    """

    @abstractmethod
    def project_point(self, image_point: PixelPoint, point_id: str = "") -> MapPoint:
        """Project single image coordinate to map coordinate.

        Args:
            image_point: Pixel coordinates in camera image
            point_id: Optional ID for the generated MapPoint (auto-generated if empty)

        Returns:
            MapPoint with pixel coordinates on the map
        """
        pass

    @abstractmethod
    def project_points(
        self, image_points: list[PixelPoint], point_id_prefix: str = "proj"
    ) -> list[MapPoint]:
        """Project multiple image points to map coordinates.

        Args:
            image_points: List of pixel coordinates
            point_id_prefix: Prefix for generated MapPoint IDs (default: "proj")

        Returns:
            List of MapPoint objects with pixel coordinates on the map
        """
        pass

    @abstractmethod
    def get_confidence(self) -> float:
        """Return confidence score [0.0, 1.0] of current homography."""
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if homography is valid and ready for projection."""
        pass


def validate_homography_matrix(
    matrix: np.ndarray | None,
    confidence: float,
    confidence_threshold: float,
    min_det_threshold: float = 1e-15,
) -> bool:
    """Validate a homography matrix for projections.

    Checks for None, identity, singularity, and confidence threshold.
    Default min_det_threshold of 1e-15 accommodates large scale differences
    between pixel coords (0-2000) and metric coords (0-500m).
    """
    if matrix is None:
        return False

    if np.allclose(matrix, np.eye(3)):
        return False

    if abs(np.linalg.det(matrix)) < min_det_threshold:
        return False

    if confidence < confidence_threshold:
        return False

    return True
