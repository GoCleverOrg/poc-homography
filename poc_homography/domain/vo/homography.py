"""Homography matrix value object for projective transformation between 2D planes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from poc_homography.domain.vo.matrix3x3 import Matrix3x3
from poc_homography.domain.vo.vector3 import Vector3

CONDITION_THRESHOLD = 1e10
DETERMINANT_THRESHOLD = 1e-10

_PRIVATE_SENTINEL = object()


@dataclass(frozen=True)
class Homography:
    """3x3 homography matrix for projective transformation between 2D planes.

    A homography (also called a projective transformation or collineation)
    maps points from one 2D plane to another. Common uses include:
    - Image-to-image transformations (e.g., panorama stitching)
    - Ground plane to image projection (e.g., camera calibration)
    - Map to image transformations

    The homography satisfies:
        [x']       [x]
        [y']  = H  [y]
        [1 ]       [1]

    Where (x, y) is the source point and (x', y') is the destination point
    (after normalization by the homogeneous coordinate).

    Use the `create()` factory method to construct instances.
    Direct constructor access is reserved for internal use.
    """

    _matrix: Matrix3x3 = field(repr=False)
    _inverse_matrix: Matrix3x3 = field(repr=False)
    _sentinel: object = field(default=None, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        """Verify construction was via create() factory or internal inverse()."""
        if self._sentinel is not _PRIVATE_SENTINEL:
            raise TypeError(
                "Homography cannot be instantiated directly. Use Homography.create() instead."
            )

    @classmethod
    def create(cls, matrix: np.ndarray) -> Homography:
        """Create a Homography from a numpy array.

        This factory method validates the matrix and computes its inverse.

        Args:
            matrix: 3x3 homography matrix.

        Returns:
            New Homography instance.

        Raises:
            ValueError: If matrix is not 3x3, contains NaN/Inf, is singular,
                or is ill-conditioned.
        """
        m = Matrix3x3.create(matrix)

        det = m.determinant
        if abs(det) < DETERMINANT_THRESHOLD:
            raise ValueError(
                f"Homography matrix is singular (det={det:.2e}). Cannot compute inverse."
            )

        cond = m.condition_number
        if cond > CONDITION_THRESHOLD:
            raise ValueError(
                f"Homography matrix is ill-conditioned (condition={cond:.2e}). "
                f"Results would be numerically unstable."
            )

        m_inv = m.inverse()

        return cls(_matrix=m, _inverse_matrix=m_inv, _sentinel=_PRIVATE_SENTINEL)

    @property
    def condition_number(self) -> float:
        """Get the condition number of the homography matrix."""
        return self._matrix.condition_number

    @property
    def determinant(self) -> float:
        """Get the determinant of the homography matrix."""
        return self._matrix.determinant

    @property
    def inverse(self) -> Homography:
        """Get the inverse homography.

        The inverse homography maps points in the opposite direction.
        Note that `h.inverse.inverse` returns a homography equivalent to `h`.

        Returns:
            A new Homography that is the inverse of this one.
        """
        return Homography(
            _matrix=self._inverse_matrix,
            _inverse_matrix=self._matrix,
            _sentinel=_PRIVATE_SENTINEL,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with matrix, condition_number, and determinant.
        """
        return {
            "matrix": self._matrix.to_list(),
            "condition_number": self.condition_number,
            "determinant": self.determinant,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Homography:
        """Create Homography from dictionary.

        Args:
            data: Dictionary with matrix (required).

        Returns:
            New Homography instance.
        """
        matrix = np.array(data["matrix"], dtype=np.float64)
        return cls.create(matrix=matrix)

    def try_world_to_image(self, Xw: float, Yw: float) -> tuple[float, float] | None:
        """Project world point to image, returning None if behind camera.

        Args:
            Xw: World x-coordinate in meters (East).
            Yw: World y-coordinate in meters (North).

        Returns:
            Tuple of (u, v) pixel coordinates, or None if point is behind camera.
        """
        pt = Vector3.create(Xw, Yw, 1.0)
        result: Vector3 = self._matrix @ pt

        if result.z <= 1e-10:
            return None

        return (result.x / result.z, result.y / result.z)

    def __hash__(self) -> int:
        """Compute hash for use in sets and as dict keys."""
        return hash((self._matrix, self._inverse_matrix))
