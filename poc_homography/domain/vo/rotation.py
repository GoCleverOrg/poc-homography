"""Rotation matrix value object for 3D rotations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from poc_homography.domain.vo.matrix3x3 import Matrix3x3
from poc_homography.types import Degrees, degrees_to_radians

_PRIVATE_SENTINEL = object()


@dataclass(frozen=True)
class Rotation:
    """Immutable 3x3 rotation matrix value object.

    Represents a rotation in 3D space using the ZYX (yaw-pitch-roll) Euler
    angle convention. The rotation matrix R transforms vectors from camera
    coordinates to world coordinates: p_world = R @ p_camera.

    Use the `from_euler()` factory method to construct instances.
    Direct constructor access is reserved for internal use.
    """

    _matrix: Matrix3x3 = field(repr=False)
    _sentinel: object = field(default=None, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        """Verify construction was via factory method."""
        if self._sentinel is not _PRIVATE_SENTINEL:
            raise TypeError(
                "Rotation cannot be instantiated directly. Use Rotation.from_euler() instead."
            )

    @classmethod
    def from_euler(
        cls,
        yaw: Degrees,
        pitch: Degrees,
        roll: Degrees,
    ) -> Rotation:
        """Create Rotation from Euler angles (ZYX convention).

        Uses the ZYX (yaw-pitch-roll) Euler angle convention:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

        Args:
            yaw: Rotation around Z-axis in degrees.
            pitch: Rotation around Y-axis in degrees.
            roll: Rotation around X-axis in degrees.

        Returns:
            New Rotation instance.

        Raises:
            ValueError: If angles are non-finite.
        """
        if not math.isfinite(float(yaw)):
            raise ValueError(f"yaw must be finite, got {yaw}")
        if not math.isfinite(float(pitch)):
            raise ValueError(f"pitch must be finite, got {pitch}")
        if not math.isfinite(float(roll)):
            raise ValueError(f"roll must be finite, got {roll}")

        yaw_rad = degrees_to_radians(yaw)
        pitch_rad = degrees_to_radians(pitch)
        roll_rad = degrees_to_radians(roll)

        matrix = Matrix3x3.from_euler_zyx(yaw_rad, pitch_rad, roll_rad)
        return cls(_matrix=matrix, _sentinel=_PRIVATE_SENTINEL)

    @classmethod
    def from_matrix(cls, matrix: Matrix3x3) -> Rotation:
        """Create Rotation from an existing Matrix3x3.

        Note: This does not validate that the matrix is a valid rotation
        matrix (orthogonal with determinant 1). Use with caution.

        Args:
            matrix: A 3x3 matrix representing a rotation.

        Returns:
            New Rotation instance.
        """
        return cls(_matrix=matrix, _sentinel=_PRIVATE_SENTINEL)

    def to_matrix(self) -> Matrix3x3:
        """Get the underlying 3x3 matrix.

        Returns:
            The rotation matrix as a Matrix3x3 value object.
        """
        return self._matrix

    def compose(self, other: Rotation) -> Rotation:
        """Compose this rotation with another.

        The result represents applying `other` first, then `self`:
        R_result = self @ other

        Args:
            other: The rotation to compose with.

        Returns:
            New Rotation representing the composition.
        """
        composed = self._matrix @ other._matrix
        return Rotation(_matrix=composed, _sentinel=_PRIVATE_SENTINEL)

    def __hash__(self) -> int:
        """Compute hash for use in sets and as dict keys."""
        return hash(self._matrix)
