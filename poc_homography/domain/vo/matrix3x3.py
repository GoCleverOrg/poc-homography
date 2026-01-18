"""Immutable 3x3 matrix value object."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, overload

import numpy as np

if TYPE_CHECKING:
    from poc_homography.domain.vo.vector3 import Vector3
    from poc_homography.types import Radians

_PRIVATE_SENTINEL = object()


@dataclass(frozen=True)
class Matrix3x3:
    """Immutable 3x3 matrix value object.

    This value object encapsulates a 3x3 numpy array, providing:
    - Immutability (frozen dataclass with bytes storage)
    - Hashability (can be used as dict key or in sets)
    - Common matrix operations (determinant, condition number, inverse)
    - Rotation matrix factories

    Use the `create()` factory method to construct instances.
    Direct constructor access is reserved for internal use.
    """

    _data: bytes = field(repr=False)
    _sentinel: object = field(default=None, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        """Verify construction was via create() factory."""
        if self._sentinel is not _PRIVATE_SENTINEL:
            raise TypeError(
                "Matrix3x3 cannot be instantiated directly. Use Matrix3x3.create() instead."
            )

    @classmethod
    def create(cls, array: np.ndarray) -> Matrix3x3:
        """Create a Matrix3x3 from a numpy array.

        Args:
            array: A 3x3 numpy array.

        Returns:
            New Matrix3x3 instance.

        Raises:
            ValueError: If array is not 3x3 or contains NaN/Infinity.
        """
        arr = np.asarray(array, dtype=np.float64)

        if arr.shape != (3, 3):
            raise ValueError(f"Matrix must be 3x3, got shape {arr.shape}")

        if not np.all(np.isfinite(arr)):
            raise ValueError("Matrix contains NaN or Infinity values")

        return cls(_data=arr.tobytes(), _sentinel=_PRIVATE_SENTINEL)

    @classmethod
    def rotation_x(cls, angle_rad: Radians) -> Matrix3x3:
        """Create rotation matrix around X axis.

        Args:
            angle_rad: Rotation angle in radians.

        Returns:
            Rotation matrix for rotation around X axis.
        """
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return cls.create(
            np.array(
                [
                    [1, 0, 0],
                    [0, c, -s],
                    [0, s, c],
                ],
                dtype=np.float64,
            )
        )

    @classmethod
    def rotation_y(cls, angle_rad: Radians) -> Matrix3x3:
        """Create rotation matrix around Y axis.

        Args:
            angle_rad: Rotation angle in radians.

        Returns:
            Rotation matrix for rotation around Y axis.
        """
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return cls.create(
            np.array(
                [
                    [c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c],
                ],
                dtype=np.float64,
            )
        )

    @classmethod
    def rotation_z(cls, angle_rad: Radians) -> Matrix3x3:
        """Create rotation matrix around Z axis.

        Args:
            angle_rad: Rotation angle in radians.

        Returns:
            Rotation matrix for rotation around Z axis.
        """
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return cls.create(
            np.array(
                [
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
        )

    @classmethod
    def from_euler_zyx(cls, yaw_rad: Radians, pitch_rad: Radians, roll_rad: Radians) -> Matrix3x3:
        """Create rotation matrix from ZYX Euler angles.

        Uses the ZYX (yaw-pitch-roll) convention:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

        This computes the combined rotation matrix directly, avoiding
        intermediate Matrix3x3 allocations.

        Args:
            yaw_rad: Rotation around Z-axis in radians.
            pitch_rad: Rotation around Y-axis in radians.
            roll_rad: Rotation around X-axis in radians.

        Returns:
            Rotation matrix as Matrix3x3.
        """
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
        cr, sr = math.cos(roll_rad), math.sin(roll_rad)

        # Direct computation of R = Rz @ Ry @ Rx
        return cls.create(
            np.array(
                [
                    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                    [-sp, cp * sr, cp * cr],
                ],
                dtype=np.float64,
            )
        )

    def _to_array(self) -> np.ndarray:
        """Get the matrix as an immutable numpy array.

        Returns:
            3x3 numpy array with writeable=False.
        """
        arr = np.frombuffer(self._data, dtype=np.float64).reshape(3, 3)
        arr.flags.writeable = False
        return arr

    @property
    def T(self) -> Matrix3x3:
        """Return the transpose of the matrix."""
        return Matrix3x3.create(self._to_array().T)

    @property
    def determinant(self) -> float:
        """Compute the determinant of the matrix."""
        return float(np.linalg.det(self._to_array()))

    @property
    def condition_number(self) -> float:
        """Compute the condition number of the matrix.

        The condition number indicates numerical stability.
        Lower values are better; high values indicate potential numerical issues.
        """
        return float(np.linalg.cond(self._to_array()))

    def inverse(self) -> Matrix3x3:
        """Compute the inverse matrix.

        Returns:
            New Matrix3x3 that is the inverse of this matrix.

        Raises:
            ValueError: If matrix is singular (determinant near zero).
        """
        arr = self._to_array()
        det = np.linalg.det(arr)

        if abs(det) < 1e-10:
            raise ValueError(f"Matrix is singular (det={det:.2e}). Cannot compute inverse.")

        inv_arr = np.linalg.inv(arr)
        return Matrix3x3.create(inv_arr)

    @overload
    def __matmul__(self, other: Matrix3x3) -> Matrix3x3: ...

    @overload
    def __matmul__(self, other: Vector3) -> Vector3: ...

    def __matmul__(self, other: Matrix3x3 | Vector3) -> Matrix3x3 | Vector3:
        """Matrix multiplication.

        Args:
            other: Matrix3x3 for matrix-matrix multiplication,
                   or Vector3 for matrix-vector multiplication.

        Returns:
            Matrix3x3 if other is Matrix3x3, Vector3 if other is Vector3.
        """
        from poc_homography.domain.vo.vector3 import Vector3

        if isinstance(other, Vector3):
            result = self._to_array() @ other._to_array()
            return Vector3.from_array(result)
        if isinstance(other, Matrix3x3):
            return Matrix3x3.create(self._to_array() @ other._to_array())
        return NotImplemented

    def to_list(self) -> list[list[float]]:
        """Convert to nested list for serialization."""
        return self._to_array().tolist()

    @classmethod
    def from_list(cls, data: list[list[float]]) -> Matrix3x3:
        """Create from nested list.

        Args:
            data: 3x3 nested list of floats.

        Returns:
            New Matrix3x3 instance.
        """
        return cls.create(np.array(data, dtype=np.float64))

    def __hash__(self) -> int:
        """Compute hash for use in sets and as dict keys."""
        return hash(self._data)
