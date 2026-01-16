"""Immutable 3x3 matrix value object."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_PRIVATE_SENTINEL = object()


@dataclass(frozen=True)
class Matrix3x3:
    """Immutable 3x3 matrix value object.

    This value object encapsulates a 3x3 numpy array, providing:
    - Immutability (frozen dataclass with bytes storage)
    - Hashability (can be used as dict key or in sets)
    - Common matrix operations (determinant, condition number, inverse)

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

    def to_array(self) -> np.ndarray:
        """Get the matrix as an immutable numpy array.

        Returns:
            3x3 numpy array with writeable=False.
        """
        arr = np.frombuffer(self._data, dtype=np.float64).reshape(3, 3)
        arr.flags.writeable = False
        return arr

    @property
    def determinant(self) -> float:
        """Compute the determinant of the matrix."""
        return float(np.linalg.det(self.to_array()))

    @property
    def condition_number(self) -> float:
        """Compute the condition number of the matrix.

        The condition number indicates numerical stability.
        Lower values are better; high values indicate potential numerical issues.
        """
        return float(np.linalg.cond(self.to_array()))

    def inverse(self) -> Matrix3x3:
        """Compute the inverse matrix.

        Returns:
            New Matrix3x3 that is the inverse of this matrix.

        Raises:
            ValueError: If matrix is singular (determinant near zero).
        """
        arr = self.to_array()
        det = np.linalg.det(arr)

        if abs(det) < 1e-10:
            raise ValueError(f"Matrix is singular (det={det:.2e}). Cannot compute inverse.")

        inv_arr = np.linalg.inv(arr)
        return Matrix3x3.create(inv_arr)

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        """Matrix multiplication with a numpy array.

        Args:
            other: Array to multiply (typically a vector or another matrix).

        Returns:
            Result of matrix multiplication.
        """
        return self.to_array() @ other

    def to_list(self) -> list[list[float]]:
        """Convert to nested list for serialization."""
        return self.to_array().tolist()

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
