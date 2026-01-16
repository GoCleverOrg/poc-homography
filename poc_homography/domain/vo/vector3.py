"""Immutable 3-element vector value object for homogeneous coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_PRIVATE_SENTINEL = object()


@dataclass(frozen=True)
class Vector3:
    """Immutable 3-element vector value object.

    Used for homogeneous coordinates in projective geometry.
    A 2D point (x, y) is represented as [x, y, 1] in homogeneous form.

    Use the `create()` factory method to construct instances.
    Direct constructor access is reserved for internal use.
    """

    _data: bytes = field(repr=False)
    _sentinel: object = field(default=None, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        """Verify construction was via create() factory."""
        if self._sentinel is not _PRIVATE_SENTINEL:
            raise TypeError(
                "Vector3 cannot be instantiated directly. Use Vector3.create() instead."
            )

    @classmethod
    def create(cls, x: float, y: float, z: float) -> Vector3:
        """Create a Vector3 from three components.

        Args:
            x: First component.
            y: Second component.
            z: Third component.

        Returns:
            New Vector3 instance.

        Raises:
            ValueError: If any component is NaN or Infinity.
        """
        arr = np.array([x, y, z], dtype=np.float64)

        if not np.all(np.isfinite(arr)):
            raise ValueError("Vector contains NaN or Infinity values")

        return cls(_data=arr.tobytes(), _sentinel=_PRIVATE_SENTINEL)

    @classmethod
    def from_array(cls, array: np.ndarray) -> Vector3:
        """Create a Vector3 from a numpy array.

        Args:
            array: A 1D numpy array with 3 elements.

        Returns:
            New Vector3 instance.

        Raises:
            ValueError: If array doesn't have exactly 3 elements or contains NaN/Infinity.
        """
        arr = np.asarray(array, dtype=np.float64).flatten()

        if arr.shape != (3,):
            raise ValueError(f"Array must have 3 elements, got shape {array.shape}")

        if not np.all(np.isfinite(arr)):
            raise ValueError("Vector contains NaN or Infinity values")

        return cls(_data=arr.tobytes(), _sentinel=_PRIVATE_SENTINEL)

    def to_array(self) -> np.ndarray:
        """Get the vector as an immutable numpy array.

        Returns:
            1D numpy array with 3 elements, writeable=False.
        """
        arr = np.frombuffer(self._data, dtype=np.float64).copy()
        arr.flags.writeable = False
        return arr

    @property
    def x(self) -> float:
        """First component of the vector."""
        return float(np.frombuffer(self._data, dtype=np.float64)[0])

    @property
    def y(self) -> float:
        """Second component of the vector."""
        return float(np.frombuffer(self._data, dtype=np.float64)[1])

    @property
    def z(self) -> float:
        """Third component of the vector."""
        return float(np.frombuffer(self._data, dtype=np.float64)[2])

    def normalized(self) -> Vector3:
        """Return a new vector normalized by the z component.

        Used to convert from homogeneous to Cartesian coordinates.

        Returns:
            New Vector3 with z=1 (or z=0 if input z was 0).

        Raises:
            ValueError: If z component is near zero (point at infinity).
        """
        if abs(self.z) < 1e-10:
            raise ValueError("Cannot normalize: z component near zero (point at infinity)")

        return Vector3.create(self.x / self.z, self.y / self.z, 1.0)

    def __hash__(self) -> int:
        """Compute hash for use in sets and as dict keys."""
        return hash(self._data)
