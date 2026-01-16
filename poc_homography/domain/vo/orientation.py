"""Orientation value object for camera yaw/pitch/roll angles."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from poc_homography.domain.vo.rotation import Rotation
from poc_homography.types import Degrees

_PRIVATE_SENTINEL = object()


@dataclass(frozen=True)
class Orientation:
    """Camera orientation in 3D space (yaw, pitch, roll).

    Used for both base installation orientation and computed final orientation.
    The context determines the meaning:
    - Base orientation: Camera direction when PTZ reports (0, 0)
    - Final orientation: Actual camera direction after applying PTZ state

    Coordinate convention:
        - Yaw: Azimuth from map-north, clockwise positive (0=North, 90=East)
        - Pitch: Elevation angle (sign convention depends on TiltConvention)
        - Roll: Rotation around optical axis, clockwise when looking forward

    Composition:
        - Use `+` for simple angle addition (fast, valid for small angles)
        - Use `compose()` for rotation matrix composition (accurate for large angles)

    Use the `create()` factory method to construct instances.
    Direct constructor access is reserved for internal use.

    Attributes:
        yaw: Azimuth angle in degrees.
        pitch: Elevation angle in degrees.
        roll: Roll angle in degrees.
    """

    yaw: Degrees
    pitch: Degrees
    roll: Degrees
    _rotation: Rotation = field(repr=False)
    _sentinel: object = field(default=None, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        """Verify construction was via create() factory."""
        if self._sentinel is not _PRIVATE_SENTINEL:
            raise TypeError(
                "Orientation cannot be instantiated directly. Use Orientation.create() instead."
            )

    @classmethod
    def create(
        cls,
        yaw: Degrees,
        pitch: Degrees,
        roll: Degrees = Degrees(0.0),
    ) -> Orientation:
        """Create Orientation with validation.

        Args:
            yaw: Azimuth angle in degrees.
            pitch: Elevation angle in degrees.
            roll: Roll angle in degrees (default 0.0).

        Returns:
            New Orientation instance.

        Raises:
            ValueError: If angles are invalid (non-finite or pitch out of range).
        """
        # Validate finite values
        if not math.isfinite(float(yaw)):
            raise ValueError(f"yaw must be finite, got {yaw}")
        if not math.isfinite(float(pitch)):
            raise ValueError(f"pitch must be finite, got {pitch}")
        if not math.isfinite(float(roll)):
            raise ValueError(f"roll must be finite, got {roll}")

        # Validate pitch range (cannot look beyond vertical)
        if not -90.0 <= float(pitch) <= 90.0:
            raise ValueError(f"pitch must be in range [-90, 90], got {pitch}")

        # Compute rotation matrix
        rotation = Rotation.from_euler(yaw, pitch, roll)

        return cls(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            _rotation=rotation,
            _sentinel=_PRIVATE_SENTINEL,
        )

    @classmethod
    def from_rotation(cls, rotation: Rotation) -> Orientation:
        """Create Orientation by extracting Euler angles from a Rotation.

        Extracts ZYX Euler angles (yaw, pitch, roll) from the rotation matrix.

        Args:
            rotation: Rotation to extract angles from.

        Returns:
            New Orientation instance.
        """
        R = rotation.to_matrix().to_array()

        # Handle gimbal lock case
        if abs(R[2, 0]) >= 1.0 - 1e-6:
            # Gimbal lock: pitch is +/- 90 degrees
            yaw = math.atan2(-R[0, 1], R[0, 2])
            pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
            roll = 0.0
        else:
            yaw = math.atan2(R[1, 0], R[0, 0])
            pitch = -math.asin(R[2, 0])
            roll = math.atan2(R[2, 1], R[2, 2])

        return cls(
            yaw=Degrees(math.degrees(yaw)),
            pitch=Degrees(math.degrees(pitch)),
            roll=Degrees(math.degrees(roll)),
            _rotation=rotation,
            _sentinel=_PRIVATE_SENTINEL,
        )

    def __add__(self, other: Orientation) -> Orientation:
        """Simple additive composition (valid for small angles).

        Adds the angles component-wise. This is fast but only accurate
        for small angle adjustments. For large angles, use `compose()`.

        Args:
            other: Orientation to add.

        Returns:
            New Orientation with summed angles.
        """
        return Orientation.create(
            yaw=Degrees(self.yaw + other.yaw),
            pitch=Degrees(self.pitch + other.pitch),
            roll=Degrees(self.roll + other.roll),
        )

    def compose(self, other: Orientation) -> Orientation:
        """Rotation matrix composition (accurate for large angles).

        Composes the rotations using proper SO(3) matrix multiplication.
        The result is: self @ other (self applied after other).

        This is more accurate than `+` for large angles but more expensive.

        Args:
            other: Orientation to compose with.

        Returns:
            New Orientation from composed rotation.
        """
        R_composed = self._rotation.compose(other._rotation)
        return Orientation.from_rotation(R_composed)

    def to_rotation(self) -> Rotation:
        """Get the rotation matrix.

        Returns:
            The rotation as a Rotation value object.
        """
        return self._rotation

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "yaw": float(self.yaw),
            "pitch": float(self.pitch),
            "roll": float(self.roll),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Orientation:
        """Create Orientation from dictionary."""
        return cls.create(
            yaw=Degrees(data["yaw"]),
            pitch=Degrees(data["pitch"]),
            roll=Degrees(data.get("roll", 0.0)),
        )
