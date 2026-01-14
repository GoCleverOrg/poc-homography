"""Orientation value object for camera yaw/pitch/roll angles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from poc_homography.types import Degrees

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

    Attributes:
        yaw: Azimuth angle in degrees.
        pitch: Elevation angle in degrees.
        roll: Roll angle in degrees.
    """

    yaw: Degrees
    pitch: Degrees
    roll: Degrees = Degrees(0.0)  # noqa: RUF009

    @property
    def rotation_matrix(self) -> NDArray[np.float64]:
        """Compute the 3x3 rotation matrix from yaw, pitch, roll.

        Uses the ZYX (yaw-pitch-roll) Euler angle convention:
        R = Rz(yaw) * Ry(pitch) * Rx(roll)

        Returns:
            3x3 rotation matrix R such that p_world = R @ p_camera
        """
        yaw_rad = math.radians(float(self.yaw))
        pitch_rad = math.radians(float(self.pitch))
        roll_rad = math.radians(float(self.roll))

        # Rotation around Z (yaw)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        Rz = np.array(
            [
                [cy, -sy, 0],
                [sy, cy, 0],
                [0, 0, 1],
            ]
        )

        # Rotation around Y (pitch)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
        Ry = np.array(
            [
                [cp, 0, sp],
                [0, 1, 0],
                [-sp, 0, cp],
            ]
        )

        # Rotation around X (roll)
        cr, sr = math.cos(roll_rad), math.sin(roll_rad)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, cr, -sr],
                [0, sr, cr],
            ]
        )

        return Rz @ Ry @ Rx

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
        return cls(
            yaw=Degrees(data["yaw"]),
            pitch=Degrees(data["pitch"]),
            roll=Degrees(data.get("roll", 0.0)),
        )
