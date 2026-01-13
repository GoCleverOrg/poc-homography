"""Final orientation value object for computed camera orientation."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from poc_homography.types import Degrees


@dataclass(frozen=True)
class FinalOrientation:
    """Computed camera orientation in world coordinates.

    This is the result of combining the base installation orientation
    with the current PTZ state. It represents the actual direction
    the camera is pointing in world coordinates.

    Coordinate convention:
        - Yaw: Azimuth from map-north, clockwise positive (0=North, 90=East)
        - Pitch: Elevation angle, positive = looking down (after convention applied)
        - Roll: Rotation around optical axis

    Attributes:
        yaw: Final azimuth angle in degrees.
        pitch: Final elevation angle in degrees.
        roll: Final roll angle in degrees.
    """

    yaw: Degrees
    pitch: Degrees
    roll: Degrees

    @property
    def rotation_matrix(self) -> NDArray[np.float64]:
        """Compute the 3x3 rotation matrix from yaw, pitch, roll.

        Uses the ZYX (yaw-pitch-roll) Euler angle convention:
        R = Rz(yaw) * Ry(pitch) * Rx(roll)

        Returns:
            3x3 rotation matrix R such that p_world = R @ p_camera
        """
        # Convert to radians
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
