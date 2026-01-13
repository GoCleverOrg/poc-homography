"""Rotation matrix orientation strategy."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from poc_homography.domain.vo import Orientation
from poc_homography.types import Degrees

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from poc_homography.domain.enums import TiltConvention
    from poc_homography.domain.vo import PTZState


class RotationMatrixStrategy:
    """Rotation matrix strategy for proper SO(3) orientation composition.

    This strategy computes final orientation by composing rotation matrices.
    It properly handles large angles and non-zero roll angles by:
    1. Converting base orientation to rotation matrix
    2. Converting PTZ state to incremental rotation matrix
    3. Composing the matrices: R_final = R_ptz @ R_base
    4. Extracting Euler angles from the final rotation matrix

    This is more accurate than simple addition for large angles or when
    roll is significant.
    """

    def compute(
        self,
        base: Orientation,
        ptz: PTZState,
        tilt_convention: TiltConvention,
    ) -> Orientation:
        """Compute final orientation using rotation matrix composition.

        Args:
            base: Base camera orientation at PTZ home position.
            ptz: Current PTZ state (pan, tilt, zoom).
            tilt_convention: Sign convention for tilt angles.

        Returns:
            Final computed orientation extracted from composed rotation matrix.
        """
        # Get base rotation matrix
        R_base = base.rotation_matrix

        # Build PTZ incremental rotation matrix
        R_ptz = self._ptz_to_rotation_matrix(ptz, tilt_convention)

        # Compose: final = PTZ applied to base
        R_final = R_ptz @ R_base

        # Extract Euler angles from final rotation matrix
        yaw, pitch, roll = self._rotation_matrix_to_euler(R_final)

        return Orientation(yaw=yaw, pitch=pitch, roll=roll)

    @staticmethod
    def _ptz_to_rotation_matrix(
        ptz: PTZState,
        tilt_convention: TiltConvention,
    ) -> NDArray[np.float64]:
        """Convert PTZ state to incremental rotation matrix.

        Args:
            ptz: Current PTZ state.
            tilt_convention: Sign convention for tilt angles.

        Returns:
            3x3 rotation matrix representing PTZ rotation.
        """
        pan_rad = math.radians(ptz.pan_raw)
        tilt_rad = math.radians(ptz.tilt_deg * tilt_convention.sign)

        # Pan rotation around Z-axis (yaw)
        cy, sy = math.cos(pan_rad), math.sin(pan_rad)
        Rz = np.array(
            [
                [cy, -sy, 0],
                [sy, cy, 0],
                [0, 0, 1],
            ]
        )

        # Tilt rotation around Y-axis (pitch)
        cp, sp = math.cos(tilt_rad), math.sin(tilt_rad)
        Ry = np.array(
            [
                [cp, 0, sp],
                [0, 1, 0],
                [-sp, 0, cp],
            ]
        )

        # PTZ rotation: tilt then pan (in extrinsic frame)
        return Rz @ Ry

    @staticmethod
    def _rotation_matrix_to_euler(R: NDArray[np.float64]) -> tuple[Degrees, Degrees, Degrees]:
        """Extract ZYX Euler angles (yaw, pitch, roll) from rotation matrix.

        Args:
            R: 3x3 rotation matrix.

        Returns:
            Tuple of (yaw, pitch, roll) in degrees.
        """
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

        return (
            Degrees(math.degrees(yaw)),
            Degrees(math.degrees(pitch)),
            Degrees(math.degrees(roll)),
        )
