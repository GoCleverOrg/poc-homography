"""Additive orientation strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.domain.vo import Orientation
from poc_homography.types import Degrees

if TYPE_CHECKING:
    from poc_homography.domain.enums import TiltConvention
    from poc_homography.domain.vo import PTZState


class AdditiveOrientationStrategy:
    """Simple additive strategy for orientation composition.

    This strategy computes final orientation by directly adding PTZ angles
    to base orientation angles. This is valid for small angles and when
    roll is negligible.

    Formula:
        final_yaw = base_yaw + ptz_pan
        final_pitch = base_pitch + ptz_tilt * tilt_convention.sign
        final_roll = base_roll (PTZ doesn't change roll)
    """

    def compute(
        self,
        base: Orientation,
        ptz: PTZState,
        tilt_convention: TiltConvention,
    ) -> Orientation:
        """Compute final orientation using simple angle addition.

        Args:
            base: Base camera orientation at PTZ home position.
            ptz: Current PTZ state (pan, tilt, zoom).
            tilt_convention: Sign convention for tilt angles.

        Returns:
            Final computed orientation with combined angles.
        """
        final_yaw = Degrees(float(base.yaw) + ptz.pan_raw)
        final_pitch = Degrees(float(base.pitch) + ptz.tilt_deg * tilt_convention.sign)
        final_roll = base.roll  # PTZ doesn't change roll

        return Orientation(yaw=final_yaw, pitch=final_pitch, roll=final_roll)
