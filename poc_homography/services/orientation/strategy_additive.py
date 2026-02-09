"""Additive orientation strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poc_homography.domain.enums import TiltConvention
    from poc_homography.domain.vo import Orientation, PTZState


class StrategyOrientationAdditive:
    """Simple additive strategy for orientation composition.

    This strategy computes final orientation by directly adding PTZ angles
    to base orientation angles. This is valid for small angles and when
    roll is negligible.
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
        return base + ptz.to_orientation(tilt_convention)
