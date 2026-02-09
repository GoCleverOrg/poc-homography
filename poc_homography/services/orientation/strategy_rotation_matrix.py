"""Rotation matrix orientation strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poc_homography.domain.enums import TiltConvention
    from poc_homography.domain.vo import Orientation, PTZState


class StrategyRotationMatrix:
    """Rotation matrix strategy for proper SO(3) orientation composition.

    This strategy computes final orientation by composing rotation matrices.
    It properly handles large angles and non-zero roll angles by:
    1. Converting base orientation to rotation matrix
    2. Converting PTZ state to rotation matrix
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
        return base.compose(ptz.to_orientation(tilt_convention))
