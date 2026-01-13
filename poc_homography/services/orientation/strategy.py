"""Orientation strategy protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from poc_homography.domain.enums import TiltConvention
    from poc_homography.domain.vo import Orientation, PTZState


class OrientationStrategy(Protocol):
    """Strategy protocol for computing final orientation from base + PTZ state."""

    def compute(
        self,
        base: Orientation,
        ptz: PTZState,
        tilt_convention: TiltConvention,
    ) -> Orientation:
        """Compute final orientation by combining base orientation with PTZ state.

        Args:
            base: Base camera orientation at PTZ home position.
            ptz: Current PTZ state (pan, tilt, zoom).
            tilt_convention: Sign convention for tilt angles.

        Returns:
            Final computed orientation.
        """
        ...
