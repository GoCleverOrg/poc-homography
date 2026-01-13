"""Orientation service for computing final camera orientation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.services.orientation.additive_strategy import AdditiveOrientationStrategy

if TYPE_CHECKING:
    from poc_homography.domain.enums import TiltConvention
    from poc_homography.domain.vo import Orientation, PTZState
    from poc_homography.services.orientation.strategy import OrientationStrategy


class ServiceOrientation:
    """Service for computing final camera orientation.

    This service combines the camera's base orientation (from calibration)
    with the current PTZ state to compute the final camera orientation
    in world coordinates.

    The service uses a pluggable strategy pattern to allow different
    composition methods (additive vs rotation matrix).

    Example:
        >>> from poc_homography.services.orientation import (
        ...     OrientationService,
        ...     AdditiveOrientationStrategy,
        ... )
        >>> service = OrientationService(AdditiveOrientationStrategy())
        >>> final = service.compute_orientation(
        ...     base_orientation=calibration.base_orientation,
        ...     ptz_state=ptz_state,
        ...     tilt_convention=config.spec.tilt_convention,
        ... )
    """

    def __init__(self, strategy: OrientationStrategy | None = None) -> None:
        """Initialize the orientation service.

        Args:
            strategy: Strategy to use for orientation computation.
                Defaults to AdditiveOrientationStrategy.
        """
        self._strategy = strategy or AdditiveOrientationStrategy()

    def compute_orientation(
        self,
        base_orientation: Orientation,
        ptz_state: PTZState,
        tilt_convention: TiltConvention,
    ) -> Orientation:
        """Compute final camera orientation from base orientation and PTZ state.

        Args:
            base_orientation: Camera orientation at PTZ home position.
            ptz_state: Current PTZ state (pan, tilt, zoom).
            tilt_convention: Sign convention for tilt angles.

        Returns:
            Final computed orientation in world coordinates.
        """
        return self._strategy.compute(base_orientation, ptz_state, tilt_convention)
