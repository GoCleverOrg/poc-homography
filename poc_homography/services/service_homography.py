"""Homography service for computing camera-to-world projections."""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.services.homography.strategy_intrinsic_extrinsic import (
    StrategyIntrinsicExtrinsic,
)
from poc_homography.services.orientation import ServiceOrientation

if TYPE_CHECKING:
    from poc_homography.domain.entities import CameraCalibration, CameraConfig, Map
    from poc_homography.domain.vo import Homography, PTZState
    from poc_homography.services.homography.strategy import StrategyHomography
    from poc_homography.services.orientation.strategy import StrategyOrientation


class ServiceHomography:
    """Service for computing camera homography matrices.

    This service computes the homography matrix H that maps between:
    - Image pixels [u, v, 1] and world ground plane coordinates [X, Y, 1]

    The service orchestrates:
    1. Computing final orientation from base orientation + PTZ state
    2. Computing homography from camera intrinsics, position, and orientation

    Both steps use pluggable strategies for flexibility.

    Example:
        >>> from poc_homography.services.homography import (
        ...     ServiceHomography,
        ...     StrategyIntrinsicExtrinsic,
        ... )
        >>> service = ServiceHomography()
        >>> homography = service.compute(
        ...     config=camera_config,
        ...     calibration=camera_calibration,
        ...     ptz_state=ptz_state,
        ...     map_entity=map_entity,
        ... )
        >>> u, v = homography.world_to_image(5.0, 10.0)
    """

    def __init__(
        self,
        homography_strategy: StrategyHomography | None = None,
        orientation_strategy: StrategyOrientation | None = None,
    ) -> None:
        """Initialize the homography service.

        Args:
            homography_strategy: Strategy for homography computation.
                Defaults to StrategyIntrinsicExtrinsic.
            orientation_strategy: Strategy for orientation computation.
                Defaults to StrategyOrientationAdditive (via ServiceOrientation default).
        """
        self._homography_strategy = homography_strategy or StrategyIntrinsicExtrinsic()
        self._orientation_service = ServiceOrientation(orientation_strategy)

    def compute(
        self,
        config: CameraConfig,
        calibration: CameraCalibration,
        ptz_state: PTZState,
        map_entity: Map,
    ) -> Homography:
        """Compute homography from camera configuration and state.

        Args:
            config: Camera configuration (spec, map_id, name, etc.).
            calibration: Camera calibration data (position, height, orientation, distortion).
            ptz_state: Current PTZ state (pan, tilt, zoom).
            map_entity: Map entity with GeoTiff metadata for coordinate transforms.

        Returns:
            Homography VO with projection methods (world_to_image, image_to_world).
        """
        # Step 1: Compute final orientation from base + PTZ
        final_orientation = self._orientation_service.compute_orientation(
            base_orientation=calibration.base_orientation,
            ptz_state=ptz_state,
            tilt_convention=config.spec.tilt_convention,
        )

        # Step 2: Compute homography using the strategy
        return self._homography_strategy.compute(
            config=config,
            calibration=calibration,
            ptz_state=ptz_state,
            map_entity=map_entity,
            final_orientation=final_orientation,
        )
