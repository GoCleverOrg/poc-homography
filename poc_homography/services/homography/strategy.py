"""Homography computation strategy protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from poc_homography.camera_parameters import CameraGeometryResult
    from poc_homography.domain.entities import CameraCalibration, CameraConfig, Map
    from poc_homography.domain.vo import Orientation, PTZState


class StrategyHomography(Protocol):
    """Strategy protocol for computing homography from camera state.

    Implementations define how to compute the homography matrix that maps
    between image pixels and world ground plane coordinates.
    """

    def compute(
        self,
        config: CameraConfig,
        calibration: CameraCalibration,
        ptz_state: PTZState,
        map_entity: Map,
        final_orientation: Orientation,
    ) -> CameraGeometryResult:
        """Compute homography from camera configuration and state.

        Args:
            config: Camera configuration (spec, map_id, etc.).
            calibration: Camera calibration data (position, height, distortion).
            ptz_state: Current PTZ state (pan, tilt, zoom).
            map_entity: Map entity with GeoTiff metadata.
            final_orientation: Pre-computed final orientation (from ServiceOrientation).

        Returns:
            CameraGeometryResult containing homography matrix and validation state.
        """
        ...
