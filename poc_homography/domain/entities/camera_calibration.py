"""Camera calibration entity for calibration data refined during the calibration process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poc_homography.domain.vo.lens_distortion import LensDistortion
    from poc_homography.domain.vo.orientation import Orientation
    from poc_homography.domain.vo.pixel_point import PixelPoint
    from poc_homography.types import Meters


@dataclass(frozen=True)
class CameraCalibration:
    """Camera calibration data refined during the calibration process.

    This entity contains data that changes during calibration workflows.
    It is separate from CameraConfig (which rarely changes) and PTZState
    (which changes constantly and is never persisted).

    The camera_id field references the CameraConfig.id of the camera
    this calibration belongs to.

    Attributes:
        camera_id: References CameraConfig.id (format: "map_id/name").
        position: Position of the camera on the map (pixel coordinates).
        height: Height of the camera above the ground plane in meters.
        base_orientation: Camera orientation at PTZ home position (when PTZ is at 0,0).
        distortion: Lens distortion coefficients (calibrated per-camera).
    """

    camera_id: str
    position: PixelPoint
    height: Meters
    base_orientation: Orientation
    distortion: LensDistortion
