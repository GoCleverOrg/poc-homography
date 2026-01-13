"""Camera installation value object for fixed installation parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from poc_homography.types import Meters

if TYPE_CHECKING:
    from poc_homography.domain.vo.orientation import Orientation
    from poc_homography.domain.vo.pixel_point import PixelPoint


@dataclass(frozen=True)
class CameraInstallation:
    """Fixed installation parameters for a camera.

    These parameters are determined during installation and calibration,
    and don't change during normal operation. They define where the camera
    is physically located and how it's oriented.

    Attributes:
        position: Position of the camera on the map (pixel coordinates).
        height: Height of the camera above the ground plane in meters.
        base_orientation: Camera orientation at PTZ home position (when PTZ is at 0,0).
    """

    position: PixelPoint
    height: Meters
    base_orientation: Orientation
