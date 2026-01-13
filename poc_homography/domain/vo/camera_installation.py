"""Camera installation value object for fixed installation parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from poc_homography.domain.enums import TiltConvention
from poc_homography.types import Meters, Unitless

if TYPE_CHECKING:
    from poc_homography.domain.vo.base_orientation import BaseOrientation
    from poc_homography.domain.vo.map_point import MapPoint


@dataclass(frozen=True)
class CameraInstallation:
    """Fixed installation parameters for a camera.

    These parameters are determined during installation and calibration,
    and don't change during normal operation. They define where the camera
    is physically located and how it's oriented.

    Attributes:
        map_id: ID of the map this camera is installed on.
        position: Position of the camera on the map (pixel coordinates).
        height: Height of the camera above the ground plane in meters.
        base_orientation: Camera orientation at PTZ home position.
        tilt_convention: How to interpret tilt sign from PTZ reports.
        k1: Radial distortion coefficient (1st order).
        k2: Radial distortion coefficient (2nd order).
        p1: Tangential distortion coefficient.
        p2: Tangential distortion coefficient.
    """

    map_id: str
    position: MapPoint
    height: Meters
    base_orientation: BaseOrientation
    tilt_convention: TiltConvention = TiltConvention.POSITIVE_DOWN

    # Lens distortion coefficients
    k1: Unitless = Unitless(0.0)  # noqa: RUF009
    k2: Unitless = Unitless(0.0)  # noqa: RUF009
    p1: Unitless = Unitless(0.0)  # noqa: RUF009
    p2: Unitless = Unitless(0.0)  # noqa: RUF009

    @property
    def has_distortion(self) -> bool:
        """True if any distortion coefficient is non-zero."""
        return self.k1 != 0.0 or self.k2 != 0.0 or self.p1 != 0.0 or self.p2 != 0.0
