from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from poc_homography.types import Degrees, Unitless

if TYPE_CHECKING:
    from poc_homography.domain.enums import TiltConvention
    from poc_homography.domain.vo.orientation import Orientation


@dataclass(frozen=True)
class PTZState:
    """Camera PTZ state (pan, tilt, zoom).

    Represents raw hardware values from a PTZ camera. Use `to_orientation()`
    to convert to world-referenced angles.

    Attributes:
        pan_raw: Pan angle in degrees (from PTZ API, before offset).
        tilt_deg: Tilt angle in degrees.
        zoom: Zoom level (1.0 = no zoom).
    """

    pan_raw: Degrees
    tilt_deg: Degrees
    zoom: Unitless

    def to_orientation(self, tilt_convention: TiltConvention) -> Orientation:
        """Convert PTZ state to orientation delta.

        Interprets the raw PTZ values according to the tilt convention
        and returns an Orientation that can be composed with a base orientation.

        Args:
            tilt_convention: Sign convention for tilt angles.

        Returns:
            Orientation representing the PTZ adjustment.
        """
        from poc_homography.domain.vo.orientation import Orientation

        return Orientation.create(
            yaw=Degrees(self.pan_raw),
            pitch=Degrees(self.tilt_deg * tilt_convention.sign),
            roll=Degrees(0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pan_raw": self.pan_raw,
            "tilt_deg": self.tilt_deg,
            "zoom": self.zoom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PTZState:
        """Create PTZState from dictionary."""
        return cls(
            pan_raw=Degrees(float(data["pan_raw"])),
            tilt_deg=Degrees(float(data["tilt_deg"])),
            zoom=Unitless(float(data["zoom"])),
        )
