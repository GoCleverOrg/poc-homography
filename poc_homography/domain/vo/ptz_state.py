from dataclasses import dataclass
from typing import Any

from poc_homography.types import Degrees, Unitless


@dataclass(frozen=True)
class PTZState:
    """Camera state when a calibration frame was captured.

    Attributes:
        pan_raw: Pan angle in degrees (from PTZ API, before offset).
        tilt_deg: Tilt angle in degrees.
        zoom: Zoom level (1.0 = no zoom).
    """

    pan_raw: Degrees
    tilt_deg: Degrees
    zoom: Unitless

    def to_dict(self) -> dict[str, Any]:
        """Convert CameraPose to a dictionary for JSON serialization.

        Returns:
            Dictionary with camera, pan_raw, tilt_deg, and zoom keys.
        """
        return {
            "pan_raw": self.pan_raw,
            "tilt_deg": self.tilt_deg,
            "zoom": self.zoom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PTZState":
        """Create CameraPose from a dictionary.

        Args:
            data: Dictionary with camera, pan_raw, tilt_deg, and zoom keys.

        Returns:
            New CameraPose instance.

        Raises:
            KeyError: If required keys are missing from data.
            ValueError: If data types are invalid.
        """
        return cls(
            pan_raw=Degrees(float(data["pan_raw"])),
            tilt_deg=Degrees(float(data["tilt_deg"])),
            zoom=Unitless(float(data["zoom"])),
        )
