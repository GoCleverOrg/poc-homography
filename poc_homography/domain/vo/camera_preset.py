"""Camera PTZ preset value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.domain.vo.ptz_state import PTZState


@dataclass(frozen=True)
class CameraPreset:
    """A named PTZ preset position.

    The preset's ``AbsoluteHigh`` raw values are scaled x10 by the hardware.
    Conversion from raw to degrees/zoom is the responsibility of the
    infrastructure adapter (it owns the unit math); this value object stores the
    already-converted :class:`PTZState`. There is therefore no ``from_element``
    parser here, keeping the domain free of the divide-by-10 conversion.

    Attributes:
        preset_id: Preset identifier.
        name: Human-readable preset name.
        ptz: PTZ state (degrees/zoom) for the preset.
    """

    preset_id: int
    name: str
    ptz: PTZState

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "preset_id": self.preset_id,
            "name": self.name,
            "ptz": self.ptz.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraPreset:
        """Create :class:`CameraPreset` from a dictionary."""
        return cls(
            preset_id=int(data["preset_id"]),
            name=data["name"],
            ptz=PTZState.from_dict(data["ptz"]),
        )
