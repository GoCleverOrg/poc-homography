"""Camera entity representing a PTZ camera with installation and state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poc_homography.domain.enums import CameraSpec
    from poc_homography.domain.vo.camera_installation import CameraInstallation
    from poc_homography.domain.vo.ptz_state import PTZState


@dataclass
class Camera:
    """A PTZ camera with installation parameters and current state.

    This entity represents a physical PTZ camera, including:
    - Fixed installation parameters (position, orientation, distortion)
    - Hardware/model-specific parameters (from CameraSpec enum)
    - Current PTZ state (pan, tilt, zoom - mutable)

    The camera ID is computed from map_id and name, following the same
    pattern as GroundControlPoint (format: "map_id/name").

    Attributes:
        map_id: ID of the map this camera is installed on.
        name: Human-readable name for the camera.
        installation: Fixed installation parameters.
        spec: Camera hardware specification (enum).
        ptz_state: Current PTZ state (mutable).
        ip_address: IP address for camera control (optional).
    """

    map_id: str
    name: str
    installation: CameraInstallation
    spec: CameraSpec
    ptz_state: PTZState
    ip_address: str | None = None

    @property
    def id(self) -> str:
        """Computed unique identifier (format: map_id/name)."""
        return f"{self.map_id}/{self.name}"

    def update_ptz_state(self, new_state: PTZState) -> None:
        """Update the current PTZ state.

        Args:
            new_state: New PTZ state from the camera hardware.
        """
        # Note: dataclass is not frozen, so we can update the field directly
        object.__setattr__(self, "ptz_state", new_state)
