"""Camera entity representing a PTZ camera with installation and state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.vo.camera_installation import CameraInstallation
    from poc_homography.domain.vo.camera_intrinsics import CameraIntrinsics
    from poc_homography.domain.vo.ptz_state import PTZState


@dataclass
class Camera:
    """A PTZ camera with installation parameters and current state.

    This entity represents a physical PTZ camera, including:
    - Fixed installation parameters (position, orientation, distortion)
    - Fixed intrinsic parameters (sensor, lens characteristics)
    - Current PTZ state (pan, tilt, zoom - mutable)

    Attributes:
        id: Unique identifier for the camera (e.g., "valte").
        name: Human-readable name for the camera.
        installation: Fixed installation parameters.
        intrinsics: Camera intrinsic parameters (sensor/lens).
        ptz_state: Current PTZ state (mutable).
        model: Camera model string (optional).
        ip_address: IP address for camera control (optional).
        calibration_table: Zoom-dependent calibration data (optional).
    """

    id: str
    name: str
    installation: CameraInstallation
    intrinsics: CameraIntrinsics
    ptz_state: PTZState
    model: str | None = None
    ip_address: str | None = None
    calibration_table: dict[float, dict[str, Any]] | None = field(default=None)

    @property
    def map_id(self) -> str:
        """ID of the map this camera is installed on."""
        return self.installation.map_id

    def update_ptz_state(self, new_state: PTZState) -> None:
        """Update the current PTZ state.

        Args:
            new_state: New PTZ state from the camera hardware.
        """
        # Note: dataclass is not frozen, so we can update the field directly
        object.__setattr__(self, "ptz_state", new_state)
