"""Camera configuration entity for static camera registration data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poc_homography.domain.enums import CameraSpec
    from poc_homography.domain.vo.credential import Credential


@dataclass(frozen=True)
class CameraConfig:
    """Camera configuration set once during registration.

    This entity contains data that rarely changes and is set when the camera
    is first registered in the system. It does NOT include calibration data
    (position, orientation, distortion) which has a different lifecycle.

    The camera ID is computed from map_id and name, following the same
    pattern as GroundControlPoint (format: "map_id/name").

    Attributes:
        map_id: ID of the map this camera is associated with.
        name: Human-readable name for the camera.
        spec: Camera hardware specification (enum).
        credential: Authentication credentials for camera access.
        ip_address: IP address for camera control (optional).
    """

    map_id: str
    name: str
    spec: CameraSpec
    credential: Credential
    ip_address: str | None = None

    @property
    def id(self) -> str:
        """Computed unique identifier (format: map_id/name)."""
        return f"{self.map_id}/{self.name}"
