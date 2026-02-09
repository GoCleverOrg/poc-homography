"""Camera configuration entity for static camera registration data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.enums import CameraSpec
    from poc_homography.domain.vo.credential import Credential


@dataclass(frozen=True)
class CameraConfig:
    """Camera configuration set once during registration.

    This entity contains data that rarely changes and is set when the camera
    is first registered in the system. It does NOT include calibration data
    (position, orientation, distortion) which has a different lifecycle.

    Attributes:
        id: Unique identifier for this camera configuration.
        map_id: ID of the map this camera is associated with.
        name: Human-readable name for the camera.
        spec: Camera hardware specification (enum).
        credential: Authentication credentials for camera access.
        ip_address: IP address for camera control (optional).
    """

    id: str
    map_id: str
    name: str
    spec: CameraSpec
    credential: Credential
    ip_address: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data: dict[str, Any] = {
            "id": self.id,
            "map_id": self.map_id,
            "name": self.name,
            "spec": self.spec.name,
            "credential": self.credential.to_dict(),
        }
        if self.ip_address:
            data["ip_address"] = self.ip_address
        return data

    def rtsp_url(self, stream_type: str = "main") -> str:
        """Build RTSP URL for this camera.

        Args:
            stream_type: "main" for high quality or "sub" for low quality

        Returns:
            Full RTSP URL for the camera stream

        Raises:
            ValueError: If camera has no IP address configured
        """
        if not self.ip_address:
            raise ValueError(f"Camera '{self.name}' has no IP address configured")
        return self.spec.rtsp_url(self.ip_address, self.credential, stream_type)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraConfig:
        """Create CameraConfig from dictionary."""
        from poc_homography.domain.enums import CameraSpec
        from poc_homography.domain.vo.credential import Credential

        return cls(
            id=data["id"],
            map_id=data["map_id"],
            name=data["name"],
            spec=CameraSpec[data["spec"]],
            credential=Credential.from_dict(data["credential"]),
            ip_address=data.get("ip_address"),
        )
