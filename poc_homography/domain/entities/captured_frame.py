"""CapturedFrame entity representing a photo with camera pose/PTZ state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.vo.ptz_state import PTZState


@dataclass(frozen=True)
class CapturedFrame:
    """A captured camera frame with associated PTZ state.

    This entity represents a photo captured from a PTZ camera at a specific
    moment in time. It stores the image path and the PTZ state when captured.

    ID Format: {map_id}/{camera_name}/{timestamp}
    - timestamp is ISO format with colons replaced by dashes for filesystem safety
    - Example: "valte/Valte/2024-01-15T10-30-45"

    File Storage:
    - Folder: data/frames/{map_id}/{camera_name}/
    - YAML: {timestamp}.yaml
    - Image: Referenced by relative path in YAML (stored alongside)

    Attributes:
        id: Unique identifier (map_id/camera_name/timestamp).
        map_id: ID of the map this frame belongs to.
        camera_name: Name of the camera that captured this frame.
        timestamp: When the frame was captured.
        ptz_state: PTZ state at capture time.
        image_path: Relative path to the image file.
    """

    id: str
    map_id: str
    camera_name: str
    timestamp: datetime
    ptz_state: PTZState
    image_path: Path

    @classmethod
    def create(
        cls,
        map_id: str,
        camera_name: str,
        timestamp: datetime,
        ptz_state: PTZState,
        image_path: Path,
    ) -> CapturedFrame:
        """Factory method to create a CapturedFrame with proper ID.

        Args:
            map_id: Map identifier.
            camera_name: Camera name.
            timestamp: Capture timestamp.
            ptz_state: PTZ state at capture.
            image_path: Relative path to image file.

        Returns:
            New CapturedFrame instance with generated ID.
        """
        ts_str = timestamp.isoformat().replace(":", "-")
        entity_id = f"{map_id}/{camera_name}/{ts_str}"

        return cls(
            id=entity_id,
            map_id=map_id,
            camera_name=camera_name,
            timestamp=timestamp,
            ptz_state=ptz_state,
            image_path=image_path,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "map_id": self.map_id,
            "camera_name": self.camera_name,
            "timestamp": self.timestamp.isoformat(),
            "ptz_state": self.ptz_state.to_dict(),
            "image_path": str(self.image_path),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapturedFrame:
        """Create CapturedFrame from dictionary."""
        from poc_homography.domain.vo.ptz_state import PTZState

        return cls(
            id=data["id"],
            map_id=data["map_id"],
            camera_name=data["camera_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            ptz_state=PTZState.from_dict(data["ptz_state"]),
            image_path=Path(data["image_path"]),
        )
