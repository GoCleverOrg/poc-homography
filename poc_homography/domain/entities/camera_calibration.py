"""Camera calibration entity for calibration data refined during the calibration process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.vo.lens_distortion import LensDistortion
    from poc_homography.domain.vo.orientation import Orientation
    from poc_homography.domain.vo.pixel_point import PixelPoint
    from poc_homography.types import Meters


@dataclass(frozen=True)
class CameraCalibration:
    """Camera calibration data refined during the calibration process.

    This entity contains data that changes during calibration workflows.
    It is separate from CameraConfig (which rarely changes) and PTZState
    (which changes constantly and is never persisted).

    The id field references the CameraConfig.id of the camera
    this calibration belongs to.

    Attributes:
        id: References CameraConfig.id (format: "map_id/name").
        position: Position of the camera on the map (pixel coordinates).
        height: Height of the camera above the ground plane in meters.
        base_orientation: Camera orientation at PTZ home position (when PTZ is at 0,0).
        distortion: Lens distortion coefficients (calibrated per-camera).
    """

    id: str
    position: PixelPoint
    height: Meters
    base_orientation: Orientation
    distortion: LensDistortion

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "position": self.position.to_dict(),
            "height": float(self.height),
            "base_orientation": self.base_orientation.to_dict(),
            "distortion": self.distortion.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraCalibration:
        """Create CameraCalibration from dictionary."""
        from poc_homography.domain.vo.lens_distortion import LensDistortion
        from poc_homography.domain.vo.orientation import Orientation
        from poc_homography.domain.vo.pixel_point import PixelPoint
        from poc_homography.types import Meters

        return cls(
            id=data["id"],
            position=PixelPoint.from_dict(data["position"]),
            height=Meters(data["height"]),
            base_orientation=Orientation.from_dict(data["base_orientation"]),
            distortion=LensDistortion.from_dict(data["distortion"]),
        )
