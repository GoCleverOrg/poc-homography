"""Domain layer containing entities, value objects, enums, and repository interfaces."""

from poc_homography.domain.entities import (
    Annotation,
    CameraCalibration,
    CameraConfig,
    GroundControlPoint,
    Map,
)
from poc_homography.domain.enums import CameraSpec, TiltConvention
from poc_homography.domain.repositories import (
    CameraCalibrationRepository,
    CameraConfigRepository,
    GroundControlPointRepository,
    MapRepository,
)
from poc_homography.domain.vo import (
    CameraIntrinsics,
    CameraSnapshot,
    Credential,
    GeoTiff,
    LensDistortion,
    MapPoint,
    Orientation,
    Photo,
    PixelPoint,
    PTZState,
)

__all__ = [
    # Entities
    "Annotation",
    "CameraCalibration",
    "CameraConfig",
    "GroundControlPoint",
    "Map",
    # Enums
    "CameraSpec",
    "TiltConvention",
    # Repositories
    "CameraCalibrationRepository",
    "CameraConfigRepository",
    "GroundControlPointRepository",
    "MapRepository",
    # Value Objects
    "CameraIntrinsics",
    "CameraSnapshot",
    "Credential",
    "GeoTiff",
    "LensDistortion",
    "MapPoint",
    "Orientation",
    "Photo",
    "PixelPoint",
    "PTZState",
]
