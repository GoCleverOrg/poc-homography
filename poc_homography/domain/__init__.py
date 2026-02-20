"""Domain layer containing entities, value objects, enums, and repository interfaces."""

from poc_homography.domain.entities import (
    Annotation,
    CameraCalibration,
    CameraConfig,
    Entity,
    GroundControlPoint,
    Map,
)
from poc_homography.domain.enums import CameraSpec, TiltConvention
from poc_homography.domain.repositories import Repo
from poc_homography.domain.vo import (
    CameraIntrinsics,
    CameraSnapshot,
    Credential,
    GeoTiff,
    HeightUncertainty,
    Homography,
    ImageDimensions,
    LensDistortion,
    LineTrace,
    MapPoint,
    Mask,
    Matrix3x3,
    Orientation,
    Photo,
    PixelPoint,
    PTZState,
    Rotation,
    Vector3,
    ZoomCalibrationEntry,
)

__all__ = [
    # Entities
    "Annotation",
    "CameraCalibration",
    "CameraConfig",
    "Entity",
    "GroundControlPoint",
    "Map",
    # Enums
    "CameraSpec",
    "TiltConvention",
    # Repositories
    "Repo",
    # Value Objects
    "CameraIntrinsics",
    "CameraSnapshot",
    "Credential",
    "GeoTiff",
    "HeightUncertainty",
    "Homography",
    "ImageDimensions",
    "LensDistortion",
    "LineTrace",
    "MapPoint",
    "Mask",
    "Matrix3x3",
    "Orientation",
    "Photo",
    "PixelPoint",
    "PTZState",
    "Rotation",
    "Vector3",
    "ZoomCalibrationEntry",
]
