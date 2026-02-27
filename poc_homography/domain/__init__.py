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
    Credential,
    GeoTiff,
    HeightUncertainty,
    ImageDimensions,
    LensDistortion,
    LineTrace,
    MapPoint,
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
    "Credential",
    "GeoTiff",
    "HeightUncertainty",
    "ImageDimensions",
    "LensDistortion",
    "LineTrace",
    "MapPoint",
    "Matrix3x3",
    "Orientation",
    "Photo",
    "PixelPoint",
    "PTZState",
    "Rotation",
    "Vector3",
    "ZoomCalibrationEntry",
]
