"""Repository interfaces for the domain layer.

Repositories provide abstractions for data access, keeping the domain
layer independent of infrastructure concerns (file I/O, databases, etc.).

Naming convention: All repository interfaces use the suffix "Repository"
(e.g., MapRepository, CameraConfigRepository).
"""

from poc_homography.domain.repositories.camera_calibration_repository import (
    CameraCalibrationRepository,
)
from poc_homography.domain.repositories.camera_config_repository import (
    CameraConfigRepository,
)
from poc_homography.domain.repositories.ground_control_point_repository import (
    GroundControlPointRepository,
)
from poc_homography.domain.repositories.map_repository import MapRepository

__all__ = [
    "CameraCalibrationRepository",
    "CameraConfigRepository",
    "GroundControlPointRepository",
    "MapRepository",
]
