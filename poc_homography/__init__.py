"""
Unified Homography Interface Package.

This package provides a unified interface for computing homography transformations
between camera image coordinates and map coordinates (pixel coordinates on reference map).

All providers implement the HomographyProvider interface, ensuring consistent API
across different approaches.
"""

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.domain.vo import (
    CameraIntrinsics,
    HeightUncertainty,
    Homography,
    LensDistortion,
    MapPoint,
    Orientation,
    Vector3,
)
from poc_homography.domain.vo.pixel_point import PixelPoint
from poc_homography.homography import (
    CoordinateSystemMode,
    FeatureMatchHomography,
    HomographyApproach,
    HomographyConfig,
    HomographyMatrix,
    HomographyProvider,
    HomographyResult,
    IntrinsicExtrinsicConfig,
    IntrinsicExtrinsicHomography,
    IntrinsicExtrinsicResult,
    MapPointComputationResult,
    MapPointHomography,
    get_default_config,
)

__all__ = [
    # Core camera geometry (VO-based API)
    "CameraGeometry",
    "CameraIntrinsics",
    "Homography",
    "Orientation",
    "Vector3",
    "LensDistortion",
    "HeightUncertainty",
    # Homography interface
    "HomographyProvider",
    "HomographyApproach",
    "HomographyMatrix",
    "HomographyResult",
    "CoordinateSystemMode",
    # Immutable homography parameter types
    "IntrinsicExtrinsicConfig",
    "IntrinsicExtrinsicResult",
    # Map points
    "MapPoint",
    "MapPointHomography",
    "MapPointComputationResult",
    # Other
    "PixelPoint",
    "IntrinsicExtrinsicHomography",
    "FeatureMatchHomography",
    "HomographyConfig",
    "get_default_config",
]

# Package metadata
__version__ = "0.1.0"
__author__ = "SmartTerminal Team"
__description__ = "Unified interface for homography computation with MapPoint coordinates"
