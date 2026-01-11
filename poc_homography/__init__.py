"""
Unified Homography Interface Package.

This package provides a unified interface for computing homography transformations
between camera image coordinates and map coordinates (pixel coordinates on reference map).

All providers implement the HomographyProvider interface, ensuring consistent API
across different approaches.
"""

from poc_homography.camera_parameters import (
    CameraGeometryResult,
    CameraParameters,
    DistortionCoefficients,
    HeightUncertainty,
)
from poc_homography.feature_match_homography import FeatureMatchHomography
from poc_homography.homography_config import HomographyConfig, get_default_config
from poc_homography.homography_interface import (
    CoordinateSystemMode,
    HomographyApproach,
    HomographyMatrix,
    HomographyProvider,
    HomographyResult,
)
from poc_homography.homography_map_points import (
    MapPointComputationResult,
    MapPointHomography,
)
from poc_homography.homography_parameters import (
    IntrinsicExtrinsicConfig,
    IntrinsicExtrinsicResult,
)
from poc_homography.intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography
from poc_homography.map_points import MapPoint, MapPointRegistry
from poc_homography.pixel_point import PixelPoint

__all__ = [
    # Immutable camera parameter types
    "CameraParameters",
    "CameraGeometryResult",
    "DistortionCoefficients",
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
    "MapPointRegistry",
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
