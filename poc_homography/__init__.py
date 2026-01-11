"""
Unified Homography Interface Package.

This package provides a unified interface for computing homography transformations
between camera image coordinates and map coordinates (pixel coordinates on reference map).

All providers implement the HomographyProvider interface, ensuring consistent API
across different approaches.
"""

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.camera_parameters import (
    CameraGeometryResult,
    CameraParameters,
    DistortionCoefficients,
    HeightUncertainty,
)
from poc_homography.coordinate_converter import (
    GCPCoordinateConverter,
    UTMConverter,
    UTMConverterConfig,
)
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
from poc_homography.map_points import MapPoint, MapPointRegistry
from poc_homography.pixel_point import PixelPoint

__all__ = [
    # Core camera geometry (immutable API)
    "CameraGeometry",
    "CameraParameters",
    "CameraGeometryResult",
    "DistortionCoefficients",
    "HeightUncertainty",
    # Coordinate converters (immutable factory pattern)
    "UTMConverter",
    "UTMConverterConfig",
    "GCPCoordinateConverter",
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
