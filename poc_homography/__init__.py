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
)
from poc_homography.domain.vo import HeightUncertainty, LensDistortion, PixelPoint
from poc_homography.homography import (
    HomographyApproach,
    HomographyProvider,
    IntrinsicExtrinsicConfig,
    IntrinsicExtrinsicHomography,
    IntrinsicExtrinsicResult,
    MapPointHomography,
)
from poc_homography.map_points import GCPRegistry, MapPoint

__all__ = [
    # Core camera geometry (immutable API)
    "CameraGeometry",
    "CameraParameters",
    "CameraGeometryResult",
    "LensDistortion",
    "HeightUncertainty",
    # Homography interface
    "HomographyProvider",
    "HomographyApproach",
    # Immutable homography parameter types
    "IntrinsicExtrinsicConfig",
    "IntrinsicExtrinsicResult",
    # Map points
    "MapPoint",
    "GCPRegistry",
    "MapPointHomography",
    # Other
    "PixelPoint",
    "IntrinsicExtrinsicHomography",
]

# Package metadata
__version__ = "0.1.0"
__author__ = "SmartTerminal Team"
__description__ = "Unified interface for homography computation with MapPoint coordinates"
