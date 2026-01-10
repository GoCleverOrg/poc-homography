"""
Unified Homography Interface Package.

This package provides a unified interface for computing homography transformations
between camera image coordinates and map coordinates (pixel coordinates on reference map).

All providers implement the HomographyProvider interface, ensuring consistent API
across different approaches.
"""

from poc_homography.feature_match_homography import FeatureMatchHomography
from poc_homography.homography_config import HomographyConfig, get_default_config
from poc_homography.homography_interface import (
    CoordinateSystemMode,
    HomographyApproach,
    HomographyProvider,
    HomographyResult,
)
from poc_homography.homography_map_points import (
    HomographyResult as MapPointHomographyResult,
)
from poc_homography.homography_map_points import (
    MapPointHomography,
)
from poc_homography.intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography
from poc_homography.map_points import MapPoint, MapPointRegistry
from poc_homography.pixel_point import PixelPoint

__all__ = [
    "HomographyProvider",
    "HomographyApproach",
    "HomographyResult",
    "CoordinateSystemMode",
    "MapPoint",
    "MapPointRegistry",
    "MapPointHomography",
    "MapPointHomographyResult",
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
