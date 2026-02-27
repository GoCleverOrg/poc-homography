"""
Homography strategy pattern implementations.

This package provides the HomographyProvider interface and multiple implementations
for computing homography transformations between camera image coordinates and
map coordinates.

Providers:
- IntrinsicExtrinsicHomography: Uses camera intrinsic/extrinsic parameters
- FeatureMatchHomography: Uses Ground Control Points (GCPs)
- MapPointHomography: Uses map point correspondences
"""

from poc_homography.homography.config import HomographyConfig
from poc_homography.homography.interface import (
    HomographyApproach,
    HomographyProvider,
)
from poc_homography.homography.intrinsic_extrinsic import IntrinsicExtrinsicHomography
from poc_homography.homography.map_points import MapPointHomography
from poc_homography.homography.parameters import (
    IntrinsicExtrinsicConfig,
    IntrinsicExtrinsicResult,
)

__all__ = [
    # Interface and enums
    "HomographyProvider",
    "HomographyApproach",
    # Configuration
    "HomographyConfig",
    # Parameters
    "IntrinsicExtrinsicConfig",
    "IntrinsicExtrinsicResult",
    # Providers
    "IntrinsicExtrinsicHomography",
    "MapPointHomography",
]
