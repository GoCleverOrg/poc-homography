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

from poc_homography.homography.config import HomographyConfig, get_default_config
from poc_homography.homography.feature_match import FeatureMatchHomography
from poc_homography.homography.interface import (
    CoordinateSystemMode,
    HomographyApproach,
    HomographyMatrix,
    HomographyProvider,
    HomographyResult,
    validate_homography_matrix,
)
from poc_homography.homography.intrinsic_extrinsic import IntrinsicExtrinsicHomography
from poc_homography.homography.map_points import (
    MapPointComputationResult,
    MapPointHomography,
)
from poc_homography.homography.parameters import (
    IntrinsicExtrinsicConfig,
    IntrinsicExtrinsicResult,
)

__all__ = [
    # Interface and enums
    "HomographyProvider",
    "HomographyApproach",
    "HomographyMatrix",
    "HomographyResult",
    "CoordinateSystemMode",
    "validate_homography_matrix",
    # Configuration
    "HomographyConfig",
    "get_default_config",
    # Parameters
    "IntrinsicExtrinsicConfig",
    "IntrinsicExtrinsicResult",
    # Providers
    "IntrinsicExtrinsicHomography",
    "FeatureMatchHomography",
    "MapPointHomography",
    "MapPointComputationResult",
]
