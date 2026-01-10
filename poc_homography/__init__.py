"""
Unified Homography Interface Package.

This package provides a unified interface for computing homography transformations
between camera image coordinates and map coordinates (pixel coordinates on reference map).

The package supports multiple homography computation approaches:
    - Intrinsic/Extrinsic: Camera calibration + pose parameters
    - Feature Matching: GCP-based feature matching
    - Learned: Neural network-based homography estimation
    - MapPoint: Map-point-based pixel coordinate transformations

All providers implement the HomographyProvider interface, ensuring consistent API
across different approaches.

Example Usage:
    >>> from poc_homography import MapPointHomography
    >>> from poc_homography.map_points import MapPointRegistry
    >>>
    >>> # Load map points
    >>> registry = MapPointRegistry.load("map_points.json")
    >>>
    >>> # Create homography provider
    >>> homography = MapPointHomography(map_id="map_valte")
    >>>
    >>> # Compute homography from GCPs
    >>> gcps = [
    ...     {"pixel_x": 800, "pixel_y": 580, "map_point_id": "A7"},
    ...     {"pixel_x": 1082, "pixel_y": 390, "map_point_id": "A6"},
    ... ]
    >>> result = homography.compute_from_gcps(gcps, registry)
    >>>
    >>> # Project camera pixel to map - returns MapPoint
    >>> map_point = homography.camera_to_map((960, 540))
    >>> print(f"Map pixel: ({map_point.pixel_x}, {map_point.pixel_y})")

Available Classes:
    Core Interface:
        - HomographyProvider: Base interface for all homography providers
        - HomographyApproach: Enum of supported approaches
        - HomographyResult: Result dataclass with matrix and metadata

    MapPoint System:
        - MapPoint: Point on a map with pixel coordinates
        - MapPointRegistry: Registry of map points
        - MapPointHomography: MapPoint-based homography provider
        - PixelPoint: Simple pixel coordinate pair

    Legacy Implementations (for backward compatibility):
        - IntrinsicExtrinsicHomography: Camera-based approach
        - FeatureMatchHomography: GCP feature matching
        - LearnedHomography: Placeholder for learned approach
"""

# Core interface and data structures
# Legacy homography provider implementations (for backward compatibility)
from poc_homography.feature_match_homography import FeatureMatchHomography

# Configuration and factory
from poc_homography.homography_config import HomographyConfig, get_default_config
from poc_homography.homography_factory import HomographyFactory
from poc_homography.homography_interface import (
    CoordinateSystemMode,
    HomographyApproach,
    HomographyProvider,
    HomographyResult,
)

# MapPoint system
from poc_homography.homography_map_points import (
    HomographyResult as MapPointHomographyResult,
)
from poc_homography.homography_map_points import (
    MapPointHomography,
)
from poc_homography.intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography
from poc_homography.learned_homography import LearnedHomography
from poc_homography.map_points import MapPoint, MapPointRegistry
from poc_homography.pixel_point import PixelPoint

# Define public API
__all__ = [
    # Core interface
    "HomographyProvider",
    "HomographyApproach",
    "HomographyResult",
    "CoordinateSystemMode",
    # MapPoint system
    "MapPoint",
    "MapPointRegistry",
    "MapPointHomography",
    "MapPointHomographyResult",
    "PixelPoint",
    # Legacy implementations
    "IntrinsicExtrinsicHomography",
    "FeatureMatchHomography",
    "LearnedHomography",
    # Configuration and factory
    "HomographyConfig",
    "HomographyFactory",
    "get_default_config",
]

# Package metadata
__version__ = "0.1.0"
__author__ = "SmartTerminal Team"
__description__ = "Unified interface for homography computation with MapPoint coordinates"
