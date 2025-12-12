"""
Unified Homography Interface Package.

This package provides a unified interface for computing homography transformations
between image coordinates and world coordinates (GPS or local map coordinates).

The package supports multiple homography computation approaches:
    - Intrinsic/Extrinsic: Camera calibration + pose parameters
    - Feature Matching: SIFT/ORB/LoFTR-based feature detection and matching
    - Learned: Neural network-based homography estimation

All providers implement the HomographyProvider or HomographyProviderExtended
interface, ensuring consistent API across different approaches.

Example Usage:
    >>> from homography_interface import (
    ...     HomographyProvider,
    ...     HomographyApproach,
    ...     WorldPoint,
    ...     MapCoordinate
    ... )
    >>> from intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography
    >>>
    >>> # Create provider
    >>> provider = IntrinsicExtrinsicHomography(width=1920, height=1080)
    >>> provider.set_camera_gps_position(lat=40.7128, lon=-74.0060)
    >>>
    >>> # Compute homography
    >>> result = provider.compute_homography(
    ...     frame=image,
    ...     reference={
    ...         'camera_matrix': K,
    ...         'camera_position': np.array([0, 0, 5.0]),
    ...         'pan_deg': 0.0,
    ...         'tilt_deg': -30.0,
    ...         'map_width': 640,
    ...         'map_height': 640
    ...     }
    ... )
    >>>
    >>> # Project points
    >>> if provider.is_valid():
    ...     world_pt = provider.project_point((1280, 720))
    ...     print(f"GPS: {world_pt.latitude}, {world_pt.longitude}")

Available Classes:
    Core Interface:
        - HomographyProvider: Base interface for all homography providers
        - HomographyProviderExtended: Extended interface with map projections
        - HomographyApproach: Enum of supported approaches
        - HomographyResult: Result dataclass with matrix and metadata
        - WorldPoint: GPS coordinate with confidence
        - MapCoordinate: Local metric coordinate with confidence

    Implementations:
        - IntrinsicExtrinsicHomography: Fully implemented camera-based approach
        - FeatureMatchHomography: Placeholder for feature matching (issue #14)
        - LearnedHomography: Placeholder for learned approach (issue #14)
"""

# Core interface and data structures
from poc_homography.homography_interface import (
    HomographyProvider,
    HomographyProviderExtended,
    HomographyApproach,
    HomographyResult,
    WorldPoint,
    MapCoordinate
)

# Homography provider implementations
from poc_homography.intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography
from poc_homography.feature_match_homography import FeatureMatchHomography
from poc_homography.learned_homography import LearnedHomography

# Configuration and factory
from poc_homography.homography_config import HomographyConfig, get_default_config
from poc_homography.homography_factory import HomographyFactory

# Define public API
__all__ = [
    # Core interface
    'HomographyProvider',
    'HomographyProviderExtended',
    'HomographyApproach',
    'HomographyResult',
    'WorldPoint',
    'MapCoordinate',

    # Implementations
    'IntrinsicExtrinsicHomography',
    'FeatureMatchHomography',
    'LearnedHomography',

    # Configuration and factory
    'HomographyConfig',
    'HomographyFactory',
    'get_default_config',
]

# Package metadata
__version__ = '0.1.0'
__author__ = 'SmartTerminal Team'
__description__ = 'Unified interface for homography computation with multiple approaches'
