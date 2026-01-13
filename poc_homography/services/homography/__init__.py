"""Homography service and strategies.

This module provides the ServiceHomography for computing homography matrices
that map between camera image pixels and world ground plane coordinates.

Strategies:
    - IntrinsicExtrinsicStrategy: Classical CV approach using K matrix and R|t
"""

from poc_homography.services.homography.intrinsic_extrinsic_strategy import (
    IntrinsicExtrinsicStrategy,
)
from poc_homography.services.homography.strategy import HomographyStrategy
from poc_homography.services.service_homography import ServiceHomography

__all__ = [
    "HomographyStrategy",
    "IntrinsicExtrinsicStrategy",
    "ServiceHomography",
]
