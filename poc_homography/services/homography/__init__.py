"""Homography service and strategies.

This module provides the ServiceHomography for computing homography matrices
that map between camera image pixels and world ground plane coordinates.

Strategies:
    - StrategyIntrinsicExtrinsic: Classical CV approach using K matrix and R|t
"""

from poc_homography.services.homography.strategy import StrategyHomography
from poc_homography.services.homography.strategy_intrinsic_extrinsic import (
    StrategyIntrinsicExtrinsic,
)
from poc_homography.services.service_homography import ServiceHomography

__all__ = [
    "ServiceHomography",
    "StrategyHomography",
    "StrategyIntrinsicExtrinsic",
]
