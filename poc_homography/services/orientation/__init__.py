"""Orientation service and strategies.

This module provides the ServiceOrientation for computing final camera
orientation from base orientation and PTZ state, using pluggable strategies.

Strategies:
    - StrategyOrientationAdditive: Simple angle addition (for small angles)
    - StrategyRotationMatrix: Proper SO(3) matrix composition (for large angles/roll)
"""

from poc_homography.services.orientation.strategy import StrategyOrientation
from poc_homography.services.orientation.strategy_additive import StrategyOrientationAdditive
from poc_homography.services.orientation.strategy_rotation_matrix import StrategyRotationMatrix
from poc_homography.services.service_orientation import ServiceOrientation

__all__ = [
    "ServiceOrientation",
    "StrategyOrientation",
    "StrategyOrientationAdditive",
    "StrategyRotationMatrix",
]
