"""Orientation service and strategies.

This module provides the OrientationService for computing final camera
orientation from base orientation and PTZ state, using pluggable strategies.

Strategies:
    - AdditiveOrientationStrategy: Simple angle addition (for small angles)
    - RotationMatrixStrategy: Proper SO(3) matrix composition (for large angles/roll)
"""

from poc_homography.services.orientation.additive_strategy import AdditiveOrientationStrategy
from poc_homography.services.orientation.rotation_matrix_strategy import RotationMatrixStrategy
from poc_homography.services.orientation.strategy import OrientationStrategy
from poc_homography.services.service_orientation import ServiceOrientation

__all__ = [
    "AdditiveOrientationStrategy",
    "ServiceOrientation",
    "OrientationStrategy",
    "RotationMatrixStrategy",
]
