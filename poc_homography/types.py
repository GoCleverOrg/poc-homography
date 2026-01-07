"""
Unit type annotations for type-safe numeric parameters.

This module defines NewType aliases for all physical units used across the
poc_homography codebase. These provide zero-overhead type hints that help
catch unit mismatches at static analysis time while remaining transparent
at runtime.

Type Safety Benefits:
    - Prevents mixing incompatible units (e.g., passing degrees where radians expected)
    - Documents expected units in function signatures
    - Enables static type checkers (mypy) to catch unit errors
    - Zero runtime overhead (NewType is erased at runtime)

Usage Example:
    >>> from poc_homography.types import Degrees, Meters, Pixels
    >>>
    >>> def get_fov(zoom: Unitless, sensor: Millimeters) -> Degrees:
    ...     # Function signature clearly documents units
    ...     pass
    >>>
    >>> # Type checker ensures correct units
    >>> fov: Degrees = get_fov(Unitless(5.0), Millimeters(6.78))
"""

from typing import NewType

# Angular units
Degrees = NewType('Degrees', float)
"""Angle in degrees (e.g., pan, tilt, latitude, longitude, bearing)"""

Radians = NewType('Radians', float)
"""Angle in radians (e.g., intermediate trigonometric calculations)"""

# Distance/position units
Meters = NewType('Meters', float)
"""Distance or position in meters (e.g., X, Y, Z coordinates, camera height, GPS distances)"""

# Image coordinate units
Pixels = NewType('Pixels', int)
"""Image coordinates or dimensions in pixels (e.g., u, v, width, height)"""

PixelsFloat = NewType('PixelsFloat', float)
"""Floating-point image coordinates in pixels (e.g., subpixel coordinates)"""

# Physical sensor dimensions
Millimeters = NewType('Millimeters', float)
"""Physical dimensions in millimeters (e.g., sensor width, focal length)"""

# Dimensionless quantities
Unitless = NewType('Unitless', float)
"""Dimensionless scalar (e.g., zoom factor, scale factor, confidence, ratios)"""
