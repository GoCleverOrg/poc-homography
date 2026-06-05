"""PTZ unit conversions for Hikvision ISAPI.

Hikvision ISAPI encodes pan/tilt/zoom as integers scaled x10 (e.g. raw 600 =
60.0 degrees). Raw tilt is positive=down.

This module is the single home for the divide-by-10 / multiply-by-10 PTZ unit
math. No other module may perform these conversions directly. ``round()`` is
used on the encode path so the round-trip is symmetric for representable inputs
(multiples of 0.1): ``raw_to_degrees(degrees_to_raw(18.2)) == 18.2``.
"""

from __future__ import annotations

from poc_homography.types import Degrees, Unitless

_SCALE = 10


def raw_to_degrees(raw: int) -> Degrees:
    """Convert a raw x10 integer to degrees."""
    return Degrees(raw / _SCALE)


def degrees_to_raw(deg: float) -> int:
    """Convert degrees to a raw x10 integer."""
    return round(deg * _SCALE)


def raw_to_zoom(raw: int) -> Unitless:
    """Convert a raw x10 integer to a dimensionless zoom factor."""
    return Unitless(raw / _SCALE)


def zoom_to_raw(zoom: float) -> int:
    """Convert a dimensionless zoom factor to a raw x10 integer."""
    return round(zoom * _SCALE)
