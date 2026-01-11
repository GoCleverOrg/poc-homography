"""
GPS coordinate conversion utilities.

This module provides functions for converting between different coordinate formats.
"""

import re


def dms_to_dd(dms_str: str) -> float:
    """
    Convert DMS (degrees, minutes, seconds) string to decimal degrees.

    Supports formats like:
    - "39°38'25.72\"N"
    - "0°13'48.63\"W"

    Args:
        dms_str: DMS coordinate string

    Returns:
        Decimal degrees (negative for S/W)

    Raises:
        ValueError: If DMS format is invalid
    """
    # Pattern to match DMS format
    pattern = r"""(\d+)°(\d+)'([\d.]+)"?([NSEW])"""
    match = re.match(pattern, dms_str)
    if not match:
        raise ValueError(f"Invalid DMS format: {dms_str}")

    degrees = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    direction = match.group(4)

    dd = degrees + minutes / 60 + seconds / 3600

    if direction in ("S", "W"):
        dd = -dd

    return dd
