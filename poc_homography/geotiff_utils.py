#!/usr/bin/env python3
"""
Utility functions for GeoTIFF coordinate transformations.

Implements GDAL's standard 6-parameter affine GeoTransform for converting
between pixel coordinates and geographic/projected coordinates (e.g., UTM).

References:
    - GDAL GeoTransform: https://gdal.org/tutorials/geotransforms_tut.html
    - Issue #133: Fix GeoTIFF pixel→UTM transform to use full 6-parameter affine
"""

from typing import List, Tuple


def apply_geotransform(px: float, py: float, gt: List[float]) -> Tuple[float, float]:
    """
    Apply GDAL 6-parameter affine geotransform to convert pixel to geographic coordinates.

    Implements the GDAL GeoTransform formula:
        Xgeo = GT[0] + P*GT[1] + L*GT[2]
        Ygeo = GT[3] + P*GT[4] + L*GT[5]

    Where:
        GT[0]: X-coordinate of upper-left corner (origin easting/longitude)
        GT[1]: Pixel width (meters/degrees per pixel in X direction)
        GT[2]: Row rotation (typically 0 for north-up images)
        GT[3]: Y-coordinate of upper-left corner (origin northing/latitude)
        GT[4]: Column rotation (typically 0 for north-up images)
        GT[5]: Pixel height (meters/degrees per pixel in Y direction, typically negative)

    Pixel Origin Convention:
        GDAL GeoTransform references the UPPER-LEFT CORNER of a pixel.
        To get pixel CENTER coordinates, add 0.5 to both px and py before calling.

    Args:
        px: Pixel X coordinate (column), 0-indexed from left
        py: Pixel Y coordinate (row), 0-indexed from top
        gt: GeoTransform array [GT0, GT1, GT2, GT3, GT4, GT5]

    Returns:
        Tuple of (easting, northing) or (longitude, latitude) in the coordinate
        reference system of the GeoTIFF.

    Examples:
        >>> # North-up raster with 0.15m pixels
        >>> gt = [737575.05, 0.15, 0, 4391595.45, 0, -0.15]
        >>> easting, northing = apply_geotransform(10, 20, gt)
        >>> print(f"({easting:.2f}, {northing:.2f})")
        (737576.55, 4391592.45)

        >>> # Rotated raster (22.5° clockwise)
        >>> gt_rotated = [500000, 0.1387, 0.0574, 4400000, 0.0574, -0.1387]
        >>> easting, northing = apply_geotransform(100, 0, gt_rotated)
        >>> print(f"({easting:.2f}, {northing:.2f})")
        (500013.87, 4400005.74)

        >>> # Pixel center coordinates (add 0.5 offset)
        >>> easting_center, northing_center = apply_geotransform(0.5, 0.5, gt)
        >>> print(f"Pixel (0,0) center: ({easting_center:.3f}, {northing_center:.3f})")
        Pixel (0,0) center: (737575.125, 4391595.375)

    Notes:
        - For north-up images (no rotation), GT[2]=0 and GT[4]=0, simplifying to:
          easting = GT[0] + px*GT[1]
          northing = GT[3] + py*GT[5]

        - GT[5] is typically negative because image rows increase downward (Y+)
          while northing increases upward in most projected coordinate systems.

        - This function validates that gt has exactly 6 elements and raises
          ValueError otherwise.
    """
    if len(gt) != 6:
        raise ValueError(f"geotransform must have exactly 6 elements, got {len(gt)}")

    easting = gt[0] + px * gt[1] + py * gt[2]
    northing = gt[3] + px * gt[4] + py * gt[5]
    return easting, northing
