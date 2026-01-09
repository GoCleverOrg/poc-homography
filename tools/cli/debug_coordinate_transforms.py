#!/usr/bin/env python3
"""CLI for coordinate transformation debug tool."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.debug_coordinate_transforms import (
    analyze_transformation_discrepancy,
    load_geotiff_params,
    test_specific_point,
)


def main():
    parser = argparse.ArgumentParser(
        description="Debug tool to analyze coordinate transformation discrepancies"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="Valte",
        help="Camera name to load georeferencing parameters from (default: Valte)",
    )
    parser.add_argument(
        "px", type=float, nargs="?", help="Pixel X coordinate for specific point test (optional)"
    )
    parser.add_argument(
        "py", type=float, nargs="?", help="Pixel Y coordinate for specific point test (optional)"
    )
    args = parser.parse_args()

    # Load georeferencing parameters from camera config
    origin_easting, origin_northing, gsd, utm_crs = load_geotiff_params(args.camera)

    print(f"Loaded georeferencing parameters from camera '{args.camera}':")
    print(f"  Origin Easting: {origin_easting}")
    print(f"  Origin Northing: {origin_northing}")
    print(f"  GSD: {gsd} m/pixel")
    print(f"  UTM CRS: {utm_crs}")
    print()

    analyze_transformation_discrepancy()

    if args.px is not None and args.py is not None:
        test_specific_point(args.px, args.py)


if __name__ == "__main__":
    main()
