#!/usr/bin/env python3
"""CLI for Y scaling debug tool."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.debug_y_scaling import (
    analyze_gsd_sign,
    check_capture_gcps_local_xy,
    check_origin_interpretation,
    load_geotiff_params,
    trace_pixel_to_all,
)


def main():
    parser = argparse.ArgumentParser(
        description="Debug tool to trace Y scaling issues in coordinate transformations"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="Valte",
        help="Camera name to load georeferencing parameters from (default: Valte)",
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

    check_origin_interpretation()
    analyze_gsd_sign()

    # Trace specific pixels
    trace_pixel_to_all(0, 0)  # Top-left
    trace_pixel_to_all(840, 458)  # Center
    trace_pixel_to_all(100, 100)  # Near top-left

    check_capture_gcps_local_xy()


if __name__ == "__main__":
    main()
