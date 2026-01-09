#!/usr/bin/env python3
"""CLI for KML point extraction tool."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.extract_kml_points import run_server

from poc_homography.camera_config import get_camera_by_name, get_camera_configs
from poc_homography.kml import GeoConfig


def main():
    parser = argparse.ArgumentParser(
        description="Extract reference points from georeferenced image to KML"
    )
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument(
        "--camera",
        type=str,
        default="Valte",
        help="Camera name to load configuration from (default: Valte)",
    )
    parser.add_argument(
        "--origin-e",
        type=float,
        default=None,
        help="Origin easting (UTM) - overrides camera config",
    )
    parser.add_argument(
        "--origin-n",
        type=float,
        default=None,
        help="Origin northing (UTM) - overrides camera config",
    )
    parser.add_argument(
        "--gsd",
        type=float,
        default=None,
        help="Ground sample distance in meters - overrides camera config",
    )
    parser.add_argument(
        "--crs", default=None, help="Coordinate reference system - overrides camera config"
    )
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")

    args = parser.parse_args()

    # Load camera configuration (required - single source of truth)
    camera_config = get_camera_by_name(args.camera)

    if camera_config is None:
        print(f"Error: Camera '{args.camera}' not found in configuration.")
        print(f"Available cameras: {', '.join([c['name'] for c in get_camera_configs()])}")
        sys.exit(1)

    # Check if camera has geotiff_params
    if "geotiff_params" not in camera_config:
        print(f"Error: Camera '{args.camera}' does not have 'geotiff_params' defined.")
        print("Please update the camera configuration in poc_homography/camera_config.py")
        sys.exit(1)

    geotiff_params = camera_config["geotiff_params"]

    # Check for new geotransform format vs old format
    if "geotransform" in geotiff_params:
        # New format: use geotransform array directly
        gt = list(geotiff_params["geotransform"])
        crs = geotiff_params["utm_crs"]
        print(
            f"Loaded georeferencing parameters from camera: {args.camera} (new geotransform format)"
        )
    else:
        # Old format: build geotransform from separate parameters
        gt = [
            geotiff_params["origin_easting"],
            geotiff_params["pixel_size_x"],
            0.0,  # row_rotation (assumed 0 for legacy format)
            geotiff_params["origin_northing"],
            0.0,  # col_rotation (assumed 0 for legacy format)
            geotiff_params["pixel_size_y"],
        ]
        crs = geotiff_params["utm_crs"]
        print(
            f"Loaded georeferencing parameters from camera: {args.camera} (legacy format, converted to geotransform)"
        )

    # Command-line arguments override camera config
    if args.origin_e is not None or args.origin_n is not None or args.gsd is not None:
        # Override specific values
        if args.origin_e is not None:
            gt[0] = args.origin_e
        if args.origin_n is not None:
            gt[3] = args.origin_n
        if args.gsd is not None:
            gt[1] = args.gsd
            gt[5] = -args.gsd
        print("Applied command-line overrides to geotransform")

    if args.crs is not None:
        crs = args.crs

    geo_config = GeoConfig(
        crs=crs,
        geotransform=(gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]),
    )
    run_server(args.image, geo_config, args.port)


if __name__ == "__main__":
    main()
