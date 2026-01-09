#!/usr/bin/env python3
"""CLI for comprehensive calibration tool."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.comprehensive_calibration import (
    CAMERA_CONFIGS,
    GCP,
    parse_gcps_from_yaml,
    print_results,
    run_calibration,
)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive calibration for sub-5px projection accuracy"
    )
    parser.add_argument("--camera", "-c", type=str, required=True, help="Camera name (e.g., Valte)")
    parser.add_argument("--gcps", "-g", type=str, help="Path to YAML file with GCPs")
    parser.add_argument(
        "--gcp",
        "-p",
        action="append",
        nargs=6,
        metavar=("LAT", "LON", "U", "V", "PAN", "TILT"),
        help="Single GCP: LAT LON PIXEL_U PIXEL_V PAN_RAW TILT_DEG (can be repeated)",
    )
    parser.add_argument("--no-gps", action="store_true", help="Do not optimize camera GPS position")
    parser.add_argument("--no-focal", action="store_true", help="Do not optimize focal length")
    parser.add_argument(
        "--no-pan", action="store_true", help="Do not optimize pan offset (use value from config)"
    )
    parser.add_argument("--no-tilt", action="store_true", help="Do not optimize tilt offset")
    parser.add_argument(
        "--no-distortion", action="store_true", help="Do not optimize lens distortion coefficients"
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Zoom factor for all GCPs if using --gcp (default: 1.0)",
    )
    parser.add_argument(
        "--width", type=int, default=1920, help="Image width in pixels (default: 1920)"
    )
    parser.add_argument(
        "--height", type=int, default=1080, help="Image height in pixels (default: 1080)"
    )

    args = parser.parse_args()

    # Get camera config
    if args.camera not in CAMERA_CONFIGS:
        print(f"Error: Unknown camera '{args.camera}'. Available: {list(CAMERA_CONFIGS.keys())}")
        sys.exit(1)

    camera_config = CAMERA_CONFIGS[args.camera]

    # Parse GCPs
    gcps = []

    if args.gcps:
        gcps = parse_gcps_from_yaml(args.gcps)

    if args.gcp:
        for gcp_args in args.gcp:
            lat, lon, u, v, pan, tilt = map(float, gcp_args)
            gcps.append(
                GCP(
                    lat=lat,
                    lon=lon,
                    pixel_u=u,
                    pixel_v=v,
                    pan_raw=pan,
                    tilt_deg=tilt,
                    zoom=args.zoom,
                )
            )

    if not gcps:
        print("Error: No GCPs provided. Use --gcps FILE or --gcp LAT LON U V PAN TILT")
        print("\nExample:")
        print("  python comprehensive_calibration.py --camera Valte \\")
        print("    --gcp 39.640500 -0.230100 960 540 0.0 30.0 \\")
        print("    --gcp 39.640600 -0.230000 1200 400 5.0 25.0")
        sys.exit(1)

    if len(gcps) < 2:
        print("Warning: Only 1 GCP provided. Using at least 3-4 GCPs is recommended")
        print("         for reliable calibration of all parameters.")

    # Run calibration
    optimized, mean_error, individual_errors = run_calibration(
        camera_config,
        gcps,
        optimize_gps=not args.no_gps,
        optimize_focal=not args.no_focal,
        optimize_pan=not args.no_pan,
        optimize_tilt=not args.no_tilt,
        optimize_distortion=not args.no_distortion,
        image_width=args.width,
        image_height=args.height,
    )

    # Print results
    print_results(camera_config, optimized, mean_error, individual_errors, gcps)


if __name__ == "__main__":
    main()
