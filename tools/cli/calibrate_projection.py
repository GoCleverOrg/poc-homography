#!/usr/bin/env python3
"""CLI for projection calibration tool."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.calibrate_projection import CAMERA_CONFIGS, analyze_projection_error


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate projection parameters using a known reference point"
    )
    parser.add_argument("--camera", "-c", type=str, required=True, help="Camera name (e.g., Valte)")
    parser.add_argument(
        "--reference-point",
        "-r",
        type=str,
        required=True,
        help="GPS coordinates of reference point: LAT,LON",
    )
    parser.add_argument(
        "--pixel",
        "-p",
        type=str,
        required=True,
        help="Actual pixel location of reference point in image: U,V",
    )
    parser.add_argument(
        "--pan-raw", type=float, default=0.0, help="Raw pan value from camera PTZ (default: 0)"
    )
    parser.add_argument(
        "--tilt", type=float, default=30.0, help="Tilt angle in degrees (default: 30)"
    )
    parser.add_argument("--zoom", type=float, default=1.0, help="Zoom factor (default: 1.0)")

    args = parser.parse_args()

    # Parse reference point
    try:
        ref_lat, ref_lon = map(float, args.reference_point.split(","))
    except ValueError:
        print("Error: Invalid reference point format. Use: LAT,LON")
        sys.exit(1)

    # Parse pixel
    try:
        actual_u, actual_v = map(float, args.pixel.split(","))
    except ValueError:
        print("Error: Invalid pixel format. Use: U,V")
        sys.exit(1)

    # Get camera config
    if args.camera not in CAMERA_CONFIGS:
        print(f"Error: Unknown camera '{args.camera}'. Available: {list(CAMERA_CONFIGS.keys())}")
        sys.exit(1)

    camera_config = CAMERA_CONFIGS[args.camera]

    analyze_projection_error(
        camera_config, ref_lat, ref_lon, actual_u, actual_v, args.pan_raw, args.tilt, args.zoom
    )


if __name__ == "__main__":
    main()
