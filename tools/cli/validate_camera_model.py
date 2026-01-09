#!/usr/bin/env python3
"""CLI for camera model validation tool."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.validate_camera_model import load_gcps_from_yaml, validate_model


def main():
    parser = argparse.ArgumentParser(description="Validate camera projection model")
    parser.add_argument("--camera", "-c", required=True, help="Camera name (e.g., Valte)")
    parser.add_argument("--gcps", "-g", help="Path to YAML file with GCPs")
    parser.add_argument(
        "--gcp",
        "-p",
        action="append",
        nargs=6,
        metavar=("LAT", "LON", "U", "V", "PAN", "TILT"),
        help="Single GCP: LAT LON PIXEL_U PIXEL_V PAN_DEG TILT_DEG",
    )
    parser.add_argument("--zoom", type=float, default=1.0, help="Zoom factor (default: 1.0)")

    args = parser.parse_args()

    gcps = []

    if args.gcps:
        gcps = load_gcps_from_yaml(args.gcps)

    if args.gcp:
        for gcp_args in args.gcp:
            lat, lon, u, v, pan, tilt = map(float, gcp_args)
            gcps.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "pixel_u": u,
                    "pixel_v": v,
                    "pan": pan,
                    "tilt": tilt,
                    "zoom": args.zoom,
                    "name": f"Manual ({lat:.4f}, {lon:.4f})",
                }
            )

    if not gcps:
        print("Error: No GCPs provided. Use --gcps FILE or --gcp LAT LON U V PAN TILT")
        sys.exit(1)

    validate_model(args.camera, gcps)


if __name__ == "__main__":
    main()
