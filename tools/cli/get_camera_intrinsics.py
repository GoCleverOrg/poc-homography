#!/usr/bin/env python3
"""CLI for camera intrinsics tool."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.get_camera_intrinsics import (
    CAMERAS,
    DEFAULT_BASE_FOCAL_LENGTH_MM,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_SENSOR_WIDTH_MM,
    format_human_readable,
    format_json,
    format_yaml,
    get_camera_intrinsics,
)


def main():
    parser = argparse.ArgumentParser(
        description="Get current camera intrinsics from PTZ status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("camera", type=str, help="Camera name (e.g., Valte, Setram)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--yaml", action="store_true", help="Output as YAML snippet for config file"
    )
    parser.add_argument(
        "--width",
        "-W",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help=f"Image width in pixels (default: {DEFAULT_IMAGE_WIDTH})",
    )
    parser.add_argument(
        "--height",
        "-H",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help=f"Image height in pixels (default: {DEFAULT_IMAGE_HEIGHT})",
    )
    parser.add_argument(
        "--sensor-width",
        type=float,
        default=DEFAULT_SENSOR_WIDTH_MM,
        help=f"Sensor width in mm (default: {DEFAULT_SENSOR_WIDTH_MM})",
    )
    parser.add_argument(
        "--base-focal",
        type=float,
        default=DEFAULT_BASE_FOCAL_LENGTH_MM,
        help=f"Base focal length in mm at 1x zoom (default: {DEFAULT_BASE_FOCAL_LENGTH_MM})",
    )
    parser.add_argument(
        "--list-cameras", action="store_true", help="List available cameras and exit"
    )

    args = parser.parse_args()

    # List cameras mode
    if args.list_cameras:
        print("Available cameras:")
        for cam in CAMERAS:
            print(f"  - {cam['name']} ({cam['ip']})")
        sys.exit(0)

    try:
        result = get_camera_intrinsics(
            camera_name=args.camera,
            image_width=args.width,
            image_height=args.height,
            sensor_width_mm=args.sensor_width,
            base_focal_length_mm=args.base_focal,
        )

        if args.json:
            print(format_json(result))
        elif args.yaml:
            print(format_yaml(result))
        else:
            print(format_human_readable(result))

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
