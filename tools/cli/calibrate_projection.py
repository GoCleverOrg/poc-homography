#!/usr/bin/env python3
"""CLI for projection calibration tool.

Example usage:
    python tools/cli/calibrate_projection.py Valte Z1 960 540 0.0 30.0 1.0 --map-points map_points.json
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.calibrate_projection import CAMERA_CONFIGS, analyze_projection_error

from poc_homography.map_points import MapPointRegistry


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate projection parameters using a known Map Point reference",
        epilog="Example: python tools/cli/calibrate_projection.py Valte Z1 960 540 0.0 30.0 1.0 --map-points map_points.json",
    )
    parser.add_argument("camera", type=str, help="Camera name (e.g., Valte)")
    parser.add_argument("map_point_id", type=str, help="Map Point ID (e.g., Z1, P5, A3)")
    parser.add_argument(
        "actual_u", type=float, help="Actual pixel U coordinate where point appears in image"
    )
    parser.add_argument(
        "actual_v", type=float, help="Actual pixel V coordinate where point appears in image"
    )
    parser.add_argument("pan_raw", type=float, help="Raw pan value from camera PTZ")
    parser.add_argument("tilt", type=float, help="Tilt angle in degrees")
    parser.add_argument("zoom", type=float, help="Zoom factor")
    parser.add_argument(
        "--map-points",
        type=str,
        default="map_points.json",
        help="Path to map_points.json file (default: map_points.json)",
    )

    args = parser.parse_args()

    # Get camera config
    if args.camera not in CAMERA_CONFIGS:
        print(f"Error: Unknown camera '{args.camera}'. Available: {list(CAMERA_CONFIGS.keys())}")
        sys.exit(1)

    camera_config = CAMERA_CONFIGS[args.camera]

    # Load map points registry
    try:
        registry = MapPointRegistry.load(args.map_points)
    except FileNotFoundError:
        print(f"Error: Map points file not found: {args.map_points}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading map points file: {e}")
        sys.exit(1)

    # Get the map point
    if args.map_point_id not in registry.points:
        available_ids = list(registry.points.keys())[:10]
        print(f"Error: Unknown map point '{args.map_point_id}'.")
        print(f"Available points (first 10): {available_ids}")
        sys.exit(1)

    point = registry.points[args.map_point_id]

    analyze_projection_error(
        camera_config,
        point.pixel_x,
        point.pixel_y,
        args.actual_u,
        args.actual_v,
        args.pan_raw,
        args.tilt,
        args.zoom,
    )


if __name__ == "__main__":
    main()
