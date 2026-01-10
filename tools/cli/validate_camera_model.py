#!/usr/bin/env python3
"""CLI for camera model validation tool.

Example usage:
    python tools/cli/validate_camera_model.py --camera Valte \\
        --gcps gcps.yaml --map-points map_points.json

Where gcps.yaml contains:
    gcps:
      - map_point_id: Z1
        pixel_u: 960
        pixel_v: 540
        pan_raw: 0.0
        tilt_deg: 30.0
        zoom: 1.0
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.validate_camera_model import load_gcps_from_yaml, validate_model

from poc_homography.map_points import MapPointRegistry


def main():
    parser = argparse.ArgumentParser(description="Validate camera projection model")
    parser.add_argument("--camera", "-c", required=True, help="Camera name (e.g., Valte)")
    parser.add_argument("--gcps", "-g", help="Path to YAML file with GCPs")
    parser.add_argument(
        "--gcp",
        "-p",
        action="append",
        nargs=5,
        metavar=("MAP_POINT_ID", "U", "V", "PAN", "TILT"),
        help="Single GCP: MAP_POINT_ID PIXEL_U PIXEL_V PAN_DEG TILT_DEG (can be repeated)",
    )
    parser.add_argument(
        "--map-points",
        type=str,
        default="map_points.json",
        help="Path to map points JSON file (default: map_points.json)",
    )
    parser.add_argument("--zoom", type=float, default=1.0, help="Zoom factor (default: 1.0)")

    args = parser.parse_args()

    # Load map points registry
    try:
        registry = MapPointRegistry.load(args.map_points)
    except FileNotFoundError:
        print(f"Error: Map points file not found: {args.map_points}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading map points: {e}")
        sys.exit(1)

    gcps = []

    if args.gcps:
        gcps = load_gcps_from_yaml(args.gcps)

    if args.gcp:
        for gcp_args in args.gcp:
            map_point_id, u, v, pan, tilt = gcp_args
            # Validate map point exists
            if map_point_id not in registry.points:
                print(f"Error: Map point '{map_point_id}' not found in {args.map_points}")
                print(f"Available points: {list(registry.points.keys())[:10]}...")
                sys.exit(1)
            gcps.append(
                {
                    "map_point_id": map_point_id,
                    "pixel_u": float(u),
                    "pixel_v": float(v),
                    "pan": float(pan),
                    "tilt": float(tilt),
                    "zoom": args.zoom,
                }
            )

    if not gcps:
        print("Error: No GCPs provided. Use --gcps FILE or --gcp MAP_POINT_ID U V PAN TILT")
        print("\nExample:")
        print("  python validate_camera_model.py --camera Valte \\")
        print("    --gcp Z1 960 540 0.0 30.0 \\")
        print("    --map-points map_points.json")
        sys.exit(1)

    # Validate all GCP map points exist
    for gcp in gcps:
        if gcp["map_point_id"] not in registry.points:
            print(f"Error: Map point '{gcp['map_point_id']}' not found in {args.map_points}")
            sys.exit(1)

    validate_model(args.camera, gcps, registry)


if __name__ == "__main__":
    main()
