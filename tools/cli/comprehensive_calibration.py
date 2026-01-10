#!/usr/bin/env python3
"""CLI for comprehensive calibration tool.

Example usage:
    python tools/cli/comprehensive_calibration.py --camera Valte \\
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

from tools.comprehensive_calibration import (
    CAMERA_CONFIGS,
    GCP,
    parse_gcps_from_yaml,
    print_results,
    run_calibration,
)

from poc_homography.map_points import MapPointRegistry


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
        nargs=5,
        metavar=("MAP_POINT_ID", "U", "V", "PAN", "TILT"),
        help="Single GCP: MAP_POINT_ID PIXEL_U PIXEL_V PAN_RAW TILT_DEG (can be repeated)",
    )
    parser.add_argument(
        "--map-points",
        type=str,
        default="map_points.json",
        help="Path to map points JSON file (default: map_points.json)",
    )
    parser.add_argument(
        "--no-position", action="store_true", help="Do not optimize camera position offset"
    )
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

    # Load map points registry
    try:
        registry = MapPointRegistry.load(args.map_points)
    except FileNotFoundError:
        print(f"Error: Map points file not found: {args.map_points}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading map points: {e}")
        sys.exit(1)

    # Parse GCPs
    gcps = []

    if args.gcps:
        gcps = parse_gcps_from_yaml(args.gcps)

    if args.gcp:
        for gcp_args in args.gcp:
            map_point_id, u, v, pan, tilt = gcp_args
            # Validate map point exists
            if map_point_id not in registry.points:
                print(f"Error: Map point '{map_point_id}' not found in {args.map_points}")
                print(f"Available points: {list(registry.points.keys())[:10]}...")
                sys.exit(1)
            gcps.append(
                GCP(
                    map_point_id=map_point_id,
                    pixel_u=float(u),
                    pixel_v=float(v),
                    pan_raw=float(pan),
                    tilt_deg=float(tilt),
                    zoom=args.zoom,
                )
            )

    if not gcps:
        print("Error: No GCPs provided. Use --gcps FILE or --gcp MAP_POINT_ID U V PAN TILT")
        print("\nExample:")
        print("  python comprehensive_calibration.py --camera Valte \\")
        print("    --gcp Z1 960 540 0.0 30.0 \\")
        print("    --gcp Z2 1200 400 5.0 25.0 \\")
        print("    --map-points map_points.json")
        sys.exit(1)

    if len(gcps) < 2:
        print("Warning: Only 1 GCP provided. Using at least 3-4 GCPs is recommended")
        print("         for reliable calibration of all parameters.")

    # Validate all GCP map points exist in registry
    for gcp in gcps:
        if gcp.map_point_id not in registry.points:
            print(f"Error: Map point '{gcp.map_point_id}' not found in {args.map_points}")
            sys.exit(1)

    # Run calibration
    optimized, mean_error, individual_errors = run_calibration(
        camera_config,
        gcps,
        registry,
        optimize_position=not args.no_position,
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
