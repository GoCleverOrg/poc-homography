#!/usr/bin/env python3
"""CLI for camera model validation tool.

Example usage:
    python tools/cli/validate_camera_model.py --camera Valte \
        --gcps gcps.yaml --map-points map_points.json

Where gcps.yaml contains:
    capture:
      context:
        camera: Valte
        pan_raw: 0.0
        tilt_deg: 30.0
        zoom: 1.0
      annotations:
        - gcp_id: Z1
          pixel:
            x: 960.0
            y: 540.0
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.validate_camera_model import load_gcps_from_yaml, validate_model

from poc_homography.calibration.annotation import Annotation, CaptureContext
from poc_homography.map_points import MapPointRegistry
from poc_homography.pixel_point import PixelPoint


def main():
    parser = argparse.ArgumentParser(description="Validate camera projection model")
    parser.add_argument("--camera", "-c", required=True, help="Camera name (e.g., Valte)")
    parser.add_argument("--gcps", "-g", help="Path to YAML file with GCPs")
    parser.add_argument(
        "--gcp",
        "-p",
        action="append",
        nargs=5,
        metavar=("GCP_ID", "X", "Y", "PAN", "TILT"),
        help="Single GCP: GCP_ID PIXEL_X PIXEL_Y PAN_RAW TILT_DEG (can be repeated)",
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

    context = None
    annotations = []

    if args.gcps:
        context, annotations = load_gcps_from_yaml(args.gcps)

    if args.gcp:
        # Create context from command-line args if not from YAML
        if context is None:
            context = CaptureContext(
                camera=args.camera,
                pan_raw=0.0,  # Will be overridden by first GCP
                tilt_deg=0.0,  # Will be overridden by first GCP
                zoom=args.zoom,
            )

        for gcp_args in args.gcp:
            gcp_id, x, y, pan, tilt = gcp_args
            # Validate map point exists
            if gcp_id not in registry.points:
                print(f"Error: Map point '{gcp_id}' not found in {args.map_points}")
                print(f"Available points: {list(registry.points.keys())[:10]}...")
                sys.exit(1)

            # Update context with first GCP's PTZ values
            if not annotations:
                context = CaptureContext(
                    camera=args.camera,
                    pan_raw=float(pan),
                    tilt_deg=float(tilt),
                    zoom=args.zoom,
                )

            annotations.append(
                Annotation(
                    gcp_id=gcp_id,
                    pixel=PixelPoint(x=float(x), y=float(y)),
                )
            )

    if not annotations:
        print("Error: No GCPs provided. Use --gcps FILE or --gcp GCP_ID X Y PAN TILT")
        print("\nExample:")
        print("  python validate_camera_model.py --camera Valte \\")
        print("    --gcp Z1 960 540 0.0 30.0 \\")
        print("    --map-points map_points.json")
        sys.exit(1)

    # Validate all GCP map points exist
    for annotation in annotations:
        if annotation.gcp_id not in registry.points:
            print(f"Error: Map point '{annotation.gcp_id}' not found in {args.map_points}")
            sys.exit(1)

    validate_model(context, annotations, registry)


if __name__ == "__main__":
    main()
