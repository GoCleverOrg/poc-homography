#!/usr/bin/env python3
"""CLI for GCP GPS verification tool."""

import argparse
import os
import sys
import webbrowser
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.verify_gcp_gps import (
    calculate_bearing,
    calculate_distance,
    generate_map_html,
    get_camera_config_decimal,
    load_gcps_from_yaml,
)


def main():
    parser = argparse.ArgumentParser(
        description="Verify GCP GPS coordinates on an interactive map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_gcp_gps.py --gcps my_gcps.yaml
  python verify_gcp_gps.py --gcps gcps.yaml --camera Valte --output map.html
  python verify_gcp_gps.py --gcps gcps.yaml --camera Valte --no-browser
        """,
    )
    parser.add_argument("--gcps", "-g", required=True, help="Path to YAML file with GCPs")
    parser.add_argument("--camera", "-c", help="Camera name to show position and FOV")
    parser.add_argument("--output", "-o", help="Output HTML file (default: gcp_map.html)")
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open browser automatically"
    )

    args = parser.parse_args()

    # Load GCPs
    print(f"Loading GCPs from: {args.gcps}")
    gcps, ptz_info, metadata = load_gcps_from_yaml(args.gcps)
    print(f"  Loaded {len(gcps)} GCPs")

    if metadata.get("coordinate_system") is None:
        print("  Note: Converted from legacy leaflet_y format")

    # Get camera config if specified
    camera_config = None
    if args.camera:
        camera_config = get_camera_config_decimal(args.camera)
        if camera_config:
            print(
                f"  Camera: {args.camera} at ({camera_config['lat']:.6f}, {camera_config['lon']:.6f})"
            )
        else:
            print(f"  Warning: Unknown camera '{args.camera}'")

    # Generate map
    title = f"GCP Verification - {os.path.basename(args.gcps)}"
    html = generate_map_html(gcps, camera_config, ptz_info, metadata, title)

    # Write output
    output_path = args.output or "gcp_map.html"
    with open(output_path, "w") as f:
        f.write(html)
    print(f"\nMap saved to: {output_path}")

    # Calculate statistics
    if camera_config and gcps:
        distances = [
            calculate_distance(camera_config["lat"], camera_config["lon"], g["lat"], g["lon"])
            for g in gcps
        ]
        bearings = [
            calculate_bearing(camera_config["lat"], camera_config["lon"], g["lat"], g["lon"])
            for g in gcps
        ]

        print("\nGCP Statistics:")
        print(f"  Distance range: {min(distances):.1f}m - {max(distances):.1f}m")
        print(f"  Bearing range: {min(bearings):.1f}° - {max(bearings):.1f}°")

        if ptz_info and "pan" in ptz_info:
            expected_bearing = ptz_info["pan"] + camera_config.get("pan_offset_deg", 0)
            print(f"  Expected center bearing: {expected_bearing:.1f}°")

    # Open in browser
    if not args.no_browser:
        abs_path = os.path.abspath(output_path)
        print("\nOpening in browser...")
        webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()
