#!/usr/bin/env python3
"""Convert KML file with geographic coordinates to map_points.json with pixel coordinates only.

This tool reads a KML file, extracts pixel coordinates using georeferencing configuration,
and outputs a JSON file containing only pixel coordinates and point IDs.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from poc_homography.kml import GeoConfig, GeoPointRegistry, Kml
from poc_homography.map_points.kml_converter import convert_geo_registry_to_map_points


def main() -> None:
    """Main entry point for KML to map points conversion."""
    parser = argparse.ArgumentParser(
        description="Convert KML file to map_points.json format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python convert_kml_to_map_points.py \\
    --kml Cartografia_valencia_recreated.kml \\
    --output map_points.json \\
    --map-id map_valte \\
    --crs EPSG:25830 \\
    --geotransform 725140.0 0.05 0.0 4373490.0 0.0 -0.05
        """,
    )

    parser.add_argument(
        "--kml",
        required=True,
        type=Path,
        help="Path to input KML file with geographic coordinates",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to output JSON file (e.g., map_points.json)",
    )
    parser.add_argument(
        "--map-id",
        required=True,
        type=str,
        help='Map identifier (e.g., "map_valte")',
    )
    parser.add_argument(
        "--crs",
        required=True,
        type=str,
        help="Coordinate reference system (e.g., EPSG:25830)",
    )
    parser.add_argument(
        "--geotransform",
        required=True,
        type=float,
        nargs=6,
        metavar=("X0", "DX", "RX", "Y0", "RY", "DY"),
        help="GDAL geotransform parameters: origin_x pixel_width row_rotation origin_y col_rotation pixel_height",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.kml.exists():
        print(f"Error: KML file not found: {args.kml}", file=sys.stderr)
        sys.exit(1)

    # Read and parse KML
    print(f"Reading KML file: {args.kml}")
    kml_text = args.kml.read_text(encoding="utf-8")
    kml_doc = Kml(kml_text)

    print(f"Found {len(kml_doc.points)} points in KML")

    # Create GeoConfig from command line arguments
    geo_config = GeoConfig(
        crs=args.crs,
        geotransform=tuple(args.geotransform),  # type: ignore
    )

    # Create GeoPointRegistry to get pixel coordinates
    print(f"Converting geographic coordinates to pixels using {args.crs}")
    geo_registry = GeoPointRegistry.from_kml_points(geo_config, kml_doc.points)

    # Convert to map points (pixel only)
    print(f"Extracting pixel coordinates for map_id: {args.map_id}")
    map_registry = convert_geo_registry_to_map_points(geo_registry, args.map_id)

    # Save to JSON
    print(f"Saving to: {args.output}")
    map_registry.save(args.output)

    print(f"✓ Successfully converted {len(map_registry.points)} points")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
