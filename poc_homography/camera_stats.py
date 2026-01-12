#!/usr/bin/env python3
"""
Camera stats utility.

Retrieve and display camera statistics based on camera name.
Can also fetch live PTZ status from the camera API.

Usage:
    python -m poc_homography.camera_stats <camera_name>
    python -m poc_homography.camera_stats Valte
    python -m poc_homography.camera_stats Valte --live
    python -m poc_homography.camera_stats --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from poc_homography.camera_config import get_camera_by_name, get_camera_configs
from poc_homography.camera.intrinsics import get_ptz_status, PTZStatus


def fetch_live_ptz_status(ip: str, timeout: float = 5.0) -> PTZStatus | None:
    """
    Fetch live PTZ status from camera.

    Args:
        ip: Camera IP address
        timeout: Request timeout in seconds

    Returns:
        PTZStatus or None if credentials not set or connection failed
    """
    username = os.getenv("CAMERA_USERNAME")
    password = os.getenv("CAMERA_PASSWORD")

    if not username or not password:
        return None

    try:
        return get_ptz_status(ip, username, password, timeout)
    except RuntimeError:
        return None


def get_camera_stats(camera_name: str, include_live: bool = False) -> dict[str, Any] | None:
    """
    Get all stats for a camera by name.

    Args:
        camera_name: Name of the camera (e.g., "Valte", "Setram")
        include_live: If True, fetch live PTZ status from camera API

    Returns:
        Dictionary with camera stats or None if not found
    """
    camera = get_camera_by_name(camera_name)
    if camera is None:
        return None

    # Organize stats into categories
    stats = {
        "name": camera.get("name"),
        "model": camera.get("model"),
        "ip": camera.get("ip"),
        "description": camera.get("description"),
        "location": {
            "latitude": camera.get("lat"),
            "longitude": camera.get("lon"),
            "height_m": camera.get("height_m"),
        },
        "calibration": {
            "pan_offset_deg": camera.get("pan_offset_deg"),
            "tilt_offset_deg": camera.get("tilt_offset_deg"),
        },
        "distortion": {
            "k1": camera.get("k1"),
            "k2": camera.get("k2"),
            "p1": camera.get("p1"),
            "p2": camera.get("p2"),
        },
        "sensor": {
            "sensor_width_mm": camera.get("sensor_width_mm"),
            "base_focal_length_mm": camera.get("base_focal_length_mm"),
        },
        "geotiff_params": camera.get("geotiff_params"),
        "calibration_table": camera.get("calibration_table"),
        "live_status": None,
    }

    # Fetch live PTZ status if requested
    if include_live:
        ptz = fetch_live_ptz_status(camera["ip"])
        if ptz:
            stats["live_status"] = {
                "pan_deg": ptz.pan,
                "tilt_deg": ptz.tilt,
                "zoom": ptz.zoom,
            }

    return stats


def format_stats_human(stats: dict[str, Any]) -> str:
    """Format stats for human-readable output."""
    lines = []
    lines.append(f"Camera: {stats['name']}")
    lines.append("=" * 50)

    lines.append(f"\nModel: {stats['model']}")
    lines.append(f"IP: {stats['ip']}")
    lines.append(f"Description: {stats['description']}")

    lines.append("\nLocation:")
    loc = stats["location"]
    lines.append(f"  Latitude:  {loc['latitude']}")
    lines.append(f"  Longitude: {loc['longitude']}")
    lines.append(f"  Height:    {loc['height_m']} m")

    lines.append("\nCalibration Offsets:")
    cal = stats["calibration"]
    lines.append(f"  Pan offset:  {cal['pan_offset_deg']}°")
    lines.append(f"  Tilt offset: {cal['tilt_offset_deg']}°")

    lines.append("\nLens Distortion (OpenCV model):")
    dist = stats["distortion"]
    lines.append(f"  k1: {dist['k1']}")
    lines.append(f"  k2: {dist['k2']}")
    lines.append(f"  p1: {dist['p1']}")
    lines.append(f"  p2: {dist['p2']}")

    lines.append("\nSensor Parameters:")
    sensor = stats["sensor"]
    lines.append(f"  Sensor width:      {sensor['sensor_width_mm']} mm")
    lines.append(f"  Base focal length: {sensor['base_focal_length_mm']} mm")

    if stats["geotiff_params"]:
        lines.append("\nGeoTIFF Parameters:")
        geo = stats["geotiff_params"]
        lines.append(f"  GeoTransform: {geo.get('geotransform')}")
        lines.append(f"  UTM CRS:      {geo.get('utm_crs')}")

    if stats["calibration_table"]:
        lines.append("\nCalibration Table:")
        lines.append(f"  Zoom levels: {list(stats['calibration_table'].keys())}")
    else:
        lines.append("\nCalibration Table: Not configured (using linear approximation)")

    # Live PTZ status
    if stats.get("live_status"):
        live = stats["live_status"]
        lines.append("\nLive PTZ Status:")
        lines.append(f"  Pan:  {live['pan_deg']:.2f}°")
        lines.append(f"  Tilt: {live['tilt_deg']:.2f}°")
        lines.append(f"  Zoom: {live['zoom']:.1f}x")

    return "\n".join(lines)


def list_cameras() -> list[str]:
    """Get list of all available camera names."""
    return [cam["name"] for cam in get_camera_configs()]


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Get camera statistics by name",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s Valte              # Show stats for Valte camera
    %(prog)s Valte --live       # Show stats with live PTZ status
    %(prog)s Setram --json      # Show stats as JSON
    %(prog)s --list             # List all available cameras
        """,
    )
    parser.add_argument(
        "camera_name",
        nargs="?",
        help="Name of the camera to get stats for",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available camera names",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output stats as JSON",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch live PTZ status from camera (requires CAMERA_USERNAME and CAMERA_PASSWORD env vars)",
    )

    args = parser.parse_args()

    if args.list:
        cameras = list_cameras()
        print("Available cameras:")
        for name in cameras:
            print(f"  - {name}")
        return 0

    if not args.camera_name:
        parser.print_help()
        return 1

    stats = get_camera_stats(args.camera_name, include_live=args.live)

    if stats is None:
        available = list_cameras()
        print(f"Error: Camera '{args.camera_name}' not found.", file=sys.stderr)
        print(f"Available cameras: {', '.join(available)}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(format_stats_human(stats))
        # Show warning if live status was requested but failed
        if args.live and stats.get("live_status") is None:
            print(
                "\nWarning: Could not fetch live PTZ status. "
                "Check CAMERA_USERNAME/CAMERA_PASSWORD env vars and network connectivity.",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
