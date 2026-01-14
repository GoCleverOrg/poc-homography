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
import sys
from typing import Any

from poc_homography.application import ApplicationContext
from poc_homography.camera.intrinsics import PTZStatus, get_ptz_status


def fetch_live_ptz_status(
    ip: str, username: str, password: str, timeout: float = 5.0
) -> PTZStatus | None:
    """
    Fetch live PTZ status from camera.

    Args:
        ip: Camera IP address
        username: Camera username
        password: Camera password
        timeout: Request timeout in seconds

    Returns:
        PTZStatus or None if connection failed
    """
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
    ctx = ApplicationContext.default()

    all_configs = ctx.camera_config_repo.get_all()
    camera = next((c for c in all_configs if c.name == camera_name), None)
    if camera is None:
        return None

    # Get calibration data
    calibration = ctx.camera_calibration_repo.get(camera.id)

    # Build stats from domain entities
    stats: dict[str, Any] = {
        "name": camera.name,
        "model": camera.spec.model_name,
        "ip": camera.ip_address,
        "map_id": camera.map_id,
        "sensor": {
            "sensor_width_mm": float(camera.spec.sensor_width),
            "base_focal_length_mm": float(camera.spec.base_focal_length),
            "image_width": int(camera.spec.image_width),
            "image_height": int(camera.spec.image_height),
            "max_zoom": camera.spec.max_zoom,
        },
        "live_status": None,
    }

    # Add calibration data if available
    if calibration:
        stats["position"] = {
            "x": float(calibration.position.x),
            "y": float(calibration.position.y),
        }
        stats["height_m"] = float(calibration.height)
        stats["calibration"] = {
            "pan_offset_deg": calibration.base_orientation.yaw,
            "tilt_offset_deg": calibration.base_orientation.pitch,
        }
        stats["distortion"] = {
            "k1": calibration.distortion.k1,
            "k2": calibration.distortion.k2,
            "p1": calibration.distortion.p1,
            "p2": calibration.distortion.p2,
        }
    else:
        stats["position"] = None
        stats["height_m"] = None
        stats["calibration"] = None
        stats["distortion"] = None

    # Fetch live PTZ status if requested
    if include_live and camera.ip_address:
        ptz = fetch_live_ptz_status(
            camera.ip_address,
            camera.credential.username,
            camera.credential.password,
        )
        if ptz:
            stats["live_status"] = {
                "pan_deg": float(ptz.pan),
                "tilt_deg": float(ptz.tilt),
                "zoom": float(ptz.zoom),
            }

    return stats


def format_stats_human(stats: dict[str, Any]) -> str:
    """Format stats for human-readable output."""
    lines = []
    lines.append(f"Camera: {stats['name']}")
    lines.append("=" * 50)

    lines.append(f"\nModel: {stats['model']}")
    lines.append(f"IP: {stats['ip']}")
    lines.append(f"Map ID: {stats['map_id']}")

    if stats.get("position"):
        lines.append("\nPosition on Map:")
        pos = stats["position"]
        lines.append(f"  X: {pos['x']}")
        lines.append(f"  Y: {pos['y']}")

    if stats.get("height_m"):
        lines.append(f"\nHeight: {stats['height_m']} m")

    if stats.get("calibration"):
        lines.append("\nCalibration Offsets:")
        cal = stats["calibration"]
        lines.append(f"  Pan offset:  {cal['pan_offset_deg']}deg")
        lines.append(f"  Tilt offset: {cal['tilt_offset_deg']}deg")

    if stats.get("distortion"):
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
    lines.append(f"  Image size:        {sensor['image_width']}x{sensor['image_height']}")
    lines.append(f"  Max zoom:          {sensor['max_zoom']}x")

    # Live PTZ status
    if stats.get("live_status"):
        live = stats["live_status"]
        lines.append("\nLive PTZ Status:")
        lines.append(f"  Pan:  {live['pan_deg']:.2f}deg")
        lines.append(f"  Tilt: {live['tilt_deg']:.2f}deg")
        lines.append(f"  Zoom: {live['zoom']:.1f}x")

    return "\n".join(lines)


def list_cameras() -> list[str]:
    """Get list of all available camera names."""
    ctx = ApplicationContext.default()
    return [cam.name for cam in ctx.camera_config_repo.get_all()]


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
        help="Fetch live PTZ status from camera (uses credentials from camera config)",
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
                "Check credentials in camera config and network connectivity.",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
