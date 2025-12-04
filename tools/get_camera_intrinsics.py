#!/usr/bin/env python3
"""
Get current camera intrinsics from PTZ status.

This utility queries a camera's current PTZ position and computes the
intrinsic matrix based on the zoom level and sensor parameters.

Usage:
    python tools/get_camera_intrinsics.py Valte
    python tools/get_camera_intrinsics.py Setram --json
    python tools/get_camera_intrinsics.py Valte --yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests
from requests.auth import HTTPDigestAuth
import xml.etree.ElementTree as ET

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from poc_homography.camera_config import (
        get_camera_by_name, USERNAME, PASSWORD, CAMERAS
    )
    CAMERA_CONFIG_AVAILABLE = True
except (ValueError, ImportError) as e:
    CAMERA_CONFIG_AVAILABLE = False
    print(f"Error: Camera config not available ({e})")
    print("Set CAMERA_USERNAME and CAMERA_PASSWORD environment variables.")
    sys.exit(1)

# Default camera parameters (can be overridden via CLI)
DEFAULT_SENSOR_WIDTH_MM = 7.18
DEFAULT_BASE_FOCAL_LENGTH_MM = 5.9
DEFAULT_IMAGE_WIDTH = 2560
DEFAULT_IMAGE_HEIGHT = 1440


def get_ptz_status(ip: str, username: str, password: str, timeout: float = 5.0) -> dict:
    """
    Get current PTZ status from camera.

    Args:
        ip: Camera IP address
        username: Camera username
        password: Camera password
        timeout: Request timeout in seconds

    Returns:
        Dictionary with 'pan', 'tilt', 'zoom' keys

    Raises:
        RuntimeError: If cannot connect or parse response
    """
    url = f"http://{ip}/ISAPI/PTZCtrl/channels/1/status"

    try:
        response = requests.get(
            url,
            auth=HTTPDigestAuth(username, password),
            timeout=timeout
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to connect to camera: {e}")

    if response.status_code != 200:
        raise RuntimeError(f"Camera returned status {response.status_code}")

    try:
        root = ET.fromstring(response.text)
        ns = {'h': 'http://www.hikvision.com/ver20/XMLSchema'}

        azimuth = root.find('.//h:azimuth', ns)
        elevation = root.find('.//h:elevation', ns)
        abs_zoom = root.find('.//h:absoluteZoom', ns)

        if azimuth is None or elevation is None or abs_zoom is None:
            raise RuntimeError("Failed to parse PTZ status XML")

        return {
            'pan': float(azimuth.text) / 10,
            'tilt': float(elevation.text) / 10,
            'zoom': float(abs_zoom.text) / 10,
        }
    except ET.ParseError as e:
        raise RuntimeError(f"Failed to parse XML response: {e}")


def compute_intrinsics(
    zoom: float,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM,
    base_focal_length_mm: float = DEFAULT_BASE_FOCAL_LENGTH_MM
) -> dict:
    """
    Compute camera intrinsics from zoom and sensor parameters.

    Args:
        zoom: Zoom factor (1.0 = no zoom)
        image_width: Image width in pixels
        image_height: Image height in pixels
        sensor_width_mm: Sensor width in millimeters
        base_focal_length_mm: Base focal length in millimeters (at 1x zoom)

    Returns:
        Dictionary with intrinsic parameters and matrix
    """
    # Compute focal length
    f_mm = base_focal_length_mm * zoom
    f_px = f_mm * (image_width / sensor_width_mm)

    # Principal point (image center)
    cx = image_width / 2.0
    cy = image_height / 2.0

    # Intrinsic matrix
    K = np.array([
        [f_px, 0, cx],
        [0, f_px, cy],
        [0, 0, 1]
    ])

    return {
        'focal_length_mm': f_mm,
        'focal_length_px': f_px,
        'principal_point': {'cx': cx, 'cy': cy},
        'sensor_width_mm': sensor_width_mm,
        'base_focal_length_mm': base_focal_length_mm,
        'image_width': image_width,
        'image_height': image_height,
        'K': K,
    }


def get_camera_intrinsics(
    camera_name: str,
    image_width: int = None,
    image_height: int = None,
    sensor_width_mm: float = None,
    base_focal_length_mm: float = None
) -> dict:
    """
    Get current intrinsics for a camera.

    Args:
        camera_name: Camera name from camera_config
        image_width: Override image width (default: 2560)
        image_height: Override image height (default: 1440)
        sensor_width_mm: Override sensor width (default: 7.18)
        base_focal_length_mm: Override base focal length (default: 5.9)

    Returns:
        Dictionary with PTZ status and computed intrinsics
    """
    cam_info = get_camera_by_name(camera_name)
    if not cam_info:
        available = [c['name'] for c in CAMERAS]
        raise ValueError(
            f"Camera '{camera_name}' not found. "
            f"Available cameras: {', '.join(available)}"
        )

    # Get PTZ status
    ptz_status = get_ptz_status(cam_info['ip'], USERNAME, PASSWORD)

    # Compute intrinsics
    intrinsics = compute_intrinsics(
        zoom=ptz_status['zoom'],
        image_width=image_width or DEFAULT_IMAGE_WIDTH,
        image_height=image_height or DEFAULT_IMAGE_HEIGHT,
        sensor_width_mm=sensor_width_mm or DEFAULT_SENSOR_WIDTH_MM,
        base_focal_length_mm=base_focal_length_mm or DEFAULT_BASE_FOCAL_LENGTH_MM,
    )

    return {
        'camera_name': camera_name,
        'camera_ip': cam_info['ip'],
        'ptz': ptz_status,
        'intrinsics': intrinsics,
    }


def format_human_readable(result: dict) -> str:
    """Format result for human-readable output."""
    ptz = result['ptz']
    intr = result['intrinsics']
    K = intr['K']

    lines = [
        "=" * 60,
        f"{result['camera_name'].upper()} CAMERA - Current Status & Intrinsics",
        "=" * 60,
        "",
        "📍 PTZ Position:",
        f"  Pan (azimuth):    {ptz['pan']:.1f}°",
        f"  Tilt (elevation): {ptz['tilt']:.1f}° (Hikvision: positive = down)",
        f"  Zoom factor:      {ptz['zoom']:.1f}x",
        "",
        "📷 Camera Intrinsics:",
        f"  Sensor width:       {intr['sensor_width_mm']} mm",
        f"  Base focal length:  {intr['base_focal_length_mm']} mm",
        f"  Focal length (mm):  {intr['focal_length_mm']:.2f} mm (at {ptz['zoom']:.1f}x zoom)",
        f"  Focal length (px):  {intr['focal_length_px']:.2f} px",
        f"  Principal point:    ({intr['principal_point']['cx']:.1f}, {intr['principal_point']['cy']:.1f})",
        "",
        "📐 Intrinsic Matrix K:",
        f"  [{K[0,0]:10.2f}  {K[0,1]:10.2f}  {K[0,2]:10.2f}]",
        f"  [{K[1,0]:10.2f}  {K[1,1]:10.2f}  {K[1,2]:10.2f}]",
        f"  [{K[2,0]:10.2f}  {K[2,1]:10.2f}  {K[2,2]:10.2f}]",
        "",
        "📋 For homography_config.yaml camera_capture_context:",
        "  ptz_position:",
        f"    pan: {ptz['pan']:.1f}",
        f"    tilt: {ptz['tilt']:.1f}",
        f"    zoom: {ptz['zoom']:.1f}",
        "  intrinsics:",
        f"    focal_length_px: {intr['focal_length_px']:.2f}",
        "    principal_point:",
        f"      cx: {intr['principal_point']['cx']:.1f}",
        f"      cy: {intr['principal_point']['cy']:.1f}",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def format_json(result: dict) -> str:
    """Format result as JSON."""
    # Convert numpy array to list for JSON serialization
    output = {
        'camera_name': result['camera_name'],
        'camera_ip': result['camera_ip'],
        'ptz': result['ptz'],
        'intrinsics': {
            k: v for k, v in result['intrinsics'].items()
            if k != 'K'
        },
        'K': result['intrinsics']['K'].tolist(),
    }
    return json.dumps(output, indent=2)


def format_yaml(result: dict) -> str:
    """Format result as YAML snippet for config file."""
    ptz = result['ptz']
    intr = result['intrinsics']

    lines = [
        "# Camera capture context for homography_config.yaml",
        "camera_capture_context:",
        f"  camera_name: \"{result['camera_name']}\"",
        f"  image_width: {intr['image_width']}",
        f"  image_height: {intr['image_height']}",
        "  ptz_position:",
        f"    pan: {ptz['pan']:.1f}",
        f"    tilt: {ptz['tilt']:.1f}",
        f"    zoom: {ptz['zoom']:.1f}",
        "  intrinsics:",
        f"    focal_length_px: {intr['focal_length_px']:.2f}",
        "    principal_point:",
        f"      cx: {intr['principal_point']['cx']:.1f}",
        f"      cy: {intr['principal_point']['cy']:.1f}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Get current camera intrinsics from PTZ status',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'camera',
        type=str,
        help='Camera name (e.g., Valte, Setram)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--yaml',
        action='store_true',
        help='Output as YAML snippet for config file'
    )
    parser.add_argument(
        '--width', '-W',
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help=f'Image width in pixels (default: {DEFAULT_IMAGE_WIDTH})'
    )
    parser.add_argument(
        '--height', '-H',
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help=f'Image height in pixels (default: {DEFAULT_IMAGE_HEIGHT})'
    )
    parser.add_argument(
        '--sensor-width',
        type=float,
        default=DEFAULT_SENSOR_WIDTH_MM,
        help=f'Sensor width in mm (default: {DEFAULT_SENSOR_WIDTH_MM})'
    )
    parser.add_argument(
        '--base-focal',
        type=float,
        default=DEFAULT_BASE_FOCAL_LENGTH_MM,
        help=f'Base focal length in mm at 1x zoom (default: {DEFAULT_BASE_FOCAL_LENGTH_MM})'
    )
    parser.add_argument(
        '--list-cameras',
        action='store_true',
        help='List available cameras and exit'
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


if __name__ == '__main__':
    main()
