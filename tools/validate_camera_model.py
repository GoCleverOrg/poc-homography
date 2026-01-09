#!/usr/bin/env python3
"""
Simple camera model validation script.

Tests projection accuracy with a few known GCPs to validate that the
camera geometry model works correctly before running full calibration.

Usage:
    python validate_camera_model.py --camera Valte --gcps gcps.yaml
    python validate_camera_model.py --camera Valte --gcp LAT LON U V PAN TILT
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poc_homography.camera_config import get_camera_by_name_safe
from poc_homography.camera_geometry import CameraGeometry
from poc_homography.coordinate_converter import gps_to_local_xy
from poc_homography.gps_distance_calculator import dms_to_dd

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_gcps_from_yaml(yaml_path: str, image_height: int = 1080) -> list:
    """Load GCPs from YAML file.

    Handles coordinate system conversion:
    - 'image_v' format: V=0 at top (standard)
    - Legacy format (no coordinate_system): V=0 at bottom, needs conversion
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML required for YAML file loading")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    gcps = []
    coordinate_system = None

    # Handle different YAML formats
    if "gcps" in data:
        # Simple format: gcps: [{lat, lon, pixel_u, pixel_v, ...}]
        for gcp in data["gcps"]:
            gcps.append(
                {
                    "lat": gcp["lat"],
                    "lon": gcp["lon"],
                    "pixel_u": gcp["pixel_u"],
                    "pixel_v": gcp["pixel_v"],
                    "pan": gcp.get("pan_raw", 0.0),
                    "tilt": gcp.get("tilt_deg", 30.0),
                    "zoom": gcp.get("zoom", 1.0),
                    "name": gcp.get("name", "GCP"),
                }
            )
    elif "homography" in data:
        # Complex format from capture tool
        ctx = data["homography"]["feature_match"]["camera_capture_context"]
        ptz = ctx["ptz_position"]
        coordinate_system = ctx.get("coordinate_system")  # 'image_v' or None (legacy)

        for gcp in data["homography"]["feature_match"]["ground_control_points"]:
            v = gcp["image"]["v"]

            # Convert legacy leaflet_y format to image_v
            if coordinate_system is None:
                # Legacy: V was stored as leaflet_y (0 at bottom)
                # Convert to image_v (0 at top): v = image_height - leaflet_y
                v = image_height - v

            gcps.append(
                {
                    "lat": gcp["gps"]["latitude"],
                    "lon": gcp["gps"]["longitude"],
                    "pixel_u": gcp["image"]["u"],
                    "pixel_v": v,
                    "pan": ptz["pan"],
                    "tilt": ptz["tilt"],
                    "zoom": ptz.get("zoom", 1.0),
                    "name": gcp.get("metadata", {}).get("description", "GCP"),
                }
            )

        if coordinate_system is None:
            print(f"Note: Converted {len(gcps)} GCPs from legacy leaflet_y to image_v format")

    return gcps


def project_gps_to_pixel(
    lat: float,
    lon: float,
    camera_lat: float,
    camera_lon: float,
    camera_height: float,
    pan_deg: float,
    tilt_deg: float,
    zoom: float,
    pan_offset_deg: float = 0.0,
    tilt_offset_deg: float = 0.0,
    focal_multiplier: float = 1.0,
    sensor_width_mm: float = 6.78,
    k1: float = 0.0,
    k2: float = 0.0,
    image_width: int = 1920,
    image_height: int = 1080,
) -> tuple:
    """
    Project GPS coordinate to pixel using camera model.

    Returns (u, v, success, error_msg)
    """
    try:
        # Convert GPS to local XY (meters from camera)
        x_m, y_m = gps_to_local_xy(camera_lat, camera_lon, lat, lon)

        # Build intrinsic matrix
        effective_focal_mm = 5.9 * zoom * focal_multiplier
        f_px = effective_focal_mm * (image_width / sensor_width_mm)
        cx, cy = image_width / 2.0, image_height / 2.0
        K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]])

        # Compute pan and tilt with offsets
        actual_pan = pan_deg + pan_offset_deg
        actual_tilt = tilt_deg + tilt_offset_deg

        # Create geometry
        geo = CameraGeometry(w=image_width, h=image_height)
        w_pos = np.array([0.0, 0.0, camera_height])

        geo.set_camera_parameters(K, w_pos, actual_pan, actual_tilt, 640, 640)

        # Set distortion if provided
        if k1 != 0.0 or k2 != 0.0:
            geo.set_distortion_coefficients(k1=k1, k2=k2)

        # Project world point to image
        world_pt = np.array([[x_m], [y_m], [1.0]])
        img_pt = geo.H @ world_pt

        if img_pt[2, 0] <= 0:
            return None, None, False, "Point behind camera"

        u = img_pt[0, 0] / img_pt[2, 0]
        v = img_pt[1, 0] / img_pt[2, 0]

        # Apply distortion if set
        if k1 != 0.0 or k2 != 0.0:
            u, v = geo.distort_point(u, v)

        return u, v, True, None

    except Exception as e:
        return None, None, False, str(e)


def validate_model(camera_name: str, gcps: list, verbose: bool = True):
    """
    Validate camera model with GCPs.

    Returns (mean_error, individual_errors)
    """
    # Get camera config
    cam_config = get_camera_by_name_safe(camera_name)
    if not cam_config:
        print(f"Error: Unknown camera '{camera_name}'")
        return None, []

    # Convert DMS coordinates to decimal degrees
    camera_lat = dms_to_dd(cam_config["lat"])
    camera_lon = dms_to_dd(cam_config["lon"])
    camera_height = cam_config.get("height_m", 5.0)
    pan_offset = cam_config.get("pan_offset_deg", 0.0)
    tilt_offset = cam_config.get("tilt_offset_deg", 0.0)
    focal_mult = cam_config.get("focal_multiplier", 1.0)
    k1 = cam_config.get("k1", 0.0)
    k2 = cam_config.get("k2", 0.0)

    if verbose:
        print("\n" + "=" * 70)
        print("CAMERA MODEL VALIDATION")
        print("=" * 70)
        print(f"\nCamera: {camera_name}")
        print(f"  Position: {camera_lat:.6f}, {camera_lon:.6f}")
        print(f"  Height: {camera_height:.2f}m")
        print(f"  Pan offset: {pan_offset:.1f}°")
        print(f"  Tilt offset: {tilt_offset:+.2f}°")
        print(f"  Focal multiplier: {focal_mult:.4f}")
        print(f"  Distortion: k1={k1:.4f}, k2={k2:.4f}")
        print(f"\nNumber of GCPs: {len(gcps)}")
        print("-" * 70)

    errors = []
    results = []

    for i, gcp in enumerate(gcps):
        proj_u, proj_v, success, err_msg = project_gps_to_pixel(
            gcp["lat"],
            gcp["lon"],
            camera_lat,
            camera_lon,
            camera_height,
            gcp["pan"],
            gcp["tilt"],
            gcp["zoom"],
            pan_offset,
            tilt_offset,
            focal_mult,
            6.78,
            k1,
            k2,
        )

        if success:
            error = math.sqrt((proj_u - gcp["pixel_u"]) ** 2 + (proj_v - gcp["pixel_v"]) ** 2)
            errors.append(error)

            if verbose:
                status = "✓" if error < 5 else "✗"
                print(
                    f"GCP {i + 1}: {gcp['name'][:20]:20s} | "
                    f"Expected: ({gcp['pixel_u']:7.1f}, {gcp['pixel_v']:7.1f}) | "
                    f"Projected: ({proj_u:7.1f}, {proj_v:7.1f}) | "
                    f"Error: {error:6.1f}px {status}"
                )

            results.append({"gcp": gcp, "projected": (proj_u, proj_v), "error": error})
        else:
            if verbose:
                print(f"GCP {i + 1}: {gcp['name'][:20]:20s} | FAILED: {err_msg}")
            errors.append(1000.0)

    if errors:
        mean_error = np.mean([e for e in errors if e < 1000])
        if verbose:
            print("-" * 70)
            print(f"\nMean error: {mean_error:.2f} pixels")
            good = sum(1 for e in errors if e < 5)
            print(f"GCPs with <5px error: {good}/{len(errors)}")

            if mean_error < 5:
                print("\n✓ Camera model is well-calibrated!")
            elif mean_error < 20:
                print("\n⚠ Camera model needs minor calibration adjustment")
            else:
                print("\n✗ Camera model needs significant calibration")
                print("  Possible causes:")
                print("  - Incorrect camera GPS position")
                print("  - Incorrect pan offset")
                print("  - GCP coordinates may be inaccurate")

        return mean_error, results

    return None, []


def main():
    parser = argparse.ArgumentParser(description="Validate camera projection model")
    parser.add_argument("--camera", "-c", required=True, help="Camera name (e.g., Valte)")
    parser.add_argument("--gcps", "-g", help="Path to YAML file with GCPs")
    parser.add_argument(
        "--gcp",
        "-p",
        action="append",
        nargs=6,
        metavar=("LAT", "LON", "U", "V", "PAN", "TILT"),
        help="Single GCP: LAT LON PIXEL_U PIXEL_V PAN_DEG TILT_DEG",
    )
    parser.add_argument("--zoom", type=float, default=1.0, help="Zoom factor (default: 1.0)")

    args = parser.parse_args()

    gcps = []

    if args.gcps:
        gcps = load_gcps_from_yaml(args.gcps)

    if args.gcp:
        for gcp_args in args.gcp:
            lat, lon, u, v, pan, tilt = map(float, gcp_args)
            gcps.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "pixel_u": u,
                    "pixel_v": v,
                    "pan": pan,
                    "tilt": tilt,
                    "zoom": args.zoom,
                    "name": f"Manual ({lat:.4f}, {lon:.4f})",
                }
            )

    if not gcps:
        print("Error: No GCPs provided. Use --gcps FILE or --gcp LAT LON U V PAN TILT")
        sys.exit(1)

    validate_model(args.camera, gcps)


if __name__ == "__main__":
    main()
