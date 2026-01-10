"""
Calibration tool to verify and fix projection parameters.

This tool helps identify which parameter is causing projection misalignment:
1. Pan offset
2. Camera height
3. Focal length / intrinsics

The tool will:
1. Take a known Map Point that you've manually marked in the image
2. Calculate what parameters would make that projection correct
3. Compare against current parameters to identify the error
"""

import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poc_homography.camera_config import get_camera_configs
from poc_homography.camera_geometry import CameraGeometry


def dms_to_dd(dms_str: str) -> float:
    """
    Convert DMS (degrees, minutes, seconds) string to decimal degrees.

    Supports formats like:
    - "39°38'25.72\"N"
    - "0°13'48.63\"W"

    Args:
        dms_str: DMS coordinate string

    Returns:
        Decimal degrees (negative for S/W)
    """
    # Pattern to match DMS format
    pattern = r"""(\d+)°(\d+)'([\d.]+)"?([NSEW])"""
    match = re.match(pattern, dms_str)
    if not match:
        raise ValueError(f"Invalid DMS format: {dms_str}")

    degrees = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    direction = match.group(4)

    dd = degrees + minutes / 60 + seconds / 3600

    # Negative for South and West
    if direction in ("S", "W"):
        dd = -dd

    return dd


# Load camera configs from canonical source and convert DMS to decimal degrees
def _convert_camera_config(cam):
    """Convert camera config from DMS format to decimal degrees for this tool."""
    return {
        "lat": dms_to_dd(cam["lat"]),
        "lon": dms_to_dd(cam["lon"]),
        "height_m": cam["height_m"],
        "pan_offset_deg": cam["pan_offset_deg"],
    }


CAMERA_CONFIGS = {cam["name"]: _convert_camera_config(cam) for cam in get_camera_configs()}


def analyze_projection_error(
    camera_config: dict,
    map_point_x: float,
    map_point_y: float,
    actual_u: float,
    actual_v: float,
    pan_raw: float,
    tilt_deg: float,
    zoom: float,
    image_width: int = 1920,
    image_height: int = 1080,
):
    """
    Analyze the projection error for a known reference point.

    Args:
        camera_config: Camera configuration dictionary
        map_point_x: X coordinate from MapPoint (world/local coordinate)
        map_point_y: Y coordinate from MapPoint (world/local coordinate)
        actual_u: Actual image pixel U coordinate where user clicked
        actual_v: Actual image pixel V coordinate where user clicked
        pan_raw: Raw pan value from camera PTZ
        tilt_deg: Tilt angle in degrees
        zoom: Zoom factor
        image_width: Image width in pixels
        image_height: Image height in pixels
    """
    camera_lat = camera_config["lat"]
    camera_lon = camera_config["lon"]
    height_m = camera_config["height_m"]
    pan_offset_deg = camera_config["pan_offset_deg"]

    # Current pan calculation
    pan_deg = pan_raw + pan_offset_deg

    print("\n" + "=" * 70)
    print("PROJECTION ERROR ANALYSIS")
    print("=" * 70)

    print("\nReference Point:")
    print(f"  Map coordinates: ({map_point_x:.2f}, {map_point_y:.2f})")
    print(f"  Actual pixel (marked by user): ({actual_u:.1f}, {actual_v:.1f})")

    print("\nCamera Parameters:")
    print(f"  GPS: ({camera_lat:.6f}, {camera_lon:.6f})")
    print(f"  Height: {height_m}m")
    print(f"  Pan raw: {pan_raw}°, Offset: {pan_offset_deg}°, Applied: {pan_deg}°")
    print(f"  Tilt: {tilt_deg}°, Zoom: {zoom}x")

    # Use map point coordinates directly as local XY
    x_m, y_m = map_point_x, map_point_y
    distance = math.sqrt(x_m**2 + y_m**2)
    bearing = math.degrees(math.atan2(x_m, y_m))  # Bearing from camera to point

    print("\nLocal Coordinates:")
    print(f"  X (East): {x_m:.2f}m")
    print(f"  Y (North): {y_m:.2f}m")
    print(f"  Distance: {distance:.2f}m")
    print(f"  Bearing from camera: {bearing:.1f}°")

    # Get intrinsics
    K = CameraGeometry.get_intrinsics(zoom, image_width, image_height, 7.18)

    # Project with current parameters
    geo = CameraGeometry(w=image_width, h=image_height)
    w_pos = np.array([0.0, 0.0, height_m])
    geo.set_camera_parameters(K, w_pos, pan_deg, tilt_deg, 640, 640)

    world_pt = np.array([[x_m], [y_m], [1.0]])
    img_pt = geo.H @ world_pt
    if img_pt[2, 0] > 0:
        projected_u = img_pt[0, 0] / img_pt[2, 0]
        projected_v = img_pt[1, 0] / img_pt[2, 0]
    else:
        projected_u, projected_v = None, None

    print("\nProjection Result:")
    if projected_u is not None:
        print(f"  Current projection: ({projected_u:.1f}, {projected_v:.1f})")
        error_u = actual_u - projected_u
        error_v = actual_v - projected_v
        error_dist = math.sqrt(error_u**2 + error_v**2)
        print(f"  Actual pixel: ({actual_u:.1f}, {actual_v:.1f})")
        print(f"  Error: ({error_u:.1f}, {error_v:.1f}) = {error_dist:.1f} pixels")
    else:
        print("  Point projects behind camera!")

    # Now analyze what parameters would fix the projection
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP ANALYSIS")
    print("=" * 70)

    # Sweep pan offset
    print("\n1. Pan Offset Sweep (finding best match):")
    best_pan_error = float("inf")
    best_pan_offset = pan_offset_deg

    for test_offset in np.arange(-180, 180, 1):
        test_pan = pan_raw + test_offset
        geo.set_camera_parameters(K, w_pos, test_pan, tilt_deg, 640, 640)
        img_pt = geo.H @ world_pt
        if img_pt[2, 0] > 0:
            u = img_pt[0, 0] / img_pt[2, 0]
            v = img_pt[1, 0] / img_pt[2, 0]
            error = math.sqrt((actual_u - u) ** 2 + (actual_v - v) ** 2)
            if error < best_pan_error:
                best_pan_error = error
                best_pan_offset = test_offset

    print(f"  Current pan offset: {pan_offset_deg}°")
    print(f"  Best pan offset: {best_pan_offset}°")
    print(f"  Best error at this offset: {best_pan_error:.1f} pixels")
    if abs(best_pan_offset - pan_offset_deg) > 5:
        print(f"  --> SUGGESTION: Change pan_offset_deg from {pan_offset_deg} to {best_pan_offset}")

    # Sweep height
    print("\n2. Height Sweep (at current pan offset):")
    best_height_error = float("inf")
    best_height = height_m

    for test_height in np.arange(1.0, 20.0, 0.1):
        test_w_pos = np.array([0.0, 0.0, test_height])
        geo.set_camera_parameters(K, test_w_pos, pan_deg, tilt_deg, 640, 640)
        img_pt = geo.H @ world_pt
        if img_pt[2, 0] > 0:
            u = img_pt[0, 0] / img_pt[2, 0]
            v = img_pt[1, 0] / img_pt[2, 0]
            error = math.sqrt((actual_u - u) ** 2 + (actual_v - v) ** 2)
            if error < best_height_error:
                best_height_error = error
                best_height = test_height

    print(f"  Current height: {height_m}m")
    print(f"  Best height: {best_height:.2f}m")
    print(f"  Best error at this height: {best_height_error:.1f} pixels")
    if abs(best_height - height_m) > 0.5:
        print(f"  --> SUGGESTION: Change height_m from {height_m} to {best_height:.1f}")

    # Sweep both pan offset and height together
    print("\n3. Joint Pan+Height Optimization:")
    best_joint_error = float("inf")
    best_joint_pan = pan_offset_deg
    best_joint_height = height_m

    for test_offset in np.arange(-180, 180, 5):  # Coarser grid for speed
        test_pan = pan_raw + test_offset
        for test_height in np.arange(1.0, 20.0, 0.5):
            test_w_pos = np.array([0.0, 0.0, test_height])
            try:
                geo.set_camera_parameters(K, test_w_pos, test_pan, tilt_deg, 640, 640)
            except ValueError:
                continue
            img_pt = geo.H @ world_pt
            if img_pt[2, 0] > 0:
                u = img_pt[0, 0] / img_pt[2, 0]
                v = img_pt[1, 0] / img_pt[2, 0]
                error = math.sqrt((actual_u - u) ** 2 + (actual_v - v) ** 2)
                if error < best_joint_error:
                    best_joint_error = error
                    best_joint_pan = test_offset
                    best_joint_height = test_height

    print(f"  Best pan offset: {best_joint_pan}°")
    print(f"  Best height: {best_joint_height:.2f}m")
    print(f"  Best error: {best_joint_error:.1f} pixels")

    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if best_joint_error < 50:
        print("\nTo fix projection, update camera_config.py:")
        print(f"  'height_m': {best_joint_height:.1f},")
        print(f"  'pan_offset_deg': {best_joint_pan:.1f},")
    else:
        print("\nCould not find parameters that reduce error below 50 pixels.")
        print(f"Best achievable error: {best_joint_error:.1f} pixels")
        print("\nPossible issues:")
        print("  - Intrinsic matrix may be wrong (focal length, sensor size)")
        print("  - Camera GPS position may be inaccurate")
        print("  - Tilt angle may be incorrectly reported by camera")
        print("  - Map point coordinates may be inaccurate")
