"""
Simple camera model validation script.

Tests projection accuracy with a few known GCPs to validate that the
camera geometry model works correctly before running full calibration.

GCPs are defined in YAML format with the new capture.context and
capture.annotations structure:

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

Usage:
    from tools.validate_camera_model import load_gcps_from_yaml, validate_model
    from poc_homography.map_points import MapPointRegistry

    context, annotations = load_gcps_from_yaml("gcps.yaml")
    registry = MapPointRegistry.load("map_points.json")
    mean_error, results = validate_model(context, annotations, registry)
"""

import math
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poc_homography.calibration.annotation import Annotation, CaptureContext
from poc_homography.camera_config import get_camera_by_name_safe
from poc_homography.camera_geometry import CameraGeometry
from poc_homography.map_points import MapPointRegistry
from poc_homography.pixel_point import PixelPoint

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_gcps_from_yaml(yaml_path: str) -> tuple[CaptureContext, list[Annotation]]:
    """Load capture context and annotations from YAML file.

    Supports both the new format (capture.context + capture.annotations) and
    the legacy format (gcps list) for backward compatibility.

    New YAML format:
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

    Legacy YAML format (deprecated):
        gcps:
          - map_point_id: Z1
            pixel_u: 960
            pixel_v: 540
            pan_raw: 0.0
            tilt_deg: 30.0
            zoom: 1.0

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        Tuple of (CaptureContext, list of Annotations).

    Raises:
        ImportError: If PyYAML is not available.
        ValueError: If YAML format is invalid.
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML required for YAML file loading")

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Check for new format first
    if "capture" in data:
        capture_data = data["capture"]

        if "context" not in capture_data:
            raise ValueError(
                "New format requires 'capture.context' with camera, pan_raw, tilt_deg, zoom"
            )

        if "annotations" not in capture_data:
            raise ValueError("New format requires 'capture.annotations' list")

        # Parse context
        context = CaptureContext.from_dict(capture_data["context"])

        # Parse annotations
        annotations = [Annotation.from_dict(ann_data) for ann_data in capture_data["annotations"]]

        return context, annotations

    # Fall back to legacy format
    if "gcps" in data:
        warnings.warn(
            "Legacy 'gcps' format is deprecated. Please migrate to new 'capture.context' "
            "and 'capture.annotations' format. See issue #165 for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )

        gcps = data["gcps"]

        if not gcps:
            raise ValueError("Legacy format: 'gcps' list is empty")

        # Extract context from first GCP (assumes all GCPs share same PTZ values)
        first_gcp = gcps[0]

        # Try to infer camera name from file or default to "Unknown"
        camera_name = first_gcp.get("camera", "Unknown")

        context = CaptureContext(
            camera=camera_name,
            pan_raw=first_gcp.get("pan_raw", 0.0),
            tilt_deg=first_gcp.get("tilt_deg", 30.0),
            zoom=first_gcp.get("zoom", 1.0),
        )

        # Convert GCPs to annotations
        annotations = [
            Annotation(
                gcp_id=gcp["map_point_id"],
                pixel=PixelPoint(
                    x=float(gcp["pixel_u"]),
                    y=float(gcp["pixel_v"]),
                ),
            )
            for gcp in gcps
        ]

        return context, annotations

    raise ValueError(
        "YAML file must contain either 'capture' (new format) or 'gcps' (legacy format)"
    )


def project_map_point_to_pixel(
    map_point_x: float,
    map_point_y: float,
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
    Project Map Point coordinate to image pixel using camera model.

    Args:
        map_point_x: X coordinate from MapPoint (world/local coordinate in meters)
        map_point_y: Y coordinate from MapPoint (world/local coordinate in meters)
        camera_height: Camera height in meters
        pan_deg: Pan angle in degrees
        tilt_deg: Tilt angle in degrees
        zoom: Zoom factor
        pan_offset_deg: Pan offset calibration in degrees
        tilt_offset_deg: Tilt offset calibration in degrees
        focal_multiplier: Focal length multiplier for calibration
        sensor_width_mm: Sensor width in millimeters
        k1: Radial distortion coefficient k1
        k2: Radial distortion coefficient k2
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        (u, v, success, error_msg) tuple where u,v are image pixel coordinates
    """
    try:
        # Use map point coordinates directly as local XY (meters from camera)
        x_m, y_m = map_point_x, map_point_y

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


def validate_model(
    context: CaptureContext,
    annotations: list[Annotation],
    registry: MapPointRegistry,
    verbose: bool = True,
):
    """
    Validate camera model with annotations using Map Point coordinates.

    Args:
        context: CaptureContext containing camera name and PTZ values.
        annotations: List of Annotation instances with gcp_id and pixel coordinates.
        registry: MapPointRegistry containing the map point coordinates.
        verbose: Whether to print detailed output.

    Returns:
        (mean_error, individual_errors) tuple
    """
    # Get camera config
    cam_config = get_camera_by_name_safe(context.camera)
    if not cam_config:
        print(f"Error: Unknown camera '{context.camera}'")
        return None, []

    # Extract camera parameters
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
        print(f"\nCamera: {context.camera}")
        print(f"  Height: {camera_height:.2f}m")
        print(f"  Pan offset: {pan_offset:.1f}deg")
        print(f"  Tilt offset: {tilt_offset:+.2f}deg")
        print(f"  Focal multiplier: {focal_mult:.4f}")
        print(f"  Distortion: k1={k1:.4f}, k2={k2:.4f}")
        print("\nCapture PTZ:")
        print(f"  Pan (raw): {context.pan_raw:.1f}")
        print(f"  Tilt: {context.tilt_deg:.1f}deg")
        print(f"  Zoom: {context.zoom:.1f}")
        print(f"\nNumber of annotations: {len(annotations)}")
        print("-" * 70)

    errors = []
    results = []

    for i, annotation in enumerate(annotations):
        gcp_id = annotation.gcp_id

        # Look up map point coordinates from registry
        if gcp_id not in registry.points:
            if verbose:
                print(
                    f"Annotation {i + 1}: {gcp_id[:20]:20s} | FAILED: Map point '{gcp_id}' not found"
                )
            errors.append(1000.0)
            continue

        point = registry.points[gcp_id]

        proj_u, proj_v, success, err_msg = project_map_point_to_pixel(
            point.pixel_x,
            point.pixel_y,
            camera_height,
            context.pan_raw,
            context.tilt_deg,
            context.zoom,
            pan_offset,
            tilt_offset,
            focal_mult,
            6.78,
            k1,
            k2,
        )

        if success:
            error = math.sqrt(
                (proj_u - annotation.pixel.x) ** 2 + (proj_v - annotation.pixel.y) ** 2
            )
            errors.append(error)

            if verbose:
                status = "OK" if error < 5 else "FAIL"
                print(
                    f"Annotation {i + 1}: {gcp_id[:20]:20s} | "
                    f"Expected: ({annotation.pixel.x:7.1f}, {annotation.pixel.y:7.1f}) | "
                    f"Projected: ({proj_u:7.1f}, {proj_v:7.1f}) | "
                    f"Error: {error:6.1f}px {status}"
                )

            results.append(
                {
                    "annotation": annotation,
                    "projected": (proj_u, proj_v),
                    "error": error,
                }
            )
        else:
            if verbose:
                print(f"Annotation {i + 1}: {gcp_id[:20]:20s} | FAILED: {err_msg}")
            errors.append(1000.0)

    if errors:
        mean_error = np.mean([e for e in errors if e < 1000])
        if verbose:
            print("-" * 70)
            print(f"\nMean error: {mean_error:.2f} pixels")
            good = sum(1 for e in errors if e < 5)
            print(f"Annotations with <5px error: {good}/{len(errors)}")

            if mean_error < 5:
                print("\n[OK] Camera model is well-calibrated!")
            elif mean_error < 20:
                print("\n[WARN] Camera model needs minor calibration adjustment")
            else:
                print("\n[FAIL] Camera model needs significant calibration")
                print("  Possible causes:")
                print("  - Incorrect camera position/height")
                print("  - Incorrect pan offset")
                print("  - Map point coordinates may be inaccurate")

        return mean_error, results

    return None, []
