"""
Simple camera model validation.

Tests projection accuracy with a few known GCPs to validate that the
camera geometry model works correctly before running full calibration.

GCP observations are provided as GCPData objects, typically converted from
Annotation entities in the captured frame repository.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.domain.vo import CameraIntrinsics, LensDistortion, Orientation, Vector3
from poc_homography.types import Degrees, Meters, Millimeters, Pixels, PixelsFloat, Unitless

if TYPE_CHECKING:
    from poc_homography.domain.entities.ground_control_point import GroundControlPoint


class GCPData(NamedTuple):
    """Ground Control Point data for validation."""

    map_point_id: str
    pixel_u: PixelsFloat
    pixel_v: PixelsFloat
    pan: Degrees
    tilt: Degrees
    zoom: Unitless
    name: str


class ProjectionResult(NamedTuple):
    """Result of projecting a map point to pixel coordinates."""

    u: PixelsFloat | None
    v: PixelsFloat | None
    success: bool
    error_msg: str | None


class ValidationResult(NamedTuple):
    """Individual GCP validation result."""

    gcp: GCPData
    projected_u: PixelsFloat | None
    projected_v: PixelsFloat | None
    error_px: float


def project_map_point_to_pixel(
    map_point_x: Meters,
    map_point_y: Meters,
    camera_height: Meters,
    pan_deg: Degrees,
    tilt_deg: Degrees,
    zoom: Unitless,
    pan_offset_deg: Degrees = Degrees(0.0),
    tilt_offset_deg: Degrees = Degrees(0.0),
    focal_multiplier: Unitless = Unitless(1.0),
    sensor_width_mm: float = 6.78,
    k1: float = 0.0,
    k2: float = 0.0,
    image_width: Pixels = Pixels(1920),
    image_height: Pixels = Pixels(1080),
) -> ProjectionResult:
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
        ProjectionResult with u, v coordinates and success status
    """
    try:
        # Use map point coordinates directly as local XY (meters from camera)
        x_m, y_m = float(map_point_x), float(map_point_y)

        # Build intrinsic matrix
        effective_focal_mm = 5.9 * float(zoom) * float(focal_multiplier)
        f_px = effective_focal_mm * (float(image_width) / sensor_width_mm)
        cx, cy = float(image_width) / 2.0, float(image_height) / 2.0
        K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]])

        # Compute pan and tilt with offsets
        actual_pan = Degrees(float(pan_deg) + float(pan_offset_deg))
        actual_tilt = Degrees(float(tilt_deg) + float(tilt_offset_deg))

        # Set up distortion if needed
        distortion = None
        if k1 != 0.0 or k2 != 0.0:
            distortion = LensDistortion.radial_only(k1=Unitless(k1), k2=Unitless(k2))

        # Create VOs for the new API
        intrinsics = CameraIntrinsics.from_K_matrix(
            K=K,
            image_width=image_width,
            image_height=image_height,
            sensor_width=Millimeters(sensor_width_mm),
        )
        camera_position = Vector3.create(0.0, 0.0, float(camera_height))
        orientation = Orientation.create(yaw=actual_pan, pitch=actual_tilt, roll=Degrees(0.0))

        # Compute homography using new VO-based API
        homography = CameraGeometry.compute_from_vo(
            intrinsics=intrinsics,
            camera_position=camera_position,
            orientation=orientation,
            distortion=distortion,
        )
        H = homography.to_matrix()._to_array()

        # Project world point to image
        world_pt = np.array([[x_m], [y_m], [1.0]])
        img_pt = H @ world_pt

        if img_pt[2, 0] <= 0:
            return ProjectionResult(None, None, False, "Point behind camera")

        u = img_pt[0, 0] / img_pt[2, 0]
        v = img_pt[1, 0] / img_pt[2, 0]

        # Apply distortion if set
        if k1 != 0.0 or k2 != 0.0:
            # Simple distortion model (approximate)
            dx = u - cx
            dy = v - cy
            r2 = dx * dx + dy * dy
            distort_factor = 1.0 + k1 * r2 / (f_px * f_px) + k2 * r2 * r2 / (f_px**4)
            u = cx + dx * distort_factor
            v = cy + dy * distort_factor

        return ProjectionResult(PixelsFloat(u), PixelsFloat(v), True, None)

    except Exception as e:
        return ProjectionResult(None, None, False, str(e))


def validate_model(
    camera_config: dict[str, float | str],
    gcps: list[GCPData],
    registry: dict[str, GroundControlPoint],
    verbose: bool = True,
) -> tuple[float | None, list[ValidationResult]]:
    """
    Validate camera model with GCPs using Map Point coordinates.

    Args:
        camera_config: Camera configuration dictionary with height_m, pan_offset_deg, etc.
        gcps: List of GCPData objects
        registry: dict[str, GroundControlPoint] containing the map point coordinates
        verbose: Whether to print detailed output

    Returns:
        Tuple of (mean_error, individual_results)
        mean_error is None if no valid projections
    """
    # Extract camera parameters
    camera_name = str(camera_config.get("name", "Unknown"))
    camera_height = Meters(float(camera_config.get("height_m", 5.0)))
    pan_offset = Degrees(float(camera_config.get("pan_offset_deg", 0.0)))
    tilt_offset = Degrees(float(camera_config.get("tilt_offset_deg", 0.0)))
    focal_mult = Unitless(float(camera_config.get("focal_multiplier", 1.0)))
    k1 = float(camera_config.get("k1", 0.0))
    k2 = float(camera_config.get("k2", 0.0))

    if verbose:
        print("\n" + "=" * 70)
        print("CAMERA MODEL VALIDATION")
        print("=" * 70)
        print(f"\nCamera: {camera_name}")
        print(f"  Height: {camera_height:.2f}m")
        print(f"  Pan offset: {pan_offset:.1f}deg")
        print(f"  Tilt offset: {tilt_offset:+.2f}deg")
        print(f"  Focal multiplier: {focal_mult:.4f}")
        print(f"  Distortion: k1={k1:.4f}, k2={k2:.4f}")
        print(f"\nNumber of GCPs: {len(gcps)}")
        print("-" * 70)

    errors: list[float] = []
    results: list[ValidationResult] = []

    for i, gcp in enumerate(gcps):
        # Look up map point coordinates from registry
        if gcp.map_point_id not in registry:
            if verbose:
                print(
                    f"GCP {i + 1}: {gcp.name[:20]:20s} | FAILED: "
                    f"Map point '{gcp.map_point_id}' not found"
                )
            errors.append(1000.0)
            results.append(ValidationResult(gcp, None, None, 1000.0))
            continue

        point = registry[gcp.map_point_id].map_point

        proj_result = project_map_point_to_pixel(
            Meters(float(point.pixel_point.x)),
            Meters(float(point.pixel_point.y)),
            camera_height,
            gcp.pan,
            gcp.tilt,
            gcp.zoom,
            pan_offset,
            tilt_offset,
            focal_mult,
            6.78,
            k1,
            k2,
        )

        if proj_result.success and proj_result.u is not None and proj_result.v is not None:
            error = math.sqrt(
                (float(proj_result.u) - float(gcp.pixel_u)) ** 2
                + (float(proj_result.v) - float(gcp.pixel_v)) ** 2
            )
            errors.append(error)

            if verbose:
                status = "OK" if error < 5 else "FAIL"
                print(
                    f"GCP {i + 1}: {gcp.name[:20]:20s} | "
                    f"Expected: ({gcp.pixel_u:7.1f}, {gcp.pixel_v:7.1f}) | "
                    f"Projected: ({proj_result.u:7.1f}, {proj_result.v:7.1f}) | "
                    f"Error: {error:6.1f}px {status}"
                )

            results.append(ValidationResult(gcp, proj_result.u, proj_result.v, error))
        else:
            if verbose:
                print(f"GCP {i + 1}: {gcp.name[:20]:20s} | FAILED: {proj_result.error_msg}")
            errors.append(1000.0)
            results.append(ValidationResult(gcp, None, None, 1000.0))

    if errors:
        # Filter out error sentinel values (1000.0) when computing mean
        valid_errors = [e for e in errors if e < 1000]
        if not valid_errors:
            return None, results

        mean_error = float(np.mean(valid_errors))

        if verbose:
            print("-" * 70)
            print(f"\nMean error: {mean_error:.2f} pixels")
            good = sum(1 for e in errors if e < 5)
            print(f"GCPs with <5px error: {good}/{len(errors)}")

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
