"""
Comprehensive Calibration Tool for Sub-5px Projection Accuracy.

This tool optimizes ALL parameters that affect projection accuracy:
1. Pan offset (camera home position bearing)
2. Camera height
3. Camera position offset (X/Y in map coordinates)
4. Focal length multiplier (effective focal length)
5. Tilt offset (systematic tilt error)

Unlike the basic calibration tool which only sweeps pan_offset and height,
this tool uses scipy.optimize to find the optimal combination of ALL parameters
that minimizes projection error across multiple reference points.

Where gcps.yaml contains:
    gcps:
      - map_point_id: Z1
        pixel_u: 960
        pixel_v: 540
        pan_raw: 0.0
        tilt_deg: 30.0
        zoom: 1.0
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress verbose logging during optimization
import logging

logging.getLogger("poc_homography").setLevel(logging.ERROR)

import contextlib
import io


@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


from poc_homography.camera_config import get_camera_configs
from poc_homography.camera_geometry import CameraGeometry
from poc_homography.map_points import MapPointRegistry

# Try to import yaml, fallback to manual parsing if not available
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class GCP:
    """Ground Control Point with map point ID and pixel coordinates."""

    map_point_id: str
    pixel_u: float
    pixel_v: float
    pan_raw: float
    tilt_deg: float
    zoom: float


@dataclass
class CalibrationParams:
    """Parameters to optimize during calibration."""

    camera_x: float  # Camera X position in map coordinates
    camera_y: float  # Camera Y position in map coordinates
    height_m: float
    pan_offset_deg: float
    focal_multiplier: float  # Multiplier for base focal length (5.9mm)
    tilt_offset_deg: float  # Offset added to reported tilt
    sensor_width_mm: float  # Sensor width (6.78mm from FOV spec)
    # Lens distortion coefficients (OpenCV model)
    k1: float = 0.0  # Primary radial distortion
    k2: float = 0.0  # Secondary radial distortion


# Import camera configs from canonical source
# Note: camera_x and camera_y should be provided in map coordinates
_camera_configs_list = get_camera_configs()
CAMERA_CONFIGS = {}
for cam in _camera_configs_list:
    config = {
        "name": cam["name"],
        "camera_x": cam.get("camera_x", 0.0),  # Camera X in map coordinates
        "camera_y": cam.get("camera_y", 0.0),  # Camera Y in map coordinates
        "height_m": cam.get("height_m", 5.0),
        "pan_offset_deg": cam.get("pan_offset_deg", 0.0),
        "focal_multiplier": 1.0,  # Default multiplier, can be calibrated
        "k1": cam.get("k1", 0.0),
        "k2": cam.get("k2", 0.0),
    }
    CAMERA_CONFIGS[cam["name"]] = config


def parse_gcps_from_yaml(yaml_path: str) -> list[GCP]:
    """Parse GCPs from a YAML file."""
    if YAML_AVAILABLE:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    else:
        # Simple manual YAML parsing for flat structure
        data = {"gcps": []}
        with open(yaml_path) as f:
            current_gcp = {}
            for line in f:
                line = line.strip()
                if line.startswith("- map_point_id:"):
                    if current_gcp:
                        data["gcps"].append(current_gcp)
                    current_gcp = {"map_point_id": line.split(":")[1].strip()}
                elif ":" in line and current_gcp:
                    key, val = line.split(":", 1)
                    key = key.strip().lstrip("- ")
                    val = val.strip()
                    try:
                        current_gcp[key] = float(val)
                    except ValueError:
                        current_gcp[key] = val
            if current_gcp:
                data["gcps"].append(current_gcp)

    gcps = []
    for gcp_data in data.get("gcps", []):
        gcps.append(
            GCP(
                map_point_id=gcp_data["map_point_id"],
                pixel_u=gcp_data["pixel_u"],
                pixel_v=gcp_data["pixel_v"],
                pan_raw=gcp_data.get("pan_raw", 0.0),
                tilt_deg=gcp_data.get("tilt_deg", 30.0),
                zoom=gcp_data.get("zoom", 1.0),
            )
        )
    return gcps


def undistort_point_simple(
    u: float,
    v: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    iterations: int = 10,
) -> tuple[float, float]:
    """
    Undistort a single point using iterative method.
    Standalone function to avoid CameraGeometry dependency during optimization.
    """
    if k1 == 0.0 and k2 == 0.0:
        return (u, v)

    # Convert to normalized camera coordinates
    x_d = (u - cx) / fx
    y_d = (v - cy) / fy

    # Iterative undistortion
    x = x_d
    y = y_d

    for _ in range(iterations):
        r2 = x * x + y * y
        r4 = r2 * r2
        radial = 1.0 + k1 * r2 + k2 * r4
        x = x_d / radial
        y = y_d / radial

    # Convert back to pixel coordinates
    return (x * fx + cx, y * fy + cy)


def compute_projection_error(
    params: CalibrationParams,
    gcps: list[GCP],
    registry: MapPointRegistry,
    image_width: int = 1920,
    image_height: int = 1080,
) -> tuple[float, list[float]]:
    """
    Compute total projection error for given parameters and GCPs.

    Args:
        params: Calibration parameters including camera position.
        gcps: List of ground control points with map_point_id references.
        registry: MapPointRegistry containing the map point coordinates.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Tuple of (mean_error_pixels, list_of_individual_errors)
    """
    errors = []

    for gcp in gcps:
        try:
            # Look up map point coordinates from registry
            map_point = registry.points[gcp.map_point_id]
            # Compute local XY relative to camera position in map coordinates
            x_m = map_point.pixel_x - params.camera_x
            y_m = map_point.pixel_y - params.camera_y

            # Build intrinsic matrix with optimized parameters
            effective_focal_mm = 5.9 * gcp.zoom * params.focal_multiplier
            f_px = effective_focal_mm * (image_width / params.sensor_width_mm)
            cx, cy = image_width / 2.0, image_height / 2.0
            K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]])

            # Compute pan and tilt with offsets
            pan_deg = gcp.pan_raw + params.pan_offset_deg
            tilt_deg = gcp.tilt_deg + params.tilt_offset_deg

            # Validate tilt is positive (pointing down)
            if tilt_deg <= 0:
                errors.append(1000.0)  # Penalty for invalid tilt
                continue

            # Create geometry and compute homography
            geo = CameraGeometry(w=image_width, h=image_height)
            w_pos = np.array([0.0, 0.0, params.height_m])

            try:
                with suppress_stdout():
                    geo.set_camera_parameters(K, w_pos, pan_deg, tilt_deg, 640, 640)
            except ValueError:
                errors.append(1000.0)  # Penalty for invalid parameters
                continue

            # Project world point to image (gives undistorted coordinates)
            world_pt = np.array([[x_m], [y_m], [1.0]])
            img_pt = geo.H @ world_pt

            if img_pt[2, 0] <= 0:
                errors.append(1000.0)  # Penalty for point behind camera
                continue

            projected_u = img_pt[0, 0] / img_pt[2, 0]
            projected_v = img_pt[1, 0] / img_pt[2, 0]

            # Undistort the observed GCP pixel coordinates
            # (GCP pixels are in distorted image space, projected coords are undistorted)
            gcp_u_undist, gcp_v_undist = undistort_point_simple(
                gcp.pixel_u, gcp.pixel_v, f_px, f_px, cx, cy, params.k1, params.k2
            )

            # Compute pixel error in undistorted space
            error = math.sqrt((gcp_u_undist - projected_u) ** 2 + (gcp_v_undist - projected_v) ** 2)
            errors.append(error)

        except Exception:
            errors.append(1000.0)  # Penalty for any error

    if not errors:
        return 1000.0, []

    return np.mean(errors), errors


def objective_function(
    x: np.ndarray,
    gcps: list[GCP],
    registry: MapPointRegistry,
    base_params: CalibrationParams,
    optimize_position: bool,
    optimize_focal: bool,
    optimize_pan: bool,
    optimize_tilt: bool,
    optimize_distortion: bool,
    image_width: int,
    image_height: int,
) -> float:
    """
    Objective function for optimization.

    x contains the parameters being optimized in order:
    [height_m, (pan_offset_deg), (x_offset, y_offset), (focal_multiplier), (tilt_offset), (k1, k2)]
    """
    idx = 0

    # Always optimize height
    height_m = x[idx]
    idx += 1

    # Pan offset (optional)
    if optimize_pan:
        pan_offset_deg = x[idx]
        idx += 1
    else:
        pan_offset_deg = base_params.pan_offset_deg

    # Position offset (in map coordinates)
    if optimize_position:
        x_offset = x[idx]
        idx += 1
        y_offset = x[idx]
        idx += 1
    else:
        x_offset = 0.0
        y_offset = 0.0

    # Focal length multiplier
    if optimize_focal:
        focal_multiplier = x[idx]
        idx += 1
    else:
        focal_multiplier = base_params.focal_multiplier

    # Tilt offset
    if optimize_tilt:
        tilt_offset_deg = x[idx]
        idx += 1
    else:
        tilt_offset_deg = base_params.tilt_offset_deg

    # Distortion coefficients
    if optimize_distortion:
        k1 = x[idx]
        idx += 1
        k2 = x[idx]
        idx += 1
    else:
        k1 = base_params.k1
        k2 = base_params.k2

    # Build params
    params = CalibrationParams(
        camera_x=base_params.camera_x + x_offset,
        camera_y=base_params.camera_y + y_offset,
        height_m=height_m,
        pan_offset_deg=pan_offset_deg,
        focal_multiplier=focal_multiplier,
        tilt_offset_deg=tilt_offset_deg,
        sensor_width_mm=base_params.sensor_width_mm,
        k1=k1,
        k2=k2,
    )

    mean_error, _ = compute_projection_error(params, gcps, registry, image_width, image_height)
    return mean_error


def run_calibration(
    camera_config: dict[str, Any],
    gcps: list[GCP],
    registry: MapPointRegistry,
    optimize_position: bool = True,
    optimize_focal: bool = True,
    optimize_pan: bool = True,
    optimize_tilt: bool = True,
    optimize_distortion: bool = True,
    image_width: int = 1920,
    image_height: int = 1080,
) -> tuple[CalibrationParams, float, list[float]]:
    """
    Run comprehensive calibration to find optimal parameters.

    Args:
        camera_config: Initial camera configuration
        gcps: List of ground control points
        registry: MapPointRegistry containing map point coordinates
        optimize_position: Whether to optimize camera position (X/Y in map coordinates)
        optimize_focal: Whether to optimize focal length multiplier
        optimize_pan: Whether to optimize pan offset
        optimize_tilt: Whether to optimize tilt offset
        optimize_distortion: Whether to optimize lens distortion (k1, k2)

    Returns:
        Tuple of (optimized_params, mean_error, individual_errors)
    """
    # Initial parameters
    base_params = CalibrationParams(
        camera_x=camera_config.get("camera_x", 0.0),
        camera_y=camera_config.get("camera_y", 0.0),
        height_m=camera_config.get("height_m", 5.0),
        pan_offset_deg=camera_config.get("pan_offset_deg", 0.0),
        focal_multiplier=camera_config.get("focal_multiplier", 1.0),
        tilt_offset_deg=0.0,
        sensor_width_mm=6.78,  # Calculated from 59.8° FOV at 5.9mm focal length
        k1=camera_config.get("k1", 0.0),
        k2=camera_config.get("k2", 0.0),
    )

    # Build initial guess and bounds
    x0 = [base_params.height_m]
    bounds = [(1.0, 30.0)]  # height

    if optimize_pan:
        x0.append(base_params.pan_offset_deg)
        bounds.append((-180.0, 180.0))

    if optimize_position:
        x0.extend([0.0, 0.0])  # x/y offsets in map coordinates
        bounds.extend([(-50.0, 50.0), (-50.0, 50.0)])  # +/-50 units position adjustment

    if optimize_focal:
        x0.append(base_params.focal_multiplier)  # focal multiplier
        bounds.extend([(0.5, 2.0)])  # +/-50-100% focal length adjustment (wider for exploration)

    if optimize_tilt:
        x0.append(0.0)  # tilt offset
        bounds.extend([(-10.0, 10.0)])  # +/-10 deg tilt adjustment

    if optimize_distortion:
        x0.extend([base_params.k1, base_params.k2])  # k1, k2
        bounds.extend([(-1.0, 1.0), (-1.0, 1.0)])  # Extended distortion range

    print("\n" + "=" * 70)
    print("COMPREHENSIVE CALIBRATION")
    print("=" * 70)
    print("\nOptimizing parameters:")
    print("  - Height: Yes")
    print(
        f"  - Pan offset: {'Yes' if optimize_pan else 'No (fixed at ' + str(base_params.pan_offset_deg) + '°)'}"
    )
    print(f"  - Camera position: {'Yes' if optimize_position else 'No'}")
    print(f"  - Focal length: {'Yes' if optimize_focal else 'No'}")
    print(f"  - Tilt offset: {'Yes' if optimize_tilt else 'No'}")
    print(f"  - Lens distortion: {'Yes' if optimize_distortion else 'No'}")
    print(f"\nNumber of GCPs: {len(gcps)}")
    sys.stdout.flush()

    # Compute initial error
    initial_error, _ = compute_projection_error(
        base_params, gcps, registry, image_width, image_height
    )
    print(f"\nInitial mean error: {initial_error:.1f} pixels")
    sys.stdout.flush()

    # Progress callback for differential evolution
    iteration_count = [0]  # Use list for mutable in closure

    def progress_callback(xk, convergence):
        iteration_count[0] += 1
        if iteration_count[0] % 10 == 0:
            print(f"  Iteration {iteration_count[0]}, convergence: {convergence:.4f}")
            sys.stdout.flush()

    # Run differential evolution (global optimizer)
    print("\nRunning global optimization (differential evolution)...")
    sys.stdout.flush()

    result_de = differential_evolution(
        objective_function,
        bounds,
        args=(
            gcps,
            registry,
            base_params,
            optimize_position,
            optimize_focal,
            optimize_pan,
            optimize_tilt,
            optimize_distortion,
            image_width,
            image_height,
        ),
        seed=42,
        maxiter=200,  # Reduced for faster convergence
        tol=0.01,
        polish=True,  # Use local optimizer to polish result
        workers=1,
        disp=False,
        callback=progress_callback,
    )

    print(f"Global optimization complete. Error: {result_de.fun:.2f} pixels")

    # Refine with local optimizer
    print("\nRefining with local optimization (L-BFGS-B)...")

    result = minimize(
        objective_function,
        result_de.x,
        args=(
            gcps,
            registry,
            base_params,
            optimize_position,
            optimize_focal,
            optimize_pan,
            optimize_tilt,
            optimize_distortion,
            image_width,
            image_height,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-8},
    )

    # Extract optimized parameters
    idx = 0
    height_m = result.x[idx]
    idx += 1

    if optimize_pan:
        pan_offset_deg = result.x[idx]
        idx += 1
    else:
        pan_offset_deg = base_params.pan_offset_deg

    if optimize_position:
        x_offset = result.x[idx]
        idx += 1
        y_offset = result.x[idx]
        idx += 1
    else:
        x_offset = 0.0
        y_offset = 0.0

    if optimize_focal:
        focal_multiplier = result.x[idx]
        idx += 1
    else:
        focal_multiplier = base_params.focal_multiplier

    if optimize_tilt:
        tilt_offset_deg = result.x[idx]
        idx += 1
    else:
        tilt_offset_deg = 0.0

    if optimize_distortion:
        k1 = result.x[idx]
        idx += 1
        k2 = result.x[idx]
        idx += 1
    else:
        k1 = base_params.k1
        k2 = base_params.k2

    optimized_params = CalibrationParams(
        camera_x=base_params.camera_x + x_offset,
        camera_y=base_params.camera_y + y_offset,
        height_m=height_m,
        pan_offset_deg=pan_offset_deg,
        focal_multiplier=focal_multiplier,
        tilt_offset_deg=tilt_offset_deg,
        sensor_width_mm=base_params.sensor_width_mm,
        k1=k1,
        k2=k2,
    )

    # Compute final errors
    mean_error, individual_errors = compute_projection_error(
        optimized_params, gcps, registry, image_width, image_height
    )

    return optimized_params, mean_error, individual_errors


def print_results(
    base_config: dict[str, Any],
    optimized: CalibrationParams,
    mean_error: float,
    individual_errors: list[float],
    gcps: list[GCP],
):
    """Print calibration results."""
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)

    print(f"\nFinal mean error: {mean_error:.2f} pixels")
    if mean_error < 5:
        print("  [OK] Target accuracy achieved (< 5 pixels)")
    else:
        print("  [FAIL] Target accuracy NOT achieved (need < 5 pixels)")

    print("\nIndividual GCP errors:")
    for i, (gcp, error) in enumerate(zip(gcps, individual_errors)):
        status = "[OK]" if error < 5 else "[FAIL]"
        print(f"  GCP {i + 1}: {error:.2f}px {status} (map_point_id: {gcp.map_point_id})")

    print("\n" + "-" * 70)
    print("PARAMETER CHANGES")
    print("-" * 70)

    # Height
    height_change = optimized.height_m - base_config.get("height_m", 5.0)
    print("\nCamera Height:")
    print(f"  Original: {base_config.get('height_m', 5.0):.2f}m")
    print(f"  Optimized: {optimized.height_m:.2f}m")
    print(f"  Change: {height_change:+.2f}m")

    # Pan offset
    pan_change = optimized.pan_offset_deg - base_config.get("pan_offset_deg", 0.0)
    print("\nPan Offset:")
    print(f"  Original: {base_config.get('pan_offset_deg', 0.0):.1f} deg")
    print(f"  Optimized: {optimized.pan_offset_deg:.1f} deg")
    print(f"  Change: {pan_change:+.1f} deg")

    # Camera position in map coordinates
    base_x = base_config.get("camera_x", 0.0)
    base_y = base_config.get("camera_y", 0.0)
    x_change = optimized.camera_x - base_x
    y_change = optimized.camera_y - base_y

    print("\nCamera Position (map coordinates):")
    print(f"  Original: ({base_x:.2f}, {base_y:.2f})")
    print(f"  Optimized: ({optimized.camera_x:.2f}, {optimized.camera_y:.2f})")
    print(f"  Change: ({x_change:+.2f}, {y_change:+.2f})")

    # Focal multiplier
    print("\nFocal Length Multiplier:")
    print("  Original: 1.00")
    print(f"  Optimized: {optimized.focal_multiplier:.4f}")
    print(f"  Effect: Base focal {5.9 * optimized.focal_multiplier:.2f}mm (was 5.9mm)")

    # Tilt offset
    print("\nTilt Offset:")
    print("  Original: 0.0 deg")
    print(f"  Optimized: {optimized.tilt_offset_deg:+.2f} deg")
    print(f"  Effect: Add {optimized.tilt_offset_deg:+.2f} deg to reported tilt")

    # Lens distortion
    print("\nLens Distortion (radial):")
    print(f"  k1: {optimized.k1:+.6f}")
    print(f"  k2: {optimized.k2:+.6f}")
    if optimized.k1 < 0:
        print("  Type: Barrel distortion (edges curve outward)")
    elif optimized.k1 > 0:
        print("  Type: Pincushion distortion (edges curve inward)")
    else:
        print("  Type: No significant distortion")

    # Print configuration update instructions
    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATION UPDATE")
    print("=" * 70)
    print("\nUpdate camera_config.py with:")
    print(f"""
    {{
        "camera_x": {optimized.camera_x:.2f},
        "camera_y": {optimized.camera_y:.2f},
        "height_m": {optimized.height_m:.2f},
        "pan_offset_deg": {optimized.pan_offset_deg:.1f},
        "focal_multiplier": {optimized.focal_multiplier:.4f},
        "k1": {optimized.k1:.6f},
        "k2": {optimized.k2:.6f},
        # tilt_offset_deg: {optimized.tilt_offset_deg:.2f}
    }}
    """)
