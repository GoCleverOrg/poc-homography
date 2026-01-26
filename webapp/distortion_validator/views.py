"""Views for the distortion validator Django app."""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

# Paths
WEBAPP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEBAPP_DIR.parent
CALIBRATION_DIR = PROJECT_ROOT / "calibration_results"
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "homography" / "test_data"
SURVEY_DIR = WEBAPP_DIR / "survey"

logger = logging.getLogger(__name__)


def index(request: HttpRequest) -> HttpResponse:
    """Serve the main HTML page."""
    return render(request, "distortion_validator/index.html")


@require_GET
def api_calibration_files(request: HttpRequest) -> JsonResponse:
    """List available calibration YAML files."""
    try:
        files = []
        if CALIBRATION_DIR.exists():
            for f in sorted(CALIBRATION_DIR.glob("*.yaml")):
                files.append({
                    "name": f.name,
                    "path": str(f),
                })
        return JsonResponse({"files": files})
    except Exception as e:
        logger.exception("Failed to list calibration files")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def api_load_calibration(request: HttpRequest) -> JsonResponse:
    """Load a calibration file and return its contents."""
    try:
        data = json.loads(request.body)
        filename = data.get("filename", "")

        if not filename:
            return JsonResponse({"error": "Missing filename"}, status=400)

        # Try calibration directory first
        filepath = CALIBRATION_DIR / filename
        if not filepath.exists():
            filepath = Path(filename)
            if not filepath.exists():
                return JsonResponse({"error": f"File not found: {filename}"}, status=404)

        from poc_homography.calibration.lens_distortion.calibration_table import (
            CameraCalibrationTable,
        )

        table = CameraCalibrationTable.load(filepath)

        # Convert to JSON-serializable format
        entries = []
        for zoom, entry in table.entries.items():
            entries.append({
                "zoom_factor": entry.zoom_factor,
                "coefficients": {
                    "k1": float(entry.k1),
                    "k2": float(entry.k2),
                    "k3": float(entry.k3),
                    "p1": float(entry.p1),
                    "p2": float(entry.p2),
                },
                "calibration_date": entry.calibration_date,
                "validation_rmse": entry.validation_rmse,
                "num_lines_used": entry.num_lines_used,
            })

        return JsonResponse({
            "camera_id": table.camera_id,
            "entries": entries,
        })

    except Exception as e:
        logger.exception("Failed to load calibration file")
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def api_images(request: HttpRequest) -> JsonResponse:
    """List available test images."""
    try:
        images = []

        # Check test data directory (same as camera_line_annotator)
        if TEST_DATA_DIR.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                for f in sorted(TEST_DATA_DIR.glob(ext)):
                    images.append({
                        "name": f.name,
                        "path": str(f),
                    })

        # Also check survey directory
        if SURVEY_DIR.exists():
            for date_dir in sorted(SURVEY_DIR.iterdir(), reverse=True):
                if not date_dir.is_dir():
                    continue
                for session_dir in sorted(date_dir.iterdir(), reverse=True):
                    if not session_dir.is_dir():
                        continue
                    for ext in ["*.jpg", "*.jpeg", "*.png"]:
                        for f in list(session_dir.glob(ext))[:5]:  # Limit per session
                            images.append({
                                "name": f"survey/{date_dir.name}/{session_dir.name}/{f.name}",
                                "path": str(f),
                            })

        return JsonResponse({"images": images})
    except Exception as e:
        logger.exception("Failed to list images")
        return JsonResponse({"error": str(e)}, status=500)


def _undistort_points_solver_method(
    points: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Apply inverse distortion model to points.

    This matches EXACTLY the method used in distortion_solver.py to ensure
    consistency between calibration and validation.

    Uses the Brown-Conrady distortion model with iterative Newton-Raphson.

    Args:
        points: Nx2 array of (u, v) pixel coordinates.
        k1, k2, k3: Radial distortion coefficients.
        p1, p2: Tangential distortion coefficients.
        fx, fy: Focal lengths.
        cx, cy: Principal point coordinates.

    Returns:
        Nx2 array of undistorted (u, v) coordinates.
    """
    # Convert to normalized coordinates
    x = (points[:, 0] - cx) / fx
    y = (points[:, 1] - cy) / fy

    # Apply undistortion iteratively (Newton-Raphson approximation)
    # Start with distorted coordinates as initial guess
    x_u = x.copy()
    y_u = y.copy()

    # Newton-Raphson typically converges in 3-5 iterations for reasonable distortion
    max_undistort_iterations = 10
    # Minimum radial factor to prevent division by zero
    min_radial_factor = 1e-6

    for _ in range(max_undistort_iterations):
        r2 = x_u * x_u + y_u * y_u
        r4 = r2 * r2
        r6 = r4 * r2

        # Radial distortion factor (clamped to prevent division by zero)
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
        radial = np.maximum(np.abs(radial), min_radial_factor) * np.sign(radial + 1e-10)

        # Tangential distortion
        dx_tangential = 2 * p1 * x_u * y_u + p2 * (r2 + 2 * x_u * x_u)
        dy_tangential = p1 * (r2 + 2 * y_u * y_u) + 2 * p2 * x_u * y_u

        # Update estimate
        x_u = (x - dx_tangential) / radial
        y_u = (y - dy_tangential) / radial

    # Convert back to pixel coordinates
    undistorted = np.column_stack([
        x_u * fx + cx,
        y_u * fy + cy,
    ])

    return undistorted


def _undistort_image_using_solver_method(
    image: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Undistort an image using the same method as the calibration solver.

    This creates an undistorted image by:
    1. For each pixel in the OUTPUT (undistorted) image
    2. Apply the FORWARD distortion model to find where it came from in the INPUT (distorted) image
    3. Sample that pixel value

    This ensures consistency with how the solver measures straightness.
    """
    h, w = image.shape[:2]

    # Create coordinate grids for the output image
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float64)

    # Flatten for processing
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()

    # Convert to normalized coordinates (these are undistorted coordinates)
    x_norm = (x_flat - cx) / fx
    y_norm = (y_flat - cy) / fy

    # Apply FORWARD distortion model to find source coordinates
    r2 = x_norm * x_norm + y_norm * y_norm
    r4 = r2 * r2
    r6 = r4 * r2

    # Radial distortion
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6

    # Tangential distortion
    x_tangential = 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm * x_norm)
    y_tangential = p1 * (r2 + 2 * y_norm * y_norm) + 2 * p2 * x_norm * y_norm

    # Apply distortion to get source coordinates
    x_distorted = x_norm * radial + x_tangential
    y_distorted = y_norm * radial + y_tangential

    # Convert back to pixel coordinates
    src_x = x_distorted * fx + cx
    src_y = y_distorted * fy + cy

    # Reshape for remap
    map_x = src_x.reshape(h, w).astype(np.float32)
    map_y = src_y.reshape(h, w).astype(np.float32)

    # Use remap to sample from the distorted image
    undistorted = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return undistorted


@csrf_exempt
@require_http_methods(["POST"])
def api_undistort(request: HttpRequest) -> JsonResponse:
    """Undistort an image using provided coefficients.

    Returns both original and undistorted images as base64-encoded JPEGs.

    IMPORTANT: This now uses the SAME undistortion method as the calibration
    solver to ensure RMSE measurements are consistent.
    """
    try:
        data = json.loads(request.body)

        image_path = data.get("image_path", "")
        coefficients = data.get("coefficients", {})
        intrinsics = data.get("intrinsics", {})
        use_opencv = data.get("use_opencv", False)  # Option to use OpenCV method for comparison

        if not image_path:
            return JsonResponse({"error": "Missing image_path"}, status=400)

        # Load image
        img_path = Path(image_path)
        if not img_path.exists():
            # Try relative to test data dir
            img_path = TEST_DATA_DIR / image_path
            if not img_path.exists():
                # Try relative to survey dir
                img_path = SURVEY_DIR / image_path
                if not img_path.exists():
                    return JsonResponse({"error": f"Image not found: {image_path}"}, status=404)

        image = cv2.imread(str(img_path))
        if image is None:
            return JsonResponse({"error": f"Could not load image: {image_path}"}, status=400)

        h, w = image.shape[:2]

        # Get coefficients
        k1 = coefficients.get("k1", 0.0)
        k2 = coefficients.get("k2", 0.0)
        k3 = coefficients.get("k3", 0.0)
        p1 = coefficients.get("p1", 0.0)
        p2 = coefficients.get("p2", 0.0)

        # Get intrinsics - CRITICAL: Must match what was used during calibration!
        # Default to reasonable values but warn if not provided
        fx = intrinsics.get("fx", 1000.0)  # Default focal length
        fy = intrinsics.get("fy", 1000.0)
        cx = intrinsics.get("cx", w / 2)   # Default to image center
        cy = intrinsics.get("cy", h / 2)

        if use_opencv:
            # Original OpenCV method (for comparison)
            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)

            # Distortion coefficients in OpenCV format: [k1, k2, p1, p2, k3]
            dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

            # Don't use getOptimalNewCameraMatrix - keep same camera matrix
            undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)
            method_used = "opencv"
        else:
            # Use the SAME method as the calibration solver
            undistorted = _undistort_image_using_solver_method(
                image, k1, k2, k3, p1, p2, fx, fy, cx, cy
            )
            method_used = "solver"

        # Encode images as base64 JPEG
        def encode_image(img: np.ndarray) -> str:
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode('utf-8')

        return JsonResponse({
            "original": encode_image(image),
            "undistorted": encode_image(undistorted),
            "width": w,
            "height": h,
            "method_used": method_used,
            "coefficients_used": {
                "k1": k1, "k2": k2, "k3": k3, "p1": p1, "p2": p2
            },
            "intrinsics_used": {
                "fx": fx, "fy": fy, "cx": cx, "cy": cy
            }
        })

    except Exception as e:
        logger.exception("Failed to undistort image")
        return JsonResponse({"error": str(e)}, status=500)


def _measure_line_straightness(pts: np.ndarray) -> dict:
    """Measure straightness of a set of points.

    This matches EXACTLY the method used in distortion_solver.py.

    Args:
        pts: Nx2 array of (x, y) coordinates.

    Returns:
        Dictionary with rmse, max_deviation, r_squared, etc.
    """
    if len(pts) < 2:
        return {
            "rmse_pixels": 0.0,
            "max_deviation_pixels": 0.0,
            "r_squared": 1.0,
            "num_points": len(pts),
        }

    # Fit line using SVD (total least squares)
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid

    # SVD of centered points
    _, s, Vt = np.linalg.svd(centered)

    # Direction of the line is the first principal component
    line_direction = Vt[0]

    # Normal to the line
    line_normal = np.array([-line_direction[1], line_direction[0]])

    # Perpendicular distances
    distances = np.abs(centered @ line_normal)

    # Calculate metrics
    rmse = float(np.sqrt(np.mean(distances ** 2)))
    max_deviation = float(np.max(distances))

    # Calculate R² (how well points fit a line)
    total_variance = np.sum(s ** 2)
    explained_variance = s[0] ** 2
    r_squared = explained_variance / total_variance if total_variance > 0 else 1.0

    return {
        "rmse_pixels": rmse,
        "max_deviation_pixels": max_deviation,
        "r_squared": float(r_squared),
        "num_points": len(pts),
    }


@csrf_exempt
@require_http_methods(["POST"])
def api_measure_straightness(request: HttpRequest) -> JsonResponse:
    """Measure the straightness of a set of points.

    Returns the RMSE (perpendicular distance from best-fit line).

    Can optionally undistort points first using the same method as the calibration solver.

    Request body:
    {
        "points": [[x1, y1], [x2, y2], ...],
        "undistort": false,  // Optional: apply undistortion before measuring
        "coefficients": {k1, k2, k3, p1, p2},  // Required if undistort=true
        "intrinsics": {fx, fy, cx, cy}  // Required if undistort=true
    }
    """
    try:
        data = json.loads(request.body)
        points = data.get("points", [])

        if len(points) < 2:
            return JsonResponse({"error": "Need at least 2 points"}, status=400)

        # Convert to numpy array
        pts = np.array(points, dtype=np.float64)

        # Check if we should undistort points first
        should_undistort = data.get("undistort", False)

        if should_undistort:
            coefficients = data.get("coefficients", {})
            intrinsics = data.get("intrinsics", {})

            k1 = coefficients.get("k1", 0.0)
            k2 = coefficients.get("k2", 0.0)
            k3 = coefficients.get("k3", 0.0)
            p1 = coefficients.get("p1", 0.0)
            p2 = coefficients.get("p2", 0.0)

            # Intrinsics MUST match what was used during calibration
            fx = intrinsics.get("fx", 1000.0)
            fy = intrinsics.get("fy", 1000.0)
            cx = intrinsics.get("cx", 960.0)
            cy = intrinsics.get("cy", 540.0)

            # Undistort points using the SAME method as the calibration solver
            pts = _undistort_points_solver_method(
                pts, k1, k2, k3, p1, p2, fx, fy, cx, cy
            )

        # Measure straightness
        result = _measure_line_straightness(pts)

        # Add quality assessment
        rmse = result["rmse_pixels"]
        result["is_straight"] = rmse < 2.0
        result["quality"] = (
            "excellent" if rmse < 0.5
            else "good" if rmse < 2.0
            else "acceptable" if rmse < 5.0
            else "poor"
        )
        result["undistorted"] = should_undistort

        return JsonResponse(result)

    except Exception as e:
        logger.exception("Failed to measure straightness")
        return JsonResponse({"error": str(e)}, status=500)
