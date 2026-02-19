"""Views for the distortion validator Django app.

This app provides a web UI for **evaluating and visualizing** lens distortion
calibration results.  Users load a calibration file, select test images, and
visually inspect how well the calibration undistorts the image.  It also
measures line straightness to provide quantitative feedback.

Distinct from ``lens_calibration``, which *performs* the calibration, this app
is purely for validation / QA of existing calibrations.

CSRF protection
---------------
POST endpoints use Django's default CSRF protection.  The ``index`` view is
decorated with ``@ensure_csrf_cookie`` so the JavaScript frontend receives
the CSRF token cookie on initial page load.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods
from homography_web.calibration_utils import (
    api_compute_intrinsics as api_compute_intrinsics,
)
from homography_web.calibration_utils import (
    list_calibration_ids,
    load_calibration_from_repo,
    serialize_calibration_entry,
)
from homography_web.calibration_utils import (
    resolve_safe_path as _resolve_safe_path,
)
from homography_web.frame_utils import (
    get_frame_image_path,
    image_filename_to_frame,
    list_image_filenames,
)

# Paths
WEBAPP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEBAPP_DIR.parent
CALIBRATIONS_DIR = PROJECT_ROOT / "data" / "lens_calibrations"
SURVEY_DIR = WEBAPP_DIR / "survey"
RESULT_IMAGE_DIR = WEBAPP_DIR / "distortion_validator" / "_result_images"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


@ensure_csrf_cookie
def index(request: HttpRequest) -> HttpResponse:
    """Serve the main HTML page."""
    return render(request, "distortion_validator/index.html")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@require_GET
def api_calibration_files(request: HttpRequest) -> JsonResponse:
    """List available camera_ids in the calibration repo."""
    try:
        camera_ids = list_calibration_ids(CALIBRATIONS_DIR)
        return JsonResponse({"camera_ids": camera_ids})
    except Exception:
        logger.exception("Failed to list calibrations")
        return JsonResponse({"error": "Failed to list calibrations"}, status=500)


@require_http_methods(["POST"])
def api_load_calibration(request: HttpRequest) -> JsonResponse:
    """Load a calibration by camera_id from the DDD repo."""
    try:
        data = json.loads(request.body)
        camera_id = data.get("camera_id", "")
        if not camera_id:
            return JsonResponse({"error": "Missing camera_id"}, status=400)

        entity = load_calibration_from_repo(camera_id, CALIBRATIONS_DIR)
        if entity is None:
            return JsonResponse({"error": f"No calibration found for {camera_id}"}, status=404)

        entries = [serialize_calibration_entry(entry) for entry in entity.entries]

        return JsonResponse(
            {
                "camera_id": entity.id,
                "entries": entries,
            }
        )

    except Exception:
        logger.exception("Failed to load calibration")
        return JsonResponse({"error": "Failed to load calibration"}, status=500)


@require_GET
def api_images(request: HttpRequest) -> JsonResponse:
    """List available test images."""
    try:
        images: list[dict[str, str]] = []

        # Images from captured-frame repo
        for filename in list_image_filenames():
            images.append(
                {
                    "name": filename,
                    "source": "captured_frame",
                }
            )

        # Survey images (operational, not test data)
        if SURVEY_DIR.exists():
            for date_dir in sorted(SURVEY_DIR.iterdir(), reverse=True):
                if not date_dir.is_dir():
                    continue
                for session_dir in sorted(date_dir.iterdir(), reverse=True):
                    if not session_dir.is_dir():
                        continue
                    for ext in ["*.jpg", "*.jpeg", "*.png"]:
                        for f in list(session_dir.glob(ext))[:5]:
                            images.append(
                                {
                                    "name": f"survey/{date_dir.name}/{session_dir.name}/{f.name}",
                                    "source": "survey",
                                }
                            )

        return JsonResponse({"images": images})
    except Exception:
        logger.exception("Failed to list images")
        return JsonResponse({"error": "Failed to list images"}, status=500)


def _resolve_image_path(image_path: str) -> Path | None:
    """Resolve an image path safely against known directories."""
    # Try as filename in the CapturedFrame repo
    frame = image_filename_to_frame(image_path)
    if frame is not None:
        fp = get_frame_image_path(frame)
        if fp.exists():
            return fp

    # Try as relative path under SURVEY_DIR (e.g. "survey/2024-01-01/session/img.jpg")
    survey_rel = image_path
    if survey_rel.startswith("survey/"):
        survey_rel = survey_rel[len("survey/") :]
    try:
        candidate = (SURVEY_DIR / survey_rel).resolve()
        if candidate.is_relative_to(SURVEY_DIR.resolve()) and candidate.exists():
            return candidate
    except (ValueError, RuntimeError):
        pass

    return None


@require_http_methods(["POST"])
def api_undistort(request: HttpRequest) -> JsonResponse:
    """Undistort an image using provided coefficients.

    Returns both original and undistorted images as base64-encoded JPEGs,
    or serves the undistorted image via a temp-file URL when available.
    """
    try:
        data = json.loads(request.body)

        image_path = data.get("image_path", "")
        coefficients = data.get("coefficients", {})
        intrinsics = data.get("intrinsics", {})
        use_opencv = data.get("use_opencv", False)

        if not image_path:
            return JsonResponse({"error": "Missing image_path"}, status=400)

        img_path = _resolve_image_path(image_path)
        if img_path is None:
            return JsonResponse({"error": "Image not found"}, status=404)

        image = cv2.imread(str(img_path))
        if image is None:
            return JsonResponse({"error": "Could not load image"}, status=400)

        h, w = image.shape[:2]

        k1 = coefficients.get("k1", 0.0)
        k2 = coefficients.get("k2", 0.0)
        k3 = coefficients.get("k3", 0.0)
        p1 = coefficients.get("p1", 0.0)
        p2 = coefficients.get("p2", 0.0)

        fx = intrinsics.get("fx", 1000.0)
        fy = intrinsics.get("fy", 1000.0)
        cx = intrinsics.get("cx", w / 2)
        cy = intrinsics.get("cy", h / 2)

        if use_opencv:
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
            undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)
            method_used = "opencv"
        else:
            from poc_homography.calibration.lens_distortion.apply_calibration import (
                undistort_image,
            )

            undistorted = undistort_image(image, k1, k2, k3, p1, p2, fx, fy, cx, cy)
            method_used = "solver"

        # Try to serve via temp-file URL for performance
        result_url = _save_result_image(undistorted, img_path.stem)

        # Encode images as base64 JPEG
        def encode_image(img: np.ndarray) -> str:
            _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode("utf-8")

        response_data: dict[str, Any] = {
            "original": encode_image(image),
            "width": w,
            "height": h,
            "method_used": method_used,
            "coefficients_used": {"k1": k1, "k2": k2, "k3": k3, "p1": p1, "p2": p2},
            "intrinsics_used": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        }

        if result_url is not None:
            response_data["undistorted_url"] = result_url
        else:
            response_data["undistorted"] = encode_image(undistorted)

        return JsonResponse(response_data)

    except Exception:
        logger.exception("Failed to undistort image")
        return JsonResponse({"error": "Failed to undistort image"}, status=500)


def _save_result_image(image: np.ndarray, stem: str) -> str | None:
    """Save undistorted image to temp dir and return a relative URL, or None on failure."""
    try:
        RESULT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"{stem}_undistorted.jpg"
        out_path = RESULT_IMAGE_DIR / fname
        cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return f"api/result-image/{fname}"
    except Exception:
        logger.debug("Could not save result image to disk, falling back to base64")
        return None


@require_GET
def api_serve_result_image(request: HttpRequest, filename: str) -> HttpResponse:
    """Serve a previously generated undistorted result image."""
    resolved = _resolve_safe_path(filename, RESULT_IMAGE_DIR)
    if resolved is None or not resolved.exists():
        return HttpResponse(status=404)

    with open(resolved, "rb") as f:
        return HttpResponse(f.read(), content_type="image/jpeg")


@require_http_methods(["POST"])
def api_transform_points(request: HttpRequest) -> JsonResponse:
    """Transform points between distorted and undistorted coordinate spaces.

    Args (JSON body):
        points: List of [x, y] coordinates
        direction: 'distort' or 'undistort'
        coefficients: {k1, k2, k3, p1, p2}
        intrinsics: {fx, fy, cx, cy}

    Returns:
        JSON with transformed points array
    """
    MAX_POINTS = 10000  # Prevent DoS via excessive input

    try:
        data = json.loads(request.body)
        points = data.get("points", [])
        direction = data.get("direction", "undistort")

        if not points:
            return JsonResponse({"error": "Need at least 1 point"}, status=400)

        if len(points) > MAX_POINTS:
            return JsonResponse({"error": f"Too many points (max {MAX_POINTS})"}, status=400)

        if direction not in ("distort", "undistort"):
            return JsonResponse({"error": "direction must be 'distort' or 'undistort'"}, status=400)

        pts = np.array(points, dtype=np.float64)

        coefficients = data.get("coefficients", {})
        intrinsics = data.get("intrinsics", {})

        k1 = coefficients.get("k1", 0.0)
        k2 = coefficients.get("k2", 0.0)
        k3 = coefficients.get("k3", 0.0)
        p1 = coefficients.get("p1", 0.0)
        p2 = coefficients.get("p2", 0.0)

        fx = intrinsics.get("fx", 1000.0)
        fy = intrinsics.get("fy", 1000.0)
        cx = intrinsics.get("cx", 960.0)
        cy = intrinsics.get("cy", 540.0)

        if direction == "undistort":
            from poc_homography.calibration.lens_distortion.apply_calibration import (
                undistort_points,
            )

            transformed = undistort_points(pts, k1, k2, k3, p1, p2, fx, fy, cx, cy)
        else:
            from poc_homography.calibration.lens_distortion.apply_calibration import (
                distort_points,
            )

            transformed = distort_points(pts, k1, k2, k3, p1, p2, fx, fy, cx, cy)

        return JsonResponse(
            {
                "points": transformed.tolist(),
                "direction": direction,
            }
        )

    except Exception:
        logger.exception("Failed to transform points")
        return JsonResponse({"error": "Failed to transform points"}, status=500)


@require_http_methods(["POST"])
def api_measure_straightness(request: HttpRequest) -> JsonResponse:
    """Measure the straightness of a set of points."""
    try:
        data = json.loads(request.body)
        points = data.get("points", [])

        if len(points) < 2:
            return JsonResponse({"error": "Need at least 2 points"}, status=400)

        pts = np.array(points, dtype=np.float64)

        should_undistort = data.get("undistort", False)

        if should_undistort:
            coefficients = data.get("coefficients", {})
            intrinsics = data.get("intrinsics", {})

            k1 = coefficients.get("k1", 0.0)
            k2 = coefficients.get("k2", 0.0)
            k3 = coefficients.get("k3", 0.0)
            p1 = coefficients.get("p1", 0.0)
            p2 = coefficients.get("p2", 0.0)

            fx = intrinsics.get("fx", 1000.0)
            fy = intrinsics.get("fy", 1000.0)
            cx = intrinsics.get("cx", 960.0)
            cy = intrinsics.get("cy", 540.0)

            from poc_homography.calibration.lens_distortion.apply_calibration import (
                undistort_points,
            )

            pts = undistort_points(pts, k1, k2, k3, p1, p2, fx, fy, cx, cy)

        from poc_homography.calibration.lens_distortion.apply_calibration import (
            measure_line_straightness,
        )

        result = measure_line_straightness(pts)

        rmse = result["rmse_pixels"]
        result["is_straight"] = rmse < 2.0
        result["quality"] = (
            "excellent"
            if rmse < 0.5
            else "good"
            if rmse < 2.0
            else "acceptable"
            if rmse < 5.0
            else "poor"
        )
        result["undistorted"] = should_undistort

        return JsonResponse(result)

    except Exception:
        logger.exception("Failed to measure straightness")
        return JsonResponse({"error": "Failed to measure straightness"}, status=500)
