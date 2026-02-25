"""Views for the lens calibration Django app.

This app provides a web UI for **performing** lens distortion calibration.
Users supply calibration lines (either directly or from the camera_line_annotator
tool), run the distortion solver, and save the resulting calibration coefficients
to YAML files for later use.

Distinct from ``distortion_validator``, which only *evaluates* existing
calibrations, this app runs the actual optimisation and persists results.

CSRF protection
---------------
POST endpoints use Django's default CSRF protection.  The ``index`` view is
decorated with ``@ensure_csrf_cookie`` so the JavaScript frontend receives
the CSRF token cookie on initial page load.
"""

from __future__ import annotations

import json
import logging
from typing import Any

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
    save_calibration_to_repo,
    serialize_calibration_entry,
)
from homography_web.frame_utils import CALIBRATION_LINE_TRACES_DIR, CALIBRATIONS_DIR

logger = logging.getLogger(__name__)

# Cached repo
_line_trace_set_repo = None


def _get_line_trace_set_repo():
    """Return a cached RepoYamlCalibrationLineTraceSet instance."""
    global _line_trace_set_repo
    if _line_trace_set_repo is None:
        from poc_homography.infrastructure.repositories import (
            RepoYamlCalibrationLineTraceSet,
        )

        _line_trace_set_repo = RepoYamlCalibrationLineTraceSet(CALIBRATION_LINE_TRACES_DIR)
    return _line_trace_set_repo


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


@ensure_csrf_cookie
def index(request: HttpRequest) -> HttpResponse:
    """Serve the main HTML page."""
    return render(request, "lens_calibration/index.html")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


def _build_intrinsic_matrix(data: dict) -> tuple[Any, dict]:
    """Build intrinsic matrix from request data, computing from specs if requested.

    Returns (intrinsic_matrix, intrinsics_dict).
    """
    import numpy as np

    intrinsics = data["intrinsics"]

    # If auto_intrinsics is explicitly requested, compute from specs
    if data.get("auto_intrinsics"):
        from poc_homography.camera.intrinsics import compute_intrinsics
        from poc_homography.camera_config import (
            DEFAULT_BASE_FOCAL_LENGTH_MM,
            DEFAULT_SENSOR_WIDTH_MM,
        )

        zoom = float(data.get("zoom", intrinsics.get("zoom", 1.0)))
        result = compute_intrinsics(
            zoom=zoom,
            image_width=int(intrinsics.get("image_width", 1920)),
            image_height=int(intrinsics.get("image_height", 1080)),
            sensor_width_mm=float(intrinsics.get("sensor_width_mm", DEFAULT_SENSOR_WIDTH_MM)),
            base_focal_length_mm=float(
                intrinsics.get("base_focal_length_mm", DEFAULT_BASE_FOCAL_LENGTH_MM)
            ),
        )
        intrinsics = {
            "fx": float(result.focal_length_px),
            "fy": float(result.focal_length_px),
            "cx": float(result.cx),
            "cy": float(result.cy),
        }

    fx = intrinsics.get("fx", 1000.0)
    fy = intrinsics.get("fy", 1000.0)
    cx = intrinsics.get("cx", 960.0)
    cy = intrinsics.get("cy", 540.0)

    intrinsic_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]
    )

    return intrinsic_matrix, {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


@require_http_methods(["POST"])
def api_calibrate_annotated_lines(request: HttpRequest) -> JsonResponse:
    """Run distortion calibration using manually annotated N-point line traces.

    Request body::

        {
            "camera_line_annotations": [
                {"line_id": "L1", "points": [[x1,y1], [x2,y2], ...]},
                ...
            ],
            "intrinsics": {"fx": ..., "fy": ..., "cx": ..., "cy": ...},
            "config": {"train_split_ratio": 0.7}  // optional
        }
    """
    MAX_LINE_ANNOTATIONS = 500  # Prevent DoS via excessive input

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "intrinsics" not in data:
        return JsonResponse({"error": "Missing intrinsics"}, status=400)

    # Validate camera_line_annotations exists and is non-empty
    annotations = data.get("camera_line_annotations")
    if not annotations or not isinstance(annotations, list):
        return JsonResponse({"error": "Missing or invalid camera_line_annotations"}, status=400)

    if len(annotations) > MAX_LINE_ANNOTATIONS:
        return JsonResponse(
            {"error": f"Too many line annotations (max {MAX_LINE_ANNOTATIONS})"}, status=400
        )

    try:
        from poc_homography.calibration.lens_distortion.annotated_line_solver import (
            AnnotatedLineSolver,
            AnnotatedLineSolverConfig,
            build_camera_line_annotations,
        )

        intrinsic_matrix, intrinsics_used = _build_intrinsic_matrix(data)

        lines = build_camera_line_annotations(annotations)

        config_data = data.get("config", {})
        solver_config = AnnotatedLineSolverConfig(
            train_split_ratio=config_data.get("train_split_ratio", 0.7),
            use_radial_only=config_data.get("use_radial_only", False),
        )
        solver = AnnotatedLineSolver(config=solver_config)

        result = solver.solve(lines, intrinsic_matrix)

        response_data: dict[str, Any] = {
            "success": result.success,
            "message": result.message,
            "iterations": result.iterations,
            "initial_error": result.initial_error,
            "final_error": result.final_error,
            "overall_rmse": result.overall_rmse,
            "coefficients": {
                "k1": float(result.distortion.k1),
                "k2": float(result.distortion.k2),
                "k3": float(result.distortion.k3),
                "p1": float(result.distortion.p1),
                "p2": float(result.distortion.p2),
            },
            "intrinsics_used": intrinsics_used,
            "quality": (
                "good"
                if result.overall_rmse < 2.0
                else "acceptable"
                if result.overall_rmse < 5.0
                else "poor"
            ),
            "line_errors": result.line_errors[:20],
        }

        if result.success:
            response_data["improvement_percent"] = (1 - result.improvement_ratio()) * 100
        else:
            response_data["improvement_percent"] = 0.0

        if result.intrinsics:
            response_data["intrinsics"] = result.intrinsics

        return JsonResponse(response_data)

    except ImportError:
        logger.exception("Annotated line solver module not available")
        return JsonResponse(
            {"error": "Annotated line solver module not available"},
            status=500,
        )
    except Exception:
        logger.exception("Annotated line calibration failed")
        return JsonResponse(
            {"error": "Annotated line calibration failed"},
            status=500,
        )


@require_http_methods(["POST"])
def api_validate(request: HttpRequest) -> JsonResponse:
    """Validate calibration by computing straightness RMSE on test lines."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
        import numpy as np

        from poc_homography.calibration.lens_distortion.distortion_solver import (
            straightness_rmse,
        )
        from poc_homography.calibration.lens_distortion.models import (
            CameraLine,
            PTZPosition,
        )
        from poc_homography.domain.vo import LensDistortion
        from poc_homography.types import Degrees, Unitless

        intrinsics = data["intrinsics"]
        intrinsic_matrix = np.array(
            [
                [intrinsics.get("fx", 1000.0), 0.0, intrinsics.get("cx", 960.0)],
                [0.0, intrinsics.get("fy", 1000.0), intrinsics.get("cy", 540.0)],
                [0.0, 0.0, 1.0],
            ]
        )

        lines_data = data.get("lines", [])
        camera_lines = []
        for i, line in enumerate(lines_data):
            ptz = PTZPosition(
                pan_deg=Degrees(line.get("pan", 0.0)),
                tilt_deg=Degrees(line.get("tilt", 30.0)),
                zoom_factor=line.get("zoom", 1.0),
            )
            # Pass through edge_pixels if provided by the client
            edge_pixels = None
            points = line.get("points")
            if points and len(points) >= 2:
                edge_pixels = tuple(tuple(pt) for pt in points)
            camera_line = CameraLine(
                line_id=line.get("line_id", f"line_{i:04d}"),
                image_path=line.get("image_path", ""),
                start_pixel=(line["start_x"], line["start_y"]),
                end_pixel=(line["end_x"], line["end_y"]),
                ptz_position=ptz,
                edge_pixels=edge_pixels,
            )
            camera_lines.append(camera_line)

        if not camera_lines:
            return JsonResponse({"error": "No lines provided"}, status=400)

        baseline_rmse = straightness_rmse(camera_lines, intrinsic_matrix)

        coeffs = data.get("coefficients", {})
        distortion = LensDistortion(
            k1=Unitless(coeffs.get("k1", 0.0)),
            k2=Unitless(coeffs.get("k2", 0.0)),
            k3=Unitless(coeffs.get("k3", 0.0)),
            p1=Unitless(coeffs.get("p1", 0.0)),
            p2=Unitless(coeffs.get("p2", 0.0)),
        )
        corrected_rmse = straightness_rmse(camera_lines, intrinsic_matrix, distortion=distortion)

        improvement = (
            (baseline_rmse - corrected_rmse) / baseline_rmse * 100 if baseline_rmse > 0 else 0
        )

        return JsonResponse(
            {
                "baseline_rmse": baseline_rmse,
                "corrected_rmse": corrected_rmse,
                "improvement_percent": improvement,
                "num_lines": len(camera_lines),
            }
        )

    except ImportError:
        logger.exception("Calibration module not available")
        return JsonResponse({"error": "Calibration module not available"}, status=500)
    except Exception:
        logger.exception("Validation failed")
        return JsonResponse({"error": "Validation failed"}, status=500)


@require_http_methods(["POST"])
def api_save(request: HttpRequest) -> JsonResponse:
    """Save calibration results via the DDD repo."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
        from poc_homography.calibration.lens_distortion.calibration_table import (
            CameraCalibrationTable,
            ZoomCalibrationEntry,
        )
        from poc_homography.domain.vo import LensDistortion
        from poc_homography.types import Unitless

        camera_id = data.get("camera_id", "unknown_camera")
        zoom = data.get("zoom", 1.0)
        coeffs = data.get("coefficients", {})
        validation_rmse = data.get("validation_rmse", 0.0)
        intrinsics = data.get("intrinsics", {})

        distortion = LensDistortion(
            k1=Unitless(coeffs.get("k1", 0.0)),
            k2=Unitless(coeffs.get("k2", 0.0)),
            k3=Unitless(coeffs.get("k3", 0.0)),
            p1=Unitless(coeffs.get("p1", 0.0)),
            p2=Unitless(coeffs.get("p2", 0.0)),
        )

        table = CameraCalibrationTable(camera_id=camera_id)
        entry = ZoomCalibrationEntry.from_solver_result(
            zoom_factor=zoom,
            distortion=distortion,
            validation_rmse=validation_rmse,
            source_images=[],
            num_lines_used=data.get("num_lines", 0),
            fx=float(intrinsics.get("fx", 0.0)),
            fy=float(intrinsics.get("fy", 0.0)),
            cx=float(intrinsics.get("cx", 0.0)),
            cy=float(intrinsics.get("cy", 0.0)),
        )
        table.add_entry(entry)

        save_calibration_to_repo(table, CALIBRATIONS_DIR)

        return JsonResponse(
            {
                "success": True,
                "camera_id": camera_id,
            }
        )

    except ImportError:
        logger.exception("Calibration module not available")
        return JsonResponse({"error": "Calibration module not available"}, status=500)
    except Exception:
        logger.exception("Save failed")
        return JsonResponse({"error": "Save failed"}, status=500)


@require_http_methods(["POST"])
def api_load(request: HttpRequest) -> JsonResponse:
    """Load calibration from DDD repo by camera_id."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
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

    except ImportError:
        logger.exception("Calibration module not available")
        return JsonResponse({"error": "Calibration module not available"}, status=500)
    except Exception:
        logger.exception("Load failed")
        return JsonResponse({"error": "Load failed"}, status=500)


@require_GET
def api_calibration_ids(request: HttpRequest) -> JsonResponse:
    """List available camera_ids in the calibration repo."""
    try:
        camera_ids = list_calibration_ids(CALIBRATIONS_DIR)
        return JsonResponse({"camera_ids": camera_ids})
    except Exception:
        logger.exception("Failed to list calibration IDs")
        return JsonResponse({"error": "Failed to list calibration IDs"}, status=500)


# ---------------------------------------------------------------------------
# Line trace sets API (DDD repo)
# ---------------------------------------------------------------------------


@require_GET
def api_line_trace_sets(request: HttpRequest) -> JsonResponse:
    """List available line trace sources.

    Merges CalibrationLineTraceSet entity names with LineAnnotation frame
    groups so that annotations created via camera_line_annotator also appear.
    """
    try:
        repo = _get_line_trace_set_repo()
        entities = repo.get_all()
        names = sorted(e.name for e in entities)

        # Also include LineAnnotation entities grouped by frame_id
        from homography_web.frame_utils import get_line_annotation_repo

        frame_ids = sorted({ann.frame_id for ann in get_line_annotation_repo().get_all()})

        return JsonResponse({"names": names + frame_ids})
    except Exception:
        logger.exception("Failed to list line trace sets")
        return JsonResponse({"error": "Failed to list line trace sets"}, status=500)


@require_GET
def api_line_trace_set_detail(request: HttpRequest) -> JsonResponse:
    """Load line traces by name.

    First tries CalibrationLineTraceSet entities, then falls back to
    LineAnnotation entities grouped by frame_id.
    """
    name = request.GET.get("name", "")
    if not name:
        return JsonResponse({"error": "Missing name"}, status=400)

    try:
        # Try CalibrationLineTraceSet first
        repo = _get_line_trace_set_repo()
        entity = repo.get(name)
        if entity is not None:
            return JsonResponse(
                {
                    "name": entity.name,
                    "line_traces": [lt.to_dict() for lt in entity.line_traces],
                }
            )

        # Fall back to LineAnnotation entities matching this frame_id
        from homography_web.frame_utils import get_line_annotation_repo

        frame_anns = [
            ann for ann in get_line_annotation_repo().get_all() if ann.frame_id == name
        ]
        if frame_anns:
            line_traces = []
            for ann in frame_anns:
                if ann.points is not None:
                    points = [[float(p.x), float(p.y)] for p in ann.points]
                else:
                    points = [
                        [float(ann.start_pixel.x), float(ann.start_pixel.y)],
                        [float(ann.end_pixel.x), float(ann.end_pixel.y)],
                    ]
                line_traces.append({"line_id": ann.line_id, "points": points})
            return JsonResponse({"name": name, "line_traces": line_traces})

        return JsonResponse({"error": f"Not found: {name}"}, status=404)
    except Exception:
        logger.exception("Failed to load line trace set %s", name)
        return JsonResponse({"error": "Failed to load line trace set"}, status=500)
