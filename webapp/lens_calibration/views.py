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
from pathlib import Path
from typing import Any

import yaml
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods
from homography_web.calibration_utils import (
    get_cached_calibration_table as _get_cached_calibration_table,
)
from homography_web.calibration_utils import (
    resolve_safe_path as _resolve_safe_path,
)

# Paths
WEBAPP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEBAPP_DIR.parent
SURVEY_DIR = WEBAPP_DIR / "survey"
CALIBRATION_DIR = PROJECT_ROOT / "calibration_results"
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "homography" / "test_data"

logger = logging.getLogger(__name__)


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

@require_GET
def api_survey_sessions(request: HttpRequest) -> JsonResponse:
    """List available survey sessions with images."""
    try:
        sessions = []
        if SURVEY_DIR.exists():
            for date_dir in sorted(SURVEY_DIR.iterdir(), reverse=True):
                if not date_dir.is_dir():
                    continue
                for session_dir in sorted(date_dir.iterdir(), reverse=True):
                    if not session_dir.is_dir():
                        continue
                    images = list(session_dir.glob("*.jpg")) + list(session_dir.glob("*.png"))
                    manifest_path = session_dir / "manifest.yaml"
                    manifest = {}
                    if manifest_path.exists():
                        with open(manifest_path) as f:
                            manifest = yaml.safe_load(f) or {}

                    sessions.append(
                        {
                            "date": date_dir.name,
                            "session_id": session_dir.name,
                            "image_count": len(images),
                            "manifest": manifest,
                        }
                    )

        return JsonResponse({"sessions": sessions})
    except Exception:
        logger.exception("Failed to list survey sessions")
        return JsonResponse({"error": "Failed to list survey sessions"}, status=500)


from homography_web.calibration_utils import (
    api_compute_intrinsics,  # noqa: F401 - re-exported for URL routing
    serialize_calibration_entry,
)
api_compute_intrinsics = api_compute_intrinsics  # make linter happy


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
            sensor_width_mm=float(
                intrinsics.get("sensor_width_mm", DEFAULT_SENSOR_WIDTH_MM)
            ),
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

    intrinsic_matrix = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ])

    return intrinsic_matrix, {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


@require_http_methods(["POST"])
def api_calibrate(request: HttpRequest) -> JsonResponse:
    # TODO: Move to background task (Celery/Django-Q) for production use
    """Run distortion calibration on provided lines or images."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "intrinsics" not in data:
        return JsonResponse({"error": "Missing intrinsics"}, status=400)

    source = data.get("source", "lines")

    try:
        import numpy as np

        from poc_homography.calibration.lens_distortion.distortion_solver import (
            DistortionSolver,
            SolverConfig,
        )
        from poc_homography.calibration.lens_distortion.line_detection import (
            LineDetectionConfig,
            LineDetector,
        )
        from poc_homography.calibration.lens_distortion.models import (
            CameraLine,
            PTZPosition,
        )
        from poc_homography.types import Degrees

        intrinsic_matrix, intrinsics_used = _build_intrinsic_matrix(data)

        MAX_ITERATIONS_CAP = 10000
        config_data = data.get("config", {})
        solver_config = SolverConfig(
            use_radial_only=config_data.get("radial_only", False),
            max_iterations=min(config_data.get("max_iterations", 1000), MAX_ITERATIONS_CAP),
            optimize_intrinsics=config_data.get("optimize_intrinsics", False),
        )

        camera_lines: list[CameraLine] = []

        if source == "lines":
            lines_data = data.get("lines", [])
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
                    confidence=line.get("confidence", 1.0),
                    edge_pixels=edge_pixels,
                )
                camera_lines.append(camera_line)

        elif source == "images":
            TEST_DATA_DIR = PROJECT_ROOT / "tests" / "homography" / "test_data"
            images_path_str = data.get("images_path", "")
            # Only allow paths relative to known directories
            images_path = None
            for base in (SURVEY_DIR, TEST_DATA_DIR):
                try:
                    candidate = (base / images_path_str).resolve()
                    if candidate.is_relative_to(base.resolve()) and candidate.exists():
                        images_path = candidate
                        break
                except (ValueError, RuntimeError):
                    continue

            if images_path is None or not images_path.exists():
                return JsonResponse({"error": "Images path not found"}, status=400)

            detection_config = LineDetectionConfig(
                min_line_length=config_data.get("min_line_length", 100),
                min_confidence=config_data.get("min_confidence", 0.3),
            )
            detector = LineDetector(config=detection_config)

            image_extensions = {".jpg", ".jpeg", ".png"}
            images = []
            for ext in image_extensions:
                images.extend(images_path.glob(f"*{ext}"))
                images.extend(images_path.glob(f"*{ext.upper()}"))

            max_lines_per_image = config_data.get("max_lines_per_image", 10)
            line_counter = 0

            for img_path in sorted(images):
                try:
                    candidates = detector.detect_from_file(img_path)
                    ptz = PTZPosition(
                        pan_deg=Degrees(0.0),
                        tilt_deg=Degrees(30.0),
                        zoom_factor=1.0,
                    )
                    for c in candidates[:max_lines_per_image]:
                        camera_line = c.to_camera_line(
                            line_id=f"line_{line_counter:04d}",
                            image_path=str(img_path.name),
                            ptz_position=ptz,
                        )
                        camera_lines.append(camera_line)
                        line_counter += 1
                except Exception:
                    logger.warning("Failed to process image during calibration", exc_info=True)

        else:
            return JsonResponse({"error": "Invalid source"}, status=400)

        if len(camera_lines) < 1:
            return JsonResponse({"error": "No lines provided or detected"}, status=400)

        solver = DistortionSolver(config=solver_config)
        result = solver.solve(camera_lines, intrinsic_matrix)

        response_data: dict[str, Any] = {
            "success": result.success,
            "message": result.message,
            "iterations": result.iterations,
            "num_lines": len(camera_lines),
            "initial_error": result.initial_error,
            "final_error": result.final_error,
            "improvement_percent": (1 - result.improvement_ratio()) * 100,
            "overall_rmse": result.overall_rmse,
            "coefficients": {
                "k1": float(result.distortion.k1),
                "k2": float(result.distortion.k2),
                "k3": float(result.distortion.k3),
                "p1": float(result.distortion.p1),
                "p2": float(result.distortion.p2),
            },
            "intrinsics_used": intrinsics_used,
            "quality": "good" if result.overall_rmse < 2.0 else "acceptable" if result.overall_rmse < 5.0 else "poor",
            "line_errors": result.line_errors[:20],
        }

        if result.intrinsics:
            response_data["optimized_intrinsics"] = result.intrinsics

        return JsonResponse(response_data)

    except ImportError:
        logger.exception("Calibration module not available")
        return JsonResponse({"error": "Calibration module not available"}, status=500)
    except ValueError:
        logger.exception("Calibration validation error")
        return JsonResponse({"error": "Calibration validation error"}, status=400)
    except Exception:
        logger.exception("Calibration failed")
        return JsonResponse({"error": "Calibration failed"}, status=500)


@require_http_methods(["POST"])
def api_calibrate_from_calibration_files(request: HttpRequest) -> JsonResponse:
    # TODO: Move to background task (Celery/Django-Q) for production use
    """Run distortion calibration from camera_line_annotator calibration JSON files."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "intrinsics" not in data:
        return JsonResponse({"error": "Missing intrinsics"}, status=400)

    if "files" not in data or not data["files"]:
        return JsonResponse({"error": "No calibration files provided"}, status=400)

    try:
        import numpy as np

        from poc_homography.calibration.lens_distortion.distortion_solver import (
            DistortionSolver,
            SolverConfig,
        )
        from poc_homography.calibration.lens_distortion.models import (
            CameraLine,
            PTZPosition,
        )
        from poc_homography.types import Degrees

        intrinsic_matrix, intrinsics_used = _build_intrinsic_matrix(data)

        MAX_ITERATIONS_CAP = 10000
        config_data = data.get("config", {})
        solver_config = SolverConfig(
            use_radial_only=config_data.get("radial_only", False),
            max_iterations=min(config_data.get("max_iterations", 1000), MAX_ITERATIONS_CAP),
            optimize_intrinsics=config_data.get("optimize_intrinsics", False),
        )

        camera_lines: list[CameraLine] = []
        frame_info: list[dict] = []

        for file_data in data["files"]:
            image_name = file_data.get("image", "unknown")
            camera_status = file_data.get("camera_status", {})
            calibration_lines = file_data.get("calibration_lines", [])

            ptz = PTZPosition(
                pan_deg=Degrees(camera_status.get("pan", 0.0)),
                tilt_deg=Degrees(camera_status.get("tilt", 30.0)),
                zoom_factor=camera_status.get("zoom", 1.0),
            )

            frame_info.append({
                "image": image_name,
                "ptz": {"pan": float(ptz.pan_deg), "tilt": float(ptz.tilt_deg), "zoom": ptz.zoom_factor},
                "num_lines": len(calibration_lines),
            })

            for cal_line in calibration_lines:
                points = cal_line.get("points", [])
                if len(points) < 3:
                    continue

                line_id = cal_line.get("line_id") or f"line_{cal_line.get('index', 0):04d}"
                edge_pixels = tuple(tuple(pt) for pt in points)
                start_pixel = tuple(points[0])
                end_pixel = tuple(points[-1])

                camera_line = CameraLine(
                    line_id=f"{line_id}_{image_name}",
                    image_path=image_name,
                    start_pixel=start_pixel,
                    end_pixel=end_pixel,
                    ptz_position=ptz,
                    confidence=1.0,
                    edge_pixels=edge_pixels,
                )
                camera_lines.append(camera_line)

        if len(camera_lines) < 1:
            return JsonResponse({"error": "No valid calibration lines found"}, status=400)

        solver = DistortionSolver(config=solver_config)
        result = solver.solve(camera_lines, intrinsic_matrix)

        response_data: dict[str, Any] = {
            "success": result.success,
            "message": result.message,
            "iterations": result.iterations,
            "num_lines": len(camera_lines),
            "num_frames": len(data["files"]),
            "frame_info": frame_info,
            "initial_error": result.initial_error,
            "final_error": result.final_error,
            "improvement_percent": (1 - result.improvement_ratio()) * 100,
            "overall_rmse": result.overall_rmse,
            "coefficients": {
                "k1": float(result.distortion.k1),
                "k2": float(result.distortion.k2),
                "k3": float(result.distortion.k3),
                "p1": float(result.distortion.p1),
                "p2": float(result.distortion.p2),
            },
            "intrinsics_used": intrinsics_used,
            "quality": "good" if result.overall_rmse < 2.0 else "acceptable" if result.overall_rmse < 5.0 else "poor",
            "line_errors": result.line_errors[:20],
        }

        if result.intrinsics:
            response_data["optimized_intrinsics"] = result.intrinsics

        return JsonResponse(response_data)

    except ImportError:
        logger.exception("Calibration module not available")
        return JsonResponse({"error": "Calibration module not available"}, status=500)
    except ValueError:
        logger.exception("Calibration validation error")
        return JsonResponse({"error": "Calibration validation error"}, status=400)
    except Exception:
        logger.exception("Calibration failed")
        return JsonResponse({"error": "Calibration failed"}, status=500)


@csrf_exempt
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
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "intrinsics" not in data:
        return JsonResponse({"error": "Missing intrinsics"}, status=400)

    try:
        from poc_homography.calibration.lens_distortion.opencv_solver import (
            AnnotatedLineSolver,
            AnnotatedLineSolverConfig,
            build_camera_line_annotations,
        )

        intrinsic_matrix, intrinsics_used = _build_intrinsic_matrix(data)

        lines = build_camera_line_annotations(
            data.get("camera_line_annotations", []),
        )

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
                "good" if result.overall_rmse < 2.0
                else "acceptable" if result.overall_rmse < 5.0
                else "poor"
            ),
            "line_errors": result.line_errors[:20],
        }

        if result.success:
            response_data["improvement_percent"] = (
                (1 - result.improvement_ratio()) * 100
            )
        else:
            response_data["improvement_percent"] = 0.0

        if result.intrinsics:
            response_data["intrinsics"] = result.intrinsics

        return JsonResponse(response_data)

    except ImportError:
        logger.exception("Annotated line solver module not available")
        return JsonResponse(
            {"error": "Annotated line solver module not available"}, status=500,
        )
    except Exception:
        logger.exception("Annotated line calibration failed")
        return JsonResponse(
            {"error": "Annotated line calibration failed"}, status=500,
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
        from poc_homography.camera_parameters import DistortionCoefficients
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
        distortion = DistortionCoefficients(
            k1=Unitless(coeffs.get("k1", 0.0)),
            k2=Unitless(coeffs.get("k2", 0.0)),
            k3=Unitless(coeffs.get("k3", 0.0)),
            p1=Unitless(coeffs.get("p1", 0.0)),
            p2=Unitless(coeffs.get("p2", 0.0)),
        )
        corrected_rmse = straightness_rmse(camera_lines, intrinsic_matrix, distortion=distortion)

        improvement = (baseline_rmse - corrected_rmse) / baseline_rmse * 100 if baseline_rmse > 0 else 0

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
    """Save calibration results to YAML file."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
        from poc_homography.calibration.lens_distortion.calibration_table import (
            CameraCalibrationTable,
            ZoomCalibrationEntry,
        )
        from poc_homography.camera_parameters import DistortionCoefficients
        from poc_homography.types import Unitless

        camera_id = data.get("camera_id", "unknown_camera")
        zoom = data.get("zoom", 1.0)
        coeffs = data.get("coefficients", {})
        validation_rmse = data.get("validation_rmse", 0.0)
        intrinsics = data.get("intrinsics", {})

        # Validate output filename
        filename = data.get("filename", f"{camera_id}_calibration.yaml")
        resolved = _resolve_safe_path(filename, CALIBRATION_DIR)
        if resolved is None:
            return JsonResponse({"error": "Invalid filename"}, status=400)

        distortion = DistortionCoefficients(
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

        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        table.save(resolved)

        return JsonResponse(
            {
                "success": True,
                "filename": filename,
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
    """Load calibration from YAML file."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
        filename = data.get("filename", "")

        resolved = _resolve_safe_path(filename, CALIBRATION_DIR)
        if resolved is None or not resolved.exists():
            return JsonResponse({"error": "Invalid or missing filename"}, status=400)

        table = _get_cached_calibration_table(resolved)

        entries = [
            serialize_calibration_entry(entry)
            for entry in table.entries.values()
        ]

        return JsonResponse({
            "camera_id": table.camera_id,
            "entries": entries,
        })

    except ImportError:
        logger.exception("Calibration module not available")
        return JsonResponse({"error": "Calibration module not available"}, status=500)
    except Exception:
        logger.exception("Load failed")
        return JsonResponse({"error": "Load failed"}, status=500)


# ---------------------------------------------------------------------------
# Test data files API
# ---------------------------------------------------------------------------

@require_GET
def api_test_data_files(request: HttpRequest) -> JsonResponse:
    """List YAML files in the test data directory."""
    if not TEST_DATA_DIR.exists():
        return JsonResponse({"files": []})

    files = sorted(f.name for f in TEST_DATA_DIR.glob("*.yaml"))
    return JsonResponse({"files": files})


@require_GET
def api_test_data_file_content(request: HttpRequest) -> JsonResponse:
    """Read and return parsed YAML content of a test data file."""
    filename = request.GET.get("filename", "")
    if not filename:
        return JsonResponse({"error": "Missing filename"}, status=400)

    # Prevent path traversal
    filepath = (TEST_DATA_DIR / filename).resolve()
    if not filepath.is_relative_to(TEST_DATA_DIR.resolve()) or not filepath.exists():
        return JsonResponse({"error": "File not found"}, status=404)

    try:
        with open(filepath) as f:
            data = yaml.safe_load(f)
        return JsonResponse({"filename": filename, "data": data})
    except Exception:
        logger.exception("Failed to read test data file %s", filename)
        return JsonResponse({"error": "Failed to read file"}, status=500)
