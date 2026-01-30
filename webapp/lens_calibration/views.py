"""Views for the lens calibration Django app.

This app provides a web UI for **performing** lens distortion calibration.
Users supply calibration lines (either directly or from the camera_line_annotator
tool), run the distortion solver, and save the resulting calibration coefficients
to YAML files for later use.

Distinct from ``distortion_validator``, which only *evaluates* existing
calibrations, this app runs the actual optimisation and persists results.

CSRF exemption rationale
------------------------
All ``@csrf_exempt`` endpoints in this module are internal development / lab
tools.  The Django server is bound to ``localhost`` only, there is no user
authentication, and all data is non-sensitive calibration imagery.  CSRF
protection is therefore unnecessary and would complicate programmatic API
access from the companion JavaScript frontend.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

# Paths
WEBAPP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEBAPP_DIR.parent
SURVEY_DIR = WEBAPP_DIR / "survey"
CALIBRATION_DIR = PROJECT_ROOT / "calibration_results"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filename validation helpers
# ---------------------------------------------------------------------------

def _validate_filename(filename: str) -> bool:
    """Validate filename to prevent path traversal attacks."""
    if not filename:
        return False
    if "/" in filename or ".." in filename or "\\" in filename:
        return False
    return True


def _resolve_safe_path(filename: str, base_dir: Path) -> Path | None:
    """Resolve *filename* under *base_dir*, returning ``None`` on traversal."""
    if not _validate_filename(filename):
        return None
    try:
        resolved = (base_dir / filename).resolve()
        if not resolved.is_relative_to(base_dir.resolve()):
            return None
        return resolved
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Calibration file cache
# ---------------------------------------------------------------------------

def _load_calibration_table_cached(filepath: Path):
    """Load a calibration table with mtime-based caching."""
    from poc_homography.calibration.lens_distortion.calibration_table import (
        CameraCalibrationTable,
    )
    return CameraCalibrationTable.load(filepath)


_calibration_cache: dict[tuple[str, float], Any] = {}


def _get_cached_calibration_table(filepath: Path):
    """Return a cached CameraCalibrationTable, invalidated by mtime."""
    key_path = str(filepath)
    mtime = filepath.stat().st_mtime
    cache_key = (key_path, mtime)
    if cache_key not in _calibration_cache:
        # Evict stale entries for this path
        _calibration_cache.pop(
            next((k for k in _calibration_cache if k[0] == key_path), None),  # type: ignore[arg-type]
            None,
        )
        _calibration_cache[cache_key] = _load_calibration_table_cached(filepath)
    return _calibration_cache[cache_key]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

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


@csrf_exempt
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

        intrinsics = data["intrinsics"]
        intrinsic_matrix = np.array(
            [
                [intrinsics.get("fx", 1000.0), 0.0, intrinsics.get("cx", 960.0)],
                [0.0, intrinsics.get("fy", 1000.0), intrinsics.get("cy", 540.0)],
                [0.0, 0.0, 1.0],
            ]
        )

        config_data = data.get("config", {})
        solver_config = SolverConfig(
            use_radial_only=config_data.get("radial_only", False),
            max_iterations=config_data.get("max_iterations", 1000),
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
                camera_line = CameraLine(
                    line_id=line.get("line_id", f"line_{i:04d}"),
                    image_path=line.get("image_path", ""),
                    start_pixel=(line["start_x"], line["start_y"]),
                    end_pixel=(line["end_x"], line["end_y"]),
                    ptz_position=ptz,
                    confidence=line.get("confidence", 1.0),
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
            "quality": "good" if result.overall_rmse < 2.0 else "acceptable" if result.overall_rmse < 5.0 else "poor",
            "line_errors": result.line_errors[:20],
        }

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

        intrinsics = data["intrinsics"]
        intrinsic_matrix = np.array(
            [
                [intrinsics.get("fx", 1000.0), 0.0, intrinsics.get("cx", 960.0)],
                [0.0, intrinsics.get("fy", 1000.0), intrinsics.get("cy", 540.0)],
                [0.0, 0.0, 1.0],
            ]
        )

        config_data = data.get("config", {})
        solver_config = SolverConfig(
            use_radial_only=config_data.get("radial_only", False),
            max_iterations=config_data.get("max_iterations", 1000),
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
                if len(points) < 2:
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
            "quality": "good" if result.overall_rmse < 2.0 else "acceptable" if result.overall_rmse < 5.0 else "poor",
            "line_errors": result.line_errors[:20],
        }

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
            camera_line = CameraLine(
                line_id=line.get("line_id", f"line_{i:04d}"),
                image_path=line.get("image_path", ""),
                start_pixel=(line["start_x"], line["start_y"]),
                end_pixel=(line["end_x"], line["end_y"]),
                ptz_position=ptz,
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


@csrf_exempt
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


@csrf_exempt
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

        entries = []
        for zoom, entry in table.entries.items():
            entries.append(
                {
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
                }
            )

        return JsonResponse(
            {
                "camera_id": table.camera_id,
                "entries": entries,
            }
        )

    except ImportError:
        logger.exception("Calibration module not available")
        return JsonResponse({"error": "Calibration module not available"}, status=500)
    except Exception:
        logger.exception("Load failed")
        return JsonResponse({"error": "Load failed"}, status=500)
