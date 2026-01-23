"""Views for the lens calibration Django app."""

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


def index(request: HttpRequest) -> HttpResponse:
    """Serve the main HTML page."""
    return render(request, "lens_calibration/index.html")


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
                    # Count images in session
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
                            "path": str(session_dir),
                            "image_count": len(images),
                            "manifest": manifest,
                        }
                    )

        return JsonResponse({"sessions": sessions})
    except Exception as e:
        logger.exception("Failed to list survey sessions")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def api_calibrate(request: HttpRequest) -> JsonResponse:
    """Run distortion calibration on provided lines or images.

    Request body:
    {
        "source": "lines" | "images",
        "lines": [...] (if source=lines),
        "images_path": "..." (if source=images),
        "intrinsics": {
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": 960.0,
            "cy": 540.0
        },
        "config": {
            "radial_only": false,
            "max_iterations": 1000,
            "min_line_length": 100,
            "min_confidence": 0.3,
            "max_lines_per_image": 10
        }
    }
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Validate required fields
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

        # Build intrinsic matrix
        intrinsics = data["intrinsics"]
        intrinsic_matrix = np.array(
            [
                [intrinsics.get("fx", 1000.0), 0.0, intrinsics.get("cx", 960.0)],
                [0.0, intrinsics.get("fy", 1000.0), intrinsics.get("cy", 540.0)],
                [0.0, 0.0, 1.0],
            ]
        )

        # Get config
        config_data = data.get("config", {})
        solver_config = SolverConfig(
            use_radial_only=config_data.get("radial_only", False),
            max_iterations=config_data.get("max_iterations", 1000),
        )

        # Get lines
        camera_lines: list[CameraLine] = []

        if source == "lines":
            # Lines provided directly
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
            # Detect lines from images
            images_path = Path(data.get("images_path", ""))
            if not images_path.exists():
                return JsonResponse({"error": f"Images path not found: {images_path}"}, status=400)

            detection_config = LineDetectionConfig(
                min_line_length=config_data.get("min_line_length", 100),
                min_confidence=config_data.get("min_confidence", 0.3),
            )
            detector = LineDetector(config=detection_config)

            # Find images
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
                    # Default PTZ (could extract from filename)
                    ptz = PTZPosition(
                        pan_deg=Degrees(0.0),
                        tilt_deg=Degrees(30.0),
                        zoom_factor=1.0,
                    )

                    for c in candidates[:max_lines_per_image]:
                        camera_line = c.to_camera_line(
                            line_id=f"line_{line_counter:04d}",
                            image_path=str(img_path),
                            ptz_position=ptz,
                        )
                        camera_lines.append(camera_line)
                        line_counter += 1
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")

        else:
            return JsonResponse({"error": f"Invalid source: {source}"}, status=400)

        if len(camera_lines) < 1:
            return JsonResponse({"error": "No lines provided or detected"}, status=400)

        # Run calibration
        solver = DistortionSolver(config=solver_config)
        result = solver.solve(camera_lines, intrinsic_matrix)

        # Build response
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
            "line_errors": result.line_errors[:20],  # Top 20 worst lines
        }

        return JsonResponse(response_data)

    except ImportError as e:
        logger.exception("Calibration module not available")
        return JsonResponse({"error": f"Calibration module not available: {e}"}, status=500)
    except ValueError as e:
        logger.exception("Calibration validation error")
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("Calibration failed")
        return JsonResponse({"error": f"Calibration failed: {e}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def api_validate(request: HttpRequest) -> JsonResponse:
    """Validate calibration by computing straightness RMSE on test lines.

    Request body:
    {
        "lines": [...],
        "intrinsics": {...},
        "coefficients": {...}
    }
    """
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

        # Build intrinsic matrix
        intrinsics = data["intrinsics"]
        intrinsic_matrix = np.array(
            [
                [intrinsics.get("fx", 1000.0), 0.0, intrinsics.get("cx", 960.0)],
                [0.0, intrinsics.get("fy", 1000.0), intrinsics.get("cy", 540.0)],
                [0.0, 0.0, 1.0],
            ]
        )

        # Build lines
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

        # Compute RMSE without distortion correction (baseline)
        baseline_rmse = straightness_rmse(camera_lines, intrinsic_matrix)

        # Compute RMSE with distortion correction
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

    except ImportError as e:
        return JsonResponse({"error": f"Calibration module not available: {e}"}, status=500)
    except Exception as e:
        logger.exception("Validation failed")
        return JsonResponse({"error": f"Validation failed: {e}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def api_save(request: HttpRequest) -> JsonResponse:
    """Save calibration results to YAML file.

    Request body:
    {
        "camera_id": "my_camera",
        "zoom": 1.0,
        "coefficients": {...},
        "intrinsics": {...},
        "validation_rmse": 1.5,
        "filename": "calibration.yaml" (optional)
    }
    """
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

        distortion = DistortionCoefficients(
            k1=Unitless(coeffs.get("k1", 0.0)),
            k2=Unitless(coeffs.get("k2", 0.0)),
            k3=Unitless(coeffs.get("k3", 0.0)),
            p1=Unitless(coeffs.get("p1", 0.0)),
            p2=Unitless(coeffs.get("p2", 0.0)),
        )

        # Create calibration table
        table = CameraCalibrationTable(camera_id=camera_id)
        entry = ZoomCalibrationEntry.from_solver_result(
            zoom_factor=zoom,
            distortion=distortion,
            validation_rmse=validation_rmse,
            source_images=[],
            num_lines_used=data.get("num_lines", 0),
        )
        table.add_entry(entry)

        # Ensure output directory exists
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

        # Save to file
        filename = data.get("filename", f"{camera_id}_calibration.yaml")
        output_path = CALIBRATION_DIR / filename
        table.save(output_path)

        return JsonResponse(
            {
                "success": True,
                "path": str(output_path),
                "filename": filename,
            }
        )

    except ImportError as e:
        return JsonResponse({"error": f"Calibration module not available: {e}"}, status=500)
    except Exception as e:
        logger.exception("Save failed")
        return JsonResponse({"error": f"Save failed: {e}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def api_load(request: HttpRequest) -> JsonResponse:
    """Load calibration from YAML file.

    Request body:
    {
        "filename": "calibration.yaml"
    }
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
        from poc_homography.calibration.lens_distortion.calibration_table import (
            CameraCalibrationTable,
        )

        filename = data.get("filename", "")
        if not filename:
            return JsonResponse({"error": "Missing filename"}, status=400)

        # Check calibration directory first
        filepath = CALIBRATION_DIR / filename
        if not filepath.exists():
            # Try as absolute path
            filepath = Path(filename)
            if not filepath.exists():
                return JsonResponse({"error": f"File not found: {filename}"}, status=404)

        table = CameraCalibrationTable.load(filepath)

        # Convert to JSON-serializable format
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

    except ImportError as e:
        return JsonResponse({"error": f"Calibration module not available: {e}"}, status=500)
    except Exception as e:
        logger.exception("Load failed")
        return JsonResponse({"error": f"Load failed: {e}"}, status=500)
