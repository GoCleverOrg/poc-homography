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
        intrinsics_data = data["intrinsics"]

        # Define default intrinsics
        default_fx, default_fy, default_cx, default_cy = 1000.0, 1000.0, 960.0, 540.0

        # Extract intrinsics with defaults
        fx = intrinsics_data.get("fx", default_fx)
        fy = intrinsics_data.get("fy", default_fy)
        cx = intrinsics_data.get("cx", default_cx)
        cy = intrinsics_data.get("cy", default_cy)

        # Check if using default intrinsics and warn
        using_defaults = (
            fx == default_fx and fy == default_fy and cx == default_cx and cy == default_cy
        )
        if using_defaults:
            logger.warning(
                "Using default intrinsics (fx=1000, fy=1000, cx=960, cy=540). "
                "For best results, provide actual camera intrinsics."
            )

        # Store the actual intrinsics used for saving later
        intrinsics_used = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

        intrinsic_matrix = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
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
            "training_rmse": result.training_rmse,
            "rmse_note": "Training RMSE is in-sample error. Use separate validation lines to measure generalization.",
            "coefficients": {
                "k1": float(result.distortion.k1),
                "k2": float(result.distortion.k2),
                "k3": float(result.distortion.k3),
                "p1": float(result.distortion.p1),
                "p2": float(result.distortion.p2),
            },
            "intrinsics_used": intrinsics_used,
            "quality": "good" if result.training_rmse < 2.0 else "acceptable" if result.training_rmse < 5.0 else "poor",
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
def api_calibrate_from_calibration_files(request: HttpRequest) -> JsonResponse:
    """Run distortion calibration from camera_line_annotator calibration JSON files.

    This endpoint accepts multiple calibration JSON files (from the camera_line_annotator
    tool) and uses the n-point calibration lines for lens distortion calibration.
    Lines with the same line_id across multiple frames are combined.

    Request body:
    {
        "files": [
            {
                "image": "filename.jpg",
                "camera_status": {"pan": 30.0, "tilt": 20.6, "zoom": 1.0},
                "calibration_lines": [
                    {"index": 0, "point_count": 4, "points": [[x,y],...], "line_id": "L5"},
                    ...
                ]
            },
            ...
        ],
        "intrinsics": {
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": 960.0,
            "cy": 540.0
        },
        "config": {
            "radial_only": false,
            "max_iterations": 1000
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

    if "files" not in data or not data["files"]:
        return JsonResponse({"error": "No calibration files provided"}, status=400)

    try:
        import numpy as np

        from poc_homography.calibration.lens_distortion.distortion_solver import (
            DistortionSolver,
            SolverConfig,
            calculate_validation_rmse,
        )
        from poc_homography.calibration.lens_distortion.models import (
            CameraLine,
            PTZPosition,
        )
        from poc_homography.types import Degrees

        # Build intrinsic matrix
        intrinsics_data = data["intrinsics"]

        # Define default intrinsics
        default_fx, default_fy, default_cx, default_cy = 1000.0, 1000.0, 960.0, 540.0

        # Extract intrinsics with defaults
        fx = intrinsics_data.get("fx", default_fx)
        fy = intrinsics_data.get("fy", default_fy)
        cx = intrinsics_data.get("cx", default_cx)
        cy = intrinsics_data.get("cy", default_cy)

        # Check if using default intrinsics and warn
        using_defaults = (
            fx == default_fx and fy == default_fy and cx == default_cx and cy == default_cy
        )
        if using_defaults:
            logger.warning(
                "Using default intrinsics (fx=1000, fy=1000, cx=960, cy=540). "
                "For best results, provide actual camera intrinsics."
            )

        # Store the actual intrinsics used for saving later
        intrinsics_used = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

        intrinsic_matrix = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ]
        )

        # Get config
        config_data = data.get("config", {})
        solver_config = SolverConfig(
            use_radial_only=config_data.get("radial_only", False),
            max_iterations=config_data.get("max_iterations", 1000),
        )

        # Get optional validation split (e.g., 0.5 = 50% training, 50% validation)
        validation_split = data.get("validation_split")

        # Parse calibration files and create CameraLine objects
        camera_lines: list[CameraLine] = []
        frame_info: list[dict] = []
        lines_skipped_no_edge_pixels = 0
        points_per_line: list[int] = []

        for file_data in data["files"]:
            image_name = file_data.get("image", "unknown")
            camera_status = file_data.get("camera_status", {})
            calibration_lines = file_data.get("calibration_lines", [])

            # Create PTZ position from camera status
            ptz = PTZPosition(
                pan_deg=Degrees(camera_status.get("pan", 0.0)),
                tilt_deg=Degrees(camera_status.get("tilt", 30.0)),
                zoom_factor=camera_status.get("zoom", 1.0),
            )

            frame_lines_loaded = 0

            for cal_line in calibration_lines:
                points = cal_line.get("points", [])
                if len(points) < 2:
                    continue

                line_id = cal_line.get("line_id") or f"line_{cal_line.get('index', 0):04d}"

                # Convert points to tuples for edge_pixels
                edge_pixels = tuple(tuple(pt) for pt in points)

                # Use first and last point as start/end
                start_pixel = tuple(points[0])
                end_pixel = tuple(points[-1])

                camera_line = CameraLine(
                    line_id=f"{line_id}_{image_name}",  # Make unique per frame
                    image_path=image_name,
                    start_pixel=start_pixel,
                    end_pixel=end_pixel,
                    ptz_position=ptz,
                    confidence=1.0,
                    edge_pixels=edge_pixels,
                )

                # Validate edge_pixels was loaded correctly
                if not camera_line.has_edge_pixels:
                    logger.error(
                        "Line %s from image %s has no edge_pixels data - skipping",
                        line_id,
                        image_name,
                    )
                    lines_skipped_no_edge_pixels += 1
                    continue

                # Log debug info for each loaded line
                logger.debug(
                    "Loaded line %s from %s: %d edge_pixels, angle=%.1f deg",
                    line_id,
                    image_name,
                    len(camera_line.edge_pixels),  # type: ignore[arg-type]
                    camera_line.angle_degrees,
                )

                camera_lines.append(camera_line)
                frame_lines_loaded += 1
                points_per_line.append(len(camera_line.edge_pixels))  # type: ignore[arg-type]

            frame_info.append({
                "image": image_name,
                "ptz": {"pan": float(ptz.pan_deg), "tilt": float(ptz.tilt_deg), "zoom": ptz.zoom_factor},
                "num_lines": frame_lines_loaded,
            })

        # Log summary statistics
        total_lines_attempted = len(camera_lines) + lines_skipped_no_edge_pixels
        if points_per_line:
            min_points = min(points_per_line)
            max_points = max(points_per_line)
            avg_points = sum(points_per_line) / len(points_per_line)
        else:
            min_points = max_points = 0
            avg_points = 0.0

        logger.info(
            "Calibration data loading summary: "
            "%d lines loaded successfully, %d lines skipped (missing edge_pixels)",
            len(camera_lines),
            lines_skipped_no_edge_pixels,
        )
        if points_per_line:
            logger.info(
                "Points per line: min=%d, max=%d, avg=%.1f",
                min_points,
                max_points,
                avg_points,
            )
        for fi in frame_info:
            logger.info(
                "Frame %s: %d lines loaded",
                fi["image"],
                fi["num_lines"],
            )

        # Validate that we have usable data
        if total_lines_attempted > 0 and len(camera_lines) == 0:
            return JsonResponse(
                {
                    "error": f"All {total_lines_attempted} lines have missing edge_pixels data. "
                    "Cannot perform calibration without edge pixel coordinates."
                },
                status=400,
            )

        if len(camera_lines) < 1:
            return JsonResponse({"error": "No valid calibration lines found (need at least 2 points per line)"}, status=400)

        # Split into training and validation sets if validation_split is specified
        training_lines: list[CameraLine] = camera_lines
        validation_lines: list[CameraLine] = []

        if validation_split is not None and 0 < validation_split < 1:
            split_index = int(len(camera_lines) * (1 - validation_split))
            if split_index < 1:
                return JsonResponse({"error": "Not enough lines for training after validation split"}, status=400)
            if split_index >= len(camera_lines):
                return JsonResponse({"error": "Not enough lines for validation after split"}, status=400)

            training_lines = camera_lines[:split_index]
            validation_lines = camera_lines[split_index:]
            logger.info(
                "Train/validation split: %d training lines, %d validation lines (split=%.2f)",
                len(training_lines),
                len(validation_lines),
                validation_split,
            )

        # Run calibration on training lines
        solver = DistortionSolver(config=solver_config)
        result = solver.solve(training_lines, intrinsic_matrix)

        # Build response
        response_data: dict[str, Any] = {
            "success": result.success,
            "message": result.message,
            "iterations": result.iterations,
            "num_lines": len(training_lines),
            "num_frames": len(data["files"]),
            "frame_info": frame_info,
            "initial_error": result.initial_error,
            "final_error": result.final_error,
            "improvement_percent": (1 - result.improvement_ratio()) * 100,
            "training_rmse": result.training_rmse,
            "rmse_note": "Training RMSE is in-sample error. Use separate validation lines to measure generalization.",
            "coefficients": {
                "k1": float(result.distortion.k1),
                "k2": float(result.distortion.k2),
                "k3": float(result.distortion.k3),
                "p1": float(result.distortion.p1),
                "p2": float(result.distortion.p2),
            },
            "intrinsics_used": intrinsics_used,
            "quality": "good" if result.training_rmse < 2.0 else "acceptable" if result.training_rmse < 5.0 else "poor",
            "line_errors": result.line_errors[:20],  # Top 20 worst lines
        }

        # Add validation metrics if validation split was used
        if validation_lines:
            validation_rmse, baseline_rmse = calculate_validation_rmse(
                validation_lines,
                intrinsic_matrix,
                result.distortion,
                num_samples=solver_config.num_samples_per_line,
            )
            response_data["validation_rmse"] = validation_rmse
            response_data["baseline_rmse"] = baseline_rmse
            response_data["num_training_lines"] = len(training_lines)
            response_data["num_validation_lines"] = len(validation_lines)
            response_data["validation_split"] = validation_split
            # Update quality based on validation RMSE when available
            response_data["quality"] = "good" if validation_rmse < 2.0 else "acceptable" if validation_rmse < 5.0 else "poor"

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
        intrinsics = data.get("intrinsics", {})

        # Warn if no intrinsics provided - they will be missing in saved file
        if not intrinsics:
            logger.warning(
                "Saving calibration without intrinsics. "
                "Validation may use different intrinsics, causing mismatch."
            )

        distortion = DistortionCoefficients(
            k1=Unitless(coeffs.get("k1", 0.0)),
            k2=Unitless(coeffs.get("k2", 0.0)),
            k3=Unitless(coeffs.get("k3", 0.0)),
            p1=Unitless(coeffs.get("p1", 0.0)),
            p2=Unitless(coeffs.get("p2", 0.0)),
        )

        # Create calibration table with intrinsics
        table = CameraCalibrationTable(camera_id=camera_id)
        entry = ZoomCalibrationEntry.from_solver_result(
            zoom_factor=zoom,
            distortion=distortion,
            validation_rmse=validation_rmse,
            source_images=[],
            num_lines_used=data.get("num_lines", 0),
            intrinsics=intrinsics if intrinsics else None,
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
            entry_data = {
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
            # Include intrinsics if stored
            if entry.has_intrinsics():
                entry_data["intrinsics"] = entry.get_intrinsics()
            entries.append(entry_data)

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


@csrf_exempt
@require_http_methods(["POST"])
def api_debug_data(request: HttpRequest) -> JsonResponse:
    """Debug endpoint to inspect loaded calibration data before running calibration.

    Takes the same input as api_calibrate_from_calibration_files but returns
    detailed JSON about what data was loaded, allowing developers to inspect
    the data before running calibration.

    Request body: Same as api_calibrate_from_calibration_files

    Response:
    {
        "frames": [
            {
                "image": "filename.jpg",
                "ptz": {"pan": 30, "tilt": 20, "zoom": 1},
                "lines": [
                    {
                        "line_id": "L5",
                        "has_edge_pixels": true,
                        "num_points": 15,
                        "start_pixel": [100, 200],
                        "end_pixel": [500, 210],
                        "angle_degrees": 1.14
                    }
                ]
            }
        ],
        "summary": {
            "total_frames": 5,
            "total_lines": 20,
            "lines_with_edge_pixels": 18,
            "lines_without_edge_pixels": 2,
            "min_points": 6,
            "max_points": 25,
            "avg_points": 15.2
        }
    }
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "files" not in data or not data["files"]:
        return JsonResponse({"error": "No calibration files provided"}, status=400)

    try:
        from poc_homography.calibration.lens_distortion.models import (
            CameraLine,
            PTZPosition,
        )
        from poc_homography.types import Degrees

        frames: list[dict[str, Any]] = []
        total_lines = 0
        lines_with_edge_pixels = 0
        lines_without_edge_pixels = 0
        points_per_line: list[int] = []

        for file_data in data["files"]:
            image_name = file_data.get("image", "unknown")
            camera_status = file_data.get("camera_status", {})
            calibration_lines = file_data.get("calibration_lines", [])

            # Create PTZ position from camera status
            ptz = PTZPosition(
                pan_deg=Degrees(camera_status.get("pan", 0.0)),
                tilt_deg=Degrees(camera_status.get("tilt", 30.0)),
                zoom_factor=camera_status.get("zoom", 1.0),
            )

            frame_lines: list[dict[str, Any]] = []

            for cal_line in calibration_lines:
                points = cal_line.get("points", [])
                line_id = cal_line.get("line_id") or f"line_{cal_line.get('index', 0):04d}"

                total_lines += 1

                if len(points) < 2:
                    # Line has no valid edge pixels
                    frame_lines.append({
                        "line_id": line_id,
                        "has_edge_pixels": False,
                        "num_points": len(points),
                        "start_pixel": points[0] if len(points) > 0 else None,
                        "end_pixel": points[-1] if len(points) > 0 else None,
                        "angle_degrees": None,
                    })
                    lines_without_edge_pixels += 1
                    continue

                # Convert points to tuples for edge_pixels
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

                has_edge_pixels = camera_line.has_edge_pixels
                num_points = len(camera_line.edge_pixels) if has_edge_pixels else 0

                if has_edge_pixels:
                    lines_with_edge_pixels += 1
                    points_per_line.append(num_points)
                else:
                    lines_without_edge_pixels += 1

                frame_lines.append({
                    "line_id": line_id,
                    "has_edge_pixels": has_edge_pixels,
                    "num_points": num_points,
                    "start_pixel": list(start_pixel),
                    "end_pixel": list(end_pixel),
                    "angle_degrees": round(camera_line.angle_degrees, 2),
                })

            frames.append({
                "image": image_name,
                "ptz": {
                    "pan": float(ptz.pan_deg),
                    "tilt": float(ptz.tilt_deg),
                    "zoom": ptz.zoom_factor,
                },
                "lines": frame_lines,
            })

        # Build summary
        if points_per_line:
            min_points = min(points_per_line)
            max_points = max(points_per_line)
            avg_points = round(sum(points_per_line) / len(points_per_line), 1)
        else:
            min_points = 0
            max_points = 0
            avg_points = 0.0

        summary = {
            "total_frames": len(frames),
            "total_lines": total_lines,
            "lines_with_edge_pixels": lines_with_edge_pixels,
            "lines_without_edge_pixels": lines_without_edge_pixels,
            "min_points": min_points,
            "max_points": max_points,
            "avg_points": avg_points,
        }

        return JsonResponse({
            "frames": frames,
            "summary": summary,
        })

    except ImportError as e:
        return JsonResponse({"error": f"Calibration module not available: {e}"}, status=500)
    except Exception as e:
        logger.exception("Debug data inspection failed")
        return JsonResponse({"error": f"Debug data inspection failed: {e}"}, status=500)
