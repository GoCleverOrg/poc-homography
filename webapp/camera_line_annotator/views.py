"""Views for the camera line annotator Django app."""

from __future__ import annotations

import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import yaml
from django.http import FileResponse, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST

# Paths relative to webapp directory
WEBAPP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEBAPP_DIR.parent
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "homography" / "test_data"

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Session keys
SESSION_IMAGE_KEY = "camera_line_annotator_image"
SESSION_ANNOTATIONS_KEY = "camera_line_annotator_annotations"

# Lines registry path (set via environment variable or default)
_lines_registry_path: Path | None = None
_lines_registry_data: dict | None = None


def get_lines_registry_path() -> Path | None:
    """Get the lines registry path from environment variable."""
    global _lines_registry_path
    if _lines_registry_path is None:
        env_path = os.environ.get("LINES_REGISTRY_PATH")
        if env_path:
            _lines_registry_path = Path(env_path)
        else:
            # Default to Cartografia_valencia_lines.yaml in test data
            default_path = TEST_DATA_DIR / "Cartografia_valencia_lines.yaml"
            if default_path.exists():
                _lines_registry_path = default_path
    return _lines_registry_path


def load_lines_registry() -> dict:
    """Load and cache lines registry from YAML file."""
    global _lines_registry_data
    if _lines_registry_data is not None:
        return _lines_registry_data

    registry_path = get_lines_registry_path()
    if registry_path is None or not registry_path.exists():
        return {"map_id": "", "lines": []}

    with open(registry_path) as f:
        data = yaml.safe_load(f)

    if not data or not isinstance(data, dict):
        return {"map_id": "", "lines": []}

    # Validate structure
    if "map_id" not in data or "lines" not in data:
        return {"map_id": "", "lines": []}

    _lines_registry_data = data
    return _lines_registry_data


def get_available_images() -> list[str]:
    """Scan TEST_DATA_DIR for available image files."""
    if not TEST_DATA_DIR.exists():
        return []

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(f.name for f in TEST_DATA_DIR.glob(f"*{ext}"))
        images.extend(f.name for f in TEST_DATA_DIR.glob(f"*{ext.upper()}"))

    return sorted(set(images))


def get_current_image(request: HttpRequest) -> str | None:
    """Get the current image filename from session, or first available image."""
    session_image = request.session.get(SESSION_IMAGE_KEY)
    if session_image:
        if (TEST_DATA_DIR / session_image).exists():
            return session_image

    images = get_available_images()
    if images:
        return images[0]

    return None


def validate_image_filename(filename: str) -> bool:
    """Validate filename to prevent path traversal attacks."""
    if not filename:
        return False
    if "/" in filename or ".." in filename or "\\" in filename:
        return False
    return True


def get_session_annotations(request: HttpRequest) -> dict[str, list[dict]]:
    """Get annotations from session storage."""
    annotations = request.session.get(SESSION_ANNOTATIONS_KEY)
    if annotations is None:
        annotations = {}
        request.session[SESSION_ANNOTATIONS_KEY] = annotations
    return annotations


def save_session_annotations(request: HttpRequest, annotations: dict[str, list[dict]]) -> None:
    """Save annotations to session storage."""
    request.session[SESSION_ANNOTATIONS_KEY] = annotations
    request.session.modified = True


def extract_camera_status(filename: str) -> dict[str, Any]:
    """Extract pan/tilt/zoom from filename pattern.

    Supports two patterns:
    1. Old: valte_{pan}_{tilt}_{zoom}_{timestamp}.{ext}
       Example: valte_56.7_20.7_1_20260114_182208.jpg
    2. New: valte_valte_cam01_{date}_{time}_{pan}_{tilt}_{zoom}.{ext}
       Example: valte_valte_cam01_20260123_120715_30.0_20.6_1.0.jpg
    """
    # Try new format first (PTZ at the end before extension)
    new_pattern = r".*_([0-9.]+)_([0-9.]+)_([0-9.]+)\.[a-zA-Z]+$"
    match = re.match(new_pattern, filename)
    if match:
        return {
            "pan": float(match.group(1)),
            "tilt": float(match.group(2)),
            "zoom": int(float(match.group(3))),
        }

    # Try old format (PTZ at the beginning after valte_)
    old_pattern = r"valte_([0-9.]+)_([0-9.]+)_([0-9]+)_"
    match = re.match(old_pattern, filename)
    if match:
        return {
            "pan": float(match.group(1)),
            "tilt": float(match.group(2)),
            "zoom": int(match.group(3)),
        }
    return {"pan": None, "tilt": None, "zoom": None}


def extract_point_annotations_ref(filename: str) -> str:
    """Extract point annotations reference from filename.

    Removes timestamp and extension to get base test case name.
    Example: valte_56.7_20.7_1_20260114_182208.jpg -> valte_56.7_20.7_1_20260114
    """
    # Remove extension
    base = filename.rsplit(".", 1)[0]
    # Remove last segment (timestamp part after last underscore)
    parts = base.rsplit("_", 1)
    if len(parts) > 1:
        return parts[0]
    return base


def index(request: HttpRequest) -> HttpResponse:
    """Serve the main HTML page."""
    current_image = get_current_image(request)
    registry = load_lines_registry()
    context = {
        "image_filename": current_image or "No images available",
        "registry_filename": (
            get_lines_registry_path().name if get_lines_registry_path() else "No registry"
        ),
        "map_id": registry.get("map_id", ""),
    }
    return render(request, "camera_line_annotator/index.html", context)


@require_GET
def api_images(request: HttpRequest) -> JsonResponse:
    """Get list of available images."""
    try:
        images = get_available_images()
        return JsonResponse(images, safe=False)
    except Exception as e:
        return JsonResponse({"error": f"Failed to get available images: {e}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def api_switch_image(request: HttpRequest) -> JsonResponse:
    """Switch to a different image file."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    filename = data.get("filename", "")

    if not validate_image_filename(filename):
        return JsonResponse({"error": "Invalid filename"}, status=400)

    image_path = TEST_DATA_DIR / filename

    try:
        resolved_path = image_path.resolve()
        if not resolved_path.is_relative_to(TEST_DATA_DIR.resolve()):
            return JsonResponse({"error": "Invalid filename"}, status=400)
    except (ValueError, RuntimeError):
        return JsonResponse({"error": "Invalid filename"}, status=400)

    if not resolved_path.exists():
        return JsonResponse({"error": f"Image not found: {filename}"}, status=404)

    request.session[SESSION_IMAGE_KEY] = filename

    # Load annotations for the new image
    all_annotations = get_session_annotations(request)
    image_annotations = all_annotations.get(filename, [])

    # Extract camera status from filename
    camera_status = extract_camera_status(filename)

    return JsonResponse(
        {
            "success": True,
            "filename": filename,
            "annotations": image_annotations,
            "camera_status": camera_status,
        }
    )


@require_GET
def serve_image(request: HttpRequest) -> HttpResponse:
    """Serve the current image file."""
    current_image = get_current_image(request)
    if not current_image:
        return HttpResponse("No image available", status=404)

    image_path = TEST_DATA_DIR / current_image

    try:
        resolved_path = image_path.resolve()
        if not resolved_path.is_relative_to(TEST_DATA_DIR.resolve()):
            return HttpResponse("Invalid image path", status=400)
    except (ValueError, RuntimeError):
        return HttpResponse("Invalid image path", status=400)

    if not resolved_path.exists():
        return HttpResponse("Image not found", status=404)

    mime_type, _ = mimetypes.guess_type(str(resolved_path))
    if not mime_type:
        mime_type = "image/jpeg"

    response = FileResponse(
        open(resolved_path, "rb"),
        content_type=mime_type,
    )
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response


@require_GET
def api_line_ids(request: HttpRequest) -> JsonResponse:
    """Get available line IDs from registry."""
    try:
        registry = load_lines_registry()
        lines = registry.get("lines", [])
        line_ids = [line.get("line_id") for line in lines if line.get("line_id")]
        return JsonResponse(
            {
                "map_id": registry.get("map_id", ""),
                "line_ids": line_ids,
            }
        )
    except Exception as e:
        return JsonResponse({"error": f"Failed to load line IDs: {e}"}, status=500)


@require_GET
def api_annotations(request: HttpRequest) -> JsonResponse:
    """Get line annotations for current image."""
    current_image = get_current_image(request)
    if not current_image:
        return JsonResponse([], safe=False)

    all_annotations = get_session_annotations(request)
    image_annotations = all_annotations.get(current_image, [])
    return JsonResponse(image_annotations, safe=False)


@csrf_exempt
@require_http_methods(["POST"])
def api_annotations_create(request: HttpRequest) -> JsonResponse:
    """Create a new line annotation."""
    current_image = get_current_image(request)
    if not current_image:
        return JsonResponse({"error": "No image selected"}, status=400)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Validate required fields
    required = ["line_id", "start_pixel_x", "start_pixel_y", "end_pixel_x", "end_pixel_y"]
    missing = [f for f in required if f not in data]
    if missing:
        return JsonResponse({"error": f"Missing required fields: {missing}"}, status=422)

    # Validate coordinate types
    try:
        annotation = {
            "line_id": str(data["line_id"]),
            "start_pixel_x": float(data["start_pixel_x"]),
            "start_pixel_y": float(data["start_pixel_y"]),
            "end_pixel_x": float(data["end_pixel_x"]),
            "end_pixel_y": float(data["end_pixel_y"]),
        }
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid coordinate values"}, status=422)

    # Store annotation
    all_annotations = get_session_annotations(request)
    if current_image not in all_annotations:
        all_annotations[current_image] = []
    all_annotations[current_image].append(annotation)
    save_session_annotations(request, all_annotations)

    return JsonResponse(
        {
            "success": True,
            "annotation": annotation,
            "index": len(all_annotations[current_image]) - 1,
        }
    )


@csrf_exempt
@require_http_methods(["DELETE"])
def api_annotations_delete(request: HttpRequest, index: int) -> JsonResponse:
    """Delete a line annotation by index."""
    current_image = get_current_image(request)
    if not current_image:
        return JsonResponse({"error": "No image selected"}, status=400)

    all_annotations = get_session_annotations(request)
    image_annotations = all_annotations.get(current_image, [])

    if index < 0 or index >= len(image_annotations):
        return JsonResponse({"error": f"Invalid annotation index: {index}"}, status=404)

    deleted = image_annotations.pop(index)
    save_session_annotations(request, all_annotations)

    return JsonResponse(
        {
            "success": True,
            "deleted": deleted,
        }
    )


@require_GET
def api_camera_status(request: HttpRequest) -> JsonResponse:
    """Get extracted camera status from current image filename."""
    current_image = get_current_image(request)
    if not current_image:
        return JsonResponse({"pan": None, "tilt": None, "zoom": None})

    camera_status = extract_camera_status(current_image)
    return JsonResponse(camera_status)


@csrf_exempt
@require_http_methods(["POST"])
def api_export(request: HttpRequest) -> JsonResponse:
    """Export current annotations to YAML file."""
    current_image = get_current_image(request)
    if not current_image:
        return JsonResponse({"error": "No image selected"}, status=400)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Get camera status from request or extract from filename
    camera_status = data.get("camera_status")
    if camera_status is None:
        camera_status = extract_camera_status(current_image)

    # Validate camera status values
    try:
        pan = float(camera_status.get("pan"))
        tilt = float(camera_status.get("tilt"))
        zoom = int(camera_status.get("zoom"))
    except (TypeError, ValueError):
        return JsonResponse(
            {"error": "Invalid camera status values. pan, tilt, zoom are required."},
            status=422,
        )

    # Get annotations
    all_annotations = get_session_annotations(request)
    image_annotations = all_annotations.get(current_image, [])

    if not image_annotations:
        return JsonResponse({"error": "No annotations to export"}, status=400)

    # Build export data
    registry_path = get_lines_registry_path()
    registry_filename = registry_path.name if registry_path else "unknown_lines.yaml"

    # Generate test case name from image filename
    base_name = current_image.rsplit(".", 1)[0]
    test_case_name = f"{base_name}_lines"

    # Point annotations reference
    point_annotations_ref = extract_point_annotations_ref(current_image)

    export_data = {
        "line_registry": registry_filename,
        "test_cases": [
            {
                "name": test_case_name,
                "image": current_image,
                "camera_status": {
                    "pan": pan,
                    "tilt": tilt,
                    "zoom": zoom,
                },
                "point_annotations_ref": point_annotations_ref,
                "line_annotations": image_annotations,
            }
        ],
    }

    # Determine output filename
    output_filename = data.get("output_filename")
    if not output_filename:
        output_filename = f"{base_name}_line_annotations.yaml"

    # Validate output filename
    if not validate_image_filename(output_filename):
        return JsonResponse({"error": "Invalid output filename"}, status=400)

    output_path = TEST_DATA_DIR / output_filename

    try:
        resolved_path = output_path.resolve()
        if not resolved_path.is_relative_to(TEST_DATA_DIR.resolve()):
            return JsonResponse({"error": "Invalid output path"}, status=400)
    except (ValueError, RuntimeError):
        return JsonResponse({"error": "Invalid output path"}, status=400)

    # Write YAML file
    with open(resolved_path, "w") as f:
        yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)

    return JsonResponse(
        {
            "success": True,
            "filename": output_filename,
            "path": str(resolved_path),
            "test_case_name": test_case_name,
            "annotation_count": len(image_annotations),
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def api_detect_lines(request: HttpRequest) -> JsonResponse:
    """Run automatic line detection on the current image.

    Uses the Hough-based line detector from the lens distortion calibration
    module to find candidate parking spot lines in the image.

    Request body (optional):
    {
        "min_line_length": 100,  // Minimum line length in pixels
        "min_confidence": 0.3,   // Minimum confidence threshold
        "canny_low": 50,         // Canny edge detection lower threshold
        "canny_high": 150        // Canny edge detection upper threshold
    }

    Response:
    {
        "success": true,
        "filename": "image.jpg",
        "detected_lines": [
            {
                "start_x": 100.0,
                "start_y": 200.0,
                "end_x": 500.0,
                "end_y": 210.0,
                "confidence": 0.85,
                "angle_deg": 1.2,
                "length": 400.5,
                "cluster_id": 0
            },
            ...
        ],
        "total_detected": 15
    }
    """
    current_image = get_current_image(request)
    if not current_image:
        return JsonResponse({"error": "No image selected"}, status=400)

    image_path = TEST_DATA_DIR / current_image

    try:
        resolved_path = image_path.resolve()
        if not resolved_path.is_relative_to(TEST_DATA_DIR.resolve()):
            return JsonResponse({"error": "Invalid image path"}, status=400)
    except (ValueError, RuntimeError):
        return JsonResponse({"error": "Invalid image path"}, status=400)

    if not resolved_path.exists():
        return JsonResponse({"error": f"Image not found: {current_image}"}, status=404)

    # Parse optional configuration from request
    config_overrides = {}
    if request.body:
        try:
            data = json.loads(request.body)
            if "min_line_length" in data:
                config_overrides["min_line_length"] = int(data["min_line_length"])
            if "min_confidence" in data:
                config_overrides["min_confidence"] = float(data["min_confidence"])
            if "canny_low" in data:
                config_overrides["canny_low"] = int(data["canny_low"])
            if "canny_high" in data:
                config_overrides["canny_high"] = int(data["canny_high"])
        except json.JSONDecodeError:
            pass  # Use default config

    try:
        from PIL import Image

        from poc_homography.calibration.lens_distortion.line_detection import (
            LineDetectionConfig,
            LineDetector,
        )

        # Get image dimensions for filtering
        with Image.open(resolved_path) as img:
            img_width = img.width
            img_height = img.height

        # Create config with any overrides
        config = LineDetectionConfig(**config_overrides)
        detector = LineDetector(config)

        # Run detection
        candidates = detector.detect_from_file(resolved_path)

        # Filter: only keep lines where both endpoints are within image bounds
        detected_lines = []
        for c in candidates:
            # Check all coordinates are within image bounds
            if (
                0 <= c.start[0] <= img_width
                and 0 <= c.start[1] <= img_height
                and 0 <= c.end[0] <= img_width
                and 0 <= c.end[1] <= img_height
            ):
                detected_lines.append(
                    {
                        "start_x": c.start[0],
                        "start_y": c.start[1],
                        "end_x": c.end[0],
                        "end_y": c.end[1],
                        "confidence": c.confidence,
                        "angle_deg": c.angle_deg,
                        "length": c.length,
                        "cluster_id": c.cluster_id,
                    }
                )

        # Sort by confidence (highest first)
        detected_lines.sort(key=lambda x: x["confidence"], reverse=True)

        return JsonResponse(
            {
                "success": True,
                "filename": current_image,
                "detected_lines": detected_lines,
                "total_detected": len(detected_lines),
            }
        )

    except ImportError as e:
        return JsonResponse(
            {
                "error": f"Line detection module not available: {e}",
                "details": "Ensure opencv-python is installed",
            },
            status=500,
        )
    except Exception as e:
        return JsonResponse(
            {
                "error": f"Line detection failed: {e}",
            },
            status=500,
        )


@csrf_exempt
@require_POST
def api_detect_lines_masked(request: HttpRequest) -> JsonResponse:
    """Detect lines using SAM3 masking to filter to ground markings only.

    POST /api/detect-lines-masked/
    Optional JSON body:
        - min_line_length: Minimum line length in pixels
        - min_confidence: Minimum confidence score (0.0-1.0)
        - sam3_prompt: SAM3 segmentation prompt (default: "white lines on ground")
        - include_comparison: If true, also return count of unmasked detection

    Returns JSON with detected lines, SAM3 mask info, and comparison stats.
    """
    current_image = get_current_image(request)
    if not current_image:
        return JsonResponse({"error": "No image selected"}, status=400)

    image_path = TEST_DATA_DIR / current_image

    try:
        resolved_path = image_path.resolve()
        if not resolved_path.is_relative_to(TEST_DATA_DIR.resolve()):
            return JsonResponse({"error": "Invalid image path"}, status=400)
    except (ValueError, RuntimeError):
        return JsonResponse({"error": "Invalid image path"}, status=400)

    if not resolved_path.exists():
        return JsonResponse({"error": f"Image not found: {current_image}"}, status=404)

    # Parse configuration from request
    line_config_overrides = {}
    sam3_prompt = "white lines on ground"
    include_comparison = False

    if request.body:
        try:
            data = json.loads(request.body)
            if "min_line_length" in data:
                line_config_overrides["min_line_length"] = int(data["min_line_length"])
            if "min_confidence" in data:
                line_config_overrides["min_confidence"] = float(data["min_confidence"])
            if "sam3_prompt" in data:
                sam3_prompt = str(data["sam3_prompt"])
            if "include_comparison" in data:
                include_comparison = bool(data["include_comparison"])
        except json.JSONDecodeError:
            pass  # Use default config

    try:
        import os

        from PIL import Image

        from poc_homography.calibration.lens_distortion.line_detection import (
            LineDetectionConfig,
        )
        from poc_homography.calibration.lens_distortion.masked_line_detection import (
            MaskedLineDetectionConfig,
            MaskedLineDetector,
        )
        from poc_homography.calibration.lens_distortion.sam3_masking import SAM3Config

        # Get image dimensions
        with Image.open(resolved_path) as img:
            img_width = img.width
            img_height = img.height

        # Check for API key
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            return JsonResponse(
                {
                    "error": "ROBOFLOW_API_KEY environment variable not set",
                    "details": "Set ROBOFLOW_API_KEY to use SAM3 masked detection",
                },
                status=500,
            )

        # Create configs
        sam3_config = SAM3Config(api_key=api_key, prompt=sam3_prompt)
        line_config = LineDetectionConfig(**line_config_overrides) if line_config_overrides else None
        config = MaskedLineDetectionConfig(
            sam3_config=sam3_config,
            line_config=line_config,
        )

        # Run masked detection
        detector = MaskedLineDetector(config)
        result = detector.detect_from_file(resolved_path, include_original_count=include_comparison)

        # Filter and convert lines
        detected_lines = []
        for c in result.lines:
            if (
                0 <= c.start[0] <= img_width
                and 0 <= c.start[1] <= img_height
                and 0 <= c.end[0] <= img_width
                and 0 <= c.end[1] <= img_height
            ):
                detected_lines.append(
                    {
                        "start_x": c.start[0],
                        "start_y": c.start[1],
                        "end_x": c.end[0],
                        "end_y": c.end[1],
                        "confidence": c.confidence,
                        "angle_deg": c.angle_deg,
                        "length": c.length,
                        "cluster_id": c.cluster_id,
                    }
                )

        detected_lines.sort(key=lambda x: x["confidence"], reverse=True)

        response_data = {
            "success": True,
            "filename": current_image,
            "detected_lines": detected_lines,
            "total_detected": len(detected_lines),
            "sam3": {
                "prompt": result.sam3_result.prompt,
                "coverage_percent": result.sam3_result.coverage_percent,
                "polygon_count": len(result.sam3_result.polygons),
                "error": result.sam3_result.error,
            },
        }

        if include_comparison and result.original_line_count > 0:
            reduction = (1 - len(detected_lines) / result.original_line_count) * 100
            response_data["comparison"] = {
                "original_line_count": result.original_line_count,
                "masked_line_count": len(detected_lines),
                "reduction_percent": round(reduction, 1),
            }

        return JsonResponse(response_data)

    except ImportError as e:
        return JsonResponse(
            {
                "error": f"Masked line detection module not available: {e}",
                "details": "Ensure opencv-python and requests are installed",
            },
            status=500,
        )
    except Exception as e:
        import traceback

        return JsonResponse(
            {
                "error": f"Masked line detection failed: {e}",
                "traceback": traceback.format_exc(),
            },
            status=500,
        )


