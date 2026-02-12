"""Views for the camera annotator Django app."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path

import yaml
from django.http import FileResponse, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

# Paths relative to webapp directory
WEBAPP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEBAPP_DIR.parent
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "homography" / "test_data"
GCPS_DIR = PROJECT_ROOT / "data" / "gcps"
DEFAULT_ANNOTATIONS_FILE = TEST_DATA_DIR / "valte_annotations.yaml"

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Session key for current image
SESSION_IMAGE_KEY = "camera_annotator_image"


def get_available_images() -> list[str]:
    """Scan TEST_DATA_DIR for available image files.

    Returns:
        Sorted list of image filenames matching supported extensions.
    """
    if not TEST_DATA_DIR.exists():
        return []

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(f.name for f in TEST_DATA_DIR.glob(f"*{ext}"))
        # Also check uppercase extensions
        images.extend(f.name for f in TEST_DATA_DIR.glob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    return sorted(set(images))


def get_current_image(request: HttpRequest) -> str | None:
    """Get the current image filename from session, or first available image."""
    # Check session first
    session_image = request.session.get(SESSION_IMAGE_KEY)
    if session_image:
        # Verify it still exists
        if (TEST_DATA_DIR / session_image).exists():
            return session_image

    # Fall back to first available image
    images = get_available_images()
    if images:
        return images[0]

    return None


def load_gcps() -> list[dict]:
    """Load GCPs from the repository."""
    from poc_homography.map_points.gcp_registry import from_gcp_repo

    try:
        registry = from_gcp_repo(GCPS_DIR, "valte")
    except (KeyError, ValueError, OSError):
        return []

    return [
        {"id": pid, "pixel_x": p.pixel_x, "pixel_y": p.pixel_y}
        for pid, p in registry.points.items()
    ]


def load_existing_annotations(
    image_filename: str, annotations_file: Path = DEFAULT_ANNOTATIONS_FILE
) -> list[dict]:
    """Load existing annotations for a specific image from the annotations file."""
    if not annotations_file.exists():
        return []

    with open(annotations_file) as f:
        data = yaml.safe_load(f)

    # Handle empty or malformed YAML files
    if not data or not isinstance(data, dict):
        return []

    test_cases = data.get("test_cases", [])
    for tc in test_cases:
        # Match by image filename
        if tc.get("image") == image_filename:
            return tc.get("annotations", [])

    return []


def validate_image_filename(filename: str) -> bool:
    """Validate filename to prevent path traversal attacks.

    Returns True if filename is valid, False otherwise.
    """
    if not filename:
        return False
    # Security: Reject paths containing directory traversal or path separators
    if "/" in filename or ".." in filename or "\\" in filename:
        return False
    return True


def index(request: HttpRequest) -> HttpResponse:
    """Serve the main HTML page."""
    current_image = get_current_image(request)
    context = {
        "image_filename": current_image or "No images available",
    }
    return render(request, "camera_annotator/index.html", context)


@require_GET
def api_gcps(request: HttpRequest) -> JsonResponse:
    """Get list of available GCPs."""
    try:
        gcps = load_gcps()
        return JsonResponse(gcps, safe=False)
    except Exception as e:
        return JsonResponse({"error": f"Failed to load GCPs: {e}"}, status=500)


@require_GET
def api_annotations(request: HttpRequest) -> JsonResponse:
    """Get existing annotations for current image."""
    current_image = get_current_image(request)
    if not current_image:
        return JsonResponse([], safe=False)

    try:
        annotations = load_existing_annotations(current_image)
        return JsonResponse(annotations, safe=False)
    except Exception as e:
        return JsonResponse({"error": f"Failed to load annotations: {e}"}, status=500)


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

    # Security: Validate filename to prevent path traversal
    if not validate_image_filename(filename):
        return JsonResponse({"error": "Invalid filename"}, status=400)

    # Verify the image exists
    image_path = TEST_DATA_DIR / filename

    # Security: Ensure resolved path is within TEST_DATA_DIR (defense-in-depth)
    try:
        resolved_path = image_path.resolve()
        if not resolved_path.is_relative_to(TEST_DATA_DIR.resolve()):
            return JsonResponse({"error": "Invalid filename"}, status=400)
    except (ValueError, RuntimeError):
        return JsonResponse({"error": "Invalid filename"}, status=400)

    if not resolved_path.exists():
        return JsonResponse({"error": f"Image not found: {filename}"}, status=404)

    # Store in session
    request.session[SESSION_IMAGE_KEY] = filename

    # Load annotations for the new image
    annotations = load_existing_annotations(filename)

    return JsonResponse(
        {
            "success": True,
            "filename": filename,
            "annotations": annotations,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def api_save_annotations(request: HttpRequest) -> JsonResponse:
    """Save annotations for the current image to the annotations YAML file.

    Adds or updates the test_case entry for the current image.
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    current_image = get_current_image(request)
    if not current_image:
        return JsonResponse({"error": "No image selected"}, status=400)

    new_annotations = data.get("annotations", [])
    if not new_annotations:
        return JsonResponse({"error": "No annotations to save"}, status=400)

    # Validate annotation format
    for ann in new_annotations:
        if not all(k in ann for k in ("gcp_id", "pixel_x", "pixel_y")):
            return JsonResponse(
                {"error": "Each annotation must have gcp_id, pixel_x, pixel_y"},
                status=400,
            )

    # Load existing YAML
    if DEFAULT_ANNOTATIONS_FILE.exists():
        with open(DEFAULT_ANNOTATIONS_FILE) as f:
            yaml_data = yaml.safe_load(f) or {}
    else:
        yaml_data = {}

    test_cases = yaml_data.setdefault("test_cases", [])

    # Find existing entry for this image
    existing_tc = None
    for tc in test_cases:
        if tc.get("image") == current_image:
            existing_tc = tc
            break

    # Build clean annotation list (round to 1 decimal)
    clean_annotations = [
        {
            "pixel_x": round(float(a["pixel_x"]), 1),
            "pixel_y": round(float(a["pixel_y"]), 1),
            "gcp_id": a["gcp_id"],
        }
        for a in new_annotations
    ]

    if existing_tc:
        existing_tc["annotations"] = clean_annotations
    else:
        # Parse camera status from filename if possible (e.g. valte_102.5_20.7_1_...)
        parts = current_image.replace(".png", "").replace(".jpg", "").split("_")
        camera_status = {}
        if len(parts) >= 4:
            try:
                camera_status = {
                    "pan": float(parts[1]),
                    "tilt": float(parts[2]),
                    "zoom": int(parts[3]),
                }
            except (ValueError, IndexError):
                pass

        new_tc: dict = {
            "name": current_image.rsplit(".", 1)[0],
            "image": current_image,
        }
        if camera_status:
            new_tc["camera_status"] = camera_status
        new_tc["annotations"] = clean_annotations
        test_cases.append(new_tc)

    # Write back
    with open(DEFAULT_ANNOTATIONS_FILE, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return JsonResponse({"success": True, "saved": len(clean_annotations)})


@require_GET
def serve_image(request: HttpRequest) -> HttpResponse:
    """Serve the current image file."""
    current_image = get_current_image(request)
    if not current_image:
        return HttpResponse("No image available", status=404)

    image_path = TEST_DATA_DIR / current_image

    # Security: Verify path is within TEST_DATA_DIR
    try:
        resolved_path = image_path.resolve()
        if not resolved_path.is_relative_to(TEST_DATA_DIR.resolve()):
            return HttpResponse("Invalid image path", status=400)
    except (ValueError, RuntimeError):
        return HttpResponse("Invalid image path", status=400)

    if not resolved_path.exists():
        return HttpResponse("Image not found", status=404)

    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(str(resolved_path))
    if not mime_type:
        mime_type = "image/jpeg"

    # Return the image with no-cache headers to ensure fresh content after switch
    response = FileResponse(
        open(resolved_path, "rb"),
        content_type=mime_type,
    )
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response
