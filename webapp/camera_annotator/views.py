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
DEFAULT_GCP_FILE = TEST_DATA_DIR / "Cartografia_valencia_gcps.yaml"
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


def load_gcps(gcps_file: Path = DEFAULT_GCP_FILE) -> list[dict]:
    """Load GCP IDs from the registry file."""
    if not gcps_file.exists():
        return []

    with open(gcps_file) as f:
        data = yaml.safe_load(f)

    # Handle empty or malformed YAML files
    if not data or not isinstance(data, dict):
        return []

    points = data.get("points", [])
    gcps = []
    for p in points:
        # Validate required keys exist
        if not all(k in p for k in ("id", "pixel_x", "pixel_y")):
            continue
        gcps.append({"id": p["id"], "pixel_x": p["pixel_x"], "pixel_y": p["pixel_y"]})

    return gcps


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
