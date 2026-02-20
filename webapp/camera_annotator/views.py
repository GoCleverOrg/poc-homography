"""Views for the camera annotator Django app."""

from __future__ import annotations

import json
import mimetypes

from django.http import FileResponse, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods
from homography_web.frame_utils import (
    GCPS_DIR,
    get_default_map_id,
    get_frame_image_path,
    image_filename_to_frame,
    list_image_filenames,
    validate_image_filename,
)
from homography_web.frame_utils import (
    get_annotation_repo as _get_annotation_repo,
)

# Session key for current image
SESSION_IMAGE_KEY = "camera_annotator_image"


def get_available_images() -> list[str]:
    """Return available image filenames from the CapturedFrame repo."""
    return list_image_filenames()


def get_current_image(request: HttpRequest) -> str | None:
    """Get the current image filename from session, or first available image."""
    session_image = request.session.get(SESSION_IMAGE_KEY)
    if session_image:
        if image_filename_to_frame(session_image) is not None:
            return session_image

    images = get_available_images()
    if images:
        return images[0]

    return None


def load_gcps() -> list[dict]:
    """Load GCPs from the repository."""
    from poc_homography.map_points.gcp_registry import from_gcp_repo

    map_id = get_default_map_id()
    if map_id is None:
        return []
    try:
        registry = from_gcp_repo(GCPS_DIR, map_id)
    except (KeyError, ValueError, OSError):
        return []

    return [
        {"id": pid, "pixel_x": p.pixel_x, "pixel_y": p.pixel_y}
        for pid, p in registry.points.items()
    ]


def load_existing_annotations(image_filename: str) -> list[dict]:
    """Load existing annotations for a specific image from the Annotation repo."""
    return load_annotations_from_repo(image_filename)


# ---------------------------------------------------------------------------
# Repository adapter functions (bridge legacy test_cases <-> DDD repos)
# ---------------------------------------------------------------------------


def load_annotations_from_repo(image_filename: str) -> list[dict]:
    """Load annotations from the DDD Annotation repository.

    Args:
        image_filename: Camera image filename to match annotations for.

    Returns:
        List of annotation dicts in legacy format (gcp_id, pixel_x, pixel_y).
    """
    frame = image_filename_to_frame(image_filename)
    if frame is None:
        return []

    repo = _get_annotation_repo()
    return [
        {
            "gcp_id": ann.gcp_id,
            "pixel_x": round(float(ann.pixel.x), 1),
            "pixel_y": round(float(ann.pixel.y), 1),
        }
        for ann in repo.get_by_frame_id(frame.id)
    ]


def save_annotations_to_repo(
    image_filename: str,
    annotations: list[dict],
) -> None:
    """Save legacy annotation dicts to the DDD Annotation repository.

    Args:
        image_filename: Camera image filename (used as frame_id base).
        annotations: List of dicts with gcp_id, pixel_x, pixel_y.
    """
    from poc_homography.domain.entities.annotation import Annotation
    from poc_homography.domain.vo import PixelPoint

    frame = image_filename_to_frame(image_filename)
    if frame is None:
        return

    repo = _get_annotation_repo()

    for ann_dict in annotations:
        annotation = Annotation(
            gcp_id=ann_dict["gcp_id"],
            frame_id=frame.id,
            camera_pose=frame.ptz_state,
            pixel=PixelPoint.create(
                round(float(ann_dict["pixel_x"]), 1),
                round(float(ann_dict["pixel_y"]), 1),
            ),
        )
        repo.save(annotation)


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
    """Get existing annotations for current image from the CapturedFrame repo."""
    current_image = get_current_image(request)
    if not current_image:
        return JsonResponse([], safe=False)

    try:
        annotations = load_annotations_from_repo(current_image)
        if not annotations:
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

    if not validate_image_filename(filename):
        return JsonResponse({"error": "Invalid filename"}, status=400)

    frame = image_filename_to_frame(filename)
    if frame is None:
        return JsonResponse({"error": f"Image not found: {filename}"}, status=404)

    request.session[SESSION_IMAGE_KEY] = filename

    annotations = load_annotations_from_repo(filename)
    if not annotations:
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
    """Save annotations for the current image to the DDD repository."""
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

    # Build clean annotation list (round to 1 decimal)
    clean_annotations = [
        {
            "pixel_x": round(float(a["pixel_x"]), 1),
            "pixel_y": round(float(a["pixel_y"]), 1),
            "gcp_id": a["gcp_id"],
        }
        for a in new_annotations
    ]

    save_annotations_to_repo(current_image, clean_annotations)

    return JsonResponse({"success": True, "saved": len(clean_annotations)})


@require_GET
def serve_image(request: HttpRequest) -> HttpResponse:
    """Serve the current image file."""
    current_image = get_current_image(request)
    if not current_image:
        return HttpResponse("No image available", status=404)

    frame = image_filename_to_frame(current_image)
    if frame is None:
        return HttpResponse("Image not found", status=404)

    resolved_path = get_frame_image_path(frame)
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
