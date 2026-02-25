"""Views for the camera annotator Django app."""

from __future__ import annotations

import json

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods
from homography_web.frame_utils import (
    GCPS_DIR,
    get_available_images,
    get_current_image as _get_current_image,
    get_frame_repo,
    get_map_from_tenant_id,
    get_tenant_id,
    image_filename_to_frame,
    invalidate_cache,
    load_annotations_for_frame,
    serve_current_image,
    validate_image_filename,
)

# Session key for current image
SESSION_IMAGE_KEY = "camera_annotator_image"


def get_current_image(request: HttpRequest, map_id: str | None = None) -> str | None:
    """Get the current image filename from session, or first available image."""
    return _get_current_image(request, SESSION_IMAGE_KEY, map_id)


def load_gcps(tenant_id: str) -> list[dict]:
    """Load GCPs from the repository.

    Args:
        tenant_id: Tenant identifier for map lookup.
    """
    from poc_homography.map_points.gcp_registry import from_gcp_repo

    map_entity = get_map_from_tenant_id(tenant_id)
    if map_entity is None:
        return []
    try:
        map_id = map_entity.id
        registry = from_gcp_repo(GCPS_DIR, map_id)
    except (KeyError, ValueError, OSError):
        return []

    return [
        {"id": pid, "pixel_x": p.pixel_x, "pixel_y": p.pixel_y}
        for pid, p in registry.points.items()
    ]


def load_existing_annotations(image_filename: str) -> list[dict]:
    """Load existing annotations for a specific image from the CapturedFrame repo."""
    frame = image_filename_to_frame(image_filename)
    if frame is None:
        return []
    return load_annotations_for_frame(frame.id)


def save_annotations_to_repo(
    image_filename: str,
    annotations: list[dict],
) -> None:
    """Save annotations to the CapturedFrame repo (alongside frame YAML).

    Args:
        image_filename: Camera image filename (used as frame_id base).
        annotations: List of dicts with gcp_id, pixel_x, pixel_y.
    """
    from poc_homography.domain.entities.annotation import Annotation
    from poc_homography.domain.vo import PixelPoint

    frame = image_filename_to_frame(image_filename)
    if frame is None:
        return

    ann_entities = [
        Annotation(
            gcp_id=ann_dict["gcp_id"],
            frame_id=frame.id,
            camera_pose=frame.ptz_state,
            pixel=PixelPoint.create(
                round(float(ann_dict["pixel_x"]), 1),
                round(float(ann_dict["pixel_y"]), 1),
            ),
        )
        for ann_dict in annotations
    ]
    get_frame_repo().save_annotations(frame.id, ann_entities)


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
        gcps = load_gcps(get_tenant_id(request))
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
        annotations = load_existing_annotations(current_image)
        return JsonResponse(annotations, safe=False)
    except Exception as e:
        return JsonResponse({"error": f"Failed to load annotations: {e}"}, status=500)


@require_GET
def api_images(request: HttpRequest) -> JsonResponse:
    """Get list of available images for the current tenant."""
    try:
        map_entity = get_map_from_tenant_id(get_tenant_id(request))
        map_id = map_entity.id if map_entity else None
        images = get_available_images(map_id)
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
    invalidate_cache()

    return JsonResponse({"success": True, "saved": len(clean_annotations)})


@require_GET
def serve_image(request: HttpRequest) -> HttpResponse:
    """Serve the current image file."""
    return serve_current_image(request, SESSION_IMAGE_KEY)
