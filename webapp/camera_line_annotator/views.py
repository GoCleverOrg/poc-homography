"""Views for the camera line annotator Django app."""

from __future__ import annotations

import json
from typing import Any

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods
from homography_web.frame_utils import (
    LINES_DIR,
    get_available_images_fresh,
    get_map_from_tenant_id,
    get_tenant_id,
    image_filename_to_frame,
    invalidate_cache,
    load_line_annotations_for_frame,
    register_invalidation_callback,
    serve_current_image,
    validate_image_filename,
)
from homography_web.frame_utils import (
    get_current_image as _get_current_image,
)
from homography_web.frame_utils import (
    get_line_annotation_repo as _get_line_annotation_repo,
)

# Session keys
SESSION_IMAGE_KEY = "camera_line_annotator_image"

# Per-tenant cached line registries
_lines_registry_cache: dict[str, dict] = {}


def _invalidate_lines_cache() -> None:
    """Clear the local line registry cache."""
    _lines_registry_cache.clear()


register_invalidation_callback(_invalidate_lines_cache)


def load_lines_registry(tenant_id: str) -> dict:
    """Load and cache lines registry from the DDD repo.

    Args:
        tenant_id: Tenant identifier for map lookup.
    """
    if tenant_id in _lines_registry_cache:
        return _lines_registry_cache[tenant_id]

    from line_picker.state import from_line_repo

    try:
        map_entity = get_map_from_tenant_id(tenant_id)
        if map_entity:
            repo_lines = from_line_repo(LINES_DIR, map_entity.id)
            if repo_lines:
                result = {
                    "map_id": map_entity.id,
                    "lines": [line.to_dict() for line in repo_lines],
                }
                _lines_registry_cache[tenant_id] = result
                return result
    except OSError:
        pass

    return {"map_id": "", "lines": []}


def _get_tenant_map_id(request: HttpRequest) -> str | None:
    """Return the map_id for the current tenant, or None."""
    map_entity = get_map_from_tenant_id(get_tenant_id(request))
    return map_entity.id if map_entity else None


def get_current_image(request: HttpRequest, map_id: str | None = None) -> str | None:
    """Get the current image filename from session, or first available image."""
    return _get_current_image(request, SESSION_IMAGE_KEY, map_id)



def _load_line_annotations_from_repo(image_filename: str) -> list[dict]:
    """Load line annotations from the DDD LineAnnotation repository."""
    frame = image_filename_to_frame(image_filename)
    if frame is None:
        return []
    return load_line_annotations_for_frame(frame.id)


def _save_line_annotations_to_repo(
    image_filename: str,
    annotations: list[dict],
) -> None:
    """Persist line annotations to the DDD LineAnnotation repository.

    Args:
        image_filename: Camera image filename (used as frame_id base).
        annotations: List of annotation dicts with line_id and pixel coords.
    """
    from poc_homography.domain.entities.line_annotation import LineAnnotation
    from poc_homography.domain.vo import PixelPoint

    frame = image_filename_to_frame(image_filename)
    if frame is None:
        return

    repo = _get_line_annotation_repo()

    for ann_dict in annotations:
        raw_points = ann_dict.get("points")
        points: tuple[PixelPoint, ...] | None = None
        if raw_points and len(raw_points) >= 2:
            points = tuple(PixelPoint.create(float(p[0]), float(p[1])) for p in raw_points)

        line_ann = LineAnnotation(
            line_id=ann_dict.get("line_id", ""),
            frame_id=frame.id,
            camera_pose=frame.ptz_state,
            start_pixel=PixelPoint.create(
                float(ann_dict.get("start_pixel_x", 0)),
                float(ann_dict.get("start_pixel_y", 0)),
            ),
            end_pixel=PixelPoint.create(
                float(ann_dict.get("end_pixel_x", 0)),
                float(ann_dict.get("end_pixel_y", 0)),
            ),
            points=points,
        )
        repo.save(line_ann)


def load_existing_line_annotations(image_filename: str) -> list[dict]:
    """Load existing line annotations for a specific image from the DDD repo."""
    return _load_line_annotations_from_repo(image_filename)


def index(request: HttpRequest) -> HttpResponse:
    """Serve the main HTML page."""
    current_image = get_current_image(request, _get_tenant_map_id(request))
    registry = load_lines_registry(get_tenant_id(request))
    context = {
        "image_filename": current_image or "No images available",
        "registry_filename": "DDD line repo",
        "map_id": registry.get("map_id", ""),
    }
    return render(request, "camera_line_annotator/index.html", context)


@require_GET
def api_images(request: HttpRequest) -> JsonResponse:
    """Get list of available images for the current tenant."""
    try:
        map_entity = get_map_from_tenant_id(get_tenant_id(request))
        map_id = map_entity.id if map_entity else None
        images = get_available_images_fresh(map_id)
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

    # Validate frame belongs to the current tenant's map
    map_entity = get_map_from_tenant_id(get_tenant_id(request))
    if map_entity and frame.map_id != map_entity.id:
        return JsonResponse({"error": f"Image not found: {filename}"}, status=404)

    request.session[SESSION_IMAGE_KEY] = filename

    image_annotations = load_existing_line_annotations(filename)

    camera_status = {
        "pan": float(frame.ptz_state.pan_raw),
        "tilt": float(frame.ptz_state.tilt_deg),
        "zoom": float(frame.ptz_state.zoom),
    }

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
    return serve_current_image(request, SESSION_IMAGE_KEY, _get_tenant_map_id(request))


@require_GET
def api_line_ids(request: HttpRequest) -> JsonResponse:
    """Get available line IDs from registry."""
    try:
        registry = load_lines_registry(get_tenant_id(request))
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
    """Get line annotations for current image from the YAML repo."""
    current_image = get_current_image(request, _get_tenant_map_id(request))
    if not current_image:
        return JsonResponse([], safe=False)

    return JsonResponse(load_existing_line_annotations(current_image), safe=False)


@csrf_exempt
@require_http_methods(["POST"])
def api_annotations_create(request: HttpRequest) -> JsonResponse:
    """Create a new line annotation."""
    current_image = get_current_image(request, _get_tenant_map_id(request))
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

    # Preserve N-point data if provided
    if "points" in data and isinstance(data["points"], list) and len(data["points"]) >= 2:
        annotation["points"] = [[float(p[0]), float(p[1])] for p in data["points"]]

    # Persist directly to YAML repo
    _save_line_annotations_to_repo(current_image, [annotation])
    invalidate_cache()

    return JsonResponse(
        {
            "success": True,
            "annotation": annotation,
        }
    )


@csrf_exempt
@require_http_methods(["PUT"])
def api_annotations_update(request: HttpRequest, line_id: str) -> JsonResponse:
    """Update an existing line annotation by line_id."""
    current_image = get_current_image(request, _get_tenant_map_id(request))
    if not current_image:
        return JsonResponse({"error": "No image selected"}, status=400)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Build updated annotation
    updated: dict[str, Any] = {
        "line_id": line_id,
        "start_pixel_x": float(data["start_pixel_x"]),
        "start_pixel_y": float(data["start_pixel_y"]),
        "end_pixel_x": float(data["end_pixel_x"]),
        "end_pixel_y": float(data["end_pixel_y"]),
    }

    if "points" in data and isinstance(data["points"], list) and len(data["points"]) >= 2:
        updated["points"] = [[float(p[0]), float(p[1])] for p in data["points"]]

    # Persist directly to YAML repo (save overwrites by entity_id)
    _save_line_annotations_to_repo(current_image, [updated])
    invalidate_cache()

    return JsonResponse(
        {
            "success": True,
            "annotation": updated,
        }
    )


@csrf_exempt
@require_http_methods(["DELETE"])
def api_annotations_delete(request: HttpRequest, line_id: str) -> JsonResponse:
    """Delete a line annotation by line_id."""
    current_image = get_current_image(request, _get_tenant_map_id(request))
    if not current_image:
        return JsonResponse({"error": "No image selected"}, status=400)

    frame = image_filename_to_frame(current_image)
    if frame is None:
        return JsonResponse({"error": "Current image not found"}, status=404)

    repo = _get_line_annotation_repo()
    entity_id = f"{frame.id}/{line_id}"
    deleted = repo.delete(entity_id)
    if not deleted:
        return JsonResponse({"error": f"Annotation not found: {line_id}"}, status=404)

    invalidate_cache()

    return JsonResponse(
        {
            "success": True,
            "deleted_line_id": line_id,
        }
    )


@require_GET
def api_camera_status(request: HttpRequest) -> JsonResponse:
    """Get camera status from the CapturedFrame entity."""
    current_image = get_current_image(request, _get_tenant_map_id(request))
    if not current_image:
        return JsonResponse({"pan": None, "tilt": None, "zoom": None})

    frame = image_filename_to_frame(current_image)
    if frame is None:
        return JsonResponse({"pan": None, "tilt": None, "zoom": None})

    return JsonResponse(
        {
            "pan": float(frame.ptz_state.pan_raw),
            "tilt": float(frame.ptz_state.tilt_deg),
            "zoom": float(frame.ptz_state.zoom),
        }
    )


