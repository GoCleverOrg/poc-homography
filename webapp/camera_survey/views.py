"""Views for the camera survey Django app.

This module contains thin HTTP request handlers that delegate business logic to services.py.
"""

from __future__ import annotations

import json
import logging

from django.http import FileResponse, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from poc_homography.camera_config import (
    get_cameras_for_tenant,
    get_tenants,
)

from .models import SurveyAxis, SurveyConfig
from .services import CameraSurveyService, get_survey_presets

logger = logging.getLogger(__name__)

# Singleton service instance
_survey_service = CameraSurveyService()


def _success_response(data: dict) -> JsonResponse:
    """Create standardized success response.

    Args:
        data: Response data

    Returns:
        JsonResponse with status="success"
    """
    return JsonResponse({"status": "success", "data": data})


def _error_response(message: str, status_code: int = 400) -> JsonResponse:
    """Create standardized error response.

    Args:
        message: Error message
        status_code: HTTP status code

    Returns:
        JsonResponse with status="error"
    """
    return JsonResponse({"status": "error", "message": message}, status=status_code)


@require_GET
def index(request: HttpRequest) -> HttpResponse:
    """Serve the main camera survey HTML page."""
    return render(request, "camera_survey/index.html")


@require_GET
def api_tenants(request: HttpRequest) -> JsonResponse:
    """Get list of available tenants.

    Returns:
        JSON response with tenant list.
    """
    try:
        tenants = get_tenants()
        tenant_list = [
            {"id": t["id"], "name": t["name"], "description": t.get("description", "")}
            for t in tenants
        ]
        return _success_response({"tenants": tenant_list})
    except Exception as e:
        logger.exception("Failed to load tenants")
        return _error_response("Failed to load tenants. Check server logs for details.", 500)


@require_GET
def api_cameras(request: HttpRequest) -> JsonResponse:
    """Get list of cameras for a tenant.

    Query params:
        tenant_id: Tenant ID to filter cameras

    Returns:
        JSON response with camera list.
    """
    tenant_id = request.GET.get("tenant_id")
    if not tenant_id:
        return _error_response("tenant_id is required")

    try:
        cameras = get_cameras_for_tenant(tenant_id)
        camera_list = [
            {
                "id": cam["id"],
                "name": cam["name"],
                "ip": cam["ip"],
                "model": cam.get("model"),
            }
            for cam in cameras
        ]
        return _success_response({"cameras": camera_list})
    except Exception as e:
        logger.exception("Failed to load cameras")
        return _error_response("Failed to load cameras. Check server logs for details.", 500)


@csrf_exempt
@require_POST
def api_start_survey(request: HttpRequest) -> JsonResponse:
    """Start a new survey.

    Request body (JSON):
        tenant_id: Tenant ID
        camera_id: Camera ID
        axis: "pan", "tilt", or "zoom"
        start: Start value
        end: End value
        step: Step size
        restore_ptz: Boolean (optional, default True)
        retry_timeout: Seconds (optional, default 60)
        session_tags: List of strings (optional)

    Returns:
        JSON response with session_id and session_path.
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return _error_response("Invalid JSON body")

    # Validate required fields
    required_fields = ["tenant_id", "camera_id", "axis", "start", "end", "step"]
    for field in required_fields:
        if field not in data:
            return _error_response(f"Missing required field: {field}")

    # Validate axis
    try:
        axis = SurveyAxis(data["axis"])
    except ValueError:
        return _error_response(f"Invalid axis: {data['axis']}. Must be 'pan', 'tilt', or 'zoom'")

    # Validate numeric fields
    try:
        start = float(data["start"])
        end = float(data["end"])
        step = float(data["step"])
    except (ValueError, TypeError):
        return _error_response("start, end, and step must be numeric")

    if step <= 0:
        return _error_response("step must be greater than 0")

    # Parse session tags
    session_tags = data.get("session_tags", [])
    if isinstance(session_tags, str):
        # Handle comma-separated string
        session_tags = [t.strip() for t in session_tags.split(",") if t.strip()]

    # Create config
    config = SurveyConfig(
        tenant_id=data["tenant_id"],
        camera_id=data["camera_id"],
        axis=axis,
        start=start,
        end=end,
        step=step,
        restore_ptz=data.get("restore_ptz", True),
        retry_timeout=int(data.get("retry_timeout", 60)),
        session_tags=session_tags,
    )

    # Start survey
    session_id, error = _survey_service.start_survey(config)

    if error:
        return _error_response(error, 500)

    return _success_response(
        {
            "session_id": session_id,
            "message": "Survey started successfully",
        }
    )


@require_GET
def api_survey_status(request: HttpRequest, session_id: str) -> JsonResponse:
    """Get survey progress.

    Args:
        session_id: Survey session ID

    Returns:
        JSON response with survey progress.
    """
    progress = _survey_service.get_survey_status(session_id)

    if progress is None:
        # Check if it's a completed session
        session = _survey_service.get_session(session_id)
        if session:
            return _success_response(
                {
                    "session_id": session_id,
                    "status": session.status.value,
                    "step_count": len(session.captures),
                    "total_steps": len(session.captures),
                    "current_ptz": session.final_ptz.to_dict() if session.final_ptz else None,
                    "last_capture_path": None,
                }
            )
        return _error_response("Survey not found", 404)

    return _success_response(progress.to_dict())


@csrf_exempt
@require_POST
def api_abort_survey(request: HttpRequest, session_id: str) -> JsonResponse:
    """Abort a running survey.

    Args:
        session_id: Survey session ID

    Returns:
        JSON response with abort status.
    """
    success, error = _survey_service.abort_survey(session_id)

    if not success:
        return _error_response(error or "Failed to abort survey", 400)

    return _success_response(
        {
            "session_id": session_id,
            "message": "Survey abort requested",
        }
    )


@require_GET
def api_sessions_list(request: HttpRequest) -> JsonResponse:
    """List survey sessions.

    Query params:
        limit: Maximum number to return (default 50)
        offset: Number to skip (default 0)

    Returns:
        JSON response with sessions list.
    """
    try:
        limit = int(request.GET.get("limit", 50))
        offset = int(request.GET.get("offset", 0))
    except ValueError:
        return _error_response("limit and offset must be integers")

    sessions, total = _survey_service.list_sessions(limit=limit, offset=offset)

    return _success_response(
        {
            "sessions": sessions,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


@require_GET
def api_session_detail(request: HttpRequest, session_id: str) -> JsonResponse:
    """Get complete session details.

    Args:
        session_id: Survey session ID

    Returns:
        JSON response with session data.
    """
    session = _survey_service.get_session(session_id)

    if session is None:
        return _error_response("Session not found", 404)

    return _success_response(session.to_dict())


@require_GET
def api_session_manifest(request: HttpRequest, session_id: str) -> HttpResponse:
    """Get session manifest YAML file.

    Args:
        session_id: Survey session ID

    Returns:
        YAML file response.
    """
    manifest_path = _survey_service.get_session_manifest_path(session_id)

    if manifest_path is None:
        return JsonResponse({"status": "error", "message": "Manifest not found"}, status=404)

    return FileResponse(
        open(manifest_path, "rb"),
        content_type="application/x-yaml",
        as_attachment=True,
        filename=f"survey_{session_id}_manifest.yaml",
    )


@require_GET
def api_session_image(request: HttpRequest, session_id: str, filename: str) -> HttpResponse:
    """Serve session image file.

    Args:
        session_id: Survey session ID
        filename: Image filename

    Returns:
        JPEG image response.
    """
    image_path = _survey_service.get_session_image_path(session_id, filename)

    if image_path is None:
        return JsonResponse({"status": "error", "message": "Image not found"}, status=404)

    return FileResponse(
        open(image_path, "rb"),
        content_type="image/jpeg",
    )


@require_GET
def api_presets(request: HttpRequest) -> JsonResponse:
    """Get survey presets.

    Returns:
        JSON response with preset list.
    """
    presets = get_survey_presets()

    return _success_response(
        {
            "presets": [p.to_dict() for p in presets],
        }
    )


@csrf_exempt
@require_POST
def api_delete_session(request: HttpRequest, session_id: str) -> JsonResponse:
    """Delete a survey session.

    Args:
        session_id: Survey session ID

    Returns:
        JSON response with deletion status.
    """
    success, error = _survey_service.delete_session(session_id)

    if not success:
        return _error_response(error or "Failed to delete session", 400)

    return _success_response(
        {
            "session_id": session_id,
            "message": "Session deleted successfully",
        }
    )
