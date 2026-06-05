"""Views for the Camera Evaluation Tool Django app.

Note: Stress test views have been moved to camera_diagnostic app.
This module now only handles survey functionality.
"""

from __future__ import annotations

import json
import logging

# Import survey functionality from camera_survey app
from camera_survey.models import SurveyAxis, SurveyConfig
from camera_survey.ptz import create_ptz_camera
from camera_survey.services import get_survey_presets
from camera_survey.validation import parse_fixed_axis_values, validate_fixed_axis_ranges

# Import the shared survey service instance to avoid duplicate state
from camera_survey.views import _survey_service
from django.http import FileResponse, HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from poc_homography.camera_config import (
    get_camera_by_id,
    get_cameras_for_tenant,
    get_tenant_by_id,
    get_tenants,
)

from .services import generate_mjpeg_frames

logger = logging.getLogger(__name__)


# =============================================================================
# Response Helpers
# =============================================================================


def _success_response(data: dict) -> JsonResponse:
    """Create a standardized success response."""
    return JsonResponse({"status": "success", "data": data})


def _error_response(
    message: str, status_code: int = 400, extra: dict | None = None
) -> JsonResponse:
    """Create a standardized error response."""
    response_data = {"status": "error", "message": message}
    if extra:
        response_data.update(extra)
    return JsonResponse(response_data, status=status_code)


# =============================================================================
# Main Page View
# =============================================================================


@require_GET
def index(request: HttpRequest) -> HttpResponse:
    """Serve the main Camera Evaluation Tool HTML page."""
    from homography_web.frame_utils import get_tenant_id

    tenant_id = get_tenant_id(request)
    tenant = get_tenant_by_id(tenant_id)
    return render(
        request,
        "camera_evaluation/index.html",
        {
            "tenant_id": tenant_id,
            "tenant_name": tenant.get("name", tenant_id) if tenant else tenant_id,
        },
    )


# =============================================================================
# Common API Endpoints
# =============================================================================


@require_GET
def api_tenants(request: HttpRequest) -> JsonResponse:
    """Get list of available tenants."""
    try:
        tenants = get_tenants()
        tenant_list = [
            {"id": t["id"], "name": t["name"], "description": t.get("description", "")}
            for t in tenants
        ]
        return _success_response({"tenants": tenant_list})
    except Exception as e:
        logger.exception("Failed to load tenants")
        return _error_response(
            "Failed to load tenants. Check server logs for details.", status_code=500
        )


@require_GET
def api_cameras(request: HttpRequest) -> JsonResponse:
    """Get list of cameras for a tenant."""
    tenant_id = request.GET.get("tenant_id")
    if not tenant_id:
        return _error_response("tenant_id is required")

    try:
        cameras = get_cameras_for_tenant(tenant_id)
        camera_list = [{"id": cam["id"], "name": cam["name"], "ip": cam["ip"]} for cam in cameras]
        return _success_response({"cameras": camera_list})
    except Exception as e:
        logger.exception("Failed to load cameras")
        return _error_response(
            "Failed to load cameras. Check server logs for details.", status_code=500
        )


# =============================================================================
# Video Streaming Endpoint
# =============================================================================


@require_GET
def api_video_stream(request: HttpRequest, camera_id: str) -> StreamingHttpResponse | JsonResponse:
    """Stream MJPEG video from a camera's RTSP feed."""
    camera = get_camera_by_id(camera_id)
    if not camera:
        return _error_response(f"Camera not found: {camera_id}", status_code=404)

    response = StreamingHttpResponse(
        generate_mjpeg_frames(camera_id),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response


# =============================================================================
# Survey API Endpoints (delegating to camera_survey services)
# =============================================================================


@require_GET
def api_survey_presets(request: HttpRequest) -> JsonResponse:
    """Get survey presets."""
    presets = get_survey_presets()
    return _success_response({"presets": [p.to_dict() for p in presets]})


@csrf_exempt
@require_POST
def api_survey_start(request: HttpRequest) -> JsonResponse:
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
        fixed_pan: Fixed pan value (optional, float)
        fixed_tilt: Fixed tilt value (optional, float)
        fixed_zoom: Fixed zoom value (optional, float)
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
        session_tags = [t.strip() for t in session_tags.split(",") if t.strip()]

    # Parse and validate optional fixed axis values
    fixed_pan, fixed_tilt, fixed_zoom, parse_err = parse_fixed_axis_values(data)
    if parse_err:
        return _error_response(parse_err)

    range_err = validate_fixed_axis_ranges(
        fixed_pan, fixed_tilt, fixed_zoom, data["camera_id"], data["tenant_id"]
    )
    if range_err:
        return _error_response(range_err)

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
        fixed_pan=fixed_pan,
        fixed_tilt=fixed_tilt,
        fixed_zoom=fixed_zoom,
    )

    # Start survey
    session_id, error = _survey_service.start_survey(config)

    if error:
        return _error_response(error, status_code=500)

    return _success_response(
        {
            "session_id": session_id,
            "message": "Survey started successfully",
        }
    )


@require_GET
def api_survey_status(request: HttpRequest, session_id: str) -> JsonResponse:
    """Get survey progress."""
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
        return _error_response("Survey not found", status_code=404)

    return _success_response(progress.to_dict())


@csrf_exempt
@require_POST
def api_survey_abort(request: HttpRequest, session_id: str) -> JsonResponse:
    """Abort a running survey."""
    success, error = _survey_service.abort_survey(session_id)

    if not success:
        return _error_response(error or "Failed to abort survey", status_code=400)

    return _success_response(
        {
            "session_id": session_id,
            "message": "Survey abort requested",
        }
    )


@require_GET
def api_survey_sessions(request: HttpRequest) -> JsonResponse:
    """List survey sessions."""
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
def api_survey_session_detail(request: HttpRequest, session_id: str) -> JsonResponse:
    """Get complete session details."""
    session = _survey_service.get_session(session_id)

    if session is None:
        return _error_response("Session not found", status_code=404)

    return _success_response(session.to_dict())


@require_GET
def api_survey_session_manifest(
    request: HttpRequest, session_id: str
) -> HttpResponse | FileResponse:
    """Get session manifest YAML file."""
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
def api_survey_session_image(
    request: HttpRequest, session_id: str, filename: str
) -> HttpResponse | FileResponse:
    """Serve session image file."""
    image_path = _survey_service.get_session_image_path(session_id, filename)

    if image_path is None:
        return JsonResponse({"status": "error", "message": "Image not found"}, status=404)

    return FileResponse(
        open(image_path, "rb"),
        content_type="image/jpeg",
    )


@csrf_exempt
@require_POST
def api_survey_delete_session(request: HttpRequest, session_id: str) -> JsonResponse:
    """Delete a survey session."""
    success, error = _survey_service.delete_session(session_id)

    if not success:
        return _error_response(error or "Failed to delete session", status_code=400)

    return _success_response(
        {
            "session_id": session_id,
            "message": "Session deleted successfully",
        }
    )


# =============================================================================
# Camera PTZ API Endpoints
# =============================================================================


@require_GET
def api_camera_capabilities(request: HttpRequest) -> JsonResponse:
    """Get camera PTZ capabilities (axis ranges) from the camera.

    Query parameters:
        tenant_id: Tenant ID
        camera_id: Camera ID

    Returns:
        JSON with pan, tilt, zoom min/max ranges and minimum step size.
    """
    tenant_id = request.GET.get("tenant_id")
    camera_id = request.GET.get("camera_id")

    if not tenant_id:
        return _error_response("tenant_id is required")
    if not camera_id:
        return _error_response("camera_id is required")

    # Get camera configuration
    camera = get_camera_by_id(camera_id)
    if not camera:
        return _error_response(f"Camera not found: {camera_id}", status_code=404)

    camera_ip = camera.get("ip")
    camera_name = camera.get("name", camera_id)

    if not camera_ip:
        return _error_response(f"Camera IP not configured: {camera_id}", status_code=500)

    try:
        # Create PTZ camera instance and get capabilities via abstraction layer
        ptz_camera = create_ptz_camera(
            camera_ip=camera_ip, camera_name=camera_name, tenant_id=tenant_id
        )
        capabilities = ptz_camera.get_capabilities()

        # Build response from CameraCapabilities model
        capabilities_data = {
            "pan": {
                "min": capabilities.pan_min,
                "max": capabilities.pan_max,
            },
            "tilt": {
                "min": capabilities.tilt_min,
                "max": capabilities.tilt_max,
            },
            "zoom": {
                "min": capabilities.zoom_min,
                "max": capabilities.zoom_max,
            },
            "min_step": 0.1,  # Cameras report position in 0.1° increments
        }

        return _success_response({"capabilities": capabilities_data})

    except ValueError as e:
        # Raised by create_ptz_camera if credentials not set
        logger.exception(f"Credentials error for {camera_id}")
        return _error_response(str(e), status_code=500)
    except Exception as e:
        logger.exception(f"Error getting capabilities for {camera_id}")
        return _error_response("Failed to get camera capabilities", status_code=502)


@require_GET
def api_camera_position(request: HttpRequest) -> JsonResponse:
    """Get current camera PTZ position.

    Query parameters:
        tenant_id: Tenant ID
        camera_id: Camera ID

    Returns:
        JSON with current pan, tilt, zoom values.
    """
    tenant_id = request.GET.get("tenant_id")
    camera_id = request.GET.get("camera_id")

    if not tenant_id:
        return _error_response("tenant_id is required")
    if not camera_id:
        return _error_response("camera_id is required")

    # Get camera configuration
    camera = get_camera_by_id(camera_id)
    if not camera:
        return _error_response(f"Camera not found: {camera_id}", status_code=404)

    camera_ip = camera.get("ip")
    camera_name = camera.get("name", camera_id)

    if not camera_ip:
        return _error_response(f"Camera IP not configured: {camera_id}", status_code=500)

    try:
        # Create PTZ camera instance using existing abstraction
        ptz_camera = create_ptz_camera(
            camera_ip=camera_ip, camera_name=camera_name, tenant_id=tenant_id
        )

        # Get current position
        position = ptz_camera.get_status()

        if position is None:
            return _error_response("Failed to get camera position", status_code=502)

        return _success_response(
            {
                "pan": position.pan,
                "tilt": position.tilt,
                "zoom": position.zoom,
            }
        )

    except ValueError as e:
        # Raised by create_ptz_camera if credentials not set
        logger.exception(f"Credentials error for {camera_id}")
        return _error_response(str(e), status_code=500)
    except Exception as e:
        logger.exception(f"Error getting position for {camera_id}")
        return _error_response(f"Failed to get camera position: {e}", status_code=502)
