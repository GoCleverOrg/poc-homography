"""Views for the Camera Evaluation Tool Django app.

This module contains HTTP request handlers for stress testing and survey functionality.
"""

from __future__ import annotations

import json
import logging
import os

from django.http import FileResponse, HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from poc_homography.camera_config import (
    get_camera_by_id,
    get_cameras_for_tenant,
    get_tenants,
)

from .models import (
    STRESS_TEST_PRESETS,
    AxisMovementConfig,
    StressTestConfig,
    StressTestType,
    UserEvaluation,
)
from .services import CameraStressTestService, generate_mjpeg_frames

# Import survey functionality from camera_survey app
from camera_survey.models import SurveyAxis, SurveyConfig
from camera_survey.services import get_survey_presets

# Import the shared survey service instance to avoid duplicate state
from camera_survey.views import _survey_service

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
    return render(request, "camera_evaluation/index.html")


# =============================================================================
# Common API Endpoints (reused from camera_diagnostic)
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
# Stress Test API Endpoints
# =============================================================================


@require_GET
def api_stress_test_presets(request: HttpRequest) -> JsonResponse:
    """Get available stress test presets."""
    presets = [preset.to_dict() for preset in STRESS_TEST_PRESETS]
    return _success_response({"presets": presets})


@require_POST
def api_stress_test_start(request: HttpRequest) -> JsonResponse:
    """Start a new stress test.

    Request body:
    {
        "tenant_id": "valencia",
        "camera_id": "valencia_cam01",
        "test_type": "oscillation",
        "pan_config": {...} | null,
        "tilt_config": {...} | null,
        "repetitions": 10
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return _error_response("Invalid JSON body")

    # Validate required fields
    tenant_id = body.get("tenant_id")
    camera_id = body.get("camera_id")
    test_type_str = body.get("test_type")

    if not tenant_id:
        return _error_response("tenant_id is required")
    if not camera_id:
        return _error_response("camera_id is required")
    if not test_type_str:
        return _error_response("test_type is required")

    # Validate test type
    try:
        test_type = StressTestType(test_type_str)
    except ValueError:
        valid_types = [t.value for t in StressTestType]
        return _error_response(f"Invalid test_type. Valid values: {valid_types}")

    # Parse axis configs
    pan_config = None
    tilt_config = None
    zoom_config = None

    if body.get("pan_config"):
        try:
            pan_config = AxisMovementConfig.from_dict(body["pan_config"])
        except Exception as e:
            return _error_response(f"Invalid pan_config: {e}")

    if body.get("tilt_config"):
        try:
            tilt_config = AxisMovementConfig.from_dict(body["tilt_config"])
        except Exception as e:
            return _error_response(f"Invalid tilt_config: {e}")

    if body.get("zoom_config"):
        try:
            zoom_config = AxisMovementConfig.from_dict(body["zoom_config"])
        except Exception as e:
            return _error_response(f"Invalid zoom_config: {e}")

    repetitions = body.get("repetitions", 1)

    # Create config
    config = StressTestConfig(
        tenant_id=tenant_id,
        camera_id=camera_id,
        test_type=test_type,
        pan_config=pan_config,
        tilt_config=tilt_config,
        zoom_config=zoom_config,
        repetitions=repetitions,
    )

    # Start test
    session_id, error = CameraStressTestService.start_stress_test(config)

    if error:
        return _error_response(error, status_code=500)

    return _success_response({"session_id": session_id})


@require_GET
def api_stress_test_status(request: HttpRequest, session_id: str) -> JsonResponse:
    """Get status of a running stress test."""
    progress = CameraStressTestService.get_stress_test_status(session_id)

    if not progress:
        # Try to get completed session
        session = CameraStressTestService.get_session(session_id)
        if session:
            return _success_response(
                {
                    "session_id": session.id,
                    "status": session.status.value,
                    "message": "Test completed" if session.result else "Session found",
                    "completed": session.status.value in ("completed", "failed", "aborted"),
                }
            )
        return _error_response("Session not found", status_code=404)

    return _success_response(progress.to_dict())


@require_POST
def api_stress_test_abort(request: HttpRequest, session_id: str) -> JsonResponse:
    """Abort a running stress test."""
    success, error = CameraStressTestService.abort_stress_test(session_id)

    if not success:
        return _error_response(error or "Failed to abort", status_code=400)

    return _success_response({"message": "Test abort requested"})


@require_GET
def api_stress_test_sessions(request: HttpRequest) -> JsonResponse:
    """List stress test sessions."""
    tenant_id = request.GET.get("tenant_id")
    camera_id = request.GET.get("camera_id")
    limit = int(request.GET.get("limit", 50))
    offset = int(request.GET.get("offset", 0))

    sessions, total = CameraStressTestService.list_stress_test_sessions(
        tenant_id=tenant_id,
        camera_id=camera_id,
        limit=limit,
        offset=offset,
    )

    return _success_response(
        {
            "sessions": [s.to_dict() for s in sessions],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


@require_GET
def api_stress_test_session_detail(request: HttpRequest, session_id: str) -> JsonResponse:
    """Get details of a specific stress test session."""
    session = CameraStressTestService.get_session(session_id)

    if not session:
        return _error_response("Session not found", status_code=404)

    return _success_response({"session": session.to_dict()})


@require_POST
def api_stress_test_evaluate(request: HttpRequest, session_id: str) -> JsonResponse:
    """Save user evaluation for a stress test session.

    Request body:
    {
        "evaluation": "good" | "needs_improvement" | "bad",
        "notes": "Optional notes..."
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return _error_response("Invalid JSON body")

    evaluation_str = body.get("evaluation")
    if not evaluation_str:
        return _error_response("evaluation is required")

    try:
        evaluation = UserEvaluation(evaluation_str)
    except ValueError:
        valid_values = [e.value for e in UserEvaluation]
        return _error_response(f"Invalid evaluation. Valid values: {valid_values}")

    notes = body.get("notes", "")

    success, error = CameraStressTestService.update_user_evaluation(session_id, evaluation, notes)

    if not success:
        return _error_response(error or "Failed to update evaluation", status_code=400)

    return _success_response({"message": "Evaluation saved"})


@require_POST
def api_stress_test_delete(request: HttpRequest, session_id: str) -> JsonResponse:
    """Delete a stress test session."""
    success, error = CameraStressTestService.delete_session(session_id)

    if not success:
        return _error_response(error or "Failed to delete session", status_code=400)

    return _success_response({"message": "Session deleted"})


# =============================================================================
# Survey API Endpoints (delegating to camera_survey services)
# =============================================================================


@require_GET
def api_survey_presets(request: HttpRequest) -> JsonResponse:
    """Get survey presets."""
    presets = get_survey_presets()
    return _success_response({"presets": [p.to_dict() for p in presets]})


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
def api_survey_session_manifest(request: HttpRequest, session_id: str) -> HttpResponse:
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
def api_survey_session_image(request: HttpRequest, session_id: str, filename: str) -> HttpResponse:
    """Serve session image file."""
    image_path = _survey_service.get_session_image_path(session_id, filename)

    if image_path is None:
        return JsonResponse({"status": "error", "message": "Image not found"}, status=404)

    return FileResponse(
        open(image_path, "rb"),
        content_type="image/jpeg",
    )


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
