"""FastAPI router for camera-evaluation endpoints.

Ported from ``webapp/camera_evaluation/views.py``.  Covers tenant/camera
listing, MJPEG video streaming, survey lifecycle (presets, start, status,
abort, sessions CRUD), and camera PTZ capabilities/position queries.
"""

from __future__ import annotations

import logging
from typing import Any

from camera_survey.models import SurveyAxis, SurveyConfig
from camera_survey.ptz import create_ptz_camera
from camera_survey.services import CameraSurveyService, get_survey_presets
from camera_survey.validation import parse_fixed_axis_values, validate_fixed_axis_ranges
from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from api.deps import get_current_user
from api.schemas.camera_evaluation import SurveyStartRequest
from poc_homography.camera_config import (
    get_camera_by_id,
    get_cameras_for_tenant,
    get_tenants,
)
from poc_homography.infrastructure.models.user import UserModel
from webapp.camera_evaluation.services import generate_mjpeg_frames

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton survey service (shared with camera_survey app)
# ---------------------------------------------------------------------------

_survey_service = CameraSurveyService()

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/camera-evaluation", tags=["camera-evaluation"])


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _success_response(data: dict) -> JSONResponse:
    """Standard ``{"status": "success", "data": ...}`` response."""
    return JSONResponse({"status": "success", "data": data})


def _error_response(
    message: str, status_code: int = 400, extra: dict | None = None
) -> JSONResponse:
    """Standard ``{"status": "error", ...}`` response."""
    payload: dict = {"status": "error", "message": message}
    if extra:
        payload.update(extra)
    return JSONResponse(payload, status_code=status_code)


# =============================================================================
# Common API endpoints
# =============================================================================


@router.get("/api/tenants/")
def api_tenants(
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Get list of available tenants."""
    try:
        tenants = get_tenants()
        tenant_list = [
            {"id": t["id"], "name": t["name"], "description": t.get("description", "")}
            for t in tenants
        ]
        return _success_response({"tenants": tenant_list})
    except Exception:
        logger.exception("Failed to load tenants")
        return _error_response(
            "Failed to load tenants. Check server logs for details.", status_code=500
        )


@router.get("/api/cameras/")
def api_cameras(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Get list of cameras for a tenant."""
    try:
        cameras = get_cameras_for_tenant(tenant_id)
        camera_list = [{"id": cam["id"], "name": cam["name"], "ip": cam["ip"]} for cam in cameras]
        return _success_response({"cameras": camera_list})
    except Exception:
        logger.exception("Failed to load cameras")
        return _error_response(
            "Failed to load cameras. Check server logs for details.", status_code=500
        )


# =============================================================================
# Video streaming endpoint
# =============================================================================


@router.get("/api/video-stream/{camera_id}/", response_model=None)
def api_video_stream(
    camera_id: str,
    user: UserModel = Depends(get_current_user),
) -> StreamingResponse | JSONResponse:
    """Stream MJPEG video from a camera's RTSP feed."""
    camera = get_camera_by_id(camera_id)
    if not camera:
        return _error_response(f"Camera not found: {camera_id}", status_code=404)

    return StreamingResponse(
        generate_mjpeg_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# =============================================================================
# Survey API endpoints
# =============================================================================


@router.get("/api/survey/presets/")
def api_survey_presets(
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Get survey presets."""
    presets = get_survey_presets()
    return _success_response({"presets": [p.to_dict() for p in presets]})


@router.post("/api/survey/start/")
def api_survey_start(
    body: SurveyStartRequest,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Start a new survey.

    Accepts a JSON body with survey configuration including tenant, camera,
    axis sweep parameters, and optional fixed axis positions.
    """
    # Validate axis
    try:
        axis = SurveyAxis(body.axis)
    except ValueError:
        return _error_response(f"Invalid axis: {body.axis}. Must be 'pan', 'tilt', or 'zoom'")

    # Validate step
    if body.step <= 0:
        return _error_response("step must be greater than 0")

    # Normalise session_tags (may arrive as comma-separated string)
    session_tags: list[str]
    if isinstance(body.session_tags, str):
        session_tags = [t.strip() for t in body.session_tags.split(",") if t.strip()]
    else:
        session_tags = body.session_tags

    # Build a raw dict for the fixed-axis helpers (they expect dict access)
    raw: dict[str, Any] = {}
    if body.fixed_pan is not None:
        raw["fixed_pan"] = body.fixed_pan
    if body.fixed_tilt is not None:
        raw["fixed_tilt"] = body.fixed_tilt
    if body.fixed_zoom is not None:
        raw["fixed_zoom"] = body.fixed_zoom

    fixed_pan, fixed_tilt, fixed_zoom, parse_err = parse_fixed_axis_values(raw)
    if parse_err:
        return _error_response(parse_err)

    range_err = validate_fixed_axis_ranges(
        fixed_pan, fixed_tilt, fixed_zoom, body.camera_id, body.tenant_id
    )
    if range_err:
        return _error_response(range_err)

    # Create config
    config = SurveyConfig(
        tenant_id=body.tenant_id,
        camera_id=body.camera_id,
        axis=axis,
        start=body.start,
        end=body.end,
        step=body.step,
        restore_ptz=body.restore_ptz,
        retry_timeout=body.retry_timeout,
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


@router.get("/api/survey/{session_id}/status/")
def api_survey_status(
    session_id: str,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
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


@router.post("/api/survey/{session_id}/abort/")
def api_survey_abort(
    session_id: str,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
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


@router.get("/api/survey/sessions/")
def api_survey_sessions(
    limit: int = Query(50),
    offset: int = Query(0),
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """List survey sessions (paginated)."""
    sessions, total = _survey_service.list_sessions(limit=limit, offset=offset)

    return _success_response(
        {
            "sessions": sessions,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


@router.get("/api/survey/sessions/{session_id}/")
def api_survey_session_detail(
    session_id: str,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Get complete session details."""
    session = _survey_service.get_session(session_id)

    if session is None:
        return _error_response("Session not found", status_code=404)

    return _success_response(session.to_dict())


@router.get("/api/survey/sessions/{session_id}/manifest/", response_model=None)
def api_survey_session_manifest(
    session_id: str,
    user: UserModel = Depends(get_current_user),
) -> FileResponse | JSONResponse:
    """Get session manifest YAML file."""
    manifest_path = _survey_service.get_session_manifest_path(session_id)

    if manifest_path is None:
        return _error_response("Manifest not found", status_code=404)

    return FileResponse(
        path=str(manifest_path),
        media_type="application/x-yaml",
        filename=f"survey_{session_id}_manifest.yaml",
    )


@router.get("/api/survey/sessions/{session_id}/images/{filename}", response_model=None)
def api_survey_session_image(
    session_id: str,
    filename: str,
    user: UserModel = Depends(get_current_user),
) -> FileResponse | JSONResponse:
    """Serve session image file."""
    image_path = _survey_service.get_session_image_path(session_id, filename)

    if image_path is None:
        return _error_response("Image not found", status_code=404)

    return FileResponse(
        path=str(image_path),
        media_type="image/jpeg",
    )


@router.post("/api/survey/sessions/{session_id}/delete/")
def api_survey_delete_session(
    session_id: str,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
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
# Camera PTZ API endpoints
# =============================================================================


@router.get("/api/camera/capabilities/")
def api_camera_capabilities(
    tenant_id: str = Query(...),
    camera_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Get camera PTZ capabilities (axis ranges) from the camera.

    Returns pan, tilt, zoom min/max ranges and minimum step size.
    """
    camera = get_camera_by_id(camera_id)
    if not camera:
        return _error_response(f"Camera not found: {camera_id}", status_code=404)

    camera_ip = camera.get("ip")
    camera_name = camera.get("name", camera_id)

    if not camera_ip:
        return _error_response(f"Camera IP not configured: {camera_id}", status_code=500)

    try:
        ptz_camera = create_ptz_camera(
            camera_ip=camera_ip, camera_name=camera_name, tenant_id=tenant_id
        )
        capabilities = ptz_camera.get_capabilities()

        capabilities_data = {
            "pan": {
                "min": capabilities.pan_min,
                "max": capabilities.pan_max,
                "speed_min": capabilities.pan_speed_min,
                "speed_max": capabilities.pan_speed_max,
            },
            "tilt": {
                "min": capabilities.tilt_min,
                "max": capabilities.tilt_max,
                "speed_min": capabilities.tilt_speed_min,
                "speed_max": capabilities.tilt_speed_max,
            },
            "zoom": {
                "min": capabilities.zoom_min,
                "max": capabilities.zoom_max,
            },
            "focus": {
                "min": capabilities.focus_min,
                "max": capabilities.focus_max,
            },
            "min_step": 0.1,
        }

        return _success_response({"capabilities": capabilities_data})

    except ValueError as e:
        logger.exception("Credentials error for %s", camera_id)
        return _error_response(str(e), status_code=500)
    except Exception:
        logger.exception("Error getting capabilities for %s", camera_id)
        return _error_response("Failed to get camera capabilities", status_code=502)


@router.get("/api/camera/position/")
def api_camera_position(
    tenant_id: str = Query(...),
    camera_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Get current camera PTZ position.

    Returns current pan, tilt, zoom values.
    """
    camera = get_camera_by_id(camera_id)
    if not camera:
        return _error_response(f"Camera not found: {camera_id}", status_code=404)

    camera_ip = camera.get("ip")
    camera_name = camera.get("name", camera_id)

    if not camera_ip:
        return _error_response(f"Camera IP not configured: {camera_id}", status_code=500)

    try:
        ptz_camera = create_ptz_camera(
            camera_ip=camera_ip, camera_name=camera_name, tenant_id=tenant_id
        )

        # Read the full PTZState (pan/tilt/zoom plus focus) through the wrapper
        # pass-through rather than reaching into the concrete adapter.
        state = ptz_camera.get_ptz_state()

        return _success_response(
            {
                "pan": float(state.pan_raw),
                "tilt": float(state.tilt_deg),
                "zoom": float(state.zoom),
                "focus": int(state.focus) if state.focus is not None else None,
            }
        )

    except ValueError as e:
        logger.exception("Credentials error for %s", camera_id)
        return _error_response(str(e), status_code=500)
    except Exception as e:
        logger.exception("Error getting position for %s", camera_id)
        return _error_response(f"Failed to get camera position: {e}", status_code=502)
