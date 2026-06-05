"""FastAPI router for camera-diagnostic endpoints.

Ported from ``webapp/camera_diagnostic/views.py``.  Covers both the diagnostic
endpoints (RTSP, WebUI, PTZ tests, session CRUD) and the stress-test endpoints.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

import cv2
from camera_diagnostic.models import (
    STRESS_TEST_PRESETS,
    AxisMovementConfig,
    CameraDiagnosticResult,
    DiagnosticSession,
    DiagnosticSessionStatus,
    DiagnosticTestResult,
    DiagnosticTestStatus,
    StressTestConfig,
    StressTestType,
    UserEvaluation,
)
from camera_diagnostic.services import (
    PTZ_MOVEMENT_DURATION,
    PTZ_MOVEMENT_SPEED,
    WEBUI_CONNECTION_TIMEOUT_SEC,
    CameraErrorCategory,
    CameraStressTestService,
    _ptz_state_to_status_dict,
    _sanitize_camera_name,
    attempt_login,
    check_login_success,
    classify_ptz_error,
    classify_rtsp_error,
    classify_webui_error,
    create_rtsp_capture,
    delete_diagnostic_session,
    detect_ptz_controls,
    execute_movement_test,
    generate_mjpeg_frames,
    get_presets_list,
    get_screenshot_path,
    list_diagnostic_sessions,
    load_diagnostic_session,
    save_diagnostic_session,
    wait_for_stabilization,
)
from camera_survey.ptz import create_ptz_camera
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy.orm import Session

from api.deps import get_current_user, get_db_session
from api.schemas.camera_diagnostic import (
    CameraOut,
    CameraResultIn,
    RunDiagnosticRequest,
    SaveDiagnosticRequest,
    StressTestEvaluateRequest,
    StressTestStartRequest,
    TenantOut,
)
from poc_homography.camera_config import (
    get_camera_by_id,
    get_camera_by_name,
    get_camera_configs,
    get_cameras_for_tenant,
    get_rtsp_url,
    get_tenant_by_id,
    get_tenant_credentials,
)
from poc_homography.infrastructure.clients.hikvision import (
    HikvisionISAPIClient,
    HikvisionTransportError,
)
from poc_homography.infrastructure.models.user import UserModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/camera-diagnostic", tags=["camera-diagnostic"])


# ---------------------------------------------------------------------------
# Helpers (ported from Django views)
# ---------------------------------------------------------------------------


def _success_response(data: dict) -> JSONResponse:
    """Standard ``{"status": "success", "data": ...}`` response."""
    return JSONResponse({"status": "success", "data": data})


def _error_response(
    error_category: CameraErrorCategory,
    message: str,
    status_code: int = 500,
    extra: dict | None = None,
) -> JSONResponse:
    """Standard ``{"status": "error", ...}`` response."""
    payload: dict = {
        "status": "error",
        "error_category": error_category.value,
        "message": message,
    }
    if extra:
        payload.update(extra)
    return JSONResponse(payload, status_code=status_code)


def _validate_camera_for_rtsp(camera_name: str) -> tuple[dict, str] | JSONResponse:
    """Return ``(camera_dict, rtsp_url)`` or a ``JSONResponse`` on failure."""
    camera = get_camera_by_name(camera_name)
    if not camera:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Camera not found: {camera_name}",
            status_code=404,
        )
    try:
        rtsp_url = get_rtsp_url(camera_name)
        if not rtsp_url:
            return _error_response(
                CameraErrorCategory.CAMERA_NOT_FOUND,
                f"Camera not found: {camera_name}",
                status_code=404,
            )
    except ValueError as e:
        return _error_response(CameraErrorCategory.CREDENTIALS_NOT_SET, str(e))
    return (camera, rtsp_url)


def _validate_camera_for_webui_ptz(
    camera_name: str,
) -> tuple[dict, str, str, str] | JSONResponse:
    """Return ``(camera, ip, username, password)`` or a ``JSONResponse`` on failure."""
    camera = get_camera_by_name(camera_name)
    if not camera:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Camera not found: {camera_name}",
            status_code=404,
        )
    camera_ip = camera.get("ip")
    if not camera_ip:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            f"No IP address configured for camera: {camera_name}",
        )
    tenant_id = camera.get("tenant_id")
    username, password = get_tenant_credentials(tenant_id)
    if not username or not password:
        return _error_response(
            CameraErrorCategory.CREDENTIALS_NOT_SET,
            f"Camera credentials not set for tenant '{tenant_id}'. "
            f"Set {tenant_id.upper()}_CAMERA_USERNAME and {tenant_id.upper()}_CAMERA_PASSWORD, "
            "or global CAMERA_USERNAME/CAMERA_PASSWORD as fallback.",
        )
    return (camera, camera_ip, username, password)


# ---------------------------------------------------------------------------
# Internal diagnostic runners (ported from Django views)
# ---------------------------------------------------------------------------


def _run_rtsp_test(camera_id: str) -> DiagnosticTestResult:
    """Run RTSP test for a camera."""
    result = DiagnosticTestResult(test_type="rtsp")
    try:
        rtsp_url = get_rtsp_url(camera_id)
        if not rtsp_url:
            result.status = DiagnosticTestStatus.FAIL
            result.error_message = f"No RTSP URL for camera: {camera_id}"
            result.error_category = CameraErrorCategory.CAMERA_NOT_FOUND.value
            return result

        start_time = time.time()
        cap = create_rtsp_capture(rtsp_url)
        try:
            if not cap.isOpened():
                result.status = DiagnosticTestStatus.FAIL
                result.error_message = "Failed to connect to RTSP stream"
                result.error_category = CameraErrorCategory.CAMERA_OFFLINE.value
                return result

            latency_ms = (time.time() - start_time) * 1000
            result.response_time_ms = round(latency_ms, 2)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, _ = cap.read()
            if not ret:
                result.status = DiagnosticTestStatus.FAIL
                result.error_message = "Connected but failed to read frame"
                result.error_category = CameraErrorCategory.INVALID_RESPONSE.value
                return result
            result.status = DiagnosticTestStatus.PASS
            result.details = {
                "resolution": {"width": width, "height": height},
                "fps": fps if fps > 0 else None,
            }
        finally:
            cap.release()
    except Exception as e:
        result.status = DiagnosticTestStatus.FAIL
        result.error_message = str(e)
        result.error_category = classify_rtsp_error(e).value
    return result


def _run_webui_test(
    camera_name: str,
    camera_ip: str | None,
    username: str,
    password: str,
) -> DiagnosticTestResult:
    """Run WebUI test for a camera."""
    result = DiagnosticTestResult(test_type="webui")
    if not camera_ip:
        result.status = DiagnosticTestStatus.FAIL
        result.error_message = "No IP address for camera"
        result.error_category = CameraErrorCategory.INVALID_RESPONSE.value
        return result

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        from playwright.sync_api import sync_playwright
    except ImportError:
        result.status = DiagnosticTestStatus.FAIL
        result.error_message = "Playwright not installed"
        result.error_category = CameraErrorCategory.INVALID_RESPONSE.value
        return result

    browser = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                ignore_https_errors=True,
                viewport={"width": 1280, "height": 720},
            )
            page = context.new_page()
            page.set_default_timeout(WEBUI_CONNECTION_TIMEOUT_SEC * 1000)
            start_time = time.time()
            try:
                page.goto(
                    f"http://{camera_ip}",
                    timeout=WEBUI_CONNECTION_TIMEOUT_SEC * 1000,
                    wait_until="networkidle",
                )
            except PlaywrightTimeout:
                result.status = DiagnosticTestStatus.FAIL
                result.error_message = "Connection timeout"
                result.error_category = CameraErrorCategory.TIMEOUT.value
                return result

            result.response_time_ms = round((time.time() - start_time) * 1000, 2)
            login_attempted = attempt_login(page, username, password)
            if login_attempted:
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except PlaywrightTimeout:
                    pass
                from camera_diagnostic.services import dismiss_hikvision_warning_dialog

                try:
                    dismiss_hikvision_warning_dialog(page)
                except Exception:
                    pass

            login_success = check_login_success(page) if login_attempted else False
            ptz_controls = detect_ptz_controls(page)
            result.status = (
                DiagnosticTestStatus.PASS if login_success else DiagnosticTestStatus.FAIL
            )
            result.details = {
                "login_attempted": login_attempted,
                "login_success": login_success,
                "ptz_controls_found": len(ptz_controls),
            }
            if not login_success and login_attempted:
                from camera_diagnostic.services import get_login_error_message

                login_error = get_login_error_message(page)
                if login_error:
                    result.error_message = login_error
            browser.close()
            browser = None
    except Exception as e:
        result.status = DiagnosticTestStatus.FAIL
        result.error_message = str(e)
        result.error_category = classify_webui_error(e).value
    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
    return result


def _run_ptz_test(
    camera_name: str,
    camera_ip: str | None,
    username: str,
    password: str,
) -> DiagnosticTestResult:
    """Run PTZ API test for a camera."""
    result = DiagnosticTestResult(test_type="ptz")
    if not camera_ip:
        result.status = DiagnosticTestStatus.FAIL
        result.error_message = "No IP address for camera"
        result.error_category = CameraErrorCategory.INVALID_RESPONSE.value
        return result

    ptz = HikvisionISAPIClient(camera_ip, username, password)
    initial_status: dict | None = None
    try:
        start_time = time.time()
        initial_status = _ptz_state_to_status_dict(ptz.get_ptz_status())
        result.response_time_ms = round((time.time() - start_time) * 1000, 2)

        tests_passed = 0
        tests_failed = 0
        movement_tests = [
            ("pan_left", -PTZ_MOVEMENT_SPEED, 0, 0),
            ("pan_right", PTZ_MOVEMENT_SPEED, 0, 0),
            ("tilt_up", 0, PTZ_MOVEMENT_SPEED, 0),
            ("tilt_down", 0, -PTZ_MOVEMENT_SPEED, 0),
        ]
        for _test_name, pan, tilt, zoom in movement_tests:
            test_result = execute_movement_test(
                ptz, pan=pan, tilt=tilt, zoom=zoom, duration=PTZ_MOVEMENT_DURATION
            )
            if test_result.get("success"):
                tests_passed += 1
            else:
                tests_failed += 1

        presets_result = get_presets_list(camera_ip, username, password)
        result.status = (
            DiagnosticTestStatus.PASS if tests_failed == 0 else DiagnosticTestStatus.FAIL
        )
        result.details = {
            "initial_position": {
                "pan": initial_status["pan"],
                "tilt": initial_status["tilt"],
                "zoom": initial_status["zoom"],
            },
            "movement_tests_passed": tests_passed,
            "movement_tests_failed": tests_failed,
            "presets_count": presets_result.get("count", 0),
        }
    except HikvisionTransportError as e:
        result.status = DiagnosticTestStatus.FAIL
        result.error_message = str(e)
        result.error_category = classify_ptz_error(e).value
    except Exception as e:
        result.status = DiagnosticTestStatus.FAIL
        result.error_message = str(e)
        result.error_category = classify_ptz_error(e).value
    finally:
        if initial_status:
            try:
                ptz.move_absolute(
                    pan=initial_status["pan"],
                    tilt=initial_status["tilt"],
                    zoom=initial_status["zoom"],
                )
                wait_for_stabilization(ptz)
            except Exception as e:
                logger.warning(f"Failed to restore PTZ position: {e}")
    return result


# =============================================================================
# Diagnostic endpoints
# =============================================================================


@router.get("/api/tenants/")
def api_tenants(
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> JSONResponse:
    """List all tenants."""
    try:
        from poc_homography.infrastructure.repositories import RepoPostgresTenant

        tenant_entities = RepoPostgresTenant(session).get_all()
        tenant_list = [
            TenantOut(id=t.id, name=t.name, description=t.description).model_dump()
            for t in tenant_entities
        ]
        return _success_response({"tenants": tenant_list})
    except Exception as e:
        return _error_response(CameraErrorCategory.API_ERROR, f"Failed to load tenants: {e}")


@router.get("/api/cameras/")
def api_cameras(
    tenant_id: str = Query(None),
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """List cameras, optionally filtered by tenant."""
    try:
        if tenant_id:
            cameras = get_cameras_for_tenant(tenant_id)
        else:
            cameras = get_camera_configs()

        camera_list = [
            CameraOut(
                id=cam["id"],
                name=cam["name"],
                ip=cam["ip"],
                tenant_id=cam.get("tenant_id"),
            ).model_dump()
            for cam in cameras
        ]
        return _success_response({"cameras": camera_list})
    except Exception as e:
        return _error_response(CameraErrorCategory.API_ERROR, f"Failed to load cameras: {e}")


@router.get("/api/video-stream/{camera_name}/", response_model=None)
def api_video_stream(
    camera_name: str,
    user: UserModel = Depends(get_current_user),
) -> StreamingResponse | JSONResponse:
    """Stream MJPEG video from a camera's RTSP feed."""
    validation = _validate_camera_for_rtsp(camera_name)
    if isinstance(validation, JSONResponse):
        return validation

    return StreamingResponse(
        generate_mjpeg_frames(camera_name),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/api/test-rtsp/{camera_name}/")
def api_test_rtsp(
    camera_name: str,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Test RTSP connectivity and return metrics."""
    validation = _validate_camera_for_rtsp(camera_name)
    if isinstance(validation, JSONResponse):
        return validation
    camera, rtsp_url = validation

    start_time = time.time()
    cap = None
    try:
        cap = create_rtsp_capture(rtsp_url)
        if not cap.isOpened():
            return _error_response(
                CameraErrorCategory.CAMERA_OFFLINE,
                f"Failed to connect to RTSP stream for {camera_name}",
                status_code=503,
            )

        latency_ms = (time.time() - start_time) * 1000
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        ret, _ = cap.read()
        if not ret:
            return _error_response(
                CameraErrorCategory.INVALID_RESPONSE,
                f"Connected but failed to read frame from {camera_name}",
                status_code=503,
            )

        return _success_response(
            {
                "message": f"Successfully connected to {camera_name}",
                "metrics": {
                    "fps": fps if fps > 0 else None,
                    "resolution": {"width": width, "height": height},
                    "latency_ms": round(latency_ms, 2),
                },
            }
        )
    except Exception as e:
        error_category = classify_rtsp_error(e, rtsp_url)
        logger.error(f"RTSP test failed for {camera_name}: {e}")
        return _error_response(error_category, f"RTSP connection failed: {e}", status_code=503)
    finally:
        if cap is not None:
            cap.release()


@router.get("/api/capture-snapshot/{camera_name}/", response_model=None)
def api_capture_snapshot(
    camera_name: str,
    user: UserModel = Depends(get_current_user),
) -> Response | JSONResponse:
    """Capture a single frame from the camera and return as JPEG."""
    validation = _validate_camera_for_rtsp(camera_name)
    if isinstance(validation, JSONResponse):
        return validation
    camera, rtsp_url = validation

    cap = None
    try:
        cap = create_rtsp_capture(rtsp_url)
        if not cap.isOpened():
            return _error_response(
                CameraErrorCategory.CAMERA_OFFLINE,
                f"Failed to connect to RTSP stream for {camera_name}",
                status_code=503,
            )

        ret, frame = cap.read()
        if not ret:
            return _error_response(
                CameraErrorCategory.INVALID_RESPONSE,
                f"Failed to capture frame from {camera_name}",
                status_code=503,
            )

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            return _error_response(
                CameraErrorCategory.INVALID_RESPONSE,
                "Failed to encode frame as JPEG",
            )

        sanitized_name = _sanitize_camera_name(camera_name)
        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={"Content-Disposition": f'inline; filename="{sanitized_name}_snapshot.jpg"'},
        )
    except Exception as e:
        error_category = classify_rtsp_error(e, rtsp_url)
        logger.error(f"Snapshot capture failed for {camera_name}: {e}")
        return _error_response(error_category, f"Snapshot capture failed: {e}", status_code=503)
    finally:
        if cap is not None:
            cap.release()


@router.get("/api/test-webui/{camera_name}/")
def api_test_webui(
    camera_name: str,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Test camera web interface using Playwright."""
    validation = _validate_camera_for_webui_ptz(camera_name)
    if isinstance(validation, JSONResponse):
        return validation
    camera, camera_ip, username, password = validation

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        from playwright.sync_api import sync_playwright
    except ImportError:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            "Playwright is not installed. Install with: pip install playwright && playwright install chromium",
        )

    browser = None
    screenshot_path = None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                ignore_https_errors=True,
                viewport={"width": 1280, "height": 720},
            )
            page = context.new_page()
            page.set_default_timeout(WEBUI_CONNECTION_TIMEOUT_SEC * 1000)

            try:
                page.goto(
                    f"http://{camera_ip}",
                    timeout=WEBUI_CONNECTION_TIMEOUT_SEC * 1000,
                    wait_until="networkidle",
                )
            except PlaywrightTimeout:
                return _error_response(
                    CameraErrorCategory.TIMEOUT,
                    f"Connection timeout: Could not reach camera web interface at http://{camera_ip}",
                    status_code=503,
                )

            login_attempted = attempt_login(page, username, password)
            if login_attempted:
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except PlaywrightTimeout:
                    pass
                from camera_diagnostic.services import dismiss_hikvision_warning_dialog

                try:
                    dismiss_hikvision_warning_dialog(page)
                except Exception as e:
                    logger.debug(f"Warning dialog handling: {e}")

            login_success = check_login_success(page) if login_attempted else False
            login_error = None
            if login_attempted and not login_success:
                from camera_diagnostic.services import get_login_error_message

                login_error = get_login_error_message(page)

            ptz_controls = detect_ptz_controls(page)

            ptz_test_result = None
            if login_success and ptz_controls:
                try:
                    from camera_diagnostic.services import test_webui_ptz_controls

                    ptz_test_result = test_webui_ptz_controls(page)
                except Exception as e:
                    logger.debug(f"PTZ control test: {e}")

            screenshot_path = get_screenshot_path(camera_name)
            page.screenshot(path=str(screenshot_path), full_page=True)
            browser.close()
            browser = None

            response_data: dict = {
                "message": f"Web UI test completed for {camera_name}",
                "login_success": login_success,
                "login_attempted": login_attempted,
                "ptz_controls_found": ptz_controls,
                "screenshot_path": str(screenshot_path),
            }
            if login_error:
                response_data["login_error"] = login_error
            if ptz_test_result:
                response_data["ptz_test_result"] = ptz_test_result

            return _success_response(response_data)

    except Exception as e:
        error_category = classify_webui_error(e)
        logger.error(f"Web UI test failed for {camera_name}: {e}")
        return _error_response(
            error_category,
            f"Web UI test failed: {e}",
            status_code=503,
            extra={"screenshot_path": str(screenshot_path) if screenshot_path else None},
        )
    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass


@router.get("/api/test-ptz/{camera_name}/")
def api_test_ptz(
    camera_name: str,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Test PTZ API (status, movement, presets) and restore original position."""
    validation = _validate_camera_for_webui_ptz(camera_name)
    if isinstance(validation, JSONResponse):
        return validation
    camera, camera_ip, username, password = validation

    ptz = HikvisionISAPIClient(camera_ip, username, password)

    response_data: dict = {
        "camera_name": camera_name,
        "camera_ip": camera_ip,
        "initial_position": None,
        "position_restored": False,
        "tests": {},
    }
    initial_status: dict | None = None

    try:
        # Test 1: Get status
        status_result: dict = {
            "success": False,
            "response_time_ms": None,
            "data": None,
            "error": None,
        }
        try:
            start_time = time.time()
            initial_status = _ptz_state_to_status_dict(ptz.get_ptz_status())
            response_time = (time.time() - start_time) * 1000
            status_result["response_time_ms"] = round(response_time, 2)

            status_result["success"] = True
            status_result["data"] = {
                "pan": initial_status["pan"],
                "tilt": initial_status["tilt"],
                "zoom": initial_status["zoom"],
            }
            response_data["initial_position"] = status_result["data"].copy()
        except HikvisionTransportError as e:
            status_result["error"] = str(e)
            status_result["error_category"] = classify_ptz_error(e).value
        except Exception as e:
            status_result["error"] = str(e)
            status_result["error_category"] = classify_ptz_error(e).value

        response_data["tests"]["status"] = status_result

        if not status_result["success"]:
            return _error_response(
                CameraErrorCategory.API_ERROR,
                "Failed to get initial PTZ status",
                status_code=503,
                extra={"data": response_data},
            )

        # Test 2: Movement tests
        movement_tests = [
            ("pan_left", -PTZ_MOVEMENT_SPEED, 0, 0),
            ("pan_right", PTZ_MOVEMENT_SPEED, 0, 0),
            ("tilt_up", 0, PTZ_MOVEMENT_SPEED, 0),
            ("tilt_down", 0, -PTZ_MOVEMENT_SPEED, 0),
            ("zoom_in", 0, 0, PTZ_MOVEMENT_SPEED),
            ("zoom_out", 0, 0, -PTZ_MOVEMENT_SPEED),
        ]
        for test_name, pan, tilt, zoom in movement_tests:
            response_data["tests"][test_name] = execute_movement_test(
                ptz, pan=pan, tilt=tilt, zoom=zoom, duration=PTZ_MOVEMENT_DURATION
            )

        # Test 3: Presets
        response_data["tests"]["presets"] = get_presets_list(camera_ip, username, password)

    finally:
        if initial_status:
            try:
                ptz.move_absolute(
                    pan=initial_status["pan"],
                    tilt=initial_status["tilt"],
                    zoom=initial_status["zoom"],
                )
                wait_for_stabilization(ptz)
                response_data["position_restored"] = True
            except Exception as e:
                logger.warning(f"Failed to restore PTZ position for {camera_name}: {e}")
                response_data["position_restored"] = False
                response_data["restore_error"] = str(e)

    return _success_response(response_data)


# =============================================================================
# Diagnostic session endpoints
# =============================================================================


@router.post("/api/diagnostic/run/")
def api_run_diagnostic(
    body: RunDiagnosticRequest,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Run diagnostic tests on cameras for a tenant and save results."""
    tenant = get_tenant_by_id(body.tenant_id)
    if not tenant:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Tenant not found: {body.tenant_id}",
            status_code=404,
        )

    if body.camera_ids:
        cameras = [get_camera_by_id(cid) for cid in body.camera_ids]
        cameras = [c for c in cameras if c is not None]
    else:
        cameras = get_cameras_for_tenant(body.tenant_id)

    if not cameras:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            "No cameras found for tenant",
            status_code=404,
        )

    username, password = get_tenant_credentials(body.tenant_id)
    if not username or not password:
        return _error_response(
            CameraErrorCategory.CREDENTIALS_NOT_SET,
            f"Camera credentials not set for tenant '{body.tenant_id}'. "
            f"Set {body.tenant_id.upper()}_CAMERA_USERNAME and "
            f"{body.tenant_id.upper()}_CAMERA_PASSWORD, "
            "or global CAMERA_USERNAME/CAMERA_PASSWORD as fallback.",
        )

    session = DiagnosticSession(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        status=DiagnosticSessionStatus.RUNNING,
        tenant_id=body.tenant_id,
        tenant_name=tenant.get("name", body.tenant_id),
    )

    for camera in cameras:
        camera_id = camera.get("id")
        camera_name = camera.get("name", camera_id)
        camera_ip = camera.get("ip")

        camera_result = CameraDiagnosticResult(
            camera_id=camera_id,
            camera_name=camera_name,
            camera_ip=camera_ip or "",
        )

        if camera_ip:
            try:
                ptz_camera = create_ptz_camera(camera_ip, camera_name, tenant_id=body.tenant_id)
                camera_result.device_info = ptz_camera.get_device_info()
            except Exception as e:
                logger.warning(f"Could not get device info for {camera_name}: {e}")

        camera_result.rtsp_test = _run_rtsp_test(camera_id)
        camera_result.webui_test = _run_webui_test(camera_name, camera_ip, username, password)
        camera_result.ptz_test = _run_ptz_test(camera_name, camera_ip, username, password)
        session.camera_results.append(camera_result)

    session.status = DiagnosticSessionStatus.COMPLETED
    session.completed_at = datetime.now(timezone.utc)
    save_diagnostic_session(session)

    return _success_response(
        {
            "session_id": session.id,
            "status": session.status.value,
            "summary": session.get_summary(),
            "camera_results": [r.to_dict() for r in session.camera_results],
        }
    )


@router.post("/api/diagnostic/save/")
def api_save_diagnostic_session(
    body: SaveDiagnosticRequest,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Save a diagnostic session with pre-collected results (client-orchestrated)."""
    tenant = get_tenant_by_id(body.tenant_id)
    if not tenant:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Tenant not found: {body.tenant_id}",
            status_code=404,
        )

    if not body.camera_results:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            "camera_results is required and must not be empty",
            status_code=400,
        )

    session = DiagnosticSession(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        status=DiagnosticSessionStatus.COMPLETED,
        tenant_id=body.tenant_id,
        tenant_name=tenant.get("name", body.tenant_id),
        origin="client_orchestrated",
    )

    for cam_data in body.camera_results:
        camera_result = CameraDiagnosticResult(
            camera_id=cam_data.camera_id,
            camera_name=cam_data.camera_name,
            camera_ip=cam_data.camera_ip,
        )

        if cam_data.device_info:
            from camera_diagnostic.models import DeviceInfo

            camera_result.device_info = DeviceInfo.from_dict(cam_data.device_info)

        for test_key in ("rtsp_test", "webui_test", "ptz_test"):
            test_data: CameraResultIn | None = getattr(cam_data, test_key, None)
            if test_data is not None and isinstance(test_data, dict | CameraResultIn):
                raw = test_data.model_dump() if hasattr(test_data, "model_dump") else test_data
                try:
                    status = DiagnosticTestStatus(raw.get("status", "pending"))
                except ValueError:
                    status = DiagnosticTestStatus.PENDING
                test_result = DiagnosticTestResult(
                    test_type=test_key.replace("_test", ""),
                    status=status,
                    response_time_ms=raw.get("response_time_ms"),
                    error_message=raw.get("error_message"),
                    error_category=raw.get("error_category"),
                    details=raw.get("details", {}),
                )
                setattr(camera_result, test_key, test_result)

        session.camera_results.append(camera_result)

    save_diagnostic_session(session)

    return _success_response(
        {
            "session_id": session.id,
            "status": session.status.value,
            "summary": session.get_summary(),
        }
    )


@router.get("/api/diagnostic/sessions/")
def api_list_sessions(
    tenant_id: str = Query(None),
    limit: int = Query(50),
    offset: int = Query(0),
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """List diagnostic sessions (paginated)."""
    sessions, total = list_diagnostic_sessions(
        tenant_id=tenant_id,
        limit=limit,
        offset=offset,
    )
    return _success_response(
        {
            "sessions": [
                {
                    "id": s.id,
                    "created_at": s.created_at.isoformat(),
                    "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                    "status": s.status.value,
                    "tenant_id": s.tenant_id,
                    "tenant_name": s.tenant_name,
                    "summary": s.get_summary(),
                }
                for s in sessions
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


@router.get("/api/diagnostic/sessions/{session_id}/")
def api_get_session(
    session_id: str,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Get full diagnostic session detail."""
    session = load_diagnostic_session(session_id)
    if not session:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Session not found: {session_id}",
            status_code=404,
        )
    return _success_response(session.to_dict())


@router.post("/api/diagnostic/sessions/{session_id}/delete/")
def api_delete_session(
    session_id: str,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Delete a diagnostic session."""
    success, error = delete_diagnostic_session(session_id)
    if not success:
        return _error_response(
            CameraErrorCategory.API_ERROR,
            error or "Failed to delete session",
            status_code=404 if "not found" in (error or "").lower() else 500,
        )
    return _success_response({"message": f"Session {session_id} deleted"})


# =============================================================================
# Stress test endpoints
# =============================================================================


@router.get("/api/stress-test/presets/")
def api_stress_test_presets(
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """List available stress test presets."""
    return _success_response({"presets": [preset.to_dict() for preset in STRESS_TEST_PRESETS]})


@router.post("/api/stress-test/start/")
def api_stress_test_start(
    body: StressTestStartRequest,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Start a new stress test session."""
    if not body.tenant_id or not body.camera_id:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            "tenant_id and camera_id are required",
            status_code=400,
        )

    if body.preset_name:
        preset = next(
            (p for p in STRESS_TEST_PRESETS if p.name == body.preset_name),
            None,
        )
        if not preset:
            return _error_response(
                CameraErrorCategory.INVALID_RESPONSE,
                f"Preset not found: {body.preset_name}",
                status_code=404,
            )
        config = StressTestConfig(
            tenant_id=body.tenant_id,
            camera_id=body.camera_id,
            test_type=preset.test_type,
            pan_config=preset.pan_config,
            tilt_config=preset.tilt_config,
            zoom_config=preset.zoom_config,
            repetitions=preset.repetitions,
            max_speed=preset.max_speed,
        )
    else:
        if not body.test_type:
            return _error_response(
                CameraErrorCategory.INVALID_RESPONSE,
                "preset_name or test_type is required",
                status_code=400,
            )
        try:
            test_type = StressTestType(body.test_type)
        except ValueError:
            return _error_response(
                CameraErrorCategory.INVALID_RESPONSE,
                f"Invalid test_type: {body.test_type}",
                status_code=400,
            )

        pan_config = AxisMovementConfig.from_dict(body.pan_config) if body.pan_config else None
        tilt_config = AxisMovementConfig.from_dict(body.tilt_config) if body.tilt_config else None
        zoom_config = AxisMovementConfig.from_dict(body.zoom_config) if body.zoom_config else None

        config = StressTestConfig(
            tenant_id=body.tenant_id,
            camera_id=body.camera_id,
            test_type=test_type,
            pan_config=pan_config,
            tilt_config=tilt_config,
            zoom_config=zoom_config,
            repetitions=body.repetitions,
            max_speed=body.max_speed,
        )

    session_id, error = CameraStressTestService.start_stress_test(config)
    if error:
        return _error_response(CameraErrorCategory.API_ERROR, error, status_code=500)

    return _success_response({"session_id": session_id, "message": "Stress test started"})


@router.get("/api/stress-test/status/{session_id}/")
def api_stress_test_status(
    session_id: str,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Get current status/progress of a stress test."""
    progress = CameraStressTestService.get_stress_test_status(session_id)
    if not progress:
        session = CameraStressTestService.get_session(session_id)
        if session:
            return _success_response(
                {
                    "session_id": session.id,
                    "status": session.status.value,
                    "current_repetition": session.config.repetitions if session.config else 0,
                    "total_repetitions": session.config.repetitions if session.config else 0,
                    "message": "Test completed"
                    if session.status.value == "completed"
                    else session.status.value,
                }
            )
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Session not found: {session_id}",
            status_code=404,
        )
    return _success_response(progress.to_dict())


@router.post("/api/stress-test/abort/{session_id}/")
def api_stress_test_abort(
    session_id: str,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Abort a running stress test."""
    success, error = CameraStressTestService.abort_stress_test(session_id)
    if not success:
        return _error_response(
            CameraErrorCategory.API_ERROR,
            error or "Failed to abort test",
            status_code=404 if "not found" in (error or "").lower() else 400,
        )
    return _success_response({"message": f"Stress test {session_id} abort requested"})


@router.get("/api/stress-test/sessions/")
def api_stress_test_sessions(
    tenant_id: str = Query(None),
    camera_id: str = Query(None),
    limit: int = Query(50),
    offset: int = Query(0),
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """List stress test sessions (paginated)."""
    sessions, total = CameraStressTestService.list_stress_test_sessions(
        tenant_id=tenant_id,
        camera_id=camera_id,
        limit=limit,
        offset=offset,
    )
    return _success_response(
        {
            "sessions": [
                {
                    "id": s.id,
                    "created_at": s.created_at.isoformat(),
                    "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                    "status": s.status.value,
                    "tenant_id": s.tenant_id,
                    "camera_id": s.camera_id,
                    "camera_name": s.camera_name,
                    "test_type": s.config.test_type.value if s.config else None,
                    "user_evaluation": s.user_evaluation.value,
                    "result_success": s.result.success if s.result else None,
                }
                for s in sessions
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


@router.get("/api/stress-test/sessions/{session_id}/")
def api_stress_test_session_detail(
    session_id: str,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Get detailed stress test session info."""
    session = CameraStressTestService.get_session(session_id)
    if not session:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Session not found: {session_id}",
            status_code=404,
        )
    return _success_response(session.to_dict())


@router.post("/api/stress-test/sessions/{session_id}/evaluate/")
def api_stress_test_evaluate(
    session_id: str,
    body: StressTestEvaluateRequest,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Submit user evaluation for a stress test session."""
    try:
        evaluation = UserEvaluation(body.evaluation)
    except ValueError:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            f"Invalid evaluation value: {body.evaluation}",
            status_code=400,
        )

    success, error = CameraStressTestService.update_user_evaluation(
        session_id, evaluation, body.notes
    )
    if not success:
        return _error_response(
            CameraErrorCategory.API_ERROR,
            error or "Failed to update evaluation",
            status_code=404 if "not found" in (error or "").lower() else 500,
        )

    return _success_response(
        {
            "message": "Evaluation saved",
            "session_id": session_id,
            "evaluation": evaluation.value,
        }
    )


@router.post("/api/stress-test/sessions/{session_id}/delete/")
def api_stress_test_delete(
    session_id: str,
    user: UserModel = Depends(get_current_user),
) -> JSONResponse:
    """Delete a stress test session."""
    success, error = CameraStressTestService.delete_stress_test_session(session_id)
    if not success:
        return _error_response(
            CameraErrorCategory.API_ERROR,
            error or "Failed to delete session",
            status_code=404 if "not found" in (error or "").lower() else 500,
        )
    return _success_response({"message": f"Stress test session {session_id} deleted"})


@router.get("/api/stress-test/video-stream/{camera_id}/", response_model=None)
def api_stress_video_stream(
    camera_id: str,
    user: UserModel = Depends(get_current_user),
) -> StreamingResponse | JSONResponse:
    """Stream MJPEG video from a camera during stress testing (by camera ID)."""
    camera = get_camera_by_id(camera_id)
    if not camera:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Camera not found: {camera_id}",
            status_code=404,
        )

    return StreamingResponse(
        generate_mjpeg_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
