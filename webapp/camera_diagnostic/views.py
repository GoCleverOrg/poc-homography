"""Views for the camera diagnostic Django app.

This module contains thin HTTP request handlers that delegate business logic to services.py.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone

import cv2
import requests
from camera_survey.ptz import create_ptz_camera
from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ

from poc_homography.camera_config import (
    get_camera_by_id,
    get_camera_by_name,
    get_camera_configs,
    get_cameras_for_tenant,
    get_rtsp_url,
    get_tenant_by_id,
    get_tenant_credentials,
)

from .models import (
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
from .services import (
    PTZ_MOVEMENT_DURATION,
    PTZ_MOVEMENT_SPEED,
    WEBUI_CONNECTION_TIMEOUT_SEC,
    CameraErrorCategory,
    CameraStressTestService,
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

logger = logging.getLogger(__name__)


def _validate_camera_for_rtsp(camera_name: str) -> tuple[dict, str] | JsonResponse:
    """Validate camera exists and get RTSP URL.

    Args:
        camera_name: Name of the camera to validate

    Returns:
        Tuple of (camera_dict, rtsp_url) if valid, or JsonResponse with error
    """
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


def _validate_camera_for_webui_ptz(camera_name: str) -> tuple[dict, str, str, str] | JsonResponse:
    """Validate camera exists, has IP, and credentials are set.

    Args:
        camera_name: Name of the camera to validate

    Returns:
        Tuple of (camera_dict, camera_ip, username, password) if valid, or JsonResponse with error
    """
    camera = get_camera_by_name(camera_name)
    if not camera:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Camera not found: {camera_name}",
            status_code=404,
        )

    camera_ip = camera.ip_address
    if not camera_ip:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            f"No IP address configured for camera: {camera_name}",
        )

    # Get tenant-specific credentials (falls back to global)
    tenant_id = camera.tenant_id
    username, password = get_tenant_credentials(tenant_id)

    if not username or not password:
        return _error_response(
            CameraErrorCategory.CREDENTIALS_NOT_SET,
            f"Camera credentials not set for tenant '{tenant_id}'. "
            f"Set {tenant_id.upper()}_CAMERA_USERNAME and {tenant_id.upper()}_CAMERA_PASSWORD, "
            "or global CAMERA_USERNAME/CAMERA_PASSWORD as fallback.",
        )

    return (camera, camera_ip, username, password)


def _success_response(data: dict) -> JsonResponse:
    """Create a standardized success response.

    Args:
        data: The data to include in the response

    Returns:
        JsonResponse with status="success" and data
    """
    return JsonResponse({"status": "success", "data": data})


def _error_response(
    error_category: CameraErrorCategory,
    message: str,
    status_code: int = 500,
    extra: dict | None = None,
) -> JsonResponse:
    """Create a standardized error response.

    Args:
        error_category: The error category enum value
        message: Human-readable error message
        status_code: HTTP status code
        extra: Additional fields to include in the response

    Returns:
        JsonResponse with status="error", error_category, and message
    """
    response_data = {
        "status": "error",
        "error_category": error_category.value,
        "message": message,
    }
    if extra:
        response_data.update(extra)
    return JsonResponse(response_data, status=status_code)


@require_GET
def index(request: HttpRequest) -> HttpResponse:
    """Serve the main camera diagnostic HTML page."""
    from homography_web.frame_utils import get_tenant_id

    tenant_id = get_tenant_id(request)
    tenant = get_tenant_by_id(tenant_id)
    return render(
        request,
        "camera_diagnostic/index.html",
        {
            "tenant_id": tenant_id,
            "tenant_name": tenant.get("name", tenant_id) if tenant else tenant_id,
        },
    )


@require_GET
def api_tenants(request: HttpRequest) -> JsonResponse:
    """Get list of available tenants.

    Returns:
        JSON response with tenant list or error.
    """
    try:
        from homography_web.frame_utils import get_tenant_repo

        tenant_entities = get_tenant_repo().get_all()
        tenant_list = [
            {"id": t.id, "name": t.name, "description": t.description} for t in tenant_entities
        ]
        return _success_response({"tenants": tenant_list})
    except Exception as e:
        return _error_response(CameraErrorCategory.API_ERROR, f"Failed to load tenants: {e}")


@require_GET
def api_cameras(request: HttpRequest) -> JsonResponse:
    """Get list of available cameras, optionally filtered by tenant.

    Query params:
        tenant_id: Optional tenant ID to filter cameras

    Returns:
        JSON response with camera list or error.
    """
    try:
        tenant_id = request.GET.get("tenant_id")

        if tenant_id:
            cameras = get_cameras_for_tenant(tenant_id)
        else:
            cameras = get_camera_configs()

        # Return camera info including id, name, and ip
        camera_list = [
            {
                "id": cam.id,
                "name": cam.name,
                "ip": cam.ip_address,
                "tenant_id": cam.tenant_id,
            }
            for cam in cameras
        ]
        return _success_response({"cameras": camera_list})
    except Exception as e:
        return _error_response(CameraErrorCategory.API_ERROR, f"Failed to load cameras: {e}")


@require_GET
def api_video_stream(
    request: HttpRequest, camera_name: str
) -> StreamingHttpResponse | JsonResponse:
    """Stream MJPEG video from a camera's RTSP feed.

    Args:
        request: HTTP request
        camera_name: Name of the camera to stream

    Returns:
        StreamingHttpResponse with MJPEG content type, or JsonResponse with error
    """
    validation_result = _validate_camera_for_rtsp(camera_name)
    if isinstance(validation_result, JsonResponse):
        return validation_result
    camera, rtsp_url = validation_result

    response = StreamingHttpResponse(
        generate_mjpeg_frames(camera_name), content_type="multipart/x-mixed-replace; boundary=frame"
    )
    # Disable buffering for real-time streaming
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response


@require_GET
def api_test_rtsp(request: HttpRequest, camera_name: str) -> JsonResponse:
    """Test RTSP connectivity and return metrics.

    Args:
        request: HTTP request
        camera_name: Name of the camera to test

    Returns:
        JsonResponse with connection status and metrics
    """
    validation_result = _validate_camera_for_rtsp(camera_name)
    if isinstance(validation_result, JsonResponse):
        return validation_result
    camera, rtsp_url = validation_result

    # Measure connection time
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

        # Calculate connection latency
        connection_time = time.time()
        latency_ms = (connection_time - start_time) * 1000

        # Get stream properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Try to read a frame to verify stream is working
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


@require_GET
def api_capture_snapshot(request: HttpRequest, camera_name: str) -> HttpResponse | JsonResponse:
    """Capture a single frame from the camera and return as JPEG.

    Args:
        request: HTTP request
        camera_name: Name of the camera to capture from

    Returns:
        HttpResponse with JPEG image, or JsonResponse with error
    """
    validation_result = _validate_camera_for_rtsp(camera_name)
    if isinstance(validation_result, JsonResponse):
        return validation_result
    camera, rtsp_url = validation_result

    cap = None
    try:
        cap = create_rtsp_capture(rtsp_url)

        if not cap.isOpened():
            return _error_response(
                CameraErrorCategory.CAMERA_OFFLINE,
                f"Failed to connect to RTSP stream for {camera_name}",
                status_code=503,
            )

        # Read a frame
        ret, frame = cap.read()
        if not ret:
            return _error_response(
                CameraErrorCategory.INVALID_RESPONSE,
                f"Failed to capture frame from {camera_name}",
                status_code=503,
            )

        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            return _error_response(
                CameraErrorCategory.INVALID_RESPONSE,
                "Failed to encode frame as JPEG",
            )

        sanitized_name = _sanitize_camera_name(camera_name)
        response = HttpResponse(buffer.tobytes(), content_type="image/jpeg")
        response["Content-Disposition"] = f'inline; filename="{sanitized_name}_snapshot.jpg"'
        return response

    except Exception as e:
        error_category = classify_rtsp_error(e, rtsp_url)
        logger.error(f"Snapshot capture failed for {camera_name}: {e}")

        return _error_response(error_category, f"Snapshot capture failed: {e}", status_code=503)

    finally:
        if cap is not None:
            cap.release()


@require_GET
def api_test_webui(request: HttpRequest, camera_name: str) -> JsonResponse:
    """Test camera web interface using Playwright.

    Navigates to the camera's web interface, attempts login, and detects PTZ controls.

    Args:
        request: HTTP request
        camera_name: Name of the camera to test

    Returns:
        JsonResponse with web UI test results
    """
    validation_result = _validate_camera_for_webui_ptz(camera_name)
    if isinstance(validation_result, JsonResponse):
        return validation_result
    camera, camera_ip, username, password = validation_result

    # Import Playwright here to avoid import errors if not installed
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
            # Launch browser - headless mode configurable via settings
            # Non-headless mode often required for Hikvision camera web interfaces
            headless = getattr(settings, "CAMERA_DIAGNOSTIC_BROWSER_HEADLESS", False)
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(
                ignore_https_errors=True,  # Many cameras use self-signed certs
                viewport={"width": 1280, "height": 720},
            )
            page = context.new_page()

            # Set timeout for page operations
            page.set_default_timeout(WEBUI_CONNECTION_TIMEOUT_SEC * 1000)

            # Navigate to camera web interface
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

            # Attempt login
            login_attempted = attempt_login(page, username, password)

            # Wait for page to settle after login attempt
            if login_attempted:
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except PlaywrightTimeout:
                    pass  # Continue even if page doesn't settle

                # Handle Hikvision "low resources" warning dialog
                # This dialog appears after login and needs to be dismissed
                try:
                    from .services import dismiss_hikvision_warning_dialog

                    dismiss_hikvision_warning_dialog(page)
                except Exception as e:
                    logger.debug(f"Warning dialog handling: {e}")

            # Check login success and capture any error message
            login_success = check_login_success(page) if login_attempted else False
            login_error = None
            if login_attempted and not login_success:
                from .services import get_login_error_message

                login_error = get_login_error_message(page)

            # Detect PTZ controls
            ptz_controls = detect_ptz_controls(page)

            # If login succeeded, try to interact with PTZ controls
            ptz_test_result = None
            if login_success and ptz_controls:
                try:
                    from .services import test_webui_ptz_controls

                    ptz_test_result = test_webui_ptz_controls(page)
                except Exception as e:
                    logger.debug(f"PTZ control test: {e}")

            # Capture screenshot
            screenshot_path = get_screenshot_path(camera_name)
            page.screenshot(path=str(screenshot_path), full_page=True)

            # Close browser
            browser.close()
            browser = None

            response_data = {
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


@require_GET
def api_test_ptz(request: HttpRequest, camera_name: str) -> JsonResponse:
    """Test PTZ API functionality including status, movement, and presets.

    Performs comprehensive PTZ testing:
    1. Gets initial camera position
    2. Tests status endpoint
    3. Tests movement commands (pan left/right, tilt up/down, zoom in/out)
    4. Retrieves presets list
    5. Restores original camera position

    Args:
        request: HTTP request
        camera_name: Name of the camera to test

    Returns:
        JsonResponse with PTZ test results
    """
    validation_result = _validate_camera_for_webui_ptz(camera_name)
    if isinstance(validation_result, JsonResponse):
        return validation_result
    camera, camera_ip, username, password = validation_result

    # Initialize PTZ controller
    ptz = HikvisionPTZ(ip=camera_ip, username=username, password=password, name=camera_name)

    response_data = {
        "camera_name": camera_name,
        "camera_ip": camera_ip,
        "initial_position": None,
        "position_restored": False,
        "tests": {},
    }

    initial_status = None

    try:
        # Test 1: Get status endpoint
        status_result = {
            "success": False,
            "response_time_ms": None,
            "data": None,
            "error": None,
        }

        try:
            start_time = time.time()
            initial_status = ptz.get_status()
            response_time = (time.time() - start_time) * 1000
            status_result["response_time_ms"] = round(response_time, 2)

            if initial_status:
                status_result["success"] = True
                status_result["data"] = {
                    "pan": initial_status.get("pan"),
                    "tilt": initial_status.get("tilt"),
                    "zoom": initial_status.get("zoom"),
                }
                response_data["initial_position"] = status_result["data"].copy()
            else:
                status_result["error"] = "Failed to get PTZ status"

        except requests.exceptions.Timeout:
            status_result["error"] = "Network timeout"
            status_result["error_category"] = CameraErrorCategory.TIMEOUT.value
        except Exception as e:
            status_result["error"] = str(e)
            status_result["error_category"] = classify_ptz_error(e).value

        response_data["tests"]["status"] = status_result

        # If status failed, return early
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

        # Test 3: Get presets list
        response_data["tests"]["presets"] = get_presets_list(camera_ip, username, password)

    finally:
        # Always try to restore original position if we got initial status
        if initial_status:
            try:
                restore_success = ptz.send_ptz_return(initial_status)
                if restore_success:
                    # Wait for camera to reach restored position
                    wait_for_stabilization(ptz)
                    response_data["position_restored"] = True
            except Exception as e:
                logger.warning(f"Failed to restore PTZ position for {camera_name}: {e}")
                response_data["position_restored"] = False
                response_data["restore_error"] = str(e)

    return _success_response(response_data)


# =============================================================================
# Diagnostic Session API Endpoints
# =============================================================================


@csrf_exempt
@require_POST
def api_run_diagnostic(request: HttpRequest) -> JsonResponse:
    """Run diagnostic tests on all cameras for a tenant and save results.

    POST body:
        tenant_id: Tenant ID to run diagnostics for
        camera_ids: Optional list of specific camera IDs (if not provided, all cameras)

    Returns:
        JsonResponse with session info and results
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            "Invalid JSON request body",
            status_code=400,
        )

    tenant_id = data.get("tenant_id")
    if not tenant_id:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            "tenant_id is required",
            status_code=400,
        )

    tenant = get_tenant_by_id(tenant_id)
    if not tenant:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Tenant not found: {tenant_id}",
            status_code=404,
        )

    # Get cameras to test
    camera_ids = data.get("camera_ids")
    if camera_ids:
        cameras = [get_camera_by_id(cid) for cid in camera_ids]
        cameras = [c for c in cameras if c is not None]
    else:
        cameras = get_cameras_for_tenant(tenant_id)

    if not cameras:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            "No cameras found for tenant",
            status_code=404,
        )

    # Get tenant-specific credentials (falls back to global)
    username, password = get_tenant_credentials(tenant_id)

    if not username or not password:
        return _error_response(
            CameraErrorCategory.CREDENTIALS_NOT_SET,
            f"Camera credentials not set for tenant '{tenant_id}'. "
            f"Set {tenant_id.upper()}_CAMERA_USERNAME and {tenant_id.upper()}_CAMERA_PASSWORD, "
            "or global CAMERA_USERNAME/CAMERA_PASSWORD as fallback.",
        )

    # Create session
    session = DiagnosticSession(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        status=DiagnosticSessionStatus.RUNNING,
        tenant_id=tenant_id,
        tenant_name=tenant.get("name", tenant_id),
    )

    # Run tests for each camera
    for camera in cameras:
        camera_id = camera.id
        camera_name = camera.name
        camera_ip = camera.ip_address

        camera_result = CameraDiagnosticResult(
            camera_id=camera_id,
            camera_name=camera_name,
            camera_ip=camera_ip or "",
        )

        # Get device info if possible
        if camera_ip:
            try:
                ptz_camera = create_ptz_camera(camera_ip, camera_name, tenant_id=tenant_id)
                camera_result.device_info = ptz_camera.get_device_info()
            except Exception as e:
                logger.warning(f"Could not get device info for {camera_name}: {e}")

        # Run RTSP test
        camera_result.rtsp_test = _run_rtsp_test(camera_id)

        # Run WebUI test
        camera_result.webui_test = _run_webui_test(camera_name, camera_ip, username, password)

        # Run PTZ test
        camera_result.ptz_test = _run_ptz_test(camera_name, camera_ip, username, password)

        session.camera_results.append(camera_result)

    # Complete session
    session.status = DiagnosticSessionStatus.COMPLETED
    session.completed_at = datetime.now(timezone.utc)

    # Save session
    save_diagnostic_session(session)

    return _success_response(
        {
            "session_id": session.id,
            "status": session.status.value,
            "summary": session.get_summary(),
            "camera_results": [r.to_dict() for r in session.camera_results],
        }
    )


@csrf_exempt
@require_POST
def api_save_diagnostic_session(request: HttpRequest) -> JsonResponse:
    """Save a diagnostic session with pre-collected test results from frontend orchestration.

    POST body:
        tenant_id: Tenant ID
        camera_results: List of camera result objects, each containing:
            - camera_id: Camera identifier
            - camera_name: Camera display name
            - camera_ip: Camera IP address
            - device_info: Optional device info dict
            - rtsp_test: {status, response_time_ms, error_message, error_category, details}
            - webui_test: {status, response_time_ms, error_message, error_category, details}
            - ptz_test: {status, response_time_ms, error_message, error_category, details}

    Returns:
        JsonResponse with session_id and status
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            "Invalid JSON request body",
            status_code=400,
        )

    tenant_id = data.get("tenant_id")
    if not tenant_id:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            "tenant_id is required",
            status_code=400,
        )

    tenant = get_tenant_by_id(tenant_id)
    if not tenant:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Tenant not found: {tenant_id}",
            status_code=404,
        )

    camera_results_data = data.get("camera_results", [])
    if not camera_results_data:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            "camera_results is required and must not be empty",
            status_code=400,
        )

    # Build session from pre-collected results (client-orchestrated)
    session = DiagnosticSession(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        status=DiagnosticSessionStatus.COMPLETED,
        tenant_id=tenant_id,
        tenant_name=tenant.get("name", tenant_id),
        origin="client_orchestrated",
    )

    for cam_data in camera_results_data:
        camera_result = CameraDiagnosticResult(
            camera_id=cam_data.get("camera_id", ""),
            camera_name=cam_data.get("camera_name", ""),
            camera_ip=cam_data.get("camera_ip", ""),
        )

        if cam_data.get("device_info"):
            from .models import DeviceInfo

            camera_result.device_info = DeviceInfo.from_dict(cam_data["device_info"])

        for test_key in ("rtsp_test", "webui_test", "ptz_test"):
            test_data = cam_data.get(test_key)
            if test_data:
                try:
                    status = DiagnosticTestStatus(test_data.get("status", "pending"))
                except ValueError:
                    status = DiagnosticTestStatus.PENDING
                test_result = DiagnosticTestResult(
                    test_type=test_key.replace("_test", ""),
                    status=status,
                    response_time_ms=test_data.get("response_time_ms"),
                    error_message=test_data.get("error_message"),
                    error_category=test_data.get("error_category"),
                    details=test_data.get("details", {}),
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

            # Get stream properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Try to read a frame
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
            headless = getattr(settings, "CAMERA_DIAGNOSTIC_BROWSER_HEADLESS", False)
            browser = p.chromium.launch(headless=headless)
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

            # Attempt login
            login_attempted = attempt_login(page, username, password)

            if login_attempted:
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except PlaywrightTimeout:
                    pass

                from .services import dismiss_hikvision_warning_dialog

                try:
                    dismiss_hikvision_warning_dialog(page)
                except Exception:
                    pass

            login_success = check_login_success(page) if login_attempted else False

            # Detect PTZ controls
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
                from .services import get_login_error_message

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

    ptz = HikvisionPTZ(ip=camera_ip, username=username, password=password, name=camera_name)
    initial_status = None

    try:
        start_time = time.time()
        initial_status = ptz.get_status()
        result.response_time_ms = round((time.time() - start_time) * 1000, 2)

        if not initial_status:
            result.status = DiagnosticTestStatus.FAIL
            result.error_message = "Failed to get PTZ status"
            result.error_category = CameraErrorCategory.API_ERROR.value
            return result

        # Run movement tests
        tests_passed = 0
        tests_failed = 0
        movement_tests = [
            ("pan_left", -PTZ_MOVEMENT_SPEED, 0, 0),
            ("pan_right", PTZ_MOVEMENT_SPEED, 0, 0),
            ("tilt_up", 0, PTZ_MOVEMENT_SPEED, 0),
            ("tilt_down", 0, -PTZ_MOVEMENT_SPEED, 0),
        ]

        for test_name, pan, tilt, zoom in movement_tests:
            test_result = execute_movement_test(
                ptz, pan=pan, tilt=tilt, zoom=zoom, duration=PTZ_MOVEMENT_DURATION
            )
            if test_result.get("success"):
                tests_passed += 1
            else:
                tests_failed += 1

        # Get presets
        presets_result = get_presets_list(camera_ip, username, password)

        result.status = (
            DiagnosticTestStatus.PASS if tests_failed == 0 else DiagnosticTestStatus.FAIL
        )
        result.details = {
            "initial_position": {
                "pan": initial_status.get("pan"),
                "tilt": initial_status.get("tilt"),
                "zoom": initial_status.get("zoom"),
            },
            "movement_tests_passed": tests_passed,
            "movement_tests_failed": tests_failed,
            "presets_count": presets_result.get("count", 0),
        }

    except requests.exceptions.Timeout:
        result.status = DiagnosticTestStatus.FAIL
        result.error_message = "Network timeout"
        result.error_category = CameraErrorCategory.TIMEOUT.value
    except Exception as e:
        result.status = DiagnosticTestStatus.FAIL
        result.error_message = str(e)
        result.error_category = classify_ptz_error(e).value

    finally:
        # Restore original position
        if initial_status:
            try:
                ptz.send_ptz_return(initial_status)
                wait_for_stabilization(ptz)
            except Exception as e:
                logger.warning(f"Failed to restore PTZ position: {e}")

    return result


@require_GET
def api_list_sessions(request: HttpRequest) -> JsonResponse:
    """List diagnostic sessions.

    Query params:
        tenant_id: Optional tenant filter
        limit: Max results (default 50)
        offset: Pagination offset (default 0)

    Returns:
        JsonResponse with session list
    """
    tenant_id = request.GET.get("tenant_id")
    limit = int(request.GET.get("limit", 50))
    offset = int(request.GET.get("offset", 0))

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


@require_GET
def api_get_session(request: HttpRequest, session_id: str) -> JsonResponse:
    """Get diagnostic session details.

    Args:
        session_id: Session UUID

    Returns:
        JsonResponse with full session data
    """
    session = load_diagnostic_session(session_id)
    if not session:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Session not found: {session_id}",
            status_code=404,
        )

    return _success_response(session.to_dict())


@csrf_exempt
@require_POST
def api_delete_session(request: HttpRequest, session_id: str) -> JsonResponse:
    """Delete a diagnostic session.

    Args:
        session_id: Session UUID to delete

    Returns:
        JsonResponse with success/error status
    """
    success, error = delete_diagnostic_session(session_id)

    if not success:
        return _error_response(
            CameraErrorCategory.API_ERROR,
            error or "Failed to delete session",
            status_code=404 if "not found" in (error or "").lower() else 500,
        )

    return _success_response({"message": f"Session {session_id} deleted"})


# =============================================================================
# Stress Test API Endpoints
# =============================================================================


@require_GET
def api_stress_test_presets(request: HttpRequest) -> JsonResponse:
    """Get available stress test presets.

    Returns:
        JsonResponse with list of presets
    """
    return _success_response({"presets": [preset.to_dict() for preset in STRESS_TEST_PRESETS]})


@csrf_exempt
@require_POST
def api_stress_test_start(request: HttpRequest) -> JsonResponse:
    """Start a new stress test session.

    POST body:
        tenant_id: Tenant ID
        camera_id: Camera ID
        preset_name: Name of preset to use, OR
        test_type: Custom test type with axis configs

    Returns:
        JsonResponse with session_id
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            "Invalid JSON request body",
            status_code=400,
        )

    tenant_id = data.get("tenant_id")
    camera_id = data.get("camera_id")
    preset_name = data.get("preset_name")

    if not tenant_id or not camera_id:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            "tenant_id and camera_id are required",
            status_code=400,
        )

    # Build config from preset or custom settings
    if preset_name:
        # Find preset by name
        preset = next(
            (p for p in STRESS_TEST_PRESETS if p.name == preset_name),
            None,
        )
        if not preset:
            return _error_response(
                CameraErrorCategory.INVALID_RESPONSE,
                f"Preset not found: {preset_name}",
                status_code=404,
            )

        config = StressTestConfig(
            tenant_id=tenant_id,
            camera_id=camera_id,
            test_type=preset.test_type,
            pan_config=preset.pan_config,
            tilt_config=preset.tilt_config,
            zoom_config=preset.zoom_config,
            repetitions=preset.repetitions,
            max_speed=preset.max_speed,
        )
    else:
        # Custom config
        test_type_str = data.get("test_type")
        if not test_type_str:
            return _error_response(
                CameraErrorCategory.INVALID_RESPONSE,
                "preset_name or test_type is required",
                status_code=400,
            )

        try:
            test_type = StressTestType(test_type_str)
        except ValueError:
            return _error_response(
                CameraErrorCategory.INVALID_RESPONSE,
                f"Invalid test_type: {test_type_str}",
                status_code=400,
            )

        # Parse axis configs
        pan_config = None
        tilt_config = None
        zoom_config = None

        if data.get("pan_config"):
            pan_config = AxisMovementConfig.from_dict(data["pan_config"])
        if data.get("tilt_config"):
            tilt_config = AxisMovementConfig.from_dict(data["tilt_config"])
        if data.get("zoom_config"):
            zoom_config = AxisMovementConfig.from_dict(data["zoom_config"])

        config = StressTestConfig(
            tenant_id=tenant_id,
            camera_id=camera_id,
            test_type=test_type,
            pan_config=pan_config,
            tilt_config=tilt_config,
            zoom_config=zoom_config,
            repetitions=data.get("repetitions", 1),
            max_speed=data.get("max_speed", False),
        )

    # Start the stress test
    session_id, error = CameraStressTestService.start_stress_test(config)

    if error:
        return _error_response(
            CameraErrorCategory.API_ERROR,
            error,
            status_code=500,
        )

    return _success_response(
        {
            "session_id": session_id,
            "message": "Stress test started",
        }
    )


@require_GET
def api_stress_test_status(request: HttpRequest, session_id: str) -> JsonResponse:
    """Get current status/progress of a stress test.

    Args:
        session_id: Session UUID

    Returns:
        JsonResponse with progress info
    """
    progress = CameraStressTestService.get_stress_test_status(session_id)

    if not progress:
        # Try to get from completed session
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


@csrf_exempt
@require_POST
def api_stress_test_abort(request: HttpRequest, session_id: str) -> JsonResponse:
    """Abort a running stress test.

    Args:
        session_id: Session UUID to abort

    Returns:
        JsonResponse with success/error status
    """
    success, error = CameraStressTestService.abort_stress_test(session_id)

    if not success:
        return _error_response(
            CameraErrorCategory.API_ERROR,
            error or "Failed to abort test",
            status_code=404 if "not found" in (error or "").lower() else 400,
        )

    return _success_response({"message": f"Stress test {session_id} abort requested"})


@require_GET
def api_stress_test_sessions(request: HttpRequest) -> JsonResponse:
    """List stress test sessions.

    Query params:
        tenant_id: Optional tenant filter
        camera_id: Optional camera filter
        limit: Max results (default 50)
        offset: Pagination offset (default 0)

    Returns:
        JsonResponse with session list
    """
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


@require_GET
def api_stress_test_session_detail(request: HttpRequest, session_id: str) -> JsonResponse:
    """Get detailed stress test session info.

    Args:
        session_id: Session UUID

    Returns:
        JsonResponse with full session data
    """
    session = CameraStressTestService.get_session(session_id)
    if not session:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Session not found: {session_id}",
            status_code=404,
        )

    return _success_response(session.to_dict())


@csrf_exempt
@require_POST
def api_stress_test_evaluate(request: HttpRequest, session_id: str) -> JsonResponse:
    """Submit user evaluation for a stress test session.

    POST body:
        evaluation: "good" | "needs_improvement" | "bad"
        notes: Optional user notes

    Returns:
        JsonResponse with success/error status
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            "Invalid JSON request body",
            status_code=400,
        )

    evaluation_str = data.get("evaluation")
    notes = data.get("notes", "")

    if not evaluation_str:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            "evaluation is required",
            status_code=400,
        )

    try:
        evaluation = UserEvaluation(evaluation_str)
    except ValueError:
        return _error_response(
            CameraErrorCategory.INVALID_RESPONSE,
            f"Invalid evaluation value: {evaluation_str}",
            status_code=400,
        )

    success, error = CameraStressTestService.update_user_evaluation(session_id, evaluation, notes)

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


@csrf_exempt
@require_POST
def api_stress_test_delete(request: HttpRequest, session_id: str) -> JsonResponse:
    """Delete a stress test session.

    Args:
        session_id: Session UUID to delete

    Returns:
        JsonResponse with success/error status
    """
    success, error = CameraStressTestService.delete_stress_test_session(session_id)

    if not success:
        return _error_response(
            CameraErrorCategory.API_ERROR,
            error or "Failed to delete session",
            status_code=404 if "not found" in (error or "").lower() else 500,
        )

    return _success_response({"message": f"Stress test session {session_id} deleted"})


@require_GET
def api_stress_video_stream(
    request: HttpRequest, camera_id: str
) -> StreamingHttpResponse | JsonResponse:
    """Stream MJPEG video from a camera during stress testing.

    Args:
        request: HTTP request
        camera_id: Camera ID to stream

    Returns:
        StreamingHttpResponse with MJPEG content type
    """
    # For stress testing, we use camera_id directly (not camera_name)
    camera = get_camera_by_id(camera_id)
    if not camera:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Camera not found: {camera_id}",
            status_code=404,
        )

    response = StreamingHttpResponse(
        generate_mjpeg_frames(camera_id), content_type="multipart/x-mixed-replace; boundary=frame"
    )
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response
