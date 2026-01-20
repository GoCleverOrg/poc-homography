"""Views for the camera diagnostic Django app.

This module contains thin HTTP request handlers that delegate business logic to services.py.
"""

from __future__ import annotations

import logging
import os
import time

import cv2
import requests
from django.http import HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET
from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ

from poc_homography.camera_config import (
    get_camera_by_id,
    get_camera_by_name,
    get_camera_configs,
    get_cameras_for_tenant,
    get_rtsp_url,
    get_tenants,
)

from .services import (
    PTZ_MOVEMENT_DURATION,
    PTZ_MOVEMENT_SPEED,
    WEBUI_CONNECTION_TIMEOUT_SEC,
    CameraErrorCategory,
    _sanitize_camera_name,
    attempt_login,
    check_login_success,
    classify_ptz_error,
    classify_rtsp_error,
    classify_webui_error,
    create_rtsp_capture,
    detect_ptz_controls,
    execute_movement_test,
    generate_mjpeg_frames,
    get_presets_list,
    get_screenshot_path,
    wait_for_stabilization,
)

logger = logging.getLogger(__name__)


def _success_response(data: dict) -> JsonResponse:
    """Create a standardized success response.

    Args:
        data: The data to include in the response

    Returns:
        JsonResponse with status="success" and data
    """
    return JsonResponse({"status": "success", "data": data})


def _error_response(
    error_category: CameraErrorCategory, message: str, status_code: int = 500, extra: dict | None = None
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
    return render(request, "camera_diagnostic/index.html")


@require_GET
def api_tenants(request: HttpRequest) -> JsonResponse:
    """Get list of available tenants.

    Returns:
        JSON response with tenant list or error.
    """
    try:
        tenants = get_tenants()
        tenant_list = [
            {"id": t["id"], "name": t["name"], "description": t.get("description", "")}
            for t in tenants
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
            {"id": cam["id"], "name": cam["name"], "ip": cam["ip"], "tenant_id": cam.get("tenant_id")}
            for cam in cameras
        ]
        return _success_response({"cameras": camera_list})
    except Exception as e:
        return _error_response(CameraErrorCategory.API_ERROR, f"Failed to load cameras: {e}")


@require_GET
def api_video_stream(request: HttpRequest, camera_name: str) -> StreamingHttpResponse | JsonResponse:
    """Stream MJPEG video from a camera's RTSP feed.

    Args:
        request: HTTP request
        camera_name: Name of the camera to stream

    Returns:
        StreamingHttpResponse with MJPEG content type, or JsonResponse with error
    """
    # Validate camera exists
    camera = get_camera_by_name(camera_name)
    if not camera:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Camera not found: {camera_name}",
            status_code=404,
        )

    # Validate credentials are set
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
    # Validate camera exists
    camera = get_camera_by_name(camera_name)
    if not camera:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Camera not found: {camera_name}",
            status_code=404,
        )

    # Get RTSP URL
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

        return _success_response({
            "message": f"Successfully connected to {camera_name}",
            "metrics": {
                "fps": fps if fps > 0 else None,
                "resolution": {"width": width, "height": height},
                "latency_ms": round(latency_ms, 2),
            },
        })

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
    # Validate camera exists
    camera = get_camera_by_name(camera_name)
    if not camera:
        return _error_response(
            CameraErrorCategory.CAMERA_NOT_FOUND,
            f"Camera not found: {camera_name}",
            status_code=404,
        )

    # Get RTSP URL
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
    # Validate camera exists
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

    # Get credentials from environment variables
    username = os.getenv("CAMERA_USERNAME")
    password = os.getenv("CAMERA_PASSWORD")

    if not username or not password:
        return _error_response(
            CameraErrorCategory.CREDENTIALS_NOT_SET,
            "Camera credentials not set. Set CAMERA_USERNAME and CAMERA_PASSWORD environment variables.",
        )

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
            # Launch browser in visible mode for Hikvision cameras
            # Headless mode often fails with complex camera web interfaces
            browser = p.chromium.launch(headless=False)
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
    # Validate camera exists
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
            CameraErrorCategory.API_ERROR,
            f"No IP address configured for camera: {camera_name}",
        )

    # Get credentials from environment variables
    username = os.getenv("CAMERA_USERNAME")
    password = os.getenv("CAMERA_PASSWORD")

    if not username or not password:
        return _error_response(
            CameraErrorCategory.CREDENTIALS_NOT_SET,
            "Camera credentials not set. Set CAMERA_USERNAME and CAMERA_PASSWORD environment variables.",
        )

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
