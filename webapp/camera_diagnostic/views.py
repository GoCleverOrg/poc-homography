"""Views for the camera diagnostic Django app."""

from __future__ import annotations

import logging
import os
import socket
import time
import xml.etree.ElementTree as ET
from collections.abc import Generator
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import requests
from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET
from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ

from poc_homography.camera_config import get_camera_by_name, get_camera_configs, get_rtsp_url

logger = logging.getLogger(__name__)

# RTSP connection timeout in seconds
RTSP_CONNECTION_TIMEOUT_SEC = 10


class RTSPErrorCategory(Enum):
    """Error categories for RTSP connection failures."""

    NETWORK_UNREACHABLE = "NETWORK_UNREACHABLE"
    AUTH_FAILED = "AUTH_FAILED"
    TIMEOUT = "TIMEOUT"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    CAMERA_OFFLINE = "CAMERA_OFFLINE"
    CREDENTIALS_NOT_SET = "CREDENTIALS_NOT_SET"
    CAMERA_NOT_FOUND = "CAMERA_NOT_FOUND"


def _classify_rtsp_error(exception: Exception, rtsp_url: str | None = None) -> RTSPErrorCategory:
    """Classify an exception into an RTSP error category.

    Args:
        exception: The exception that occurred
        rtsp_url: The RTSP URL being accessed (for context)

    Returns:
        The appropriate RTSPErrorCategory
    """
    error_str = str(exception).lower()

    # Socket/network errors
    if isinstance(exception, (socket.error, OSError)):
        if "connection refused" in error_str or "no route to host" in error_str:
            return RTSPErrorCategory.NETWORK_UNREACHABLE
        if "timed out" in error_str or "timeout" in error_str:
            return RTSPErrorCategory.TIMEOUT

    # Credential errors
    if isinstance(exception, ValueError) and "credentials" in error_str:
        return RTSPErrorCategory.CREDENTIALS_NOT_SET

    # Generic timeout indicators
    if "timeout" in error_str or "timed out" in error_str:
        return RTSPErrorCategory.TIMEOUT

    # Authentication indicators
    if "401" in error_str or "403" in error_str or "unauthorized" in error_str or "auth" in error_str:
        return RTSPErrorCategory.AUTH_FAILED

    # Default to camera offline for OpenCV capture failures
    return RTSPErrorCategory.CAMERA_OFFLINE


def _create_rtsp_capture(rtsp_url: str) -> cv2.VideoCapture:
    """Create an OpenCV VideoCapture with RTSP-optimized settings.

    Args:
        rtsp_url: The RTSP URL to connect to

    Returns:
        Configured VideoCapture object
    """
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    # Set timeout for connection (in milliseconds)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_CONNECTION_TIMEOUT_SEC * 1000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_CONNECTION_TIMEOUT_SEC * 1000)

    # Use TCP for more reliable streaming
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

    return cap


def _generate_mjpeg_frames(camera_name: str) -> Generator[bytes, None, None]:
    """Generate MJPEG frames from an RTSP stream.

    Args:
        camera_name: Name of the camera to stream from

    Yields:
        MJPEG frame bytes with multipart boundary headers
    """
    try:
        rtsp_url = get_rtsp_url(camera_name)
    except ValueError as e:
        logger.error(f"Failed to get RTSP URL for {camera_name}: {e}")
        return

    if not rtsp_url:
        logger.error(f"Camera not found: {camera_name}")
        return

    cap = _create_rtsp_capture(rtsp_url)

    try:
        if not cap.isOpened():
            logger.error(f"Failed to open RTSP stream for {camera_name}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame from {camera_name}, ending stream")
                break

            # Encode frame as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            success, buffer = cv2.imencode(".jpg", frame, encode_params)
            if not success:
                logger.warning(f"Failed to encode frame from {camera_name}")
                continue

            # Yield MJPEG multipart frame
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    finally:
        cap.release()


def index(request: HttpRequest) -> HttpResponse:
    """Serve the main camera diagnostic HTML page."""
    return render(request, "camera_diagnostic/index.html")


@require_GET
def api_cameras(request: HttpRequest) -> JsonResponse:
    """Get list of available cameras with name and IP.

    Returns:
        JSON array of camera objects with 'name' and 'ip' fields.
    """
    try:
        cameras = get_camera_configs()
        # Return only name and ip fields for the dropdown
        camera_list = [{"name": cam["name"], "ip": cam["ip"]} for cam in cameras]
        return JsonResponse(camera_list, safe=False)
    except Exception as e:
        return JsonResponse({"error": f"Failed to load cameras: {e}"}, status=500)


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
        return JsonResponse(
            {"error": f"Camera not found: {camera_name}", "error_category": RTSPErrorCategory.CAMERA_NOT_FOUND.value},
            status=404,
        )

    # Validate credentials are set
    try:
        rtsp_url = get_rtsp_url(camera_name)
        if not rtsp_url:
            return JsonResponse(
                {
                    "error": f"Camera not found: {camera_name}",
                    "error_category": RTSPErrorCategory.CAMERA_NOT_FOUND.value,
                },
                status=404,
            )
    except ValueError as e:
        return JsonResponse(
            {"error": str(e), "error_category": RTSPErrorCategory.CREDENTIALS_NOT_SET.value}, status=500
        )

    response = StreamingHttpResponse(
        _generate_mjpeg_frames(camera_name), content_type="multipart/x-mixed-replace; boundary=frame"
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
        JsonResponse with connection status and metrics:
        - status: "success" or "error"
        - error_category: (only on error) one of RTSPErrorCategory values
        - message: Human-readable status message
        - metrics: (only on success)
            - fps: Frame rate
            - resolution: {"width": int, "height": int}
            - latency_ms: Connection latency in milliseconds
    """
    # Validate camera exists
    camera = get_camera_by_name(camera_name)
    if not camera:
        return JsonResponse(
            {
                "status": "error",
                "error_category": RTSPErrorCategory.CAMERA_NOT_FOUND.value,
                "message": f"Camera not found: {camera_name}",
            },
            status=404,
        )

    # Get RTSP URL
    try:
        rtsp_url = get_rtsp_url(camera_name)
        if not rtsp_url:
            return JsonResponse(
                {
                    "status": "error",
                    "error_category": RTSPErrorCategory.CAMERA_NOT_FOUND.value,
                    "message": f"Camera not found: {camera_name}",
                },
                status=404,
            )
    except ValueError as e:
        return JsonResponse(
            {
                "status": "error",
                "error_category": RTSPErrorCategory.CREDENTIALS_NOT_SET.value,
                "message": str(e),
            },
            status=500,
        )

    # Measure connection time
    start_time = time.time()
    cap = None

    try:
        cap = _create_rtsp_capture(rtsp_url)

        if not cap.isOpened():
            return JsonResponse(
                {
                    "status": "error",
                    "error_category": RTSPErrorCategory.CAMERA_OFFLINE.value,
                    "message": f"Failed to connect to RTSP stream for {camera_name}",
                },
                status=503,
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
            return JsonResponse(
                {
                    "status": "error",
                    "error_category": RTSPErrorCategory.INVALID_RESPONSE.value,
                    "message": f"Connected but failed to read frame from {camera_name}",
                },
                status=503,
            )

        return JsonResponse(
            {
                "status": "success",
                "message": f"Successfully connected to {camera_name}",
                "metrics": {
                    "fps": fps if fps > 0 else None,
                    "resolution": {"width": width, "height": height},
                    "latency_ms": round(latency_ms, 2),
                },
            }
        )

    except Exception as e:
        error_category = _classify_rtsp_error(e, rtsp_url)
        logger.error(f"RTSP test failed for {camera_name}: {e}")

        return JsonResponse(
            {
                "status": "error",
                "error_category": error_category.value,
                "message": f"RTSP connection failed: {e}",
            },
            status=503,
        )

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
        return JsonResponse(
            {
                "error": f"Camera not found: {camera_name}",
                "error_category": RTSPErrorCategory.CAMERA_NOT_FOUND.value,
            },
            status=404,
        )

    # Get RTSP URL
    try:
        rtsp_url = get_rtsp_url(camera_name)
        if not rtsp_url:
            return JsonResponse(
                {
                    "error": f"Camera not found: {camera_name}",
                    "error_category": RTSPErrorCategory.CAMERA_NOT_FOUND.value,
                },
                status=404,
            )
    except ValueError as e:
        return JsonResponse(
            {
                "error": str(e),
                "error_category": RTSPErrorCategory.CREDENTIALS_NOT_SET.value,
            },
            status=500,
        )

    cap = None
    try:
        cap = _create_rtsp_capture(rtsp_url)

        if not cap.isOpened():
            return JsonResponse(
                {
                    "error": f"Failed to connect to RTSP stream for {camera_name}",
                    "error_category": RTSPErrorCategory.CAMERA_OFFLINE.value,
                },
                status=503,
            )

        # Read a frame
        ret, frame = cap.read()
        if not ret:
            return JsonResponse(
                {
                    "error": f"Failed to capture frame from {camera_name}",
                    "error_category": RTSPErrorCategory.INVALID_RESPONSE.value,
                },
                status=503,
            )

        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            return JsonResponse(
                {
                    "error": "Failed to encode frame as JPEG",
                    "error_category": RTSPErrorCategory.INVALID_RESPONSE.value,
                },
                status=500,
            )

        response = HttpResponse(buffer.tobytes(), content_type="image/jpeg")
        response["Content-Disposition"] = f'inline; filename="{camera_name}_snapshot.jpg"'
        return response

    except Exception as e:
        error_category = _classify_rtsp_error(e, rtsp_url)
        logger.error(f"Snapshot capture failed for {camera_name}: {e}")

        return JsonResponse(
            {
                "error": f"Snapshot capture failed: {e}",
                "error_category": error_category.value,
            },
            status=503,
        )

    finally:
        if cap is not None:
            cap.release()


# Web UI test timeout in seconds
WEBUI_CONNECTION_TIMEOUT_SEC = 15

# Directory for storing screenshot artifacts
SCREENSHOTS_DIR = Path(settings.BASE_DIR) / "diagnostic_screenshots"


def _get_screenshot_path(camera_name: str) -> Path:
    """Generate a timestamped screenshot path for the camera.

    Args:
        camera_name: Name of the camera

    Returns:
        Path object for the screenshot file
    """
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCREENSHOTS_DIR / f"{camera_name}_webui_{timestamp}.png"


def _classify_webui_error(exception: Exception) -> RTSPErrorCategory:
    """Classify an exception into an error category for web UI testing.

    Args:
        exception: The exception that occurred

    Returns:
        The appropriate RTSPErrorCategory
    """
    error_str = str(exception).lower()

    # Network errors
    if "net::" in error_str or "connection refused" in error_str or "no route" in error_str:
        return RTSPErrorCategory.NETWORK_UNREACHABLE

    # Timeout errors
    if "timeout" in error_str or "timed out" in error_str:
        return RTSPErrorCategory.TIMEOUT

    # Authentication errors
    if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
        return RTSPErrorCategory.AUTH_FAILED

    # Default to invalid response for unexpected errors
    return RTSPErrorCategory.INVALID_RESPONSE


def _detect_ptz_controls(page) -> list[dict]:
    """Detect PTZ control elements on the page.

    Args:
        page: Playwright page object

    Returns:
        List of detected PTZ control elements with type and selector info
    """
    ptz_controls = []

    # Common PTZ-related selectors and patterns
    ptz_patterns = {
        "buttons": [
            # Direction controls
            'button:has-text("pan")',
            'button:has-text("tilt")',
            'button:has-text("zoom")',
            'button:has-text("left")',
            'button:has-text("right")',
            'button:has-text("up")',
            'button:has-text("down")',
            '[class*="ptz"]',
            '[class*="pan"]',
            '[class*="tilt"]',
            '[class*="zoom"]',
            '[id*="ptz"]',
            '[id*="pan"]',
            '[id*="tilt"]',
            '[id*="zoom"]',
            # Arrow/direction icons or buttons
            '[class*="arrow"]',
            '[class*="direction"]',
            '[aria-label*="pan"]',
            '[aria-label*="tilt"]',
            '[aria-label*="zoom"]',
        ],
        "sliders": [
            'input[type="range"]',
            '[class*="slider"]',
            '[role="slider"]',
            '[class*="zoom-slider"]',
        ],
        "inputs": [
            'input[name*="pan"]',
            'input[name*="tilt"]',
            'input[name*="zoom"]',
            'input[id*="pan"]',
            'input[id*="tilt"]',
            'input[id*="zoom"]',
            'input[type="number"][class*="ptz"]',
        ],
    }

    for control_type, selectors in ptz_patterns.items():
        for selector in selectors:
            try:
                elements = page.locator(selector).all()
                for element in elements:
                    try:
                        if element.is_visible():
                            ptz_controls.append({
                                "type": control_type,
                                "selector": selector,
                                "tag": element.evaluate("el => el.tagName.toLowerCase()"),
                            })
                    except Exception:
                        # Element may have become stale, skip it
                        pass
            except Exception:
                # Selector may not be valid for this page, skip it
                pass

    # Remove duplicates based on type and tag combination
    seen = set()
    unique_controls = []
    for control in ptz_controls:
        key = (control["type"], control.get("tag", ""))
        if key not in seen:
            seen.add(key)
            unique_controls.append(control)

    return unique_controls


def _attempt_login(page, username: str, password: str) -> bool:
    """Attempt to log in to the camera web interface.

    Tries multiple common login form patterns used by camera manufacturers.

    Args:
        page: Playwright page object
        username: Login username
        password: Login password

    Returns:
        True if login was attempted, False if no login form found
    """
    # Common username field selectors
    username_selectors = [
        "#username",
        "#user",
        "#userName",
        'input[name="username"]',
        'input[name="user"]',
        'input[name="userName"]',
        'input[type="text"][id*="user"]',
        'input[type="text"][name*="user"]',
        'input[placeholder*="user" i]',
        'input[placeholder*="name" i]',
    ]

    # Common password field selectors
    password_selectors = [
        "#password",
        "#pass",
        "#pwd",
        'input[name="password"]',
        'input[name="pass"]',
        'input[name="pwd"]',
        'input[type="password"]',
    ]

    # Common submit button selectors
    submit_selectors = [
        'button[type="submit"]',
        'input[type="submit"]',
        "#login",
        "#loginBtn",
        "#submit",
        'button:has-text("login")',
        'button:has-text("sign in")',
        'input[value*="login" i]',
        'input[value*="sign" i]',
    ]

    # Try to find and fill username field
    username_filled = False
    for selector in username_selectors:
        try:
            element = page.locator(selector).first
            if element.is_visible():
                element.fill(username)
                username_filled = True
                break
        except Exception:
            continue

    if not username_filled:
        return False

    # Try to find and fill password field
    password_filled = False
    for selector in password_selectors:
        try:
            element = page.locator(selector).first
            if element.is_visible():
                element.fill(password)
                password_filled = True
                break
        except Exception:
            continue

    if not password_filled:
        return False

    # Try to click submit button
    for selector in submit_selectors:
        try:
            element = page.locator(selector).first
            if element.is_visible():
                element.click()
                return True
        except Exception:
            continue

    # If no submit button found, try pressing Enter on password field
    try:
        for selector in password_selectors:
            element = page.locator(selector).first
            if element.is_visible():
                element.press("Enter")
                return True
    except Exception:
        pass

    return False


def _check_login_success(page) -> bool:
    """Check if login was successful.

    Args:
        page: Playwright page object

    Returns:
        True if login appears successful, False otherwise
    """
    # Check for absence of login form (common login success indicator)
    login_form_selectors = [
        "#loginForm",
        'form[action*="login"]',
        'form[id*="login"]',
        "#username",
        'input[name="username"]',
    ]

    for selector in login_form_selectors:
        try:
            if page.locator(selector).is_visible():
                return False
        except Exception:
            pass

    # Check for common error messages
    error_selectors = [
        '[class*="error"]',
        '[class*="alert"]',
        '[id*="error"]',
        ':text("invalid")',
        ':text("incorrect")',
        ':text("failed")',
    ]

    for selector in error_selectors:
        try:
            element = page.locator(selector).first
            if element.is_visible():
                text = element.text_content().lower()
                if any(
                    word in text
                    for word in ["invalid", "incorrect", "failed", "wrong", "error", "denied"]
                ):
                    return False
        except Exception:
            pass

    # Check for common post-login elements
    post_login_indicators = [
        '[class*="dashboard"]',
        '[class*="main"]',
        '[class*="menu"]',
        '[class*="nav"]',
        '[id*="dashboard"]',
        '[id*="main"]',
        "video",
        "canvas",
        '[class*="live"]',
        '[class*="stream"]',
    ]

    for selector in post_login_indicators:
        try:
            if page.locator(selector).first.is_visible():
                return True
        except Exception:
            pass

    # If we can't definitively tell, assume success if no login form is visible
    return True


@require_GET
def api_test_webui(request: HttpRequest, camera_name: str) -> JsonResponse:
    """Test camera web interface using Playwright.

    Navigates to the camera's web interface, attempts login, and detects PTZ controls.

    Args:
        request: HTTP request
        camera_name: Name of the camera to test

    Returns:
        JsonResponse with:
        - status: "success" or "error"
        - error_category: (only on error) one of RTSPErrorCategory values
        - message: Human-readable status message
        - login_success: (on success) boolean indicating if login succeeded
        - ptz_controls_found: (on success) list of detected PTZ control types
        - screenshot_path: (on success) path to captured screenshot
    """
    # Validate camera exists
    camera = get_camera_by_name(camera_name)
    if not camera:
        return JsonResponse(
            {
                "status": "error",
                "error_category": RTSPErrorCategory.CAMERA_NOT_FOUND.value,
                "message": f"Camera not found: {camera_name}",
            },
            status=404,
        )

    camera_ip = camera.get("ip")
    if not camera_ip:
        return JsonResponse(
            {
                "status": "error",
                "error_category": RTSPErrorCategory.INVALID_RESPONSE.value,
                "message": f"No IP address configured for camera: {camera_name}",
            },
            status=500,
        )

    # Get credentials from environment variables
    username = os.getenv("CAMERA_USERNAME")
    password = os.getenv("CAMERA_PASSWORD")

    if not username or not password:
        return JsonResponse(
            {
                "status": "error",
                "error_category": RTSPErrorCategory.CREDENTIALS_NOT_SET.value,
                "message": "Camera credentials not set. Set CAMERA_USERNAME and CAMERA_PASSWORD environment variables.",
            },
            status=500,
        )

    # Import Playwright here to avoid import errors if not installed
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        from playwright.sync_api import sync_playwright
    except ImportError:
        return JsonResponse(
            {
                "status": "error",
                "error_category": RTSPErrorCategory.INVALID_RESPONSE.value,
                "message": "Playwright is not installed. Install with: pip install playwright && playwright install chromium",
            },
            status=500,
        )

    browser = None
    screenshot_path = None

    try:
        with sync_playwright() as p:
            # Launch browser in headless mode
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                ignore_https_errors=True,  # Many cameras use self-signed certs
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
                return JsonResponse(
                    {
                        "status": "error",
                        "error_category": RTSPErrorCategory.TIMEOUT.value,
                        "message": f"Connection timeout: Could not reach camera web interface at http://{camera_ip}",
                    },
                    status=503,
                )

            # Attempt login
            login_attempted = _attempt_login(page, username, password)

            # Wait for page to settle after login attempt
            if login_attempted:
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except PlaywrightTimeout:
                    pass  # Continue even if page doesn't settle

            # Check login success
            login_success = _check_login_success(page) if login_attempted else False

            # Detect PTZ controls
            ptz_controls = _detect_ptz_controls(page)

            # Capture screenshot
            screenshot_path = _get_screenshot_path(camera_name)
            page.screenshot(path=str(screenshot_path), full_page=True)

            # Close browser
            browser.close()
            browser = None

            return JsonResponse(
                {
                    "status": "success",
                    "message": f"Web UI test completed for {camera_name}",
                    "login_success": login_success,
                    "login_attempted": login_attempted,
                    "ptz_controls_found": ptz_controls,
                    "screenshot_path": str(screenshot_path),
                }
            )

    except Exception as e:
        error_category = _classify_webui_error(e)
        logger.error(f"Web UI test failed for {camera_name}: {e}")

        return JsonResponse(
            {
                "status": "error",
                "error_category": error_category.value,
                "message": f"Web UI test failed: {e}",
                "screenshot_path": str(screenshot_path) if screenshot_path else None,
            },
            status=503,
        )

    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass


# PTZ API test constants
PTZ_MOVEMENT_SPEED = 30  # Speed for movement tests (-100 to 100)
PTZ_MOVEMENT_DURATION = 1.5  # Duration of each movement in seconds
PTZ_STABILIZATION_INTERVAL = 0.1  # Polling interval for stabilization check
PTZ_STABILIZATION_THRESHOLD = 0.5  # Seconds of no change to consider stabilized
PTZ_API_TIMEOUT = 10  # Network timeout for API calls in seconds


class PTZErrorCategory(Enum):
    """Error categories for PTZ API failures."""

    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    AUTH_FAILED = "AUTH_FAILED"
    INVALID_XML = "INVALID_XML"
    API_ERROR = "API_ERROR"
    CREDENTIALS_NOT_SET = "CREDENTIALS_NOT_SET"
    CAMERA_NOT_FOUND = "CAMERA_NOT_FOUND"


def _classify_ptz_error(exception: Exception) -> PTZErrorCategory:
    """Classify an exception into a PTZ error category.

    Args:
        exception: The exception that occurred

    Returns:
        The appropriate PTZErrorCategory
    """
    error_str = str(exception).lower()

    # Network timeout errors
    if isinstance(exception, requests.exceptions.Timeout):
        return PTZErrorCategory.NETWORK_TIMEOUT
    if "timeout" in error_str or "timed out" in error_str:
        return PTZErrorCategory.NETWORK_TIMEOUT

    # Authentication errors
    if isinstance(exception, requests.exceptions.HTTPError):
        if hasattr(exception, "response") and exception.response is not None:
            if exception.response.status_code == 401:
                return PTZErrorCategory.AUTH_FAILED

    if "401" in error_str or "unauthorized" in error_str or "authentication" in error_str:
        return PTZErrorCategory.AUTH_FAILED

    # XML parsing errors
    if isinstance(exception, ET.ParseError):
        return PTZErrorCategory.INVALID_XML
    if "xml" in error_str or "parse" in error_str:
        return PTZErrorCategory.INVALID_XML

    # Default to generic API error
    return PTZErrorCategory.API_ERROR


def _wait_for_stabilization(
    ptz: HikvisionPTZ,
    max_wait: float = 5.0,
    poll_interval: float = PTZ_STABILIZATION_INTERVAL,
    stable_threshold: float = PTZ_STABILIZATION_THRESHOLD,
) -> dict | None:
    """Wait for PTZ position to stabilize (no change for threshold duration).

    Args:
        ptz: HikvisionPTZ instance
        max_wait: Maximum time to wait for stabilization in seconds
        poll_interval: How often to check status
        stable_threshold: How long position must be unchanged to be considered stable

    Returns:
        Final stabilized status dict, or None if failed
    """
    start_time = time.time()
    last_position = None
    stable_since = None

    while time.time() - start_time < max_wait:
        status = ptz.get_status()
        if status is None:
            time.sleep(poll_interval)
            continue

        current_position = (status.get("pan"), status.get("tilt"), status.get("zoom"))

        if last_position is not None:
            # Check if position has changed
            position_changed = False
            for i, (curr, last) in enumerate(zip(current_position, last_position)):
                if curr is not None and last is not None:
                    if abs(curr - last) > 0.1:  # Tolerance for floating point
                        position_changed = True
                        break

            if not position_changed:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= stable_threshold:
                    # Position has been stable long enough
                    return status
            else:
                stable_since = None

        last_position = current_position
        time.sleep(poll_interval)

    # Return last known status even if not fully stabilized
    return ptz.get_status()


def _execute_movement_test(
    ptz: HikvisionPTZ,
    pan: int = 0,
    tilt: int = 0,
    zoom: int = 0,
    duration: float = PTZ_MOVEMENT_DURATION,
) -> dict:
    """Execute a single movement test and return results.

    Args:
        ptz: HikvisionPTZ instance
        pan: Pan speed (-100 to 100)
        tilt: Tilt speed (-100 to 100)
        zoom: Zoom speed (-100 to 100)
        duration: How long to move in seconds

    Returns:
        Dict with test results including position change and timing
    """
    result = {
        "success": False,
        "position_changed": False,
        "initial_position": None,
        "final_position": None,
        "delta": None,
        "response_time_ms": None,
        "error": None,
    }

    try:
        # Get initial position
        initial_status = ptz.get_status()
        if initial_status is None:
            result["error"] = "Failed to get initial position"
            return result

        result["initial_position"] = {
            "pan": initial_status.get("pan"),
            "tilt": initial_status.get("tilt"),
            "zoom": initial_status.get("zoom"),
        }

        # Start movement and measure response time
        start_time = time.time()
        move_success = ptz.move_continuous(pan=pan, tilt=tilt, zoom=zoom)
        response_time = (time.time() - start_time) * 1000
        result["response_time_ms"] = round(response_time, 2)

        if not move_success:
            result["error"] = "Failed to start movement"
            return result

        # Let movement run for specified duration
        time.sleep(duration)

        # Stop movement
        ptz.stop_movement()

        # Wait for position to stabilize
        final_status = _wait_for_stabilization(ptz)
        if final_status is None:
            result["error"] = "Failed to get final position"
            return result

        result["final_position"] = {
            "pan": final_status.get("pan"),
            "tilt": final_status.get("tilt"),
            "zoom": final_status.get("zoom"),
        }

        # Calculate position delta
        delta = {}
        position_changed = False
        for axis in ["pan", "tilt", "zoom"]:
            initial_val = result["initial_position"].get(axis)
            final_val = result["final_position"].get(axis)
            if initial_val is not None and final_val is not None:
                delta[axis] = round(final_val - initial_val, 2)
                if abs(delta[axis]) > 0.1:
                    position_changed = True
            else:
                delta[axis] = None

        result["delta"] = delta
        result["position_changed"] = position_changed
        result["success"] = True

    except requests.exceptions.Timeout:
        result["error"] = "Network timeout"
        result["error_category"] = PTZErrorCategory.NETWORK_TIMEOUT.value
    except Exception as e:
        result["error"] = str(e)
        result["error_category"] = _classify_ptz_error(e).value

    return result


def _get_presets_list(ptz: HikvisionPTZ) -> dict:
    """Get list of presets from camera via ISAPI.

    Args:
        ptz: HikvisionPTZ instance

    Returns:
        Dict with presets list result
    """
    result = {
        "success": False,
        "count": 0,
        "presets": [],
        "response_time_ms": None,
        "error": None,
    }

    try:
        endpoint = "/ISAPI/PTZCtrl/channels/1/presets"
        start_time = time.time()
        status_code, response = ptz._make_request("GET", endpoint)
        response_time = (time.time() - start_time) * 1000
        result["response_time_ms"] = round(response_time, 2)

        if status_code == 401:
            result["error"] = "Authentication failed"
            result["error_category"] = PTZErrorCategory.AUTH_FAILED.value
            return result

        if status_code != 200:
            result["error"] = f"HTTP {status_code}"
            result["error_category"] = PTZErrorCategory.API_ERROR.value
            return result

        # Parse XML response
        try:
            root = ET.fromstring(response)
            ns = {"h": "http://www.hikvision.com/ver20/XMLSchema"}

            presets = []
            # Try with namespace first
            preset_elements = root.findall(".//h:PTZPreset", ns)
            if not preset_elements:
                # Fall back to no namespace
                preset_elements = root.findall(".//PTZPreset")

            for preset_elem in preset_elements:
                preset_data = {}

                # Try to get preset ID
                id_elem = preset_elem.find("h:id", ns)
                if id_elem is None:
                    id_elem = preset_elem.find("id")
                if id_elem is not None:
                    preset_data["id"] = id_elem.text

                # Try to get preset name
                name_elem = preset_elem.find("h:presetName", ns)
                if name_elem is None:
                    name_elem = preset_elem.find("presetName")
                if name_elem is not None:
                    preset_data["name"] = name_elem.text

                # Try to get enabled status
                enabled_elem = preset_elem.find("h:enabled", ns)
                if enabled_elem is None:
                    enabled_elem = preset_elem.find("enabled")
                if enabled_elem is not None:
                    preset_data["enabled"] = enabled_elem.text.lower() == "true"

                if preset_data:
                    presets.append(preset_data)

            result["presets"] = presets
            result["count"] = len(presets)
            result["success"] = True

        except ET.ParseError as e:
            result["error"] = f"Invalid XML response: {e}"
            result["error_category"] = PTZErrorCategory.INVALID_XML.value

    except requests.exceptions.Timeout:
        result["error"] = "Network timeout"
        result["error_category"] = PTZErrorCategory.NETWORK_TIMEOUT.value
    except Exception as e:
        result["error"] = str(e)
        result["error_category"] = _classify_ptz_error(e).value

    return result


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
        JsonResponse with:
        - status: "success" or "error"
        - initial_position: Starting PTZ position
        - position_restored: Whether original position was restored
        - tests: Results for each test (status, movements, presets)
    """
    # Validate camera exists
    camera = get_camera_by_name(camera_name)
    if not camera:
        return JsonResponse(
            {
                "status": "error",
                "error_category": PTZErrorCategory.CAMERA_NOT_FOUND.value,
                "message": f"Camera not found: {camera_name}",
            },
            status=404,
        )

    camera_ip = camera.get("ip")
    if not camera_ip:
        return JsonResponse(
            {
                "status": "error",
                "error_category": PTZErrorCategory.API_ERROR.value,
                "message": f"No IP address configured for camera: {camera_name}",
            },
            status=500,
        )

    # Get credentials from environment variables
    username = os.getenv("CAMERA_USERNAME")
    password = os.getenv("CAMERA_PASSWORD")

    if not username or not password:
        return JsonResponse(
            {
                "status": "error",
                "error_category": PTZErrorCategory.CREDENTIALS_NOT_SET.value,
                "message": "Camera credentials not set. Set CAMERA_USERNAME and CAMERA_PASSWORD environment variables.",
            },
            status=500,
        )

    # Initialize PTZ controller
    ptz = HikvisionPTZ(ip=camera_ip, username=username, password=password, name=camera_name)

    response_data = {
        "status": "success",
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
            status_result["error_category"] = PTZErrorCategory.NETWORK_TIMEOUT.value
        except Exception as e:
            status_result["error"] = str(e)
            status_result["error_category"] = _classify_ptz_error(e).value

        response_data["tests"]["status"] = status_result

        # If status failed, return early
        if not status_result["success"]:
            response_data["status"] = "error"
            response_data["message"] = "Failed to get initial PTZ status"
            return JsonResponse(response_data, status=503)

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
            response_data["tests"][test_name] = _execute_movement_test(
                ptz, pan=pan, tilt=tilt, zoom=zoom, duration=PTZ_MOVEMENT_DURATION
            )

        # Test 3: Get presets list
        response_data["tests"]["presets"] = _get_presets_list(ptz)

    finally:
        # Always try to restore original position if we got initial status
        if initial_status:
            try:
                restore_success = ptz.send_ptz_return(initial_status)
                if restore_success:
                    # Wait for camera to reach restored position
                    _wait_for_stabilization(ptz)
                    response_data["position_restored"] = True
            except Exception as e:
                logger.warning(f"Failed to restore PTZ position for {camera_name}: {e}")
                response_data["position_restored"] = False
                response_data["restore_error"] = str(e)

    return JsonResponse(response_data)
