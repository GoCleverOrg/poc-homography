"""Business logic services for the camera diagnostic app.

This module contains service functions extracted from views to keep views thin.
"""

from __future__ import annotations

import logging
import os
import re
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
from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ
from requests.auth import HTTPDigestAuth

from poc_homography.camera_config import get_rtsp_url

logger = logging.getLogger(__name__)

# RTSP connection timeout in seconds
RTSP_CONNECTION_TIMEOUT_SEC = 10

# Web UI test timeout in seconds
WEBUI_CONNECTION_TIMEOUT_SEC = 15

# PTZ API test constants
PTZ_MOVEMENT_SPEED = 30  # Speed for movement tests (-100 to 100)
PTZ_MOVEMENT_DURATION = 1.5  # Duration of each movement in seconds
PTZ_STABILIZATION_INTERVAL = 0.1  # Polling interval for stabilization check
PTZ_STABILIZATION_THRESHOLD = 0.5  # Seconds of no change to consider stabilized
PTZ_API_TIMEOUT = 10  # Network timeout for API calls in seconds


class CameraErrorCategory(Enum):
    """Unified error categories for camera diagnostics (RTSP, WebUI, PTZ)."""

    NETWORK_UNREACHABLE = "NETWORK_UNREACHABLE"
    AUTH_FAILED = "AUTH_FAILED"
    TIMEOUT = "TIMEOUT"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    CAMERA_OFFLINE = "CAMERA_OFFLINE"
    CREDENTIALS_NOT_SET = "CREDENTIALS_NOT_SET"
    CAMERA_NOT_FOUND = "CAMERA_NOT_FOUND"
    INVALID_XML = "INVALID_XML"
    API_ERROR = "API_ERROR"


def _sanitize_camera_name(name: str) -> str:
    """Sanitize camera name for use in file paths and headers.

    Args:
        name: Raw camera name

    Returns:
        Sanitized name safe for file paths and headers
    """
    return re.sub(r"[^\w\-]", "_", name)


# =============================================================================
# RTSP Service Functions
# =============================================================================


def classify_rtsp_error(exception: Exception, rtsp_url: str | None = None) -> CameraErrorCategory:
    """Classify an exception into an RTSP error category.

    Args:
        exception: The exception that occurred
        rtsp_url: The RTSP URL being accessed (for context)

    Returns:
        The appropriate CameraErrorCategory
    """
    error_str = str(exception).lower()

    # Socket/network errors
    if isinstance(exception, (socket.error, OSError)):
        if "connection refused" in error_str or "no route to host" in error_str:
            return CameraErrorCategory.NETWORK_UNREACHABLE
        if "timed out" in error_str or "timeout" in error_str:
            return CameraErrorCategory.TIMEOUT

    # Credential errors
    if isinstance(exception, ValueError) and "credentials" in error_str:
        return CameraErrorCategory.CREDENTIALS_NOT_SET

    # Generic timeout indicators
    if "timeout" in error_str or "timed out" in error_str:
        return CameraErrorCategory.TIMEOUT

    # Authentication indicators
    if "401" in error_str or "403" in error_str or "unauthorized" in error_str or "auth" in error_str:
        return CameraErrorCategory.AUTH_FAILED

    # Default to camera offline for OpenCV capture failures
    return CameraErrorCategory.CAMERA_OFFLINE


def create_rtsp_capture(rtsp_url: str) -> cv2.VideoCapture:
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


def generate_mjpeg_frames(camera_name: str) -> Generator[bytes, None, None]:
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

    cap = create_rtsp_capture(rtsp_url)

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


# =============================================================================
# WebUI Service Functions
# =============================================================================


def get_screenshot_path(camera_name: str) -> Path:
    """Generate a timestamped screenshot path for the camera.

    Args:
        camera_name: Name of the camera

    Returns:
        Path object for the screenshot file
    """
    screenshots_dir = getattr(
        settings, "CAMERA_DIAGNOSTIC_SCREENSHOTS_DIR", Path(settings.BASE_DIR) / "diagnostic_screenshots"
    )
    screenshots_dir = Path(screenshots_dir)
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_name = _sanitize_camera_name(camera_name)
    return screenshots_dir / f"{sanitized_name}_webui_{timestamp}.png"


def classify_webui_error(exception: Exception) -> CameraErrorCategory:
    """Classify an exception into an error category for web UI testing.

    Args:
        exception: The exception that occurred

    Returns:
        The appropriate CameraErrorCategory
    """
    error_str = str(exception).lower()

    # Network errors
    if "net::" in error_str or "connection refused" in error_str or "no route" in error_str:
        return CameraErrorCategory.NETWORK_UNREACHABLE

    # Timeout errors
    if "timeout" in error_str or "timed out" in error_str:
        return CameraErrorCategory.TIMEOUT

    # Authentication errors
    if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
        return CameraErrorCategory.AUTH_FAILED

    # Default to invalid response for unexpected errors
    return CameraErrorCategory.INVALID_RESPONSE


def detect_ptz_controls(page) -> list[dict]:
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
                    except Exception as e:
                        # Element may have become stale, skip it
                        logger.debug(f"PTZ control element stale or inaccessible: {e}")
            except Exception as e:
                # Selector may not be valid for this page, skip it
                logger.debug(f"PTZ selector '{selector}' not valid for page: {e}")

    # Remove duplicates based on type and tag combination
    seen = set()
    unique_controls = []
    for control in ptz_controls:
        key = (control["type"], control.get("tag", ""))
        if key not in seen:
            seen.add(key)
            unique_controls.append(control)

    return unique_controls


def attempt_login(page, username: str, password: str) -> bool:
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

    # Common password field selectors (Hikvision-specific first)
    password_selectors = [
        # Hikvision specific - often uses placeholder or class-based identification
        'input[placeholder="Password"]',
        'input[placeholder*="password" i]',
        'input[placeholder*="Password"]',
        'input[class*="password"]',
        'input[class*="pwd"]',
        # Standard selectors
        "#password",
        "#pass",
        "#pwd",
        'input[name="password"]',
        'input[name="pass"]',
        'input[name="pwd"]',
        'input[type="password"]',
        # Fallback - second input field (after username)
        'input:nth-of-type(2)',
    ]

    # Common submit button selectors (Hikvision-specific first)
    submit_selectors = [
        # Hikvision specific
        'button:has-text("Login")',
        'a:has-text("Login")',
        'div:has-text("Login"):not(:has(*))',  # Leaf div with just "Login" text
        '[class*="login-btn"]',
        '[class*="loginBtn"]',
        '[class*="btn-login"]',
        # Generic selectors
        'button[type="submit"]',
        'input[type="submit"]',
        "#login",
        "#loginBtn",
        "#submit",
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
        except Exception as e:
            logger.debug(f"Username selector '{selector}' failed: {e}")
            continue

    if not username_filled:
        return False

    # Try multiple approaches to fill password field (Hikvision has tricky login forms)
    password_filled = False

    # Approach 1: Use JavaScript to set value directly (bypasses custom input handlers)
    for selector in password_selectors:
        try:
            element = page.locator(selector).first
            if element.is_visible():
                # Use evaluate to set value via JavaScript - pass password as argument
                page.evaluate("""([selector, pwd]) => {
                    const el = document.querySelector(selector);
                    if (el) {
                        el.value = pwd;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }""", [selector, password])
                password_filled = True
                break
        except Exception as e:
            logger.debug(f"Password selector '{selector}' failed: {e}")
            continue

    # Approach 2: Tab from username and type (fallback)
    if not password_filled:
        try:
            page.keyboard.press("Tab")
            page.wait_for_timeout(100)
            page.keyboard.type(password, delay=30)
            password_filled = True
        except Exception as e:
            logger.debug(f"Tab-to-password fallback failed: {e}")

    # Try to click submit button
    for selector in submit_selectors:
        try:
            element = page.locator(selector).first
            if element.is_visible():
                element.click()
                # Wait for navigation or page change after login
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception as e:
                    logger.debug(f"Post-login navigation wait timeout (expected): {e}")
                return True
        except Exception as e:
            logger.debug(f"Submit selector '{selector}' failed: {e}")
            continue

    # If no submit button found, try pressing Enter on password field
    try:
        for selector in password_selectors:
            element = page.locator(selector).first
            if element.is_visible():
                element.press("Enter")
                # Wait for navigation
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception as e:
                    logger.debug(f"Post-Enter navigation wait timeout (expected): {e}")
                return True
    except Exception as e:
        logger.debug(f"Enter key login fallback failed: {e}")

    return False


def get_login_error_message(page) -> str | None:
    """Try to detect and extract login error message from the page.

    Args:
        page: Playwright page object

    Returns:
        Error message text if found, None otherwise
    """
    # Text patterns that indicate real login errors
    error_keywords = [
        "invalid", "incorrect", "wrong", "failed", "denied",
        "error", "locked", "disabled", "expired", "unauthorized",
    ]

    # Text patterns to filter out (not actual errors)
    filter_patterns = [
        "privacy", "respect", "rights", "product",
        "copyright", "reserved", "technology",
    ]

    # Look for common error message patterns
    error_selectors = [
        '[class*="error"]',
        '[class*="alert"]',
        '[class*="warning"]',
        '[id*="error"]',
        '[id*="msg"]',
        '[class*="msg"]',
        '[class*="tip"]',
    ]

    for selector in error_selectors:
        try:
            elements = page.locator(selector).all()
            for element in elements:
                if element.is_visible():
                    text = element.text_content()
                    if text:
                        text_lower = text.strip().lower()
                        # Skip if it matches filter patterns
                        if any(pattern in text_lower for pattern in filter_patterns):
                            continue
                        # Return if it contains error keywords
                        if any(keyword in text_lower for keyword in error_keywords):
                            return text.strip()
        except Exception as e:
            logger.debug(f"Error selector '{selector}' failed: {e}")
            continue

    return None


def check_login_success(page) -> bool:
    """Check if login was successful.

    Args:
        page: Playwright page object

    Returns:
        True if login appears successful, False otherwise
    """
    # Primary check: URL-based detection (most reliable for Hikvision cameras)
    # After successful login, URL changes from login.asp to preview.asp or similar
    current_url = page.url.lower()
    if "login" in current_url:
        # Still on login page - check if it's due to an error or just slow redirect
        pass  # Continue with other checks
    elif any(indicator in current_url for indicator in ["preview", "live", "main", "index", "home"]):
        # Successfully redirected to post-login page
        return True

    # Secondary check: Page title (Hikvision changes from "Login" to "Live View")
    try:
        title = page.title().lower()
        if "login" in title:
            pass  # Still shows login title, continue checking
        elif any(indicator in title for indicator in ["live", "preview", "view", "camera"]):
            return True
    except Exception as e:
        logger.debug(f"Title check failed: {e}")

    # Check for absence of login form (common login success indicator)
    login_form_selectors = [
        "#loginForm",
        'form[action*="login"]',
        'form[id*="login"]',
        "#username",
        'input[name="username"]',
        # Hikvision-specific: check for User Name textbox
        'input[placeholder*="User" i]',
    ]

    for selector in login_form_selectors:
        try:
            if page.locator(selector).is_visible():
                return False
        except Exception as e:
            logger.debug(f"Login form selector '{selector}' check failed: {e}")

    # Check for common error messages (be more specific to avoid false positives)
    error_selectors = [
        '[class*="login-error"]',
        '[class*="error-msg"]',
        '[class*="alert-error"]',
        '[id*="loginError"]',
    ]

    for selector in error_selectors:
        try:
            element = page.locator(selector).first
            if element.is_visible():
                text = element.text_content().lower() if element.text_content() else ""
                if any(word in text for word in ["invalid", "incorrect", "failed", "wrong", "denied"]):
                    return False
        except Exception as e:
            logger.debug(f"Error message selector '{selector}' check failed: {e}")

    # Check for common post-login elements
    post_login_indicators = [
        '[class*="dashboard"]',
        '[class*="main-content"]',
        '[class*="menu"]',
        '[class*="nav"]',
        '[id*="dashboard"]',
        '[id*="mainContent"]',
        "video",
        "canvas",
        '[class*="live"]',
        '[class*="stream"]',
        '[class*="preview"]',
        # Hikvision-specific indicators
        '[class*="ptz"]',
        'a[href*="ptzConfig"]',
        'a[href*="preview"]',
    ]

    for selector in post_login_indicators:
        try:
            if page.locator(selector).first.is_visible():
                return True
        except Exception as e:
            logger.debug(f"Post-login indicator '{selector}' check failed: {e}")

    # If we can't definitively tell, assume success if no login form is visible
    return True


def dismiss_hikvision_warning_dialog(page) -> bool:
    """Dismiss Hikvision's 'low resources' or similar warning dialogs.

    Hikvision cameras often show a warning dialog about low browser resources
    or plugin requirements after login. This function finds and dismisses it
    by checking "Don't warn me again" and clicking OK/Close.

    Args:
        page: Playwright page object

    Returns:
        True if a dialog was dismissed, False otherwise
    """
    # Wait a moment for any dialogs to appear
    time.sleep(1)

    # Common selectors for Hikvision warning dialogs
    dialog_selectors = [
        # Dialog containers
        '[class*="dialog"]',
        '[class*="modal"]',
        '[class*="popup"]',
        '[class*="alert"]',
        '[id*="dialog"]',
        '[id*="modal"]',
        '[id*="warning"]',
    ]

    # Check if any dialog is visible
    dialog_found = False
    for selector in dialog_selectors:
        try:
            dialog = page.locator(selector).first
            if dialog.is_visible():
                dialog_found = True
                break
        except Exception as e:
            logger.debug(f"Dialog selector '{selector}' check failed: {e}")
            continue

    if not dialog_found:
        return False

    # Try to find and check "Don't warn me again" checkbox
    checkbox_selectors = [
        'input[type="checkbox"]',
        '[class*="checkbox"]',
        ':text("Don\'t")',
        ':text("again")',
        ':text("remind")',
    ]

    for selector in checkbox_selectors:
        try:
            checkbox = page.locator(selector).first
            if checkbox.is_visible():
                # Check if it's a checkbox input or contains one
                if checkbox.get_attribute("type") == "checkbox":
                    if not checkbox.is_checked():
                        checkbox.check()
                else:
                    # Try to find checkbox within the element
                    inner_checkbox = checkbox.locator('input[type="checkbox"]').first
                    if inner_checkbox.is_visible() and not inner_checkbox.is_checked():
                        inner_checkbox.check()
                break
        except Exception as e:
            logger.debug(f"Checkbox selector '{selector}' failed: {e}")
            continue

    # Try to click OK/Close/Confirm button
    button_selectors = [
        'button:text("OK")',
        'button:text("Close")',
        'button:text("Confirm")',
        'button:text("Accept")',
        '[class*="close"]',
        '[class*="confirm"]',
        '[class*="ok-btn"]',
        'button[type="button"]',
    ]

    for selector in button_selectors:
        try:
            button = page.locator(selector).first
            if button.is_visible():
                button.click()
                time.sleep(0.5)
                return True
        except Exception as e:
            logger.debug(f"Dialog button selector '{selector}' failed: {e}")
            continue

    return False


def test_webui_ptz_controls(page) -> dict:
    """Test PTZ controls in the web interface.

    Attempts to interact with PTZ direction controls to verify they're functional.

    Args:
        page: Playwright page object

    Returns:
        Dict with test results including which controls were found and tested
    """
    result = {
        "controls_found": [],
        "controls_tested": [],
        "success": False,
    }

    # Hikvision PTZ control selectors
    ptz_control_selectors = {
        "up": ['[class*="ptz-up"]', '[class*="up"]', '[title*="Up"]', 'button:text("▲")'],
        "down": ['[class*="ptz-down"]', '[class*="down"]', '[title*="Down"]', 'button:text("▼")'],
        "left": ['[class*="ptz-left"]', '[class*="left"]', '[title*="Left"]', 'button:text("◀")'],
        "right": ['[class*="ptz-right"]', '[class*="right"]', '[title*="Right"]', 'button:text("▶")'],
        "zoom_in": ['[class*="zoom-in"]', '[class*="zoomin"]', '[title*="Zoom In"]'],
        "zoom_out": ['[class*="zoom-out"]', '[class*="zoomout"]', '[title*="Zoom Out"]'],
    }

    # First, try to open PTZ panel if it's collapsed
    ptz_panel_toggles = [
        'a[href*="ptzConfig"]',
        '[class*="ptz-toggle"]',
        ':text("PTZ")',
    ]

    for selector in ptz_panel_toggles:
        try:
            toggle = page.locator(selector).first
            if toggle.is_visible():
                toggle.click()
                time.sleep(1)
                break
        except Exception as e:
            logger.debug(f"PTZ panel toggle '{selector}' failed: {e}")
            continue

    # Find and test PTZ controls
    for control_name, selectors in ptz_control_selectors.items():
        for selector in selectors:
            try:
                control = page.locator(selector).first
                if control.is_visible():
                    result["controls_found"].append(control_name)

                    # Try to click the control (mouse down/up for PTZ)
                    try:
                        control.click()
                        time.sleep(0.2)
                        result["controls_tested"].append(control_name)
                    except Exception as e:
                        logger.debug(f"PTZ control '{control_name}' click failed: {e}")
                    break
            except Exception as e:
                logger.debug(f"PTZ control selector '{selector}' failed: {e}")
                continue

    result["success"] = len(result["controls_tested"]) > 0
    return result


# =============================================================================
# PTZ Service Functions
# =============================================================================


def classify_ptz_error(exception: Exception) -> CameraErrorCategory:
    """Classify an exception into a PTZ error category.

    Args:
        exception: The exception that occurred

    Returns:
        The appropriate CameraErrorCategory
    """
    error_str = str(exception).lower()

    # Network timeout errors
    if isinstance(exception, requests.exceptions.Timeout):
        return CameraErrorCategory.TIMEOUT
    if "timeout" in error_str or "timed out" in error_str:
        return CameraErrorCategory.TIMEOUT

    # Authentication errors
    if isinstance(exception, requests.exceptions.HTTPError):
        if hasattr(exception, "response") and exception.response is not None:
            if exception.response.status_code == 401:
                return CameraErrorCategory.AUTH_FAILED

    if "401" in error_str or "unauthorized" in error_str or "authentication" in error_str:
        return CameraErrorCategory.AUTH_FAILED

    # XML parsing errors
    if isinstance(exception, ET.ParseError):
        return CameraErrorCategory.INVALID_XML
    if "xml" in error_str or "parse" in error_str:
        return CameraErrorCategory.INVALID_XML

    # Default to generic API error
    return CameraErrorCategory.API_ERROR


def wait_for_stabilization(
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


def execute_movement_test(
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
        final_status = wait_for_stabilization(ptz)
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
        result["error_category"] = CameraErrorCategory.TIMEOUT.value
    except Exception as e:
        result["error"] = str(e)
        result["error_category"] = classify_ptz_error(e).value

    return result


def get_presets_list(camera_ip: str, username: str, password: str) -> dict:
    """Get list of presets from camera via ISAPI.

    Args:
        camera_ip: IP address of the camera
        username: Authentication username
        password: Authentication password

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
        start_time = time.time()
        response = requests.get(
            f"http://{camera_ip}/ISAPI/PTZCtrl/channels/1/presets",
            auth=HTTPDigestAuth(username, password),
            timeout=PTZ_API_TIMEOUT,
        )
        response_time = (time.time() - start_time) * 1000
        result["response_time_ms"] = round(response_time, 2)

        if response.status_code == 401:
            result["error"] = "Authentication failed"
            result["error_category"] = CameraErrorCategory.AUTH_FAILED.value
            return result

        if response.status_code != 200:
            result["error"] = f"HTTP {response.status_code}"
            result["error_category"] = CameraErrorCategory.API_ERROR.value
            return result

        # Parse XML response
        try:
            root = ET.fromstring(response.text)
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
            result["error_category"] = CameraErrorCategory.INVALID_XML.value

    except requests.exceptions.Timeout:
        result["error"] = "Network timeout"
        result["error_category"] = CameraErrorCategory.TIMEOUT.value
    except Exception as e:
        result["error"] = str(e)
        result["error_category"] = classify_ptz_error(e).value

    return result
