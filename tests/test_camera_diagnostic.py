"""Unit tests for the camera diagnostic services module.

Tests cover:
- _sanitize_camera_name: File path sanitization
- classify_rtsp_error: RTSP error classification
- classify_webui_error: WebUI error classification
- classify_ptz_error: PTZ error classification
- get_screenshot_path: Screenshot path generation
- get_presets_list: PTZ presets API interaction
"""

import os
import socket
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests.exceptions

# Set up Django settings before importing from webapp
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "homography_web.settings")
sys.path.insert(0, str(Path(__file__).parent.parent / "webapp"))

import django

django.setup()

from webapp.camera_diagnostic.services import (
    CameraErrorCategory,
    _sanitize_camera_name,
    classify_ptz_error,
    classify_rtsp_error,
    classify_webui_error,
    get_presets_list,
    get_screenshot_path,
)


# =============================================================================
# Helper Classes for Mocking
# =============================================================================


class DummyResponse:
    """Mock HTTP response for testing."""

    def __init__(self, status_code=200, text="OK"):
        self.status_code = status_code
        self.text = text


class DummyHTTPError(requests.exceptions.HTTPError):
    """Mock HTTPError with response attribute."""

    def __init__(self, status_code, message=""):
        self.response = DummyResponse(status_code=status_code)
        super().__init__(message)


# =============================================================================
# Tests for _sanitize_camera_name
# =============================================================================


class TestSanitizeCameraName:
    """Tests for the _sanitize_camera_name function."""

    def test_normal_name_unchanged(self):
        """Normal alphanumeric name should pass through unchanged."""
        assert _sanitize_camera_name("Valte") == "Valte"

    def test_name_with_spaces_converted_to_underscores(self):
        """Spaces should be converted to underscores."""
        assert _sanitize_camera_name("Camera 1") == "Camera_1"

    def test_name_with_multiple_spaces(self):
        """Multiple spaces should each become underscores."""
        assert _sanitize_camera_name("My Camera Name") == "My_Camera_Name"

    def test_path_traversal_chars_sanitized(self):
        """Path traversal characters should be sanitized."""
        assert _sanitize_camera_name("../etc/passwd") == "___etc_passwd"

    def test_special_chars_sanitized(self):
        """Special characters should be replaced with underscores."""
        # Note: ^ is a word character escape sequence, so some chars may merge
        result = _sanitize_camera_name("cam@#$%^&*()")
        assert result.startswith("cam")
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert "%" not in result
        assert "&" not in result
        assert "*" not in result
        assert "(" not in result
        assert ")" not in result

    def test_dashes_allowed(self):
        """Dashes should be preserved."""
        assert _sanitize_camera_name("my-camera") == "my-camera"

    def test_underscores_allowed(self):
        """Underscores should be preserved."""
        assert _sanitize_camera_name("my_camera") == "my_camera"

    def test_numbers_allowed(self):
        """Numbers should be preserved."""
        assert _sanitize_camera_name("camera123") == "camera123"

    def test_mixed_valid_invalid_chars(self):
        """Mixed valid and invalid characters."""
        assert _sanitize_camera_name("cam-1_test@server") == "cam-1_test_server"

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert _sanitize_camera_name("") == ""

    def test_only_special_chars(self):
        """String with only special characters."""
        assert _sanitize_camera_name("@#$%") == "____"

    def test_unicode_word_chars_preserved(self):
        """Unicode word characters (like accented letters) are preserved by \\w."""
        # Note: The regex [^\w\-] uses \w which includes Unicode word characters
        # (letters, digits, underscore) across all Unicode categories.
        # So accented characters like 'e' are actually preserved.
        assert _sanitize_camera_name("camera\u00e9") == "camera\u00e9"

    def test_newline_chars_sanitized(self):
        """Newline characters should be sanitized."""
        assert _sanitize_camera_name("camera\nname") == "camera_name"


# =============================================================================
# Tests for classify_rtsp_error
# =============================================================================


class TestClassifyRtspError:
    """Tests for the classify_rtsp_error function."""

    def test_socket_error_connection_refused(self):
        """Socket error with 'connection refused' should be NETWORK_UNREACHABLE."""
        exc = socket.error("Connection refused")
        assert classify_rtsp_error(exc) == CameraErrorCategory.NETWORK_UNREACHABLE

    def test_socket_error_no_route_to_host(self):
        """Socket error with 'no route to host' should be NETWORK_UNREACHABLE."""
        exc = socket.error("No route to host")
        assert classify_rtsp_error(exc) == CameraErrorCategory.NETWORK_UNREACHABLE

    def test_socket_error_timed_out(self):
        """Socket error with 'timed out' should be TIMEOUT."""
        exc = socket.error("Connection timed out")
        assert classify_rtsp_error(exc) == CameraErrorCategory.TIMEOUT

    def test_socket_error_timeout(self):
        """Socket error with 'timeout' should be TIMEOUT."""
        exc = socket.error("Socket timeout")
        assert classify_rtsp_error(exc) == CameraErrorCategory.TIMEOUT

    def test_oserror_connection_refused(self):
        """OSError with 'connection refused' should be NETWORK_UNREACHABLE."""
        exc = OSError("Connection refused by server")
        assert classify_rtsp_error(exc) == CameraErrorCategory.NETWORK_UNREACHABLE

    def test_valueerror_credentials_not_set(self):
        """ValueError with 'credentials' should be CREDENTIALS_NOT_SET."""
        exc = ValueError("Camera credentials not configured")
        assert classify_rtsp_error(exc) == CameraErrorCategory.CREDENTIALS_NOT_SET

    def test_valueerror_without_credentials(self):
        """ValueError without 'credentials' should be CAMERA_OFFLINE."""
        exc = ValueError("Invalid camera name")
        assert classify_rtsp_error(exc) == CameraErrorCategory.CAMERA_OFFLINE

    def test_generic_exception_timeout_in_message(self):
        """Generic exception with 'timeout' in message should be TIMEOUT."""
        exc = Exception("Operation timeout occurred")
        assert classify_rtsp_error(exc) == CameraErrorCategory.TIMEOUT

    def test_generic_exception_timed_out_in_message(self):
        """Generic exception with 'timed out' in message should be TIMEOUT."""
        exc = Exception("Connection timed out")
        assert classify_rtsp_error(exc) == CameraErrorCategory.TIMEOUT

    def test_generic_exception_401_in_message(self):
        """Generic exception with '401' in message should be AUTH_FAILED."""
        exc = Exception("HTTP 401 Unauthorized")
        assert classify_rtsp_error(exc) == CameraErrorCategory.AUTH_FAILED

    def test_generic_exception_403_in_message(self):
        """Generic exception with '403' in message should be AUTH_FAILED."""
        exc = Exception("HTTP 403 Forbidden")
        assert classify_rtsp_error(exc) == CameraErrorCategory.AUTH_FAILED

    def test_generic_exception_unauthorized_in_message(self):
        """Generic exception with 'unauthorized' in message should be AUTH_FAILED."""
        exc = Exception("Unauthorized access")
        assert classify_rtsp_error(exc) == CameraErrorCategory.AUTH_FAILED

    def test_generic_exception_auth_in_message(self):
        """Generic exception with 'auth' in message should be AUTH_FAILED."""
        exc = Exception("Authentication failed")
        assert classify_rtsp_error(exc) == CameraErrorCategory.AUTH_FAILED

    def test_unknown_exception_defaults_to_camera_offline(self):
        """Unknown exception should default to CAMERA_OFFLINE."""
        exc = Exception("Unknown error")
        assert classify_rtsp_error(exc) == CameraErrorCategory.CAMERA_OFFLINE

    def test_empty_exception_message(self):
        """Exception with empty message should default to CAMERA_OFFLINE."""
        exc = Exception("")
        assert classify_rtsp_error(exc) == CameraErrorCategory.CAMERA_OFFLINE

    def test_rtsp_url_parameter_accepted(self):
        """Function should accept rtsp_url parameter (even if not used)."""
        exc = Exception("Unknown error")
        result = classify_rtsp_error(exc, rtsp_url="rtsp://192.168.1.1/stream")
        assert result == CameraErrorCategory.CAMERA_OFFLINE


# =============================================================================
# Tests for classify_webui_error
# =============================================================================


class TestClassifyWebuiError:
    """Tests for the classify_webui_error function."""

    def test_net_error(self):
        """Error with 'net::' should be NETWORK_UNREACHABLE."""
        exc = Exception("net::ERR_CONNECTION_REFUSED")
        assert classify_webui_error(exc) == CameraErrorCategory.NETWORK_UNREACHABLE

    def test_connection_refused(self):
        """Error with 'connection refused' should be NETWORK_UNREACHABLE."""
        exc = Exception("Connection refused")
        assert classify_webui_error(exc) == CameraErrorCategory.NETWORK_UNREACHABLE

    def test_no_route(self):
        """Error with 'no route' should be NETWORK_UNREACHABLE."""
        exc = Exception("No route to host")
        assert classify_webui_error(exc) == CameraErrorCategory.NETWORK_UNREACHABLE

    def test_timeout(self):
        """Error with 'timeout' should be TIMEOUT."""
        exc = Exception("Page timeout")
        assert classify_webui_error(exc) == CameraErrorCategory.TIMEOUT

    def test_timed_out(self):
        """Error with 'timed out' should be TIMEOUT."""
        exc = Exception("Request timed out")
        assert classify_webui_error(exc) == CameraErrorCategory.TIMEOUT

    def test_401_unauthorized(self):
        """Error with '401' should be AUTH_FAILED."""
        exc = Exception("HTTP 401 Unauthorized")
        assert classify_webui_error(exc) == CameraErrorCategory.AUTH_FAILED

    def test_403_forbidden(self):
        """Error with '403' should be AUTH_FAILED."""
        exc = Exception("HTTP 403 Forbidden")
        assert classify_webui_error(exc) == CameraErrorCategory.AUTH_FAILED

    def test_unauthorized(self):
        """Error with 'unauthorized' should be AUTH_FAILED."""
        exc = Exception("Unauthorized access")
        assert classify_webui_error(exc) == CameraErrorCategory.AUTH_FAILED

    def test_unknown_error_defaults_to_invalid_response(self):
        """Unknown error should default to INVALID_RESPONSE."""
        exc = Exception("Unknown browser error")
        assert classify_webui_error(exc) == CameraErrorCategory.INVALID_RESPONSE

    def test_empty_message(self):
        """Empty message should default to INVALID_RESPONSE."""
        exc = Exception("")
        assert classify_webui_error(exc) == CameraErrorCategory.INVALID_RESPONSE


# =============================================================================
# Tests for classify_ptz_error
# =============================================================================


class TestClassifyPtzError:
    """Tests for the classify_ptz_error function."""

    def test_requests_timeout(self):
        """requests.exceptions.Timeout should be TIMEOUT."""
        exc = requests.exceptions.Timeout("Connection timed out")
        assert classify_ptz_error(exc) == CameraErrorCategory.TIMEOUT

    def test_timeout_in_message(self):
        """Exception with 'timeout' in message should be TIMEOUT."""
        exc = Exception("Operation timeout")
        assert classify_ptz_error(exc) == CameraErrorCategory.TIMEOUT

    def test_timed_out_in_message(self):
        """Exception with 'timed out' in message should be TIMEOUT."""
        exc = Exception("Request timed out")
        assert classify_ptz_error(exc) == CameraErrorCategory.TIMEOUT

    def test_http_error_401(self):
        """HTTPError with 401 status code should be AUTH_FAILED."""
        exc = DummyHTTPError(401, "Unauthorized")
        assert classify_ptz_error(exc) == CameraErrorCategory.AUTH_FAILED

    def test_401_in_message(self):
        """Exception with '401' in message should be AUTH_FAILED."""
        exc = Exception("HTTP 401 error")
        assert classify_ptz_error(exc) == CameraErrorCategory.AUTH_FAILED

    def test_unauthorized_in_message(self):
        """Exception with 'unauthorized' in message should be AUTH_FAILED."""
        exc = Exception("Unauthorized request")
        assert classify_ptz_error(exc) == CameraErrorCategory.AUTH_FAILED

    def test_authentication_in_message(self):
        """Exception with 'authentication' in message should be AUTH_FAILED."""
        exc = Exception("Authentication required")
        assert classify_ptz_error(exc) == CameraErrorCategory.AUTH_FAILED

    def test_xml_parse_error(self):
        """ET.ParseError should be INVALID_XML."""
        exc = ET.ParseError("Invalid XML syntax")
        assert classify_ptz_error(exc) == CameraErrorCategory.INVALID_XML

    def test_xml_in_message(self):
        """Exception with 'xml' in message should be INVALID_XML."""
        exc = Exception("Invalid XML response")
        assert classify_ptz_error(exc) == CameraErrorCategory.INVALID_XML

    def test_parse_in_message(self):
        """Exception with 'parse' in message should be INVALID_XML."""
        exc = Exception("Failed to parse response")
        assert classify_ptz_error(exc) == CameraErrorCategory.INVALID_XML

    def test_unknown_error_defaults_to_api_error(self):
        """Unknown error should default to API_ERROR."""
        exc = Exception("Unknown PTZ error")
        assert classify_ptz_error(exc) == CameraErrorCategory.API_ERROR

    def test_empty_message(self):
        """Empty message should default to API_ERROR."""
        exc = Exception("")
        assert classify_ptz_error(exc) == CameraErrorCategory.API_ERROR

    def test_connection_error(self):
        """requests.exceptions.ConnectionError should default to API_ERROR."""
        exc = requests.exceptions.ConnectionError("Failed to connect")
        assert classify_ptz_error(exc) == CameraErrorCategory.API_ERROR


# =============================================================================
# Tests for get_screenshot_path
# =============================================================================


class TestGetScreenshotPath:
    """Tests for the get_screenshot_path function."""

    def test_returns_path_object(self, monkeypatch):
        """Should return a Path object."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.BASE_DIR = Path("/tmp/test_project")
        mock_settings.CAMERA_DIAGNOSTIC_SCREENSHOTS_DIR = Path("/tmp/screenshots")
        monkeypatch.setattr("webapp.camera_diagnostic.services.settings", mock_settings)

        # Mock mkdir to prevent filesystem operations
        monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: None)

        result = get_screenshot_path("TestCamera")
        assert isinstance(result, Path)

    def test_path_in_screenshots_dir(self, monkeypatch):
        """Path should be within the screenshots directory."""
        mock_settings = MagicMock()
        mock_settings.BASE_DIR = Path("/tmp/test_project")
        mock_settings.CAMERA_DIAGNOSTIC_SCREENSHOTS_DIR = Path("/tmp/screenshots")
        monkeypatch.setattr("webapp.camera_diagnostic.services.settings", mock_settings)
        monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: None)

        result = get_screenshot_path("TestCamera")
        assert str(result).startswith("/tmp/screenshots")

    def test_filename_contains_sanitized_camera_name(self, monkeypatch):
        """Filename should contain the sanitized camera name."""
        mock_settings = MagicMock()
        mock_settings.BASE_DIR = Path("/tmp/test_project")
        mock_settings.CAMERA_DIAGNOSTIC_SCREENSHOTS_DIR = Path("/tmp/screenshots")
        monkeypatch.setattr("webapp.camera_diagnostic.services.settings", mock_settings)
        monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: None)

        result = get_screenshot_path("My Camera")
        assert "My_Camera" in result.name

    def test_filename_contains_webui_suffix(self, monkeypatch):
        """Filename should contain 'webui' suffix."""
        mock_settings = MagicMock()
        mock_settings.BASE_DIR = Path("/tmp/test_project")
        mock_settings.CAMERA_DIAGNOSTIC_SCREENSHOTS_DIR = Path("/tmp/screenshots")
        monkeypatch.setattr("webapp.camera_diagnostic.services.settings", mock_settings)
        monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: None)

        result = get_screenshot_path("TestCamera")
        assert "_webui_" in result.name

    def test_filename_contains_timestamp_pattern(self, monkeypatch):
        """Filename should contain timestamp pattern (YYYYMMDD_HHMMSS)."""
        import re

        mock_settings = MagicMock()
        mock_settings.BASE_DIR = Path("/tmp/test_project")
        mock_settings.CAMERA_DIAGNOSTIC_SCREENSHOTS_DIR = Path("/tmp/screenshots")
        monkeypatch.setattr("webapp.camera_diagnostic.services.settings", mock_settings)
        monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: None)

        result = get_screenshot_path("TestCamera")
        # Pattern: YYYYMMDD_HHMMSS
        timestamp_pattern = r"\d{8}_\d{6}"
        assert re.search(timestamp_pattern, result.name)

    def test_filename_has_png_extension(self, monkeypatch):
        """Filename should have .png extension."""
        mock_settings = MagicMock()
        mock_settings.BASE_DIR = Path("/tmp/test_project")
        mock_settings.CAMERA_DIAGNOSTIC_SCREENSHOTS_DIR = Path("/tmp/screenshots")
        monkeypatch.setattr("webapp.camera_diagnostic.services.settings", mock_settings)
        monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: None)

        result = get_screenshot_path("TestCamera")
        assert result.suffix == ".png"

    def test_creates_directory_if_not_exists(self, monkeypatch):
        """Should call mkdir with parents=True and exist_ok=True."""
        mock_settings = MagicMock()
        mock_settings.BASE_DIR = Path("/tmp/test_project")
        mock_settings.CAMERA_DIAGNOSTIC_SCREENSHOTS_DIR = Path("/tmp/screenshots")
        monkeypatch.setattr("webapp.camera_diagnostic.services.settings", mock_settings)

        mkdir_calls = []

        def mock_mkdir(self, **kwargs):
            mkdir_calls.append({"path": self, "kwargs": kwargs})

        monkeypatch.setattr(Path, "mkdir", mock_mkdir)

        get_screenshot_path("TestCamera")
        assert len(mkdir_calls) == 1
        assert mkdir_calls[0]["kwargs"].get("parents") is True
        assert mkdir_calls[0]["kwargs"].get("exist_ok") is True

    def test_uses_default_dir_when_setting_not_defined(self, monkeypatch):
        """Should use default directory when CAMERA_DIAGNOSTIC_SCREENSHOTS_DIR not set."""
        mock_settings = MagicMock(spec=["BASE_DIR"])
        mock_settings.BASE_DIR = Path("/tmp/test_project")
        # Simulate attribute not existing
        del mock_settings.CAMERA_DIAGNOSTIC_SCREENSHOTS_DIR
        monkeypatch.setattr("webapp.camera_diagnostic.services.settings", mock_settings)
        monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: None)

        result = get_screenshot_path("TestCamera")
        assert "diagnostic_screenshots" in str(result)

    def test_special_chars_in_camera_name_sanitized(self, monkeypatch):
        """Special characters in camera name should be sanitized in filename."""
        mock_settings = MagicMock()
        mock_settings.BASE_DIR = Path("/tmp/test_project")
        mock_settings.CAMERA_DIAGNOSTIC_SCREENSHOTS_DIR = Path("/tmp/screenshots")
        monkeypatch.setattr("webapp.camera_diagnostic.services.settings", mock_settings)
        monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: None)

        result = get_screenshot_path("../etc/passwd")
        # Should not contain path traversal sequences
        assert ".." not in result.name
        assert "/" not in result.name


# =============================================================================
# Tests for get_presets_list
# =============================================================================


class TestGetPresetsList:
    """Tests for the get_presets_list function."""

    VALID_PRESETS_XML = """<?xml version="1.0" encoding="UTF-8"?>
    <PTZPresetList xmlns="http://www.hikvision.com/ver20/XMLSchema">
        <PTZPreset>
            <id>1</id>
            <presetName>Entrance</presetName>
            <enabled>true</enabled>
        </PTZPreset>
        <PTZPreset>
            <id>2</id>
            <presetName>Parking</presetName>
            <enabled>false</enabled>
        </PTZPreset>
    </PTZPresetList>"""

    VALID_PRESETS_XML_NO_NAMESPACE = """<?xml version="1.0" encoding="UTF-8"?>
    <PTZPresetList>
        <PTZPreset>
            <id>1</id>
            <presetName>Entrance</presetName>
            <enabled>true</enabled>
        </PTZPreset>
    </PTZPresetList>"""

    def test_successful_response_returns_presets(self, monkeypatch):
        """Successful XML response should return parsed presets."""

        def mock_get(*args, **kwargs):
            return DummyResponse(200, self.VALID_PRESETS_XML)

        monkeypatch.setattr("requests.get", mock_get)

        result = get_presets_list("192.168.1.100", "admin", "password")

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["presets"]) == 2
        assert result["presets"][0]["id"] == "1"
        assert result["presets"][0]["name"] == "Entrance"
        assert result["presets"][0]["enabled"] is True
        assert result["error"] is None

    def test_successful_response_without_namespace(self, monkeypatch):
        """Should parse XML without namespace."""

        def mock_get(*args, **kwargs):
            return DummyResponse(200, self.VALID_PRESETS_XML_NO_NAMESPACE)

        monkeypatch.setattr("requests.get", mock_get)

        result = get_presets_list("192.168.1.100", "admin", "password")

        assert result["success"] is True
        assert result["count"] == 1
        assert result["presets"][0]["id"] == "1"

    def test_response_time_recorded(self, monkeypatch):
        """Response time should be recorded in milliseconds."""

        def mock_get(*args, **kwargs):
            return DummyResponse(200, self.VALID_PRESETS_XML)

        monkeypatch.setattr("requests.get", mock_get)

        result = get_presets_list("192.168.1.100", "admin", "password")

        assert result["response_time_ms"] is not None
        assert isinstance(result["response_time_ms"], float)

    def test_401_response_returns_auth_failed(self, monkeypatch):
        """401 response should return AUTH_FAILED error."""

        def mock_get(*args, **kwargs):
            return DummyResponse(401, "Unauthorized")

        monkeypatch.setattr("requests.get", mock_get)

        result = get_presets_list("192.168.1.100", "admin", "wrong_password")

        assert result["success"] is False
        assert result["error"] == "Authentication failed"
        assert result["error_category"] == CameraErrorCategory.AUTH_FAILED.value

    def test_non_200_response_returns_api_error(self, monkeypatch):
        """Non-200/401 response should return API_ERROR."""

        def mock_get(*args, **kwargs):
            return DummyResponse(500, "Internal Server Error")

        monkeypatch.setattr("requests.get", mock_get)

        result = get_presets_list("192.168.1.100", "admin", "password")

        assert result["success"] is False
        assert "HTTP 500" in result["error"]
        assert result["error_category"] == CameraErrorCategory.API_ERROR.value

    def test_timeout_returns_timeout_error(self, monkeypatch):
        """requests.exceptions.Timeout should return TIMEOUT error."""

        def mock_get(*args, **kwargs):
            raise requests.exceptions.Timeout("Connection timed out")

        monkeypatch.setattr("requests.get", mock_get)

        result = get_presets_list("192.168.1.100", "admin", "password")

        assert result["success"] is False
        assert result["error"] == "Network timeout"
        assert result["error_category"] == CameraErrorCategory.TIMEOUT.value

    def test_invalid_xml_returns_invalid_xml_error(self, monkeypatch):
        """Invalid XML response should return INVALID_XML error."""

        def mock_get(*args, **kwargs):
            return DummyResponse(200, "This is not valid XML <<>>")

        monkeypatch.setattr("requests.get", mock_get)

        result = get_presets_list("192.168.1.100", "admin", "password")

        assert result["success"] is False
        assert "Invalid XML" in result["error"]
        assert result["error_category"] == CameraErrorCategory.INVALID_XML.value

    def test_connection_error_returns_api_error(self, monkeypatch):
        """requests.exceptions.ConnectionError should return API_ERROR."""

        def mock_get(*args, **kwargs):
            raise requests.exceptions.ConnectionError("Failed to connect")

        monkeypatch.setattr("requests.get", mock_get)

        result = get_presets_list("192.168.1.100", "admin", "password")

        assert result["success"] is False
        assert result["error_category"] == CameraErrorCategory.API_ERROR.value

    def test_empty_presets_list(self, monkeypatch):
        """Empty presets list should return success with count 0."""
        empty_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <PTZPresetList xmlns="http://www.hikvision.com/ver20/XMLSchema">
        </PTZPresetList>"""

        def mock_get(*args, **kwargs):
            return DummyResponse(200, empty_xml)

        monkeypatch.setattr("requests.get", mock_get)

        result = get_presets_list("192.168.1.100", "admin", "password")

        assert result["success"] is True
        assert result["count"] == 0
        assert result["presets"] == []

    def test_uses_digest_auth(self, monkeypatch):
        """Should use HTTPDigestAuth for authentication."""
        captured = {}

        def mock_get(*args, **kwargs):
            captured["auth"] = kwargs.get("auth")
            return DummyResponse(200, self.VALID_PRESETS_XML)

        monkeypatch.setattr("requests.get", mock_get)

        get_presets_list("192.168.1.100", "admin", "password")

        from requests.auth import HTTPDigestAuth

        assert captured["auth"] is not None
        assert isinstance(captured["auth"], HTTPDigestAuth)

    def test_uses_correct_url(self, monkeypatch):
        """Should use correct ISAPI URL."""
        captured = {}

        def mock_get(url, *args, **kwargs):
            captured["url"] = url
            return DummyResponse(200, self.VALID_PRESETS_XML)

        monkeypatch.setattr("requests.get", mock_get)

        get_presets_list("192.168.1.100", "admin", "password")

        assert captured["url"] == "http://192.168.1.100/ISAPI/PTZCtrl/channels/1/presets"

    def test_uses_timeout(self, monkeypatch):
        """Should use timeout for request."""
        captured = {}

        def mock_get(*args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return DummyResponse(200, self.VALID_PRESETS_XML)

        monkeypatch.setattr("requests.get", mock_get)

        get_presets_list("192.168.1.100", "admin", "password")

        assert captured["timeout"] is not None
        assert captured["timeout"] > 0

    def test_partial_preset_data(self, monkeypatch):
        """Presets with missing fields should still be parsed."""
        partial_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <PTZPresetList xmlns="http://www.hikvision.com/ver20/XMLSchema">
            <PTZPreset>
                <id>1</id>
            </PTZPreset>
        </PTZPresetList>"""

        def mock_get(*args, **kwargs):
            return DummyResponse(200, partial_xml)

        monkeypatch.setattr("requests.get", mock_get)

        result = get_presets_list("192.168.1.100", "admin", "password")

        assert result["success"] is True
        assert result["count"] == 1
        assert result["presets"][0]["id"] == "1"
        assert "name" not in result["presets"][0]
        assert "enabled" not in result["presets"][0]


# =============================================================================
# Tests for CameraErrorCategory Enum
# =============================================================================


class TestCameraErrorCategory:
    """Tests for the CameraErrorCategory enum."""

    def test_all_categories_have_string_values(self):
        """All categories should have string values."""
        for category in CameraErrorCategory:
            assert isinstance(category.value, str)

    def test_expected_categories_exist(self):
        """All expected categories should exist."""
        expected = [
            "NETWORK_UNREACHABLE",
            "AUTH_FAILED",
            "TIMEOUT",
            "INVALID_RESPONSE",
            "CAMERA_OFFLINE",
            "CREDENTIALS_NOT_SET",
            "CAMERA_NOT_FOUND",
            "INVALID_XML",
            "API_ERROR",
        ]
        for name in expected:
            assert hasattr(CameraErrorCategory, name)

    def test_categories_are_unique(self):
        """All category values should be unique."""
        values = [c.value for c in CameraErrorCategory]
        assert len(values) == len(set(values))
