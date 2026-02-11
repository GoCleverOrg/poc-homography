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
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock

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
        exc = OSError("Connection refused")
        assert classify_rtsp_error(exc) == CameraErrorCategory.NETWORK_UNREACHABLE

    def test_socket_error_no_route_to_host(self):
        """Socket error with 'no route to host' should be NETWORK_UNREACHABLE."""
        exc = OSError("No route to host")
        assert classify_rtsp_error(exc) == CameraErrorCategory.NETWORK_UNREACHABLE

    def test_socket_error_timed_out(self):
        """Socket error with 'timed out' should be TIMEOUT."""
        exc = OSError("Connection timed out")
        assert classify_rtsp_error(exc) == CameraErrorCategory.TIMEOUT

    def test_socket_error_timeout(self):
        """Socket error with 'timeout' should be TIMEOUT."""
        exc = OSError("Socket timeout")
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


# =============================================================================
# Tests for Stress Test Models
# =============================================================================

from datetime import datetime, timezone

from webapp.camera_diagnostic.models import (
    STRESS_TEST_PRESETS,
    AxisMovementConfig,
    MovementTiming,
    StressTestConfig,
    StressTestProgress,
    StressTestResult,
    StressTestSession,
    StressTestStatus,
    StressTestType,
    UserEvaluation,
)


class TestStressTestType:
    """Tests for the StressTestType enum."""

    def test_all_types_exist(self):
        """All expected test types should exist."""
        expected = [
            "OSCILLATION",
            "RANDOM_STEP_ACCURACY",
            "FULL_RANGE_SWEEP",
            "TILT_STRESS",
            "COMBINED_AXIS_LOAD",
            "POSITION_REPEATABILITY",
            "SPEED_TEST",
        ]
        for name in expected:
            assert hasattr(StressTestType, name)

    def test_type_values_are_strings(self):
        """All type values should be lowercase strings."""
        for test_type in StressTestType:
            assert isinstance(test_type.value, str)
            assert test_type.value.islower() or "_" in test_type.value


class TestStressTestStatus:
    """Tests for the StressTestStatus enum."""

    def test_all_statuses_exist(self):
        """All expected statuses should exist."""
        expected = ["PENDING", "RUNNING", "COMPLETED", "ABORTED", "FAILED"]
        for name in expected:
            assert hasattr(StressTestStatus, name)

    def test_status_values_are_strings(self):
        """All status values should be lowercase strings."""
        for status in StressTestStatus:
            assert isinstance(status.value, str)


class TestUserEvaluation:
    """Tests for the UserEvaluation enum."""

    def test_all_evaluations_exist(self):
        """All expected evaluations should exist."""
        expected = ["GOOD", "NEEDS_IMPROVEMENT", "BAD", "NOT_EVALUATED"]
        for name in expected:
            assert hasattr(UserEvaluation, name)


class TestAxisMovementConfig:
    """Tests for the AxisMovementConfig dataclass."""

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        config = AxisMovementConfig(
            axis="pan",
            start=0.0,
            end=90.0,
            step=10.0,
            step_min=5.0,
            step_max=15.0,
            use_random_steps=True,
        )
        data = config.to_dict()

        assert data["axis"] == "pan"
        assert data["start"] == 0.0
        assert data["end"] == 90.0
        assert data["step"] == 10.0
        assert data["step_min"] == 5.0
        assert data["step_max"] == 15.0
        assert data["use_random_steps"] is True

    def test_from_dict(self):
        """from_dict should deserialize all fields."""
        data = {
            "axis": "tilt",
            "start": -15.0,
            "end": 90.0,
            "step": 20.0,
            "step_min": 10.0,
            "step_max": 25.0,
            "use_random_steps": False,
        }
        config = AxisMovementConfig.from_dict(data)

        assert config.axis == "tilt"
        assert config.start == -15.0
        assert config.end == 90.0
        assert config.step == 20.0
        assert config.step_min == 10.0
        assert config.step_max == 25.0
        assert config.use_random_steps is False

    def test_from_dict_with_defaults(self):
        """from_dict should use defaults for missing optional fields."""
        data = {"axis": "pan", "start": 0.0, "end": 45.0}
        config = AxisMovementConfig.from_dict(data)

        assert config.step == 10.0
        assert config.step_min == 5.0
        assert config.step_max == 15.0
        assert config.use_random_steps is False

    def test_roundtrip(self):
        """to_dict -> from_dict should preserve all values."""
        original = AxisMovementConfig(
            axis="zoom",
            start=1.0,
            end=10.0,
            step=1.0,
            step_min=0.5,
            step_max=2.0,
            use_random_steps=True,
        )
        restored = AxisMovementConfig.from_dict(original.to_dict())

        assert restored.axis == original.axis
        assert restored.start == original.start
        assert restored.end == original.end
        assert restored.step == original.step
        assert restored.step_min == original.step_min
        assert restored.step_max == original.step_max
        assert restored.use_random_steps == original.use_random_steps


class TestMovementTiming:
    """Tests for the MovementTiming dataclass."""

    def test_to_dict(self):
        """to_dict should serialize all fields including datetime."""
        now = datetime.now(timezone.utc)
        later = datetime.now(timezone.utc)
        timing = MovementTiming(
            command_sent=now,
            stabilized=later,
            duration_ms=1500.5,
            start_position={"pan": 0.0, "tilt": 0.0, "zoom": 1.0},
            end_position={"pan": 45.0, "tilt": 30.0, "zoom": 1.0},
            target_position={"pan": 45.0, "tilt": 30.0, "zoom": 1.0},
            position_error={"pan": 0.1, "tilt": 0.2, "zoom": 0.0},
        )
        data = timing.to_dict()

        assert data["command_sent"] == now.isoformat()
        assert data["stabilized"] == later.isoformat()
        assert data["duration_ms"] == 1500.5
        assert data["start_position"]["pan"] == 0.0
        assert data["end_position"]["pan"] == 45.0
        assert data["position_error"]["pan"] == 0.1

    def test_from_dict(self):
        """from_dict should deserialize including datetime parsing."""
        now = datetime.now(timezone.utc)
        data = {
            "command_sent": now.isoformat(),
            "stabilized": now.isoformat(),
            "duration_ms": 2000.0,
            "start_position": {"pan": 10.0, "tilt": 20.0, "zoom": 2.0},
            "end_position": {"pan": 50.0, "tilt": 60.0, "zoom": 2.0},
            "target_position": {"pan": 50.0, "tilt": 60.0, "zoom": 2.0},
            "position_error": {"pan": 0.0, "tilt": 0.0, "zoom": 0.0},
        }
        timing = MovementTiming.from_dict(data)

        assert isinstance(timing.command_sent, datetime)
        assert isinstance(timing.stabilized, datetime)
        assert timing.duration_ms == 2000.0
        assert timing.start_position["pan"] == 10.0


class TestStressTestConfig:
    """Tests for the StressTestConfig dataclass."""

    def test_to_dict(self):
        """to_dict should serialize config with nested objects."""
        config = StressTestConfig(
            tenant_id="tenant1",
            camera_id="cam1",
            test_type=StressTestType.OSCILLATION,
            pan_config=AxisMovementConfig(axis="pan", start=0.0, end=10.0),
            repetitions=5,
        )
        data = config.to_dict()

        assert data["tenant_id"] == "tenant1"
        assert data["camera_id"] == "cam1"
        assert data["test_type"] == "oscillation"
        assert data["pan_config"]["axis"] == "pan"
        assert data["tilt_config"] is None
        assert data["repetitions"] == 5

    def test_from_dict(self):
        """from_dict should deserialize with nested objects."""
        data = {
            "tenant_id": "t2",
            "camera_id": "c2",
            "test_type": "full_range_sweep",
            "pan_config": {"axis": "pan", "start": 0.0, "end": 360.0},
            "tilt_config": None,
            "zoom_config": None,
            "repetitions": 3,
        }
        config = StressTestConfig.from_dict(data)

        assert config.tenant_id == "t2"
        assert config.test_type == StressTestType.FULL_RANGE_SWEEP
        assert config.pan_config is not None
        assert config.pan_config.end == 360.0
        assert config.tilt_config is None


class TestStressTestResult:
    """Tests for the StressTestResult dataclass."""

    def test_to_dict(self):
        """to_dict should serialize result including movements list."""
        result = StressTestResult(
            success=True,
            position_match=True,
            position_error={"pan": 0.1, "tilt": 0.2, "zoom": 0.0},
            total_duration_ms=5000.0,
            movements=[],
        )
        data = result.to_dict()

        assert data["success"] is True
        assert data["position_match"] is True
        assert data["position_error"]["pan"] == 0.1
        assert data["total_duration_ms"] == 5000.0
        assert data["movements"] == []

    def test_from_dict_with_error(self):
        """from_dict should handle error message."""
        data = {
            "success": False,
            "position_match": False,
            "position_error": {"pan": 0, "tilt": 0, "zoom": 0},
            "total_duration_ms": 1000.0,
            "movements": [],
            "error_message": "Connection timeout",
        }
        result = StressTestResult.from_dict(data)

        assert result.success is False
        assert result.error_message == "Connection timeout"


class TestStressTestSession:
    """Tests for the StressTestSession dataclass."""

    def test_to_dict(self):
        """to_dict should serialize complete session."""
        now = datetime.now(timezone.utc)
        session = StressTestSession(
            id="test-uuid",
            created_at=now,
            status=StressTestStatus.COMPLETED,
            tenant_id="tenant1",
            camera_id="cam1",
            camera_name="Camera 1",
            user_evaluation=UserEvaluation.GOOD,
            user_notes="Test notes",
        )
        data = session.to_dict()

        assert data["id"] == "test-uuid"
        assert data["created_at"] == now.isoformat()
        assert data["status"] == "completed"
        assert data["user_evaluation"] == "good"
        assert data["user_notes"] == "Test notes"

    def test_from_dict(self):
        """from_dict should deserialize complete session."""
        now = datetime.now(timezone.utc)
        data = {
            "id": "session-123",
            "created_at": now.isoformat(),
            "started_at": now.isoformat(),
            "completed_at": now.isoformat(),
            "status": "completed",
            "tenant_id": "t1",
            "camera_id": "c1",
            "camera_name": "Test Camera",
            "config": {
                "tenant_id": "t1",
                "camera_id": "c1",
                "test_type": "oscillation",
            },
            "result": {
                "success": True,
                "position_match": True,
                "position_error": {"pan": 0, "tilt": 0, "zoom": 0},
                "total_duration_ms": 10000.0,
            },
            "user_evaluation": "needs_improvement",
            "user_notes": "Some drift observed",
        }
        session = StressTestSession.from_dict(data)

        assert session.id == "session-123"
        assert session.status == StressTestStatus.COMPLETED
        assert session.config is not None
        assert session.result is not None
        assert session.user_evaluation == UserEvaluation.NEEDS_IMPROVEMENT

    def test_roundtrip(self):
        """to_dict -> from_dict should preserve all values."""
        now = datetime.now(timezone.utc)
        original = StressTestSession(
            id="test-roundtrip",
            created_at=now,
            started_at=now,
            completed_at=now,
            status=StressTestStatus.COMPLETED,
            tenant_id="tenant",
            camera_id="camera",
            camera_name="Test",
            config=StressTestConfig(
                tenant_id="tenant",
                camera_id="camera",
                test_type=StressTestType.OSCILLATION,
            ),
            result=StressTestResult(
                success=True,
                position_match=True,
                position_error={"pan": 0.1, "tilt": 0.1, "zoom": 0.0},
                total_duration_ms=5000.0,
            ),
            user_evaluation=UserEvaluation.GOOD,
        )
        restored = StressTestSession.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.status == original.status
        assert restored.config.test_type == original.config.test_type
        assert restored.result.success == original.result.success
        assert restored.user_evaluation == original.user_evaluation


class TestStressTestProgress:
    """Tests for the StressTestProgress dataclass."""

    def test_to_dict(self):
        """to_dict should serialize progress state."""
        progress = StressTestProgress(
            session_id="progress-123",
            status=StressTestStatus.RUNNING,
            current_repetition=3,
            total_repetitions=10,
            current_movement=5,
            total_movements=20,
            current_position={"pan": 45.0, "tilt": 30.0, "zoom": 1.0},
            message="Repetition 3/10",
        )
        data = progress.to_dict()

        assert data["session_id"] == "progress-123"
        assert data["status"] == "running"
        assert data["current_repetition"] == 3
        assert data["total_repetitions"] == 10
        assert data["current_position"]["pan"] == 45.0
        assert data["message"] == "Repetition 3/10"


class TestStressTestPresets:
    """Tests for the built-in stress test presets."""

    def test_presets_list_not_empty(self):
        """Should have at least one preset."""
        assert len(STRESS_TEST_PRESETS) > 0

    def test_all_presets_have_required_fields(self):
        """All presets should have name, description, and test_type."""
        for preset in STRESS_TEST_PRESETS:
            assert preset.name
            assert preset.description
            assert isinstance(preset.test_type, StressTestType)

    def test_all_presets_serializable(self):
        """All presets should be serializable to dict."""
        for preset in STRESS_TEST_PRESETS:
            data = preset.to_dict()
            assert "name" in data
            assert "description" in data
            assert "test_type" in data
            assert "repetitions" in data

    def test_preset_oscillation_pan_exists(self):
        """Oscillation Pan preset should exist with correct config."""
        preset = next(
            (p for p in STRESS_TEST_PRESETS if "Oscillation" in p.name and "Pan" in p.name),
            None,
        )
        assert preset is not None
        assert preset.test_type == StressTestType.OSCILLATION
        assert preset.pan_config is not None
        assert preset.repetitions > 0

    def test_preset_full_range_sweep_exists(self):
        """Full Range Sweep preset should exist."""
        preset = next(
            (p for p in STRESS_TEST_PRESETS if "Full Range" in p.name),
            None,
        )
        assert preset is not None
        assert preset.test_type == StressTestType.FULL_RANGE_SWEEP

    def test_preset_combined_axis_has_both_configs(self):
        """Combined Axis preset should have both pan and tilt configs."""
        preset = next(
            (p for p in STRESS_TEST_PRESETS if "Combined" in p.name),
            None,
        )
        assert preset is not None
        assert preset.pan_config is not None
        assert preset.tilt_config is not None


# =============================================================================
# Tests for CameraStressTestService
# =============================================================================

from webapp.camera_diagnostic.services import (
    CameraStressTestService,
    get_stress_test_storage_dir,
)


class TestCameraStressTestService:
    """Tests for the CameraStressTestService class."""

    def test_start_stress_test_camera_not_found(self, monkeypatch):
        """Should return error when camera not found."""
        monkeypatch.setattr(
            "webapp.camera_diagnostic.services.get_camera_by_id",
            lambda x: None,
        )

        config = StressTestConfig(
            tenant_id="t1",
            camera_id="nonexistent",
            test_type=StressTestType.OSCILLATION,
        )
        session_id, error = CameraStressTestService.start_stress_test(config)

        assert session_id is None
        assert "not found" in error.lower()

    def test_start_stress_test_no_ip(self, monkeypatch):
        """Should return error when camera has no IP."""
        monkeypatch.setattr(
            "webapp.camera_diagnostic.services.get_camera_by_id",
            lambda x: {"name": "test"},  # No IP
        )

        config = StressTestConfig(
            tenant_id="t1",
            camera_id="no_ip_cam",
            test_type=StressTestType.OSCILLATION,
        )
        session_id, error = CameraStressTestService.start_stress_test(config)

        assert session_id is None
        assert "ip" in error.lower()

    def test_start_stress_test_no_credentials(self, monkeypatch):
        """Should return error when credentials not set."""
        monkeypatch.setattr(
            "webapp.camera_diagnostic.services.get_camera_by_id",
            lambda x: {"name": "test", "ip": "192.168.1.100"},
        )
        monkeypatch.setenv("CAMERA_USERNAME", "")
        monkeypatch.setenv("CAMERA_PASSWORD", "")

        config = StressTestConfig(
            tenant_id="t1",
            camera_id="cam1",
            test_type=StressTestType.OSCILLATION,
        )
        session_id, error = CameraStressTestService.start_stress_test(config)

        assert session_id is None
        assert "credentials" in error.lower()

    def test_positions_match_within_tolerance(self):
        """Should return True when positions are within tolerance."""
        pos1 = {"pan": 45.0, "tilt": 30.0, "zoom": 1.0}
        pos2 = {"pan": 45.3, "tilt": 30.2, "zoom": 1.1}

        # Default tolerance is 0.5
        assert CameraStressTestService._positions_match(pos1, pos2) is True

    def test_positions_match_outside_tolerance(self):
        """Should return False when positions differ more than tolerance."""
        pos1 = {"pan": 45.0, "tilt": 30.0, "zoom": 1.0}
        pos2 = {"pan": 46.0, "tilt": 30.0, "zoom": 1.0}

        # Difference of 1.0 exceeds default 0.5 tolerance
        assert CameraStressTestService._positions_match(pos1, pos2) is False

    def test_generate_random_steps_sum_equals_total(self):
        """Generated steps should sum to approximately the total."""
        total = 90.0
        steps = CameraStressTestService._generate_random_steps(total, 5.0, 15.0)

        assert len(steps) > 0
        assert abs(sum(steps) - total) < 0.01  # Within floating point tolerance

    def test_generate_random_steps_negative_total(self):
        """Should handle negative total (reverse direction)."""
        total = -45.0
        steps = CameraStressTestService._generate_random_steps(total, 5.0, 15.0)

        assert len(steps) > 0
        assert all(s < 0 for s in steps)  # All steps should be negative
        assert abs(sum(steps) - total) < 0.01

    def test_generate_sweep_waypoints_small_range(self):
        """Small range should return single waypoint."""
        waypoints = CameraStressTestService._generate_sweep_waypoints(0.0, 45.0, 90.0)

        assert len(waypoints) == 1
        assert waypoints[0] == 45.0

    def test_generate_sweep_waypoints_large_range(self):
        """Large range should be broken into multiple waypoints."""
        waypoints = CameraStressTestService._generate_sweep_waypoints(0.0, 360.0, 90.0)

        assert len(waypoints) == 4  # 360 / 90 = 4 waypoints
        assert waypoints[-1] == 360.0

    def test_abort_nonexistent_session(self):
        """Should return error when aborting nonexistent session."""
        success, error = CameraStressTestService.abort_stress_test("nonexistent-id")

        assert success is False
        assert "not found" in error.lower()

    def test_get_status_nonexistent_session(self):
        """Should return None for nonexistent session."""
        progress = CameraStressTestService.get_stress_test_status("nonexistent-id")

        assert progress is None

    def test_delete_invalid_session_id(self):
        """Should return error for invalid session ID format."""
        success, error = CameraStressTestService.delete_stress_test_session("not-a-uuid")

        assert success is False
        assert "invalid" in error.lower()

    def test_delete_nonexistent_session(self):
        """Should return error when deleting nonexistent session."""
        import uuid

        valid_uuid = str(uuid.uuid4())
        success, error = CameraStressTestService.delete_stress_test_session(valid_uuid)

        assert success is False
        assert "not found" in error.lower()


class TestStressTestStorageDir:
    """Tests for stress test storage directory function."""

    def test_get_stress_test_storage_dir_returns_path(self, monkeypatch):
        """Should return a Path object."""
        mock_settings = MagicMock()
        mock_settings.BASE_DIR = Path("/tmp/test_project")
        monkeypatch.setattr("webapp.camera_diagnostic.services.settings", mock_settings)

        result = get_stress_test_storage_dir()

        assert isinstance(result, Path)
        assert "stress_test" in str(result)

    def test_storage_dir_in_data_directory(self, monkeypatch):
        """Storage dir should be in data/stress_test."""
        mock_settings = MagicMock()
        mock_settings.BASE_DIR = Path("/app/webapp")
        monkeypatch.setattr("webapp.camera_diagnostic.services.settings", mock_settings)

        result = get_stress_test_storage_dir()

        assert str(result).endswith("data/stress_test")
