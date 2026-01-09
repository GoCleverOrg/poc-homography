"""
Tests for test_data_generator tool.

This module tests the standalone test data generator tool that captures
camera frames and generates test data for calibration with interactive
GCP marking.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# Test Task 1: CLI Argument Parsing
class TestCLIArgumentParsing:
    """Test CLI argument parsing and validation."""

    def test_parse_args_with_camera_name(self):
        """Test parsing camera name argument."""
        from tools.test_data_generator import parse_arguments

        args = parse_arguments(["Valte"])

        assert args.camera_name == "Valte"
        assert args.output is None
        assert args.list_cameras is False

    def test_parse_args_with_output_path(self):
        """Test parsing optional output path."""
        from tools.test_data_generator import parse_arguments

        args = parse_arguments(["Valte", "--output", "/tmp/test.json"])

        assert args.camera_name == "Valte"
        assert args.output == "/tmp/test.json"

    def test_parse_args_with_list_cameras_flag(self):
        """Test parsing --list-cameras flag."""
        from tools.test_data_generator import parse_arguments

        args = parse_arguments(["--list-cameras"])

        assert args.list_cameras is True

    def test_parse_args_missing_camera_name_without_list_flag_raises_error(self):
        """Test that missing camera name raises error when not using --list-cameras."""
        from tools.test_data_generator import parse_arguments

        with pytest.raises(SystemExit):
            # argparse raises SystemExit(2) on error
            parse_arguments([])

    def test_validate_camera_name_valid_camera(self):
        """Test validating a valid camera name."""
        from tools.test_data_generator import validate_camera_name

        mock_cameras = [
            {"name": "Valte", "ip": "10.207.99.178"},
            {"name": "Setram", "ip": "10.237.100.15"},
        ]

        # Should not raise exception
        validate_camera_name("Valte", mock_cameras)

    def test_validate_camera_name_invalid_camera_raises_error(self):
        """Test validating an invalid camera name raises ValueError."""
        from tools.test_data_generator import validate_camera_name

        mock_cameras = [
            {"name": "Valte", "ip": "10.207.99.178"},
            {"name": "Setram", "ip": "10.237.100.15"},
        ]

        with pytest.raises(ValueError, match="Camera 'InvalidCam' not found"):
            validate_camera_name("InvalidCam", mock_cameras)

    def test_validate_camera_name_error_message_includes_available_cameras(self):
        """Test error message includes list of available cameras."""
        from tools.test_data_generator import validate_camera_name

        mock_cameras = [
            {"name": "Valte", "ip": "10.207.99.178"},
            {"name": "Setram", "ip": "10.237.100.15"},
        ]

        with pytest.raises(ValueError, match="Available: Valte, Setram"):
            validate_camera_name("InvalidCam", mock_cameras)


# Test Task 2: GPS Coordinate Conversion
class TestGPSCoordinateConversion:
    """Test GPS coordinate conversion from DMS to decimal degrees."""

    def test_convert_dms_to_decimal_north_latitude(self):
        """Test converting North latitude from DMS to decimal degrees."""
        from tools.test_data_generator import convert_gps_coordinates

        result = convert_gps_coordinates("39°38'25.72\"N", "0°13'48.63\"W")

        # dms_to_dd already tested in codebase - we just validate it's called correctly
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert 39.0 < result[0] < 40.0  # Approximate latitude
        assert -1.0 < result[1] < 0.0  # Approximate longitude (West is negative)

    def test_convert_dms_to_decimal_south_latitude(self):
        """Test converting South latitude from DMS to decimal degrees."""
        from tools.test_data_generator import convert_gps_coordinates

        result = convert_gps_coordinates("10°30'15\"S", "5°15'30\"E")

        assert result[0] < 0  # South is negative
        assert result[1] > 0  # East is positive

    def test_convert_dms_to_decimal_validates_latitude_range(self):
        """Test that invalid latitude range raises ValueError."""

        # Create mock that returns out-of-range value
        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            # This would require mocking dms_to_dd, but based on clarification
            # we trust dms_to_dd. Let's test the validation wrapper instead.
            from tools.test_data_generator import validate_gps_ranges

            validate_gps_ranges(95.0, 0.0)

    def test_convert_dms_to_decimal_validates_longitude_range(self):
        """Test that invalid longitude range raises ValueError."""
        from tools.test_data_generator import validate_gps_ranges

        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            validate_gps_ranges(45.0, 185.0)

    def test_validate_gps_ranges_accepts_valid_coordinates(self):
        """Test that valid GPS ranges pass validation."""
        from tools.test_data_generator import validate_gps_ranges

        # Should not raise exception
        validate_gps_ranges(39.640477, -0.230175)
        validate_gps_ranges(-90.0, -180.0)  # Edge cases
        validate_gps_ranges(90.0, 180.0)  # Edge cases
        validate_gps_ranges(0.0, 0.0)  # Origin

    def test_validate_gps_ranges_boundary_values(self):
        """Test boundary validation for GPS coordinates."""
        from tools.test_data_generator import validate_gps_ranges

        # Test exact boundaries
        validate_gps_ranges(90.0, 0.0)  # Max latitude
        validate_gps_ranges(-90.0, 0.0)  # Min latitude
        validate_gps_ranges(0.0, 180.0)  # Max longitude
        validate_gps_ranges(0.0, -180.0)  # Min longitude

        # Test just outside boundaries
        with pytest.raises(ValueError):
            validate_gps_ranges(90.1, 0.0)

        with pytest.raises(ValueError):
            validate_gps_ranges(-90.1, 0.0)

        with pytest.raises(ValueError):
            validate_gps_ranges(0.0, 180.1)

        with pytest.raises(ValueError):
            validate_gps_ranges(0.0, -180.1)
