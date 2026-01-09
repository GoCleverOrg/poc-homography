"""
Tests for test_data_generator tool.

This module tests the standalone test data generator tool that captures
camera frames and generates test data for calibration with interactive
GCP marking.
"""

import json
import sys
import tempfile
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

    def test_parse_args_with_map_points_path(self):
        """Test parsing optional --map-points path."""
        from tools.test_data_generator import parse_arguments

        args = parse_arguments(["Valte", "--map-points", "/custom/map_points.json"])

        assert args.camera_name == "Valte"
        assert args.map_points == "/custom/map_points.json"

    def test_parse_args_map_points_defaults_to_map_points_json(self):
        """Test that --map-points defaults to 'map_points.json'."""
        from tools.test_data_generator import parse_arguments

        args = parse_arguments(["Valte"])

        assert args.map_points == "map_points.json"

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


# Test Task 3: Map Point Loading
class TestMapPointLoading:
    """Test loading map points from JSON file."""

    def test_load_map_points_from_valid_file(self):
        """Test loading map points from a valid JSON file."""
        from tools.test_data_generator import load_map_points

        # Create temporary map points file
        map_data = {
            "map_id": "map_valte",
            "points": [
                {"id": "Z1", "pixel_x": 100.0, "pixel_y": 200.0, "map_id": "map_valte"},
                {"id": "Z2", "pixel_x": 150.0, "pixel_y": 250.0, "map_id": "map_valte"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(map_data, f)
            temp_path = f.name

        try:
            registry = load_map_points(temp_path)

            assert registry.map_id == "map_valte"
            assert len(registry.points) == 2
            assert "Z1" in registry.points
            assert "Z2" in registry.points
            assert registry.points["Z1"].pixel_x == 100.0
            assert registry.points["Z1"].pixel_y == 200.0
        finally:
            Path(temp_path).unlink()

    def test_load_map_points_file_not_found_raises_error(self):
        """Test that loading from non-existent file raises FileNotFoundError."""
        from tools.test_data_generator import load_map_points

        with pytest.raises(FileNotFoundError):
            load_map_points("/nonexistent/path/map_points.json")

    def test_load_map_points_invalid_json_raises_error(self):
        """Test that loading invalid JSON raises JSONDecodeError."""
        from tools.test_data_generator import load_map_points

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json {")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                load_map_points(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_map_points_converts_to_list_format(self):
        """Test that loaded map points are converted to list format for web interface."""
        from tools.test_data_generator import load_map_points, convert_map_points_to_list

        map_data = {
            "map_id": "map_valte",
            "points": [
                {"id": "Z1", "pixel_x": 100.0, "pixel_y": 200.0, "map_id": "map_valte"},
                {"id": "Z2", "pixel_x": 150.0, "pixel_y": 250.0, "map_id": "map_valte"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(map_data, f)
            temp_path = f.name

        try:
            registry = load_map_points(temp_path)
            map_points_list = convert_map_points_to_list(registry)

            assert isinstance(map_points_list, list)
            assert len(map_points_list) == 2
            assert map_points_list[0]["id"] == "Z1"
            assert map_points_list[0]["pixel_x"] == 100.0
            assert map_points_list[0]["pixel_y"] == 200.0
            assert map_points_list[0]["map_id"] == "map_valte"
        finally:
            Path(temp_path).unlink()


# Test Task 4: GCP Export with Map Point IDs
class TestGCPExportWithMapPoints:
    """Test GCP export with map point ID references."""

    def test_generate_json_output_with_map_point_ids(self):
        """Test generating JSON output with map_point_id in GCPs."""
        from tools.test_data_generator import generate_json_output

        camera_info = {
            "latitude": 39.640265,
            "longitude": -0.229972,
            "height_meters": 4.71,
            "pan_deg": 30.8,
            "tilt_deg": 13.1,
            "zoom_level": 1,
        }

        gcps = [
            {"pixel_x": 805.0, "pixel_y": 583.125, "map_point_id": "Z1"},
            {"pixel_x": 1083.0, "pixel_y": 392.125, "map_point_id": "Z2"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            result = generate_json_output(
                camera_info=camera_info,
                gcps=gcps,
                camera_name="Valte",
                output_path=output_path,
            )

            assert result["json_path"] == output_path

            # Verify JSON content
            with open(output_path, "r") as f:
                data = json.load(f)

            assert data["camera_info"] == camera_info
            assert len(data["gcps"]) == 2
            assert data["gcps"][0]["map_point_id"] == "Z1"
            assert data["gcps"][1]["map_point_id"] == "Z2"
            assert data["gcps"][0]["pixel_x"] == 805.0
            assert data["gcps"][0]["pixel_y"] == 583.125
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_generate_json_output_preserves_backward_compatibility_with_lat_lon(self):
        """Test that GCPs with lat/lon (legacy format) still work."""
        from tools.test_data_generator import generate_json_output

        camera_info = {
            "latitude": 39.640265,
            "longitude": -0.229972,
            "height_meters": 4.71,
            "pan_deg": 30.8,
            "tilt_deg": 13.1,
            "zoom_level": 1,
        }

        # Legacy format with lat/lon
        gcps = [
            {
                "pixel_x": 805.0,
                "pixel_y": 583.125,
                "latitude": 39.64050439015013,
                "longitude": -0.22998479032275593,
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            result = generate_json_output(
                camera_info=camera_info,
                gcps=gcps,
                camera_name="Valte",
                output_path=output_path,
            )

            # Verify JSON content preserves lat/lon
            with open(output_path, "r") as f:
                data = json.load(f)

            assert len(data["gcps"]) == 1
            assert data["gcps"][0]["latitude"] == 39.64050439015013
            assert data["gcps"][0]["longitude"] == -0.22998479032275593
        finally:
            Path(output_path).unlink(missing_ok=True)
