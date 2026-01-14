"""Tests for the GeoTIFF point picker application."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from poc_homography.map_points.map_point import MapPoint
from poc_homography.map_points.map_point_registry import MapPointRegistry
from poc_homography.point_picker.app import (
    ABBREVIATION_TO_TAG,
    TAG_ABBREVIATIONS,
    PointPickerState,
    _get_tag_from_id,
    normalize_array,
)


class TestTagAbbreviations:
    """Tests for tag abbreviation mappings."""

    def test_tag_abbreviations_defined(self) -> None:
        """All required tags have abbreviations."""
        assert "parking_spot" in TAG_ABBREVIATIONS
        assert "arrows" in TAG_ABBREVIATIONS
        assert "crosswalk" in TAG_ABBREVIATIONS
        assert "extra" in TAG_ABBREVIATIONS

    def test_abbreviation_values(self) -> None:
        """Abbreviations are correct."""
        assert TAG_ABBREVIATIONS["parking_spot"] == "PS"
        assert TAG_ABBREVIATIONS["arrows"] == "AR"
        assert TAG_ABBREVIATIONS["crosswalk"] == "CW"
        assert TAG_ABBREVIATIONS["extra"] == "EX"

    def test_reverse_mapping(self) -> None:
        """Reverse mapping works correctly."""
        assert ABBREVIATION_TO_TAG["PS"] == "parking_spot"
        assert ABBREVIATION_TO_TAG["AR"] == "arrows"
        assert ABBREVIATION_TO_TAG["CW"] == "crosswalk"
        assert ABBREVIATION_TO_TAG["EX"] == "extra"


class TestGetTagFromId:
    """Tests for _get_tag_from_id function."""

    def test_parking_spot_ids(self) -> None:
        """Parking spot IDs return correct tag."""
        assert _get_tag_from_id("PS1") == "parking_spot"
        assert _get_tag_from_id("PS42") == "parking_spot"
        assert _get_tag_from_id("PS999") == "parking_spot"

    def test_arrows_ids(self) -> None:
        """Arrows IDs return correct tag."""
        assert _get_tag_from_id("AR1") == "arrows"
        assert _get_tag_from_id("AR10") == "arrows"

    def test_crosswalk_ids(self) -> None:
        """Crosswalk IDs return correct tag."""
        assert _get_tag_from_id("CW1") == "crosswalk"
        assert _get_tag_from_id("CW5") == "crosswalk"

    def test_extra_ids(self) -> None:
        """Extra IDs return correct tag."""
        assert _get_tag_from_id("EX1") == "extra"
        assert _get_tag_from_id("EX100") == "extra"

    def test_unknown_prefix_returns_extra(self) -> None:
        """Unknown prefixes default to extra."""
        assert _get_tag_from_id("XY1") == "extra"
        assert _get_tag_from_id("Z99") == "extra"
        assert _get_tag_from_id("UNKNOWN5") == "extra"


class TestNormalizeArray:
    """Tests for normalize_array function."""

    def test_uint8_passthrough(self) -> None:
        """uint8 arrays pass through unchanged."""
        arr = np.array([[0, 128, 255]], dtype=np.uint8)
        result = normalize_array(arr)
        np.testing.assert_array_equal(result, arr)

    def test_float_normalization(self) -> None:
        """Float arrays are normalized to 0-255."""
        arr = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
        result = normalize_array(arr)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255
        # Middle value should be around 127-128
        assert 125 <= result[0, 1] <= 130

    def test_uint16_normalization(self) -> None:
        """uint16 arrays are normalized."""
        arr = np.array([[0, 32767, 65535]], dtype=np.uint16)
        result = normalize_array(arr)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_constant_array(self) -> None:
        """Constant arrays become zeros."""
        arr = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
        result = normalize_array(arr)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_negative_values(self) -> None:
        """Negative values are handled correctly."""
        arr = np.array([[-10.0, 0.0, 10.0]], dtype=np.float64)
        result = normalize_array(arr)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0  # min value
        assert result[0, 2] == 255  # max value


class MockTiffFile:
    """Mock TiffFile for testing without real TIFF files."""

    def __init__(self, width: int = 100, height: int = 100) -> None:
        self.pages = [MockTiffPage(width, height)]

    def __enter__(self) -> MockTiffFile:
        return self

    def __exit__(self, *args: object) -> None:
        pass


class MockTiffPage:
    """Mock TiffPage for testing."""

    def __init__(self, width: int = 100, height: int = 100) -> None:
        self.imagewidth = width
        self.imagelength = height
        self.tags = MockTiffTags()

    def asarray(self) -> np.ndarray:
        """Return mock image data."""
        return np.zeros((self.imagelength, self.imagewidth, 3), dtype=np.uint8)


class MockTiffTags:
    """Mock TiffTags for testing."""

    def values(self) -> list:
        return []


class TestPointPickerState:
    """Tests for PointPickerState class."""

    @pytest.fixture
    def mock_state(self, tmp_path: Path) -> PointPickerState:
        """Create a PointPickerState with mocked GeoTIFF."""
        geotiff_path = tmp_path / "test.tif"
        geotiff_path.touch()

        with patch("poc_homography.point_picker.app.tifffile.TiffFile") as mock_tif:
            mock_tif.return_value = MockTiffFile(width=1000, height=800)
            state = PointPickerState(geotiff_path)
        return state

    def test_initial_state(self, mock_state: PointPickerState) -> None:
        """Initial state is empty."""
        assert mock_state.map_id == "test"
        assert len(mock_state.registry.points) == 0
        assert mock_state.width == 1000
        assert mock_state.height == 800

    def test_get_next_id_initial(self, mock_state: PointPickerState) -> None:
        """Next ID starts at 1 for each tag."""
        assert mock_state.get_next_id("parking_spot") == "PS1"
        assert mock_state.get_next_id("arrows") == "AR1"
        assert mock_state.get_next_id("crosswalk") == "CW1"
        assert mock_state.get_next_id("extra") == "EX1"

    def test_get_next_id_increments(self, mock_state: PointPickerState) -> None:
        """Next ID increments after adding points."""
        mock_state.add_point("parking_spot", 100, 200)
        assert mock_state.get_next_id("parking_spot") == "PS2"

        mock_state.add_point("parking_spot", 150, 250)
        assert mock_state.get_next_id("parking_spot") == "PS3"

    def test_get_next_id_independent_tags(self, mock_state: PointPickerState) -> None:
        """Each tag has independent ID counter."""
        mock_state.add_point("parking_spot", 100, 200)
        mock_state.add_point("parking_spot", 150, 250)
        mock_state.add_point("arrows", 300, 400)

        assert mock_state.get_next_id("parking_spot") == "PS3"
        assert mock_state.get_next_id("arrows") == "AR2"
        assert mock_state.get_next_id("crosswalk") == "CW1"

    def test_get_next_id_unknown_tag(self, mock_state: PointPickerState) -> None:
        """Unknown tag raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tag"):
            mock_state.get_next_id("unknown_tag")

    def test_add_point(self, mock_state: PointPickerState) -> None:
        """Add point creates correct MapPoint."""
        point_id = mock_state.add_point("parking_spot", 100.5, 200.5)
        assert point_id == "PS1"
        assert point_id in mock_state.registry.points

        point = mock_state.registry.points[point_id]
        assert point.pixel_x == 100.5
        assert point.pixel_y == 200.5

    def test_update_point(self, mock_state: PointPickerState) -> None:
        """Update point changes coordinates."""
        point_id = mock_state.add_point("parking_spot", 100, 200)
        mock_state.update_point(point_id, 150, 250)

        point = mock_state.registry.points[point_id]
        assert point.pixel_x == 150
        assert point.pixel_y == 250

    def test_update_nonexistent_point(self, mock_state: PointPickerState) -> None:
        """Update nonexistent point raises KeyError."""
        with pytest.raises(KeyError):
            mock_state.update_point("PS999", 100, 200)

    def test_delete_point(self, mock_state: PointPickerState) -> None:
        """Delete point removes from registry."""
        point_id = mock_state.add_point("parking_spot", 100, 200)
        assert point_id in mock_state.registry.points

        mock_state.delete_point(point_id)
        assert point_id not in mock_state.registry.points

    def test_delete_nonexistent_point(self, mock_state: PointPickerState) -> None:
        """Delete nonexistent point raises KeyError."""
        with pytest.raises(KeyError):
            mock_state.delete_point("PS999")

    def test_save_and_load_registry(
        self, mock_state: PointPickerState, tmp_path: Path
    ) -> None:
        """Save and load preserves points."""
        mock_state.add_point("parking_spot", 100, 200)
        mock_state.add_point("arrows", 300, 400)

        yaml_path = tmp_path / "points.yaml"
        mock_state.save_registry(yaml_path)

        # Load into a new state
        with patch("poc_homography.point_picker.app.tifffile.TiffFile") as mock_tif:
            mock_tif.return_value = MockTiffFile()
            geotiff_path = tmp_path / "test2.tif"
            geotiff_path.touch()
            new_state = PointPickerState(geotiff_path)

        new_state.load_registry(yaml_path)

        assert len(new_state.registry.points) == 2
        assert "PS1" in new_state.registry.points
        assert "AR1" in new_state.registry.points


class TestPointPickerStateGeoCoords:
    """Tests for geographic coordinate conversion."""

    @pytest.fixture
    def state_with_geotransform(self, tmp_path: Path) -> PointPickerState:
        """Create state with geotransform."""
        geotiff_path = tmp_path / "test.tif"
        geotiff_path.touch()

        with patch("poc_homography.point_picker.app.tifffile.TiffFile") as mock_tif:
            mock_tif.return_value = MockTiffFile()
            state = PointPickerState(geotiff_path)

        # Set a simple geotransform: origin at (1000, 2000), 1 unit per pixel
        state.geotransform = [1000.0, 1.0, 0.0, 2000.0, 0.0, -1.0]
        return state

    @pytest.fixture
    def state_without_geotransform(self, tmp_path: Path) -> PointPickerState:
        """Create state without geotransform."""
        geotiff_path = tmp_path / "test.tif"
        geotiff_path.touch()

        with patch("poc_homography.point_picker.app.tifffile.TiffFile") as mock_tif:
            mock_tif.return_value = MockTiffFile()
            state = PointPickerState(geotiff_path)

        state.geotransform = None
        return state

    def test_get_geo_coords_with_geotransform(
        self, state_with_geotransform: PointPickerState
    ) -> None:
        """Geo coords calculated correctly with geotransform."""
        coords = state_with_geotransform.get_geo_coords(10, 20)
        assert coords is not None
        easting, northing = coords
        # With geotransform [1000, 1, 0, 2000, 0, -1]:
        # easting = 1000 + 10*1 + 20*0 = 1010
        # northing = 2000 + 10*0 + 20*(-1) = 1980
        assert easting == 1010.0
        assert northing == 1980.0

    def test_get_geo_coords_without_geotransform(
        self, state_without_geotransform: PointPickerState
    ) -> None:
        """Geo coords return None without geotransform."""
        coords = state_without_geotransform.get_geo_coords(10, 20)
        assert coords is None


class TestPointPickerAPI:
    """Integration tests for FastAPI endpoints using TestClient."""

    @pytest.fixture
    def test_client(self, tmp_path: Path):
        """Create a test client with a mocked GeoTIFF."""
        from fastapi.testclient import TestClient

        from poc_homography.point_picker.app import create_app

        # Create a minimal test image
        geotiff_path = tmp_path / "test.tif"
        geotiff_path.touch()

        with patch("poc_homography.point_picker.app.tifffile.TiffFile") as mock_tif:
            mock_tif.return_value = MockTiffFile(width=1000, height=800)
            app = create_app(geotiff_path)

        return TestClient(app)

    def test_get_image_info(self, test_client) -> None:
        """GET /api/image/info returns image metadata."""
        resp = test_client.get("/api/image/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["width"] == 1000
        assert data["height"] == 800
        assert data["filename"] == "test.tif"
        assert "geotransform" in data
        assert "crs" in data

    def test_get_points_empty(self, test_client) -> None:
        """GET /api/points returns empty list initially."""
        resp = test_client.get("/api/points")
        assert resp.status_code == 200
        data = resp.json()
        assert data["points"] == []

    def test_add_point_success(self, test_client) -> None:
        """POST /api/points creates a new point."""
        resp = test_client.post(
            "/api/points",
            json={"tag": "parking_spot", "pixel_x": 100.5, "pixel_y": 200.5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "PS1"
        assert data["pixel_x"] == 100.5
        assert data["pixel_y"] == 200.5
        assert data["tag"] == "parking_spot"

    def test_add_point_default_tag(self, test_client) -> None:
        """POST /api/points uses 'extra' as default tag."""
        resp = test_client.post(
            "/api/points",
            json={"pixel_x": 50, "pixel_y": 50},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "EX1"
        assert data["tag"] == "extra"

    def test_add_point_invalid_tag_returns_422(self, test_client) -> None:
        """POST /api/points with invalid tag returns 422."""
        resp = test_client.post(
            "/api/points",
            json={"tag": "invalid_tag", "pixel_x": 100, "pixel_y": 200},
        )
        assert resp.status_code == 422

    def test_add_point_missing_pixel_x_returns_422(self, test_client) -> None:
        """POST /api/points without pixel_x returns 422."""
        resp = test_client.post(
            "/api/points",
            json={"tag": "parking_spot", "pixel_y": 200},
        )
        assert resp.status_code == 422

    def test_add_point_missing_pixel_y_returns_422(self, test_client) -> None:
        """POST /api/points without pixel_y returns 422."""
        resp = test_client.post(
            "/api/points",
            json={"tag": "parking_spot", "pixel_x": 100},
        )
        assert resp.status_code == 422

    def test_get_points_after_adding(self, test_client) -> None:
        """GET /api/points returns added points."""
        # Add two points
        test_client.post(
            "/api/points",
            json={"tag": "parking_spot", "pixel_x": 100, "pixel_y": 200},
        )
        test_client.post(
            "/api/points",
            json={"tag": "arrows", "pixel_x": 300, "pixel_y": 400},
        )

        resp = test_client.get("/api/points")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["points"]) == 2

    def test_update_point_success(self, test_client) -> None:
        """PUT /api/points/{point_id} updates coordinates."""
        # Add a point first
        test_client.post(
            "/api/points",
            json={"tag": "parking_spot", "pixel_x": 100, "pixel_y": 200},
        )

        # Update it
        resp = test_client.put(
            "/api/points/PS1",
            json={"pixel_x": 150, "pixel_y": 250},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "PS1"
        assert data["pixel_x"] == 150
        assert data["pixel_y"] == 250

    def test_update_point_not_found_returns_404(self, test_client) -> None:
        """PUT /api/points/{point_id} returns 404 for nonexistent point."""
        resp = test_client.put(
            "/api/points/PS999",
            json={"pixel_x": 100, "pixel_y": 200},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_update_point_missing_pixel_returns_422(self, test_client) -> None:
        """PUT /api/points/{point_id} without pixel_x returns 422."""
        # Add a point first
        test_client.post(
            "/api/points",
            json={"tag": "parking_spot", "pixel_x": 100, "pixel_y": 200},
        )

        resp = test_client.put(
            "/api/points/PS1",
            json={"pixel_y": 250},
        )
        assert resp.status_code == 422

    def test_delete_point_success(self, test_client) -> None:
        """DELETE /api/points/{point_id} removes the point."""
        # Add a point first
        test_client.post(
            "/api/points",
            json={"tag": "parking_spot", "pixel_x": 100, "pixel_y": 200},
        )

        # Delete it
        resp = test_client.delete("/api/points/PS1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == "PS1"

        # Verify it's gone
        resp = test_client.get("/api/points")
        assert len(resp.json()["points"]) == 0

    def test_delete_point_not_found_returns_404(self, test_client) -> None:
        """DELETE /api/points/{point_id} returns 404 for nonexistent point."""
        resp = test_client.delete("/api/points/PS999")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_get_next_id(self, test_client) -> None:
        """GET /api/points/next-id returns next ID for tag."""
        resp = test_client.get("/api/points/next-id?tag=parking_spot")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tag"] == "parking_spot"
        assert data["next_id"] == "PS1"

        # Add a point and check again
        test_client.post(
            "/api/points",
            json={"tag": "parking_spot", "pixel_x": 100, "pixel_y": 200},
        )
        resp = test_client.get("/api/points/next-id?tag=parking_spot")
        assert resp.json()["next_id"] == "PS2"

    def test_get_next_id_invalid_tag_returns_400(self, test_client) -> None:
        """GET /api/points/next-id with invalid tag returns 400."""
        resp = test_client.get("/api/points/next-id?tag=invalid_tag")
        assert resp.status_code == 400

    def test_get_geo_coords_without_geotransform(self, test_client) -> None:
        """GET /api/geo-coords returns null when no geotransform."""
        resp = test_client.get("/api/geo-coords?pixel_x=100&pixel_y=200")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pixel_x"] == 100
        assert data["pixel_y"] == 200
        assert data["easting"] is None
        assert data["northing"] is None

    def test_export_points(self, test_client, tmp_path: Path) -> None:
        """POST /api/export saves points to YAML file."""
        # Add some points
        test_client.post(
            "/api/points",
            json={"tag": "parking_spot", "pixel_x": 100, "pixel_y": 200},
        )
        test_client.post(
            "/api/points",
            json={"tag": "arrows", "pixel_x": 300, "pixel_y": 400},
        )

        # Export
        export_path = tmp_path / "exported.yaml"
        resp = test_client.post(
            "/api/export",
            json={"path": str(export_path)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert export_path.exists()

    def test_import_points(self, test_client, tmp_path: Path) -> None:
        """POST /api/import loads points from YAML file."""
        # Create a YAML file with points (list format with id field)
        yaml_content = """map_id: test
points:
  - id: PS1
    pixel_x: 100.0
    pixel_y: 200.0
  - id: AR1
    pixel_x: 300.0
    pixel_y: 400.0
"""
        yaml_path = tmp_path / "import_test.yaml"
        yaml_path.write_text(yaml_content)

        # Import
        resp = test_client.post(
            "/api/import",
            json={"path": str(yaml_path)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2

        # Verify points were imported
        resp = test_client.get("/api/points")
        assert len(resp.json()["points"]) == 2

    def test_import_file_not_found_returns_404(self, test_client, tmp_path: Path) -> None:
        """POST /api/import returns 404 for nonexistent file."""
        resp = test_client.post(
            "/api/import",
            json={"path": str(tmp_path / "nonexistent.yaml")},
        )
        assert resp.status_code == 404

    def test_import_missing_path_returns_422(self, test_client) -> None:
        """POST /api/import without path returns 422."""
        resp = test_client.post(
            "/api/import",
            json={},
        )
        assert resp.status_code == 422

    def test_full_point_lifecycle(self, test_client) -> None:
        """Test complete point lifecycle: create -> read -> update -> delete."""
        # Create
        resp = test_client.post(
            "/api/points",
            json={"tag": "crosswalk", "pixel_x": 500, "pixel_y": 600},
        )
        assert resp.status_code == 200
        point_id = resp.json()["id"]
        assert point_id == "CW1"

        # Read
        resp = test_client.get("/api/points")
        points = resp.json()["points"]
        assert len(points) == 1
        assert points[0]["id"] == "CW1"

        # Update
        resp = test_client.put(
            f"/api/points/{point_id}",
            json={"pixel_x": 550, "pixel_y": 650},
        )
        assert resp.status_code == 200
        assert resp.json()["pixel_x"] == 550

        # Delete
        resp = test_client.delete(f"/api/points/{point_id}")
        assert resp.status_code == 200

        # Verify deleted
        resp = test_client.get("/api/points")
        assert len(resp.json()["points"]) == 0
