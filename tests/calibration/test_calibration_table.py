"""Tests for zoom-dependent calibration storage.

Tests the public API and behavior of calibration table without knowledge
of internal implementation details.
"""

import tempfile
from pathlib import Path

import pytest

from poc_homography.calibration.lens_distortion.calibration_table import (
    CameraCalibrationTable,
    ZoomCalibrationEntry,
    load_calibration_for_camera,
)
from poc_homography.domain.vo import LensDistortion


class TestZoomCalibrationEntry:
    """Tests for ZoomCalibrationEntry data model."""

    def test_create_valid_entry(self):
        """Should create entry with valid parameters."""
        entry = ZoomCalibrationEntry(
            zoom_factor=2.0,
            k1=-0.1,
            k2=0.05,
            k3=0.0,
            p1=0.001,
            p2=-0.001,
            calibration_date="2024-01-15T10:30:00",
        )

        assert entry.zoom_factor == 2.0
        assert entry.k1 == -0.1
        assert entry.k2 == 0.05

    def test_reject_zero_zoom_factor(self):
        """Should reject zoom_factor of zero."""
        with pytest.raises(ValueError, match="zoom_factor must be positive"):
            ZoomCalibrationEntry(
                zoom_factor=0.0,
                k1=0.0,
                k2=0.0,
                k3=0.0,
                p1=0.0,
                p2=0.0,
                calibration_date="2024-01-15",
            )

    def test_reject_negative_zoom_factor(self):
        """Should reject negative zoom_factor."""
        with pytest.raises(ValueError, match="zoom_factor must be positive"):
            ZoomCalibrationEntry(
                zoom_factor=-1.0,
                k1=0.0,
                k2=0.0,
                k3=0.0,
                p1=0.0,
                p2=0.0,
                calibration_date="2024-01-15",
            )

    def test_to_distortion_coefficients(self):
        """Should convert to LensDistortion correctly."""
        entry = ZoomCalibrationEntry(
            zoom_factor=1.0,
            k1=-0.15,
            k2=0.08,
            k3=0.01,
            p1=0.002,
            p2=-0.003,
            calibration_date="2024-01-15",
        )

        coeffs = entry.to_distortion_coefficients()

        assert isinstance(coeffs, LensDistortion)
        assert float(coeffs.k1) == -0.15
        assert float(coeffs.k2) == 0.08
        assert float(coeffs.k3) == 0.01
        assert float(coeffs.p1) == 0.002
        assert float(coeffs.p2) == -0.003

    def test_to_dict_contains_all_fields(self):
        """Should serialize all fields to dictionary."""
        entry = ZoomCalibrationEntry(
            zoom_factor=2.5,
            k1=-0.1,
            k2=0.05,
            k3=0.01,
            p1=0.001,
            p2=-0.001,
            calibration_date="2024-01-15T10:30:00",
            source_images=("img1.jpg", "img2.jpg"),
            validation_rmse=0.5,
            num_lines_used=10,
        )

        data = entry.to_dict()

        assert data["zoom_factor"] == 2.5
        assert data["k1"] == -0.1
        assert data["k2"] == 0.05
        assert data["k3"] == 0.01
        assert data["p1"] == 0.001
        assert data["p2"] == -0.001
        assert data["calibration_date"] == "2024-01-15T10:30:00"
        assert data["source_images"] == ["img1.jpg", "img2.jpg"]
        assert data["validation_rmse"] == 0.5
        assert data["num_lines_used"] == 10

    def test_from_dict_roundtrip(self):
        """Should serialize and deserialize without data loss."""
        original = ZoomCalibrationEntry(
            zoom_factor=2.5,
            k1=-0.1,
            k2=0.05,
            k3=0.01,
            p1=0.001,
            p2=-0.001,
            calibration_date="2024-01-15T10:30:00",
            source_images=("img1.jpg",),
            validation_rmse=0.5,
            num_lines_used=10,
        )

        data = original.to_dict()
        restored = ZoomCalibrationEntry.from_dict(data)

        assert restored.zoom_factor == original.zoom_factor
        assert restored.k1 == original.k1
        assert restored.k2 == original.k2
        assert restored.k3 == original.k3
        assert restored.p1 == original.p1
        assert restored.p2 == original.p2
        assert restored.calibration_date == original.calibration_date
        assert restored.validation_rmse == original.validation_rmse
        assert restored.num_lines_used == original.num_lines_used

    def test_from_dict_handles_missing_optional_fields(self):
        """Should handle missing optional fields with defaults."""
        data = {
            "zoom_factor": 1.0,
            "k1": -0.1,
            "k2": 0.05,
        }

        entry = ZoomCalibrationEntry.from_dict(data)

        assert entry.zoom_factor == 1.0
        assert entry.k1 == -0.1
        assert entry.k2 == 0.05
        assert entry.k3 == 0.0  # default
        assert entry.p1 == 0.0  # default
        assert entry.p2 == 0.0  # default
        assert entry.validation_rmse == 0.0  # default


class TestCameraCalibrationTable:
    """Tests for CameraCalibrationTable operations."""

    def test_create_empty_table(self):
        """Should create table with no entries."""
        table = CameraCalibrationTable(camera_id="camera_001")

        assert table.camera_id == "camera_001"
        assert len(table.entries) == 0
        assert table.get_zoom_levels() == []

    def test_add_entry(self):
        """Should add calibration entry to table."""
        table = CameraCalibrationTable(camera_id="camera_001")
        entry = ZoomCalibrationEntry(
            zoom_factor=1.0,
            k1=-0.1,
            k2=0.05,
            k3=0.0,
            p1=0.0,
            p2=0.0,
            calibration_date="2024-01-15",
        )

        table.add_entry(entry)

        assert len(table.entries) == 1
        assert 1.0 in table.entries

    def test_add_entry_updates_existing(self):
        """Should update existing entry for same zoom level."""
        table = CameraCalibrationTable(camera_id="camera_001")
        entry1 = ZoomCalibrationEntry(
            zoom_factor=1.0,
            k1=-0.1,
            k2=0.0,
            k3=0.0,
            p1=0.0,
            p2=0.0,
            calibration_date="2024-01-15",
        )
        entry2 = ZoomCalibrationEntry(
            zoom_factor=1.0,
            k1=-0.2,  # different value
            k2=0.0,
            k3=0.0,
            p1=0.0,
            p2=0.0,
            calibration_date="2024-01-16",
        )

        table.add_entry(entry1)
        table.add_entry(entry2)

        assert len(table.entries) == 1
        assert table.entries[1.0].k1 == -0.2

    def test_get_zoom_levels_sorted(self):
        """Should return zoom levels in sorted order."""
        table = CameraCalibrationTable(camera_id="camera_001")
        for zoom in [5.0, 1.0, 10.0, 2.5]:
            table.add_entry(
                ZoomCalibrationEntry(
                    zoom_factor=zoom,
                    k1=0.0,
                    k2=0.0,
                    k3=0.0,
                    p1=0.0,
                    p2=0.0,
                    calibration_date="2024-01-15",
                )
            )

        levels = table.get_zoom_levels()

        assert levels == [1.0, 2.5, 5.0, 10.0]

    def test_get_coefficients_exact_match(self):
        """Should return exact coefficients when zoom level matches."""
        table = CameraCalibrationTable(camera_id="camera_001")
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=2.0,
                k1=-0.15,
                k2=0.08,
                k3=0.0,
                p1=0.0,
                p2=0.0,
                calibration_date="2024-01-15",
            )
        )

        coeffs = table.get_coefficients(2.0)

        assert float(coeffs.k1) == -0.15
        assert float(coeffs.k2) == 0.08

    def test_get_coefficients_raises_when_empty(self):
        """Should raise ValueError when table is empty."""
        table = CameraCalibrationTable(camera_id="camera_001")

        with pytest.raises(ValueError, match="No calibration entries available"):
            table.get_coefficients(1.0)

    def test_get_coefficients_below_minimum_uses_minimum(self):
        """Should use minimum zoom entry when requested zoom is below minimum."""
        table = CameraCalibrationTable(camera_id="camera_001")
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=2.0,
                k1=-0.1,
                k2=0.0,
                k3=0.0,
                p1=0.0,
                p2=0.0,
                calibration_date="2024-01-15",
            )
        )
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=5.0,
                k1=-0.2,
                k2=0.0,
                k3=0.0,
                p1=0.0,
                p2=0.0,
                calibration_date="2024-01-15",
            )
        )

        # Request zoom below minimum
        coeffs = table.get_coefficients(1.0)

        assert float(coeffs.k1) == -0.1  # Uses minimum (2.0)

    def test_get_coefficients_above_maximum_uses_maximum(self):
        """Should use maximum zoom entry when requested zoom is above maximum."""
        table = CameraCalibrationTable(camera_id="camera_001")
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=2.0,
                k1=-0.1,
                k2=0.0,
                k3=0.0,
                p1=0.0,
                p2=0.0,
                calibration_date="2024-01-15",
            )
        )
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=5.0,
                k1=-0.2,
                k2=0.0,
                k3=0.0,
                p1=0.0,
                p2=0.0,
                calibration_date="2024-01-15",
            )
        )

        # Request zoom above maximum
        coeffs = table.get_coefficients(10.0)

        assert float(coeffs.k1) == -0.2  # Uses maximum (5.0)

    def test_get_coefficients_interpolates_between_levels(self):
        """Should interpolate coefficients between calibrated zoom levels."""
        table = CameraCalibrationTable(camera_id="camera_001")
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=2.0,
                k1=-0.1,
                k2=0.0,
                k3=0.0,
                p1=0.0,
                p2=0.0,
                calibration_date="2024-01-15",
            )
        )
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=4.0,
                k1=-0.3,
                k2=0.0,
                k3=0.0,
                p1=0.0,
                p2=0.0,
                calibration_date="2024-01-15",
            )
        )

        # Request zoom at midpoint (3.0)
        coeffs = table.get_coefficients(3.0)

        # Should be linear interpolation: -0.1 + 0.5 * (-0.3 - (-0.1)) = -0.2
        assert float(coeffs.k1) == pytest.approx(-0.2, abs=1e-6)

    def test_get_coefficients_interpolates_all_coefficients(self):
        """Should interpolate all coefficient values."""
        table = CameraCalibrationTable(camera_id="camera_001")
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=1.0,
                k1=0.0,
                k2=0.0,
                k3=0.0,
                p1=0.0,
                p2=0.0,
                calibration_date="2024-01-15",
            )
        )
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=3.0,
                k1=-0.2,
                k2=0.1,
                k3=0.02,
                p1=0.004,
                p2=-0.002,
                calibration_date="2024-01-15",
            )
        )

        # Request zoom at midpoint (2.0)
        coeffs = table.get_coefficients(2.0)

        # All should be at midpoint values
        assert float(coeffs.k1) == pytest.approx(-0.1, abs=1e-6)
        assert float(coeffs.k2) == pytest.approx(0.05, abs=1e-6)
        assert float(coeffs.k3) == pytest.approx(0.01, abs=1e-6)
        assert float(coeffs.p1) == pytest.approx(0.002, abs=1e-6)
        assert float(coeffs.p2) == pytest.approx(-0.001, abs=1e-6)

    def test_get_entry_returns_exact_entry(self):
        """Should return exact entry without interpolation."""
        table = CameraCalibrationTable(camera_id="camera_001")
        entry = ZoomCalibrationEntry(
            zoom_factor=2.0,
            k1=-0.1,
            k2=0.0,
            k3=0.0,
            p1=0.0,
            p2=0.0,
            calibration_date="2024-01-15",
        )
        table.add_entry(entry)

        result = table.get_entry(2.0)

        assert result == entry

    def test_get_entry_returns_none_for_missing(self):
        """Should return None when no exact match exists."""
        table = CameraCalibrationTable(camera_id="camera_001")
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=2.0,
                k1=-0.1,
                k2=0.0,
                k3=0.0,
                p1=0.0,
                p2=0.0,
                calibration_date="2024-01-15",
            )
        )

        result = table.get_entry(3.0)

        assert result is None

    def test_remove_entry(self):
        """Should remove entry from table."""
        table = CameraCalibrationTable(camera_id="camera_001")
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=2.0,
                k1=-0.1,
                k2=0.0,
                k3=0.0,
                p1=0.0,
                p2=0.0,
                calibration_date="2024-01-15",
            )
        )

        result = table.remove_entry(2.0)

        assert result is True
        assert len(table.entries) == 0

    def test_remove_entry_returns_false_for_missing(self):
        """Should return False when entry doesn't exist."""
        table = CameraCalibrationTable(camera_id="camera_001")

        result = table.remove_entry(2.0)

        assert result is False

    def test_save_and_load_roundtrip(self):
        """Should save to YAML and load back without data loss."""
        table = CameraCalibrationTable(camera_id="camera_001")
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=1.0,
                k1=-0.1,
                k2=0.05,
                k3=0.01,
                p1=0.001,
                p2=-0.001,
                calibration_date="2024-01-15T10:30:00",
                validation_rmse=0.5,
                num_lines_used=10,
            )
        )
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=5.0,
                k1=-0.2,
                k2=0.1,
                k3=0.02,
                p1=0.002,
                p2=-0.002,
                calibration_date="2024-01-15T11:00:00",
                validation_rmse=0.3,
                num_lines_used=15,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "calibration.yaml"
            table.save(filepath)
            loaded = CameraCalibrationTable.load(filepath)

        assert loaded.camera_id == table.camera_id
        assert len(loaded.entries) == 2
        assert loaded.get_zoom_levels() == [1.0, 5.0]
        assert loaded.entries[1.0].k1 == -0.1
        assert loaded.entries[5.0].k1 == -0.2

    def test_load_raises_for_missing_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            CameraCalibrationTable.load("/nonexistent/path/calibration.yaml")

    def test_summary_contains_camera_id(self):
        """Should include camera ID in summary."""
        table = CameraCalibrationTable(camera_id="camera_001")

        summary = table.summary()

        assert "camera_001" in summary

    def test_summary_shows_entry_count(self):
        """Should show number of entries in summary."""
        table = CameraCalibrationTable(camera_id="camera_001")
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=1.0,
                k1=0.0,
                k2=0.0,
                k3=0.0,
                p1=0.0,
                p2=0.0,
                calibration_date="2024-01-15",
            )
        )

        summary = table.summary()

        assert "Entries: 1" in summary


class TestGetIntrinsics:
    """Tests for CameraCalibrationTable.get_intrinsics()."""

    def test_returns_none_when_empty(self):
        """Should return None when table has no entries."""
        table = CameraCalibrationTable(camera_id="cam_001")

        assert table.get_intrinsics(1.0) is None

    def test_returns_none_when_no_intrinsics_stored(self):
        """Should return None when no entry has intrinsics (fx/fy == 0)."""
        table = CameraCalibrationTable(camera_id="cam_001")
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=1.0,
                k1=-0.1, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
                calibration_date="2024-01-15",
            )
        )

        assert table.get_intrinsics(1.0) is None

    def test_returns_intrinsics_for_exact_zoom(self):
        """Should return exact intrinsics when zoom level matches."""
        table = CameraCalibrationTable(camera_id="cam_001")
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=2.0,
                k1=-0.1, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
                calibration_date="2024-01-15",
                fx=1500.0, fy=1500.0, cx=960.0, cy=540.0,
            )
        )

        result = table.get_intrinsics(2.0)

        assert result is not None
        assert result["fx"] == 1500.0
        assert result["fy"] == 1500.0
        assert result["cx"] == 960.0
        assert result["cy"] == 540.0

    def test_clamps_below_minimum_zoom(self):
        """Should return lowest entry intrinsics when zoom is below minimum."""
        table = CameraCalibrationTable(camera_id="cam_001")
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=2.0,
                k1=-0.1, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
                calibration_date="2024-01-15",
                fx=1500.0, fy=1500.0, cx=960.0, cy=540.0,
            )
        )

        result = table.get_intrinsics(1.0)

        assert result is not None
        assert result["fx"] == 1500.0

    def test_clamps_above_maximum_zoom(self):
        """Should return highest entry intrinsics when zoom is above maximum."""
        table = CameraCalibrationTable(camera_id="cam_001")
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=2.0,
                k1=-0.1, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
                calibration_date="2024-01-15",
                fx=1500.0, fy=1500.0, cx=960.0, cy=540.0,
            )
        )

        result = table.get_intrinsics(10.0)

        assert result is not None
        assert result["fx"] == 1500.0

    def test_interpolates_between_zoom_levels(self):
        """Should linearly interpolate intrinsics between bracketing zooms."""
        table = CameraCalibrationTable(camera_id="cam_001")
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=2.0,
                k1=-0.1, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
                calibration_date="2024-01-15",
                fx=1000.0, fy=1000.0, cx=960.0, cy=540.0,
            )
        )
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=4.0,
                k1=-0.2, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
                calibration_date="2024-01-15",
                fx=2000.0, fy=2000.0, cx=960.0, cy=540.0,
            )
        )

        result = table.get_intrinsics(3.0)

        assert result is not None
        assert result["fx"] == pytest.approx(1500.0, abs=1e-6)
        assert result["fy"] == pytest.approx(1500.0, abs=1e-6)

    def test_save_load_roundtrip_preserves_intrinsics(self):
        """Intrinsics should survive YAML save/load roundtrip."""
        table = CameraCalibrationTable(camera_id="cam_001")
        table.add_entry(
            ZoomCalibrationEntry(
                zoom_factor=1.0,
                k1=-0.1, k2=0.05, k3=0.0, p1=0.0, p2=0.0,
                calibration_date="2024-01-15",
                fx=1200.0, fy=1200.0, cx=960.0, cy=540.0,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "cal.yaml"
            table.save(filepath)
            loaded = CameraCalibrationTable.load(filepath)

        result = loaded.get_intrinsics(1.0)
        assert result is not None
        assert result["fx"] == pytest.approx(1200.0, abs=1e-6)
        assert result["cx"] == pytest.approx(960.0, abs=1e-6)


class TestLoadCalibrationForCamera:
    """Tests for load_calibration_for_camera utility function."""

    def test_loads_existing_calibration(self):
        """Should load calibration file when it exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create calibration file
            table = CameraCalibrationTable(camera_id="cam_123")
            table.add_entry(
                ZoomCalibrationEntry(
                    zoom_factor=1.0,
                    k1=-0.1,
                    k2=0.0,
                    k3=0.0,
                    p1=0.0,
                    p2=0.0,
                    calibration_date="2024-01-15",
                )
            )
            table.save(Path(tmpdir) / "cam_123_calibration.yaml")

            # Load it
            loaded = load_calibration_for_camera("cam_123", tmpdir)

        assert loaded is not None
        assert loaded.camera_id == "cam_123"
        assert len(loaded.entries) == 1

    def test_returns_none_for_missing_file(self):
        """Should return None when calibration file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_calibration_for_camera("nonexistent_camera", tmpdir)

        assert result is None
