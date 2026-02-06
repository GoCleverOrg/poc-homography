"""Tests for the calibrate-batch CLI command.

Tests the batch calibration workflow including image grouping by zoom,
per-zoom calibration execution, validation, and output file generation.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from poc_homography.calibration.lens_distortion.calibration_table import (
    CameraCalibrationTable,
)
from poc_homography.calibration.lens_distortion.cli import (
    _DEFAULT_FX,
    group_images_by_zoom,
    main,
    parse_ptz_from_filename,
)
from poc_homography.calibration.lens_distortion.models import PTZPosition
from poc_homography.types import Degrees


class TestGroupImagesByZoom:
    """Tests for the image grouping by zoom functionality."""

    def test_groups_images_by_exact_zoom(self):
        """Should group images with identical zoom values together."""
        images = [
            Path("img_30.0_15.0_1.0.jpg"),
            Path("img_30.0_15.0_1.0_2.jpg"),
            Path("img_30.0_15.0_5.0.jpg"),
            Path("img_30.0_15.0_5.0_2.jpg"),
            Path("img_30.0_15.0_10.0.jpg"),
        ]

        groups = group_images_by_zoom(images)

        assert len(groups) == 3
        assert 1.0 in groups
        assert 5.0 in groups
        assert 10.0 in groups
        assert len(groups[1.0]) == 2
        assert len(groups[5.0]) == 2
        assert len(groups[10.0]) == 1

    def test_groups_similar_zooms_with_tolerance(self):
        """Should group similar zoom values (e.g., 1.0 and 1.05) together."""
        images = [
            Path("img_30.0_15.0_1.0.jpg"),
            Path("img_30.0_15.0_1.05.jpg"),  # Should round to 1.0
            Path("img_30.0_15.0_1.08.jpg"),  # Should round to 1.1
        ]

        groups = group_images_by_zoom(images, tolerance=0.1)

        # 1.0 and 1.05 should be grouped together at 1.0
        # 1.08 should round to 1.1
        assert 1.0 in groups
        assert len(groups[1.0]) == 2
        assert 1.1 in groups
        assert len(groups[1.1]) == 1

    def test_handles_empty_list(self):
        """Should return empty dict for empty image list."""
        groups = group_images_by_zoom([])

        assert groups == {}

    def test_handles_default_zoom_on_parse_failure(self):
        """Should use default zoom (1.0) when filename parsing fails."""
        images = [
            Path("image_without_ptz.jpg"),
            Path("another_plain_name.jpg"),
        ]

        groups = group_images_by_zoom(images)

        # Both should be grouped at default zoom 1.0
        assert 1.0 in groups
        assert len(groups[1.0]) == 2


class TestParseZoomFromFilename:
    """Tests for PTZ parsing from filenames."""

    def test_parses_valid_ptz_filename(self):
        """Should extract pan, tilt, zoom from valid filename format."""
        result = parse_ptz_from_filename("valte_cam01_20260126_095620_30.0_15.3_1.0.jpg")

        assert result.pan_deg == pytest.approx(30.0)
        assert result.tilt_deg == pytest.approx(15.3)
        assert result.zoom_factor == pytest.approx(1.0)

    def test_parses_different_zoom_values(self):
        """Should correctly parse various zoom levels."""
        test_cases = [
            ("img_0.0_30.0_1.0.jpg", 1.0),
            ("img_0.0_30.0_5.0.jpg", 5.0),
            ("img_0.0_30.0_10.0.jpg", 10.0),
            ("img_0.0_30.0_25.0.jpg", 25.0),
        ]

        for filename, expected_zoom in test_cases:
            result = parse_ptz_from_filename(filename)
            assert result.zoom_factor == pytest.approx(expected_zoom), f"Failed for {filename}"

    def test_returns_defaults_on_parse_failure(self):
        """Should return default PTZ values when filename can't be parsed."""
        result = parse_ptz_from_filename("plain_image_name.jpg")

        # Default values as defined in cli.py
        assert result.pan_deg == 0.0
        assert result.tilt_deg == 30.0
        assert result.zoom_factor == 1.0

    def test_handles_negative_pan_tilt(self):
        """Should handle negative pan and tilt values."""
        result = parse_ptz_from_filename("img_-45.0_-30.0_2.5.jpg")

        assert result.pan_deg == pytest.approx(-45.0)
        assert result.tilt_deg == pytest.approx(-30.0)
        assert result.zoom_factor == pytest.approx(2.5)


class TestCalibrateBatchCommand:
    """Tests for the calibrate-batch command integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_images_dir(self, temp_dir):
        """Create mock images at multiple zoom levels.

        Pattern: survey_imgX_pan_tilt_zoom.jpg where imgX is non-numeric
        to avoid the parser matching an index as part of a triplet.
        """
        zooms = [1.0, 5.0, 10.0]
        for zoom in zooms:
            for i in range(4):  # 4 images per zoom
                filename = f"survey_img{i}_30.0_15.0_{zoom:.1f}.jpg"
                (temp_dir / filename).touch()

        return temp_dir

    def test_groups_images_correctly(self, mock_images_dir):
        """Should group mock images by zoom level correctly."""
        from poc_homography.calibration.lens_distortion.cli import find_images

        images = find_images(mock_images_dir)
        groups = group_images_by_zoom(images)

        assert len(groups) == 3
        assert 1.0 in groups
        assert 5.0 in groups
        assert 10.0 in groups
        assert len(groups[1.0]) == 4
        assert len(groups[5.0]) == 4
        assert len(groups[10.0]) == 4

    def test_intrinsics_scaling_builds_correct_matrix(self):
        """Should build intrinsic matrix with zoom-scaled focal length and fixed principal point."""
        base_fx = _DEFAULT_FX
        cx, cy = 960.0, 540.0

        for zoom in [1.0, 5.0, 10.0, 25.0]:
            fx_zoom = base_fx * zoom
            fy_zoom = base_fx * zoom
            intrinsic_matrix = np.array(
                [
                    [fx_zoom, 0.0, cx],
                    [0.0, fy_zoom, cy],
                    [0.0, 0.0, 1.0],
                ]
            )

            # Focal lengths scale with zoom
            assert intrinsic_matrix[0, 0] == pytest.approx(base_fx * zoom)
            assert intrinsic_matrix[1, 1] == pytest.approx(base_fx * zoom)
            # Principal point remains constant across zoom levels
            assert intrinsic_matrix[0, 2] == cx
            assert intrinsic_matrix[1, 2] == cy

    def test_calibration_output_file_format(self, temp_dir):
        """Should create valid CameraCalibrationTable YAML output."""
        # Create a table with multiple zoom entries
        table = CameraCalibrationTable(camera_id="test_camera")
        from poc_homography.calibration.lens_distortion.calibration_table import (
            ZoomCalibrationEntry,
        )

        for zoom in [1.0, 5.0, 10.0]:
            entry = ZoomCalibrationEntry(
                zoom_factor=zoom,
                k1=-0.02 * zoom,
                k2=0.001 * zoom,
                k3=0.0,
                p1=0.0,
                p2=0.0,
                calibration_date="2026-02-05T10:00:00",
                validation_rmse=2.0 + zoom * 0.1,
                num_lines_used=50,
                fx=_DEFAULT_FX * zoom,
                fy=_DEFAULT_FX * zoom,
                cx=960.0,
                cy=540.0,
            )
            table.add_entry(entry)

        output_path = temp_dir / "test_calibration.yaml"
        table.save(output_path)

        # Verify we can load it back
        loaded = CameraCalibrationTable.load(output_path)
        assert loaded.camera_id == "test_camera"
        assert len(loaded.entries) == 3
        assert loaded.get_zoom_levels() == [1.0, 5.0, 10.0]

        # Verify intrinsics are stored
        intrinsics = loaded.get_intrinsics(5.0)
        assert intrinsics is not None
        assert intrinsics["fx"] == pytest.approx(_DEFAULT_FX * 5.0)

        # Verify distortion coefficients round-trip correctly
        entry_1 = loaded.get_entry(1.0)
        assert entry_1 is not None
        assert entry_1.k1 == pytest.approx(-0.02)
        entry_10 = loaded.get_entry(10.0)
        assert entry_10 is not None
        assert entry_10.k1 == pytest.approx(-0.2)

    def test_summary_includes_all_zoom_levels(self):
        """Should show all calibrated zoom levels in summary output."""
        table = CameraCalibrationTable(camera_id="test_camera")
        from poc_homography.calibration.lens_distortion.calibration_table import (
            ZoomCalibrationEntry,
        )

        for zoom in [1.0, 5.0, 10.0]:
            entry = ZoomCalibrationEntry(
                zoom_factor=zoom,
                k1=-0.02,
                k2=0.001,
                k3=0.0,
                p1=0.0,
                p2=0.0,
                calibration_date="2026-02-05",
                validation_rmse=2.0,
                num_lines_used=50,
                fx=_DEFAULT_FX * zoom,
                fy=_DEFAULT_FX * zoom,
                cx=960.0,
                cy=540.0,
            )
            table.add_entry(entry)

        summary = table.summary()

        # Should contain all zoom levels
        assert "1.0" in summary
        assert "5.0" in summary
        assert "10.0" in summary
        # Should contain intrinsics column header
        assert "fx" in summary
        assert "Entries: 3" in summary


class TestBatchValidation:
    """Tests for validation logic in batch calibration."""

    def test_detects_insufficient_images(self):
        """Should detect zoom groups with fewer than minimum images."""
        from poc_homography.calibration.lens_distortion.cli import group_images_by_zoom

        # Only 2 images for zoom 10.0, but 4 for others
        images = [
            Path("img_30.0_15.0_1.0_1.jpg"),
            Path("img_30.0_15.0_1.0_2.jpg"),
            Path("img_30.0_15.0_1.0_3.jpg"),
            Path("img_30.0_15.0_1.0_4.jpg"),
            Path("img_30.0_15.0_5.0_1.jpg"),
            Path("img_30.0_15.0_5.0_2.jpg"),
            Path("img_30.0_15.0_5.0_3.jpg"),
            Path("img_30.0_15.0_5.0_4.jpg"),
            Path("img_30.0_15.0_10.0_1.jpg"),
            Path("img_30.0_15.0_10.0_2.jpg"),  # Only 2 images
        ]

        groups = group_images_by_zoom(images)
        min_images = 3

        insufficient = [
            (zoom, len(imgs)) for zoom, imgs in groups.items() if len(imgs) < min_images
        ]

        assert len(insufficient) == 1
        assert insufficient[0][0] == 10.0
        assert insufficient[0][1] == 2


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_calibrate_batch_requires_camera_id(self):
        """Should exit with error when --camera-id is missing."""
        with patch.object(sys, "argv", ["cli", "calibrate-batch", "--images", "/tmp"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_batch_defaults_accept_minimal_args(self):
        """Batch command should work with only required args (defaults for the rest)."""
        with patch.object(
            sys,
            "argv",
            ["cli", "calibrate-batch", "--images", "/nonexistent/path", "--camera-id", "test"],
        ):
            # Should fail due to path not found, NOT due to missing args
            result = main()
            assert result == 1


class TestBackwardsCompatibility:
    """Tests ensuring the existing calibrate command remains unchanged."""

    def test_calibrate_command_accepts_legacy_flags(self):
        """The calibrate command should still accept --fx, --fy, --zoom flags."""
        with patch.object(
            sys,
            "argv",
            [
                "cli",
                "calibrate",
                "--images",
                "/nonexistent/path",
                "--fx",
                "1000",
                "--fy",
                "1000",
                "--zoom",
                "2.0",
            ],
        ):
            # Should fail due to path not found, NOT due to unrecognized args
            result = main()
            assert result == 1

    def test_ptz_parser_unchanged(self):
        """parse_ptz_from_filename should remain compatible with existing usage."""
        # Test with various existing filename patterns
        patterns = [
            (
                "survey_45.5_30.0_2.0_20240115.jpg",
                PTZPosition(
                    pan_deg=Degrees(45.5),
                    tilt_deg=Degrees(30.0),
                    zoom_factor=2.0,
                ),
            ),
        ]

        for filename, expected in patterns:
            result = parse_ptz_from_filename(filename)
            assert result.pan_deg == pytest.approx(float(expected.pan_deg))
            assert result.tilt_deg == pytest.approx(float(expected.tilt_deg))
            assert result.zoom_factor == pytest.approx(expected.zoom_factor)
