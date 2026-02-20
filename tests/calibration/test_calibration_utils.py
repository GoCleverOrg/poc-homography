"""Tests for shared calibration utility functions.

Tests the pure-Python helpers in calibration_utils that don't require
a Django request/response cycle.
"""

import sys
from pathlib import Path

# Add webapp/ to path so homography_web can be imported without Django setup
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "webapp"))

from homography_web.calibration_utils import (
    resolve_safe_path,
    serialize_calibration_entry,
)

from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.domain.vo.zoom_calibration_entry import ZoomCalibrationEntry
from poc_homography.types import PixelsFloat, Unitless


class TestResolveSafePath:
    """Tests for resolve_safe_path."""

    def test_resolves_valid_filename(self, tmp_path):
        result = resolve_safe_path("test.yaml", tmp_path)
        assert result is not None
        assert result.name == "test.yaml"

    def test_rejects_traversal(self, tmp_path):
        result = resolve_safe_path("../escape.yaml", tmp_path)
        assert result is None

    def test_rejects_empty(self, tmp_path):
        result = resolve_safe_path("", tmp_path)
        assert result is None


class TestSerializeCalibrationEntry:
    """Tests for serialize_calibration_entry."""

    def test_serializes_basic_entry(self):
        entry = ZoomCalibrationEntry(
            zoom_factor=Unitless(2.0),
            distortion=LensDistortion(
                k1=Unitless(-0.15),
                k2=Unitless(0.08),
                k3=Unitless(0.01),
                p1=Unitless(0.002),
                p2=Unitless(-0.003),
            ),
            calibration_date="2024-01-15",
            source_images=(),
            validation_rmse=0.5,
            num_lines_used=10,
        )

        data = serialize_calibration_entry(entry)

        assert data["zoom_factor"] == 2.0
        assert data["coefficients"]["k1"] == -0.15
        assert data["coefficients"]["k2"] == 0.08
        assert data["coefficients"]["k3"] == 0.01
        assert data["coefficients"]["p1"] == 0.002
        assert data["coefficients"]["p2"] == -0.003
        assert data["calibration_date"] == "2024-01-15"
        assert data["validation_rmse"] == 0.5
        assert data["num_lines_used"] == 10

    def test_excludes_intrinsics_when_zero(self):
        """Should not include intrinsics dict when fx/fy are zero."""
        entry = ZoomCalibrationEntry(
            zoom_factor=Unitless(1.0),
            distortion=LensDistortion(k1=Unitless(-0.1)),
            calibration_date="2024-01-15",
            source_images=(),
        )

        data = serialize_calibration_entry(entry)

        assert "intrinsics" not in data

    def test_includes_intrinsics_when_present(self):
        """Should include intrinsics dict when fx or fy is non-zero."""
        entry = ZoomCalibrationEntry(
            zoom_factor=Unitless(1.0),
            distortion=LensDistortion(k1=Unitless(-0.1)),
            calibration_date="2024-01-15",
            source_images=(),
            fx=PixelsFloat(1670.0),
            fy=PixelsFloat(1670.0),
            cx=PixelsFloat(960.0),
            cy=PixelsFloat(540.0),
        )

        data = serialize_calibration_entry(entry)

        assert "intrinsics" in data
        assert data["intrinsics"]["fx"] == 1670.0
        assert data["intrinsics"]["fy"] == 1670.0
        assert data["intrinsics"]["cx"] == 960.0
        assert data["intrinsics"]["cy"] == 540.0

    def test_includes_intrinsics_when_only_fx_nonzero(self):
        """Should include intrinsics even if only fx is non-zero."""
        entry = ZoomCalibrationEntry(
            zoom_factor=Unitless(1.0),
            distortion=LensDistortion(),
            calibration_date="2024-01-15",
            source_images=(),
            fx=PixelsFloat(1000.0),
        )

        data = serialize_calibration_entry(entry)

        assert "intrinsics" in data
