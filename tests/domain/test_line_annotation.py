"""Unit tests for LineAnnotation domain entity."""

from __future__ import annotations

import pytest

from poc_homography.domain.entities.line_annotation import LineAnnotation
from poc_homography.domain.vo.pixel_point import PixelPoint
from poc_homography.domain.vo.ptz_state import PTZState
from poc_homography.types import Degrees, Unitless


@pytest.fixture
def sample_annotation() -> LineAnnotation:
    return LineAnnotation(
        line_id="L4",
        frame_id="valte_cam01_20260126_095620",
        camera_pose=PTZState(
            pan_raw=Degrees(30.0),
            tilt_deg=Degrees(15.3),
            zoom=Unitless(1.0),
        ),
        start_pixel=PixelPoint.create(100.5, 200.5),
        end_pixel=PixelPoint.create(300.0, 400.0),
    )


class TestLineAnnotation:
    def test_id(self, sample_annotation: LineAnnotation) -> None:
        assert sample_annotation.id == "valte_cam01_20260126_095620/L4"

    def test_frozen(self, sample_annotation: LineAnnotation) -> None:
        with pytest.raises(AttributeError):
            sample_annotation.line_id = "L5"  # type: ignore[misc]

    def test_to_dict(self, sample_annotation: LineAnnotation) -> None:
        d = sample_annotation.to_dict()
        assert d["line_id"] == "L4"
        assert d["frame_id"] == "valte_cam01_20260126_095620"
        assert d["camera_pose"]["pan_raw"] == 30.0
        assert d["camera_pose"]["tilt_deg"] == 15.3
        assert d["camera_pose"]["zoom"] == 1.0
        assert d["start_pixel"] == {"x": 100.5, "y": 200.5}
        assert d["end_pixel"] == {"x": 300.0, "y": 400.0}

    def test_from_dict(self) -> None:
        data = {
            "line_id": "L2",
            "frame_id": "frame_001",
            "camera_pose": {"pan_raw": 45.0, "tilt_deg": 10.0, "zoom": 2.0},
            "start_pixel": {"x": 50.0, "y": 60.0},
            "end_pixel": {"x": 150.0, "y": 160.0},
        }
        ann = LineAnnotation.from_dict(data)
        assert ann.line_id == "L2"
        assert ann.frame_id == "frame_001"
        assert ann.id == "frame_001/L2"
        assert float(ann.camera_pose.pan_raw) == 45.0
        assert float(ann.start_pixel.x) == 50.0
        assert float(ann.end_pixel.y) == 160.0

    def test_round_trip(self, sample_annotation: LineAnnotation) -> None:
        """to_dict -> from_dict should produce an equivalent entity."""
        restored = LineAnnotation.from_dict(sample_annotation.to_dict())
        assert restored.id == sample_annotation.id
        assert restored.line_id == sample_annotation.line_id
        assert restored.frame_id == sample_annotation.frame_id
        assert float(restored.camera_pose.pan_raw) == float(sample_annotation.camera_pose.pan_raw)
        assert float(restored.start_pixel.x) == float(sample_annotation.start_pixel.x)
        assert float(restored.end_pixel.y) == float(sample_annotation.end_pixel.y)
