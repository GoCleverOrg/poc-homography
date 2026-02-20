"""Unit tests for Line domain entity."""

from __future__ import annotations

import pytest

from poc_homography.domain.entities.line import Line
from poc_homography.domain.vo.pixel_point import PixelPoint


@pytest.fixture
def sample_line() -> Line:
    return Line(
        name="L1",
        map_id="valte",
        start=PixelPoint.create(100.5, 200.5),
        end=PixelPoint.create(300.0, 400.0),
    )


class TestLine:
    def test_auto_id(self, sample_line: Line) -> None:
        assert sample_line.id == "valte/L1"

    def test_id_is_computed(self) -> None:
        line = Line(
            name="L2",
            map_id="test",
            start=PixelPoint.create(0, 0),
            end=PixelPoint.create(1, 1),
        )
        assert line.id == "test/L2"

    def test_map_id_property(self, sample_line: Line) -> None:
        assert sample_line.map_id == "valte"

    def test_to_dict(self, sample_line: Line) -> None:
        d = sample_line.to_dict()
        assert d["id"] == "valte/L1"
        assert d["name"] == "L1"
        assert d["map_id"] == "valte"
        assert d["start"] == {"x": 100.5, "y": 200.5}
        assert d["end"] == {"x": 300.0, "y": 400.0}

    def test_from_dict(self) -> None:
        data = {
            "id": "valte/L3",
            "name": "L3",
            "map_id": "valte",
            "start": {"x": 10.0, "y": 20.0},
            "end": {"x": 30.0, "y": 40.0},
        }
        line = Line.from_dict(data)
        assert line.id == "valte/L3"
        assert line.name == "L3"
        assert line.map_id == "valte"
        assert float(line.start.x) == 10.0
        assert float(line.end.y) == 40.0

    def test_round_trip(self, sample_line: Line) -> None:
        """to_dict -> from_dict should produce an equivalent entity."""
        restored = Line.from_dict(sample_line.to_dict())
        assert restored.id == sample_line.id
        assert restored.name == sample_line.name
        assert restored.map_id == sample_line.map_id
        assert float(restored.start.x) == float(sample_line.start.x)
        assert float(restored.start.y) == float(sample_line.start.y)
        assert float(restored.end.x) == float(sample_line.end.x)
        assert float(restored.end.y) == float(sample_line.end.y)

    def test_from_dict_without_id(self) -> None:
        """from_dict should auto-generate id if not provided."""
        data = {
            "name": "L5",
            "map_id": "test",
            "start": {"x": 1.0, "y": 2.0},
            "end": {"x": 3.0, "y": 4.0},
        }
        line = Line.from_dict(data)
        assert line.id == "test/L5"
