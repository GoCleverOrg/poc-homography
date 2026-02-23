"""Tests for GroundControlPoint frozen dataclass, computed ID, and serialization (Arch #7)."""

from __future__ import annotations

import pytest

from poc_homography.domain.entities.ground_control_point import GroundControlPoint
from poc_homography.domain.vo.map_point import MapPoint
from poc_homography.domain.vo.pixel_point import PixelPoint


@pytest.fixture
def gcp() -> GroundControlPoint:
    return GroundControlPoint(
        name="PS1",
        map_point=MapPoint(map_id="valte", pixel_point=PixelPoint.create(100.0, 200.0)),
    )


class TestGroundControlPointId:
    def test_id_computed_from_map_id_and_name(self, gcp: GroundControlPoint) -> None:
        assert gcp.id == "valte/PS1"

    def test_map_id_property(self, gcp: GroundControlPoint) -> None:
        assert gcp.map_id == "valte"


class TestGroundControlPointFrozen:
    def test_mutate_name_raises(self, gcp: GroundControlPoint) -> None:
        with pytest.raises(AttributeError):
            gcp.name = "PS2"  # type: ignore[misc]

    def test_mutate_map_point_raises(self, gcp: GroundControlPoint) -> None:
        new_mp = MapPoint(map_id="other", pixel_point=PixelPoint.create(0, 0))
        with pytest.raises(AttributeError):
            gcp.map_point = new_mp  # type: ignore[misc]


class TestGroundControlPointHashable:
    def test_usable_as_dict_key(self, gcp: GroundControlPoint) -> None:
        d = {gcp: "value"}
        assert d[gcp] == "value"

    def test_equal_instances_same_hash(self) -> None:
        mp = MapPoint(map_id="valte", pixel_point=PixelPoint.create(100.0, 200.0))
        a = GroundControlPoint(name="PS1", map_point=mp)
        b = GroundControlPoint(name="PS1", map_point=mp)
        assert a == b
        assert hash(a) == hash(b)


class TestGroundControlPointSerialization:
    def test_to_dict(self, gcp: GroundControlPoint) -> None:
        d = gcp.to_dict()
        assert d["id"] == "valte/PS1"
        assert d["name"] == "PS1"
        assert d["map_point"]["map_id"] == "valte"
        assert d["map_point"]["pixel_point"] == {"x": 100.0, "y": 200.0}

    def test_from_dict_round_trip(self, gcp: GroundControlPoint) -> None:
        restored = GroundControlPoint.from_dict(gcp.to_dict())
        assert restored.id == gcp.id
        assert restored.name == gcp.name
        assert restored.map_id == gcp.map_id
        assert float(restored.map_point.pixel_point.x) == float(gcp.map_point.pixel_point.x)
        assert float(restored.map_point.pixel_point.y) == float(gcp.map_point.pixel_point.y)
