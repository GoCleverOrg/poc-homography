"""Tests for Line frozen/hashable behaviour (Arch #7).

Complements test_line.py which covers serialization and ID computation.
"""

from __future__ import annotations

import pytest

from poc_homography.domain.entities.line import Line
from poc_homography.domain.vo.pixel_point import PixelPoint


@pytest.fixture
def line() -> Line:
    return Line(
        name="L1",
        map_id="valte",
        start=PixelPoint.create(100.5, 200.5),
        end=PixelPoint.create(300.0, 400.0),
    )


class TestLineFrozen:
    def test_mutate_name_raises(self, line: Line) -> None:
        with pytest.raises(AttributeError):
            line.name = "L2"  # type: ignore[misc]

    def test_mutate_map_id_raises(self, line: Line) -> None:
        with pytest.raises(AttributeError):
            line.map_id = "other"  # type: ignore[misc]

    def test_mutate_start_raises(self, line: Line) -> None:
        with pytest.raises(AttributeError):
            line.start = PixelPoint.create(0, 0)  # type: ignore[misc]


class TestLineHashable:
    def test_usable_in_set(self, line: Line) -> None:
        s = {line}
        assert line in s

    def test_usable_as_dict_key(self, line: Line) -> None:
        d = {line: 42}
        assert d[line] == 42

    def test_equal_lines_same_hash(self) -> None:
        a = Line(name="L1", map_id="valte", start=PixelPoint.create(1, 2), end=PixelPoint.create(3, 4))
        b = Line(name="L1", map_id="valte", start=PixelPoint.create(1, 2), end=PixelPoint.create(3, 4))
        assert a == b
        assert hash(a) == hash(b)
