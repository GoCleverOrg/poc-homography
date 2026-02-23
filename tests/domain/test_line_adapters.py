"""Unit tests for line adapter functions in line_picker.state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from webapp.line_picker.state import (
    Line,
    from_line_repo,
    list_line_map_ids,
    save_to_line_repo,
)

from poc_homography.domain.entities.line import Line as DomainLine
from poc_homography.domain.vo.pixel_point import PixelPoint
from poc_homography.infrastructure.repositories import RepoYamlLine

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def legacy_lines() -> list[Line]:
    return [
        Line(line_id="L1", start_x=10.0, start_y=20.0, end_x=30.0, end_y=40.0),
        Line(line_id="L2", start_x=50.0, start_y=60.0, end_x=70.0, end_y=80.0),
    ]


class TestSaveToLineRepo:
    def test_saves_all_lines(self, tmp_dir: Path, legacy_lines: list[Line]) -> None:
        save_to_line_repo(legacy_lines, "valte", tmp_dir)

        repo = RepoYamlLine(tmp_dir)
        all_lines = repo.get_all()
        assert len(all_lines) == 2
        names = {line.name for line in all_lines}
        assert names == {"L1", "L2"}

    def test_saves_correct_coordinates(self, tmp_dir: Path, legacy_lines: list[Line]) -> None:
        save_to_line_repo(legacy_lines, "valte", tmp_dir)

        repo = RepoYamlLine(tmp_dir)
        line = repo.get("valte/L1")
        assert line is not None
        assert float(line.start.x) == 10.0
        assert float(line.start.y) == 20.0
        assert float(line.end.x) == 30.0
        assert float(line.end.y) == 40.0

    def test_saves_correct_map_id(self, tmp_dir: Path, legacy_lines: list[Line]) -> None:
        save_to_line_repo(legacy_lines, "test_map", tmp_dir)

        repo = RepoYamlLine(tmp_dir)
        line = repo.get("test_map/L1")
        assert line is not None
        assert line.map_id == "test_map"


class TestFromLineRepo:
    def test_loads_all_lines_for_map(self, tmp_dir: Path) -> None:
        repo = RepoYamlLine(tmp_dir)
        for name in ("L1", "L2"):
            repo.save(
                DomainLine(
                    name=name,
                    map_id="valte",
                    start=PixelPoint.create(10, 20),
                    end=PixelPoint.create(30, 40),
                )
            )

        lines = from_line_repo(tmp_dir, "valte")
        assert len(lines) == 2
        ids = {line.line_id for line in lines}
        assert ids == {"L1", "L2"}

    def test_filters_by_map_id(self, tmp_dir: Path) -> None:
        repo = RepoYamlLine(tmp_dir)
        repo.save(
            DomainLine(
                name="L1",
                map_id="valte",
                start=PixelPoint.create(0, 0),
                end=PixelPoint.create(1, 1),
            )
        )
        repo.save(
            DomainLine(
                name="L2",
                map_id="other",
                start=PixelPoint.create(0, 0),
                end=PixelPoint.create(1, 1),
            )
        )

        lines = from_line_repo(tmp_dir, "valte")
        assert len(lines) == 1
        assert lines[0].line_id == "L1"

    def test_returns_correct_coordinates(self, tmp_dir: Path) -> None:
        repo = RepoYamlLine(tmp_dir)
        repo.save(
            DomainLine(
                name="L1",
                map_id="valte",
                start=PixelPoint.create(100.5, 200.5),
                end=PixelPoint.create(300.0, 400.0),
            )
        )

        lines = from_line_repo(tmp_dir, "valte")
        assert len(lines) == 1
        assert lines[0].start_x == 100.5
        assert lines[0].end_y == 400.0

    def test_empty_dir_returns_empty(self, tmp_dir: Path) -> None:
        lines = from_line_repo(tmp_dir, "valte")
        assert lines == []


class TestListLineMapIds:
    def test_returns_unique_sorted_map_ids(self, tmp_dir: Path) -> None:
        repo = RepoYamlLine(tmp_dir)
        for name, map_id in [("L1", "beta"), ("L2", "alpha"), ("L3", "beta")]:
            repo.save(
                DomainLine(
                    name=name,
                    map_id=map_id,
                    start=PixelPoint.create(0, 0),
                    end=PixelPoint.create(1, 1),
                )
            )

        ids = list_line_map_ids(tmp_dir)
        assert ids == ["alpha", "beta"]

    def test_empty_dir_returns_empty(self, tmp_dir: Path) -> None:
        ids = list_line_map_ids(tmp_dir)
        assert ids == []


class TestRoundTrip:
    """Verify save_to_line_repo -> from_line_repo produces equivalent lines."""

    def test_round_trip(self, tmp_dir: Path, legacy_lines: list[Line]) -> None:
        save_to_line_repo(legacy_lines, "valte", tmp_dir)
        loaded = from_line_repo(tmp_dir, "valte")

        loaded_by_id = {line.line_id: line for line in loaded}
        for original in legacy_lines:
            restored = loaded_by_id[original.line_id]
            assert restored.start_x == original.start_x
            assert restored.start_y == original.start_y
            assert restored.end_x == original.end_x
            assert restored.end_y == original.end_y
