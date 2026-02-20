"""Unit tests for RepoYamlLine and RepoYamlLineAnnotation repositories."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from poc_homography.domain.entities.line import Line
from poc_homography.domain.entities.line_annotation import LineAnnotation
from poc_homography.domain.vo.pixel_point import PixelPoint
from poc_homography.domain.vo.ptz_state import PTZState
from poc_homography.infrastructure.repositories import RepoYamlLine, RepoYamlLineAnnotation
from poc_homography.types import Degrees, Unitless


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_line() -> Line:
    return Line(
        name="L1",
        map_id="valte",
        start=PixelPoint.create(100.5, 200.5),
        end=PixelPoint.create(300.0, 400.0),
    )


@pytest.fixture
def sample_line_annotation() -> LineAnnotation:
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


class TestRepoYamlLine:
    def test_save_and_get(self, tmp_dir: Path, sample_line: Line) -> None:
        repo = RepoYamlLine(tmp_dir)
        repo.save(sample_line)

        loaded = repo.get(sample_line.id)
        assert loaded is not None
        assert loaded.id == sample_line.id
        assert loaded.name == "L1"
        assert loaded.map_id == "valte"
        assert float(loaded.start.x) == 100.5
        assert float(loaded.end.y) == 400.0

    def test_get_all(self, tmp_dir: Path) -> None:
        repo = RepoYamlLine(tmp_dir)
        for name in ("L1", "L2", "L3"):
            repo.save(
                Line(
                    name=name,
                    map_id="valte",
                    start=PixelPoint.create(0, 0),
                    end=PixelPoint.create(1, 1),
                )
            )

        all_lines = repo.get_all()
        assert len(all_lines) == 3
        names = {line.name for line in all_lines}
        assert names == {"L1", "L2", "L3"}

    def test_get_by_map(self, tmp_dir: Path) -> None:
        repo = RepoYamlLine(tmp_dir)
        repo.save(
            Line(
                name="L1",
                map_id="valte",
                start=PixelPoint.create(0, 0),
                end=PixelPoint.create(1, 1),
            )
        )
        repo.save(
            Line(
                name="L2",
                map_id="other",
                start=PixelPoint.create(0, 0),
                end=PixelPoint.create(1, 1),
            )
        )

        valte_lines = repo.get_by_map("valte")
        assert len(valte_lines) == 1
        assert "valte/L1" in valte_lines

    def test_delete(self, tmp_dir: Path, sample_line: Line) -> None:
        repo = RepoYamlLine(tmp_dir)
        repo.save(sample_line)
        assert repo.exists(sample_line.id)

        repo.delete(sample_line.id)
        assert not repo.exists(sample_line.id)
        assert repo.get(sample_line.id) is None

    def test_round_trip_via_file(self, tmp_dir: Path, sample_line: Line) -> None:
        """Save, clear cache, reload — should get identical data."""
        repo = RepoYamlLine(tmp_dir)
        repo.save(sample_line)
        repo.clear_cache()

        loaded = repo.get(sample_line.id)
        assert loaded is not None
        assert loaded.id == sample_line.id
        assert float(loaded.start.x) == float(sample_line.start.x)


class TestRepoYamlLineAnnotation:
    def test_save_and_get(self, tmp_dir: Path, sample_line_annotation: LineAnnotation) -> None:
        repo = RepoYamlLineAnnotation(tmp_dir)
        repo.save(sample_line_annotation)

        loaded = repo.get(sample_line_annotation.id)
        assert loaded is not None
        assert loaded.id == sample_line_annotation.id
        assert loaded.line_id == "L4"
        assert loaded.frame_id == "valte_cam01_20260126_095620"
        assert float(loaded.camera_pose.pan_raw) == 30.0
        assert float(loaded.start_pixel.x) == 100.5

    def test_get_all(self, tmp_dir: Path) -> None:
        repo = RepoYamlLineAnnotation(tmp_dir)
        pose = PTZState(pan_raw=Degrees(0), tilt_deg=Degrees(0), zoom=Unitless(1))
        for line_id in ("L1", "L2"):
            repo.save(
                LineAnnotation(
                    line_id=line_id,
                    frame_id="frame_001",
                    camera_pose=pose,
                    start_pixel=PixelPoint.create(0, 0),
                    end_pixel=PixelPoint.create(1, 1),
                )
            )

        all_anns = repo.get_all()
        assert len(all_anns) == 2

    def test_round_trip_via_file(
        self, tmp_dir: Path, sample_line_annotation: LineAnnotation
    ) -> None:
        repo = RepoYamlLineAnnotation(tmp_dir)
        repo.save(sample_line_annotation)
        repo.clear_cache()

        loaded = repo.get(sample_line_annotation.id)
        assert loaded is not None
        assert loaded.line_id == sample_line_annotation.line_id
        assert float(loaded.end_pixel.y) == float(sample_line_annotation.end_pixel.y)
