"""Tests for RepoYamlCapturedFrame — image_dir_for, protocol, path traversal (Arch #4, Security #10)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from poc_homography.domain.entities.captured_frame import CapturedFrame
from poc_homography.domain.vo.ptz_state import PTZState
from poc_homography.infrastructure.repositories.repo_yaml_captured_frame import (
    RepoYamlCapturedFrame,
)
from poc_homography.types import Degrees, Unitless


def _make_frame(image_path: str = "frame.jpg") -> CapturedFrame:
    return CapturedFrame.create(
        map_id="valte",
        camera_name="cam01",
        timestamp=datetime(2024, 1, 15, 10, 30, 45),
        ptz_state=PTZState(pan_raw=Degrees(30.0), tilt_deg=Degrees(15.0), zoom=Unitless(1.0)),
        image_path=Path(image_path),
    )


class TestImageDirFor:
    def test_returns_correct_path(self, tmp_path: Path) -> None:
        repo = RepoYamlCapturedFrame(tmp_path)
        assert repo.image_dir_for("valte", "cam01") == tmp_path / "valte" / "cam01"

    def test_different_cameras_different_dirs(self, tmp_path: Path) -> None:
        repo = RepoYamlCapturedFrame(tmp_path)
        dir_a = repo.image_dir_for("valte", "cam01")
        dir_b = repo.image_dir_for("valte", "cam02")
        assert dir_a != dir_b


class TestProtocolSatisfaction:
    def test_has_save_method(self) -> None:
        assert callable(getattr(RepoYamlCapturedFrame, "save", None))

    def test_has_image_dir_for_method(self) -> None:
        assert callable(getattr(RepoYamlCapturedFrame, "image_dir_for", None))


class TestDeletePathTraversal:
    def test_normal_image_deleted(self, tmp_path: Path) -> None:
        repo = RepoYamlCapturedFrame(tmp_path)
        frame = _make_frame("frame.jpg")
        repo.save(frame)

        # Create the image file alongside the YAML
        image_dir = repo.image_dir_for("valte", "cam01")
        image_file = image_dir / "frame.jpg"
        image_file.write_text("fake image")

        assert repo.delete(frame.id) is True
        assert not image_file.exists()

    def test_traversal_path_not_deleted(self, tmp_path: Path) -> None:
        """An image_path like '../../sensitive.txt' must NOT delete outside data_dir."""
        repo = RepoYamlCapturedFrame(tmp_path)
        frame = _make_frame("../../sensitive.txt")
        repo.save(frame)

        # Create the sensitive file outside data_dir
        sensitive = (tmp_path / ".." / ".." / "sensitive.txt").resolve()
        sensitive.parent.mkdir(parents=True, exist_ok=True)
        sensitive.write_text("secret")

        repo.delete(frame.id)
        # The sensitive file must still exist (path traversal blocked)
        assert sensitive.exists()
        # Clean up
        sensitive.unlink()

    def test_yaml_always_removed(self, tmp_path: Path) -> None:
        repo = RepoYamlCapturedFrame(tmp_path)
        frame = _make_frame("../../outside.jpg")
        repo.save(frame)

        yaml_path = tmp_path / "valte" / "cam01" / f"{frame.id.split('/')[-1]}.yaml"
        assert yaml_path.exists()

        repo.delete(frame.id)
        assert not yaml_path.exists()

    def test_delete_nonexistent_returns_false(self, tmp_path: Path) -> None:
        repo = RepoYamlCapturedFrame(tmp_path)
        assert repo.delete("valte/cam01/nonexistent") is False
