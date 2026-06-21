"""Offline tests for the survey-frame loader (fake repo + fake store)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from poc_homography.calibration.lens_distortion.frame_source import (
    SurveyFrame,
    iter_survey_frames,
)


@dataclass
class FakeRow:
    """A minimal stand-in for a ``CleanPlateFrameModel`` row."""

    minio_bucket: str
    minio_object_key: str
    commanded_zoom: float
    commanded_tilt: float


class FakeRepo:
    """Returns canned rows for the matching run/camera, with pagination."""

    def __init__(self, rows: list[FakeRow]) -> None:
        self._rows = rows
        self.calls: list[dict] = []

    def query_frames(
        self,
        *,
        run_id: str | None = None,
        camera_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[FakeRow], int]:
        self.calls.append(
            {"run_id": run_id, "camera_id": camera_id, "limit": limit, "offset": offset}
        )
        return self._rows[offset : offset + limit], len(self._rows)


class FakeStore:
    """Serves bytes from an in-memory ``(bucket, key) -> bytes`` map."""

    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self._objects = objects
        self.reads: list[tuple[str, str | None]] = []

    def get_frame(self, key: str, *, bucket: str | None = None) -> bytes:
        self.reads.append((key, bucket))
        return self._objects[(bucket, key)]


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    """Encode a tiny solid-color BGR image to JPEG bytes."""
    image = np.full((4, 6, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def test_iter_survey_frames_yields_decoded_frames_with_zoom_tilt() -> None:
    rows = [
        FakeRow("frames-a", "run1/cam1/0.jpg", commanded_zoom=1.5, commanded_tilt=-10.0),
        FakeRow("frames-b", "run1/cam1/1.jpg", commanded_zoom=2.0, commanded_tilt=-20.0),
    ]
    store = FakeStore(
        {
            ("frames-a", "run1/cam1/0.jpg"): _jpeg_bytes((10, 20, 30)),
            ("frames-b", "run1/cam1/1.jpg"): _jpeg_bytes((40, 50, 60)),
        }
    )
    repo = FakeRepo(rows)

    frames = list(iter_survey_frames(run_id="run1", camera_id="cam1", repo=repo, store=store))

    assert len(frames) == 2
    assert all(isinstance(f, SurveyFrame) for f in frames)
    # Decoded numpy images of the expected shape.
    assert frames[0].image.shape == (4, 6, 3)
    assert frames[1].image.shape == (4, 6, 3)
    # Zoom/tilt carried through from the rows.
    assert (frames[0].commanded_zoom, frames[0].commanded_tilt) == (1.5, -10.0)
    assert (frames[1].commanded_zoom, frames[1].commanded_tilt) == (2.0, -20.0)
    # Each image was read from the bucket recorded on its own row.
    assert store.reads == [
        ("run1/cam1/0.jpg", "frames-a"),
        ("run1/cam1/1.jpg", "frames-b"),
    ]
    # The repo was filtered by the requested run/camera.
    assert repo.calls[0]["run_id"] == "run1"
    assert repo.calls[0]["camera_id"] == "cam1"


def test_iter_survey_frames_paginates_until_drained() -> None:
    rows = [
        FakeRow("b", f"k{i}.jpg", commanded_zoom=float(i), commanded_tilt=0.0) for i in range(5)
    ]
    store = FakeStore({("b", f"k{i}.jpg"): _jpeg_bytes((i, i, i)) for i in range(5)})
    repo = FakeRepo(rows)

    frames = list(
        iter_survey_frames(run_id="r", camera_id="c", repo=repo, store=store, page_size=2)
    )

    assert [f.commanded_zoom for f in frames] == [0.0, 1.0, 2.0, 3.0, 4.0]
    # 3 pages of size 2 (2 + 2 + 1) drains 5 rows.
    assert [c["offset"] for c in repo.calls] == [0, 2, 4]


def test_iter_survey_frames_raises_on_undecodable_bytes() -> None:
    rows = [FakeRow("b", "bad.jpg", commanded_zoom=1.0, commanded_tilt=0.0)]
    store = FakeStore({("b", "bad.jpg"): b"not an image"})
    repo = FakeRepo(rows)

    with pytest.raises(ValueError, match="Failed to decode"):
        list(iter_survey_frames(run_id="r", camera_id="c", repo=repo, store=store))
