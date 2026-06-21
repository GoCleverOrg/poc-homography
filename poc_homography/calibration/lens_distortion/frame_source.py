"""Offline survey-frame loader for lens-distortion calibration.

The clean-plate sweep (#338) captures survey frames: the *image* bytes land in
MinIO (bucket ``cleanplate-frames``) and the *metadata* rows land in Postgres
(``clean_plate_frames``). This module is the read side that lens calibration
consumes: given a ``run_id``/``camera_id``, it enumerates the frame rows via
:class:`RepoPostgresCleanPlateFrame`, downloads each image via
:meth:`MinioFrameStore.get_frame`, decodes it with OpenCV, and yields the
decoded frame together with the zoom and tilt the camera was commanded to.

Both the store and the repo are injected, so the loader runs fully offline in
tests against fakes (no live MinIO, no live Postgres).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

# Default rows requested per query_frames page; the loader paginates to drain
# the full run rather than silently stopping at the first page.
_DEFAULT_PAGE_SIZE = 100


class SurveyFrame(NamedTuple):
    """One decoded survey frame plus the zoom/tilt the camera was commanded to.

    A plain tuple ``(image, commanded_zoom, commanded_tilt)`` for unpacking,
    with named fields for clarity at the call site.
    """

    image: np.ndarray
    commanded_zoom: float
    commanded_tilt: float


class FrameRow(Protocol):
    """The subset of a clean-plate frame row this loader reads."""

    minio_bucket: str
    minio_object_key: str
    commanded_zoom: float
    commanded_tilt: float


class FrameStore(Protocol):
    """The byte-readback surface this loader needs from the frame store."""

    def get_frame(self, key: str, *, bucket: str | None = None) -> bytes: ...


class FrameRepo(Protocol):
    """The query surface this loader needs from the frame repository."""

    def query_frames(
        self,
        *,
        run_id: str | None = ...,
        camera_id: str | None = ...,
        limit: int = ...,
        offset: int = ...,
    ) -> tuple[list[FrameRow], int]: ...


def iter_survey_frames(
    *,
    run_id: str,
    camera_id: str,
    repo: FrameRepo,
    store: FrameStore,
    page_size: int = _DEFAULT_PAGE_SIZE,
) -> Iterator[SurveyFrame]:
    """Yield decoded survey frames for ``run_id``/``camera_id``, newest first.

    Paginates ``repo.query_frames`` to drain every matching row, downloads each
    image from the bucket recorded on the row, and decodes it as a BGR image.

    Args:
        run_id: Survey run to load frames for.
        camera_id: Camera within the run to load frames for.
        repo: Frame metadata repository (injected; offline-testable).
        store: Frame byte store (injected; offline-testable).
        page_size: Rows fetched per ``query_frames`` call.

    Yields:
        ``SurveyFrame(image, commanded_zoom, commanded_tilt)`` per stored frame.

    Raises:
        ValueError: If a downloaded object cannot be decoded as an image.
    """
    offset = 0
    while True:
        rows, total = repo.query_frames(
            run_id=run_id, camera_id=camera_id, limit=page_size, offset=offset
        )
        if not rows:
            break
        for row in rows:
            data = store.get_frame(row.minio_object_key, bucket=row.minio_bucket)
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                msg = f"Failed to decode frame image at {row.minio_bucket}/{row.minio_object_key}"
                raise ValueError(msg)
            yield SurveyFrame(
                image=image,
                commanded_zoom=row.commanded_zoom,
                commanded_tilt=row.commanded_tilt,
            )
        offset += len(rows)
        if offset >= total:
            break


__all__ = ["FrameRepo", "FrameRow", "FrameStore", "SurveyFrame", "iter_survey_frames"]
