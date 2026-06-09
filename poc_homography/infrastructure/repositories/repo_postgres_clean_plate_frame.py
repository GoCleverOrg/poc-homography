"""PostgreSQL-backed repository for clean-plate survey frames."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from poc_homography.infrastructure.models.clean_plate_frame import CleanPlateFrameModel

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class RepoPostgresCleanPlateFrame:
    """Upsert + read clean-plate frame rows in the ``clean_plate_frames`` table.

    One row per captured ``FrameRecord``: the full record as JSONB plus
    denormalised indexed projection columns and the MinIO object location of the
    frame image. ``upsert`` is keyed on the frame id (the per-frame capture id),
    so re-running a survey is idempotent.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # Projection columns are denormalised by design (one kwarg per indexed column).
    def upsert(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        frame_id: str,
        run_id: str,
        camera_id: str,
        phase: str,
        pose_id: str,
        commanded_pan: float,
        commanded_tilt: float,
        commanded_zoom: float,
        burst_id: str | None,
        frame_index: int,
        captured_at: datetime,
        minio_bucket: str,
        minio_object_key: str,
        checksum_sha256: str | None,
        record: dict[str, Any],
    ) -> bool:
        """Insert or update one frame row by ``frame_id``. Returns success."""
        try:
            row = self._session.get(CleanPlateFrameModel, frame_id)
            if row is None:
                row = CleanPlateFrameModel(id=frame_id)
                self._session.add(row)
            row.run_id = run_id
            row.camera_id = camera_id
            row.phase = phase
            row.pose_id = pose_id
            row.commanded_pan = commanded_pan
            row.commanded_tilt = commanded_tilt
            row.commanded_zoom = commanded_zoom
            row.burst_id = burst_id
            row.frame_index = frame_index
            row.captured_at = captured_at
            row.minio_bucket = minio_bucket
            row.minio_object_key = minio_object_key
            row.checksum_sha256 = checksum_sha256
            row.record = record
            self._session.flush()
            return True
        except Exception:
            logger.exception("Failed to upsert clean-plate frame %s", frame_id)
            return False


__all__ = ["RepoPostgresCleanPlateFrame"]
