"""PostgreSQL-backed repository for clean-plate survey frames."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

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

    def query_frames(
        self,
        *,
        run_id: str | None = None,
        pose_id: str | None = None,
        camera_id: str | None = None,
        phase: str | None = None,
        captured_after: datetime | None = None,
        captured_before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[CleanPlateFrameModel], int]:
        """Query frame rows by filters, newest first, with pagination.

        Returns the page of rows (ordered by ``captured_at`` descending) and the
        total count matching the filters (ignoring ``limit``/``offset``), so the
        gallery can paginate. Consumed by the gallery API (#285).
        """
        model = CleanPlateFrameModel
        conditions = []
        if run_id is not None:
            conditions.append(model.run_id == run_id)
        if pose_id is not None:
            conditions.append(model.pose_id == pose_id)
        if camera_id is not None:
            conditions.append(model.camera_id == camera_id)
        if phase is not None:
            conditions.append(model.phase == phase)
        if captured_after is not None:
            conditions.append(model.captured_at >= captured_after)
        if captured_before is not None:
            conditions.append(model.captured_at <= captured_before)

        total = self._session.execute(
            select(func.count()).select_from(model).where(*conditions)
        ).scalar_one()

        stmt = (
            select(model)
            .where(*conditions)
            .order_by(model.captured_at.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = list(self._session.execute(stmt).scalars().all())
        return rows, total

    def list_runs(self) -> list[dict[str, Any]]:
        """List distinct runs with frame count and capture time span.

        One entry per ``run_id`` (newest run first), used by the gallery run
        picker (#285).
        """
        model = CleanPlateFrameModel
        stmt = (
            select(
                model.run_id,
                func.count().label("frame_count"),
                func.min(model.captured_at).label("first_captured_at"),
                func.max(model.captured_at).label("last_captured_at"),
            )
            .group_by(model.run_id)
            .order_by(func.max(model.captured_at).desc())
        )
        rows = self._session.execute(stmt).all()
        return [
            {
                "run_id": row.run_id,
                "frame_count": row.frame_count,
                "first_captured_at": row.first_captured_at,
                "last_captured_at": row.last_captured_at,
            }
            for row in rows
        ]


__all__ = ["RepoPostgresCleanPlateFrame"]
