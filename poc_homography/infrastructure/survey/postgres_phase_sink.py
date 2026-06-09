"""Postgres + MinIO :class:`PhaseSink`: frames to MinIO + Neon, lossless.

For every survey frame this sink uploads the image bytes to a MinIO bucket and
upserts a ``clean_plate_frames`` row (indexed projection columns + the full
``FrameRecord.to_dict()`` as JSONB + the MinIO object location). The result is
an always-on, queryable metadata index a gallery reads even while the MinIO host
(maglor) sleeps.

Inventory and video-burst records (Phase 1 / Phase 8) are not part of the
clean-plate frame model; they are forwarded to an optional ``fallback`` sink
(e.g. a :class:`YamlPhaseSink`) when one is supplied, and otherwise logged and
skipped (never silently dropped).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from poc_homography.infrastructure.database import get_session
from poc_homography.infrastructure.repositories.repo_postgres_clean_plate_frame import (
    RepoPostgresCleanPlateFrame,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session

    from poc_homography.domain.entities.survey.frame_record import FrameRecord
    from poc_homography.domain.entities.survey.inventory_record import (
        CameraInventoryRecord,
    )
    from poc_homography.domain.entities.survey.video_burst_record import (
        VideoBurstRecord,
    )
    from poc_homography.infrastructure.clients.minio_frame_store import MinioFrameStore
    from poc_homography.survey.phases.common import PhaseSink

logger = logging.getLogger(__name__)


class PostgresPhaseSink:
    """Persist frames to MinIO (image) + Postgres (metadata).

    Implements the :class:`~poc_homography.survey.phases.common.PhaseSink`
    protocol structurally.
    """

    def __init__(
        self,
        frame_store: MinioFrameStore,
        *,
        session_factory: Callable[[], AbstractContextManager[Session]] | None = None,
        fallback: PhaseSink | None = None,
    ) -> None:
        """Build the sink.

        Args:
            frame_store: MinIO/S3 store the frame images are uploaded to.
            session_factory: Callable returning a context-managed SQLAlchemy
                session that commits on exit. Defaults to the app
                :func:`~poc_homography.infrastructure.database.get_session`.
                Injectable for tests.
            fallback: Optional sink that receives inventory/burst records this
                sink does not model.
        """
        self._store = frame_store
        self._session_factory = session_factory or get_session
        self._fallback = fallback
        self._bucket_ready = False

    def save_inventory(self, record: CameraInventoryRecord) -> None:
        """Forward the Phase 1 inventory record to the fallback sink, or skip."""
        if self._fallback is not None:
            self._fallback.save_inventory(record)
        else:
            logger.warning(
                "PostgresPhaseSink does not persist inventory; skipping %s (no fallback)",
                record.id,
            )

    def save_frame(self, record: FrameRecord) -> None:
        """Upload the frame image to MinIO and upsert its metadata row."""
        if not self._bucket_ready:
            self._store.ensure_bucket()
            self._bucket_ready = True

        image_bytes = Path(record.image_data.image_path).read_bytes()
        key = self._object_key(record)
        put = self._store.put_frame(image_bytes, key)

        with self._session_factory() as session:
            repo = RepoPostgresCleanPlateFrame(session)
            repo.upsert(
                frame_id=record.id,
                run_id=record.capture.run_id,
                camera_id=record.camera.camera_id,
                phase=record.capture.phase.value,
                pose_id=record.survey_context.pose_id or "",
                commanded_pan=float(record.commanded.commanded_pan),
                commanded_tilt=float(record.commanded.commanded_tilt),
                commanded_zoom=float(record.commanded.commanded_zoom),
                burst_id=record.capture.burst_id,
                frame_index=record.capture.frame_index,
                captured_at=record.capture.timestamp_at_capture,
                minio_bucket=put.bucket,
                minio_object_key=put.key,
                checksum_sha256=put.sha256,
                record=record.to_dict(),
            )

    def save_burst(self, record: VideoBurstRecord) -> None:
        """Forward the Phase 8 burst record to the fallback sink, or skip."""
        if self._fallback is not None:
            self._fallback.save_burst(record)
        else:
            logger.warning(
                "PostgresPhaseSink does not persist video bursts; skipping %s (no fallback)",
                record.id,
            )

    @staticmethod
    def _object_key(record: FrameRecord) -> str:
        """Deterministic, human-navigable, unique MinIO object key for a frame.

        ``{run_id}/{phase}/{pose_id}/{capture_id}.jpg`` — grouped by run/phase/
        pose for browsing; the capture id makes it unique and the upload
        idempotent across re-runs.
        """
        pose = record.survey_context.pose_id or "_nopose"
        return f"{record.capture.run_id}/{record.capture.phase.value}/{pose}/{record.id}.jpg"


__all__ = ["PostgresPhaseSink"]
