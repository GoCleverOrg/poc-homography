"""SQLAlchemy ORM model for clean-plate survey frames."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — required at runtime for Mapped[] resolution

from sqlalchemy import DateTime, Float, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from poc_homography.infrastructure.database import Base


class CleanPlateFrameModel(Base):
    """Clean-plate frames table — one row per captured survey frame.

    Stores the full :meth:`FrameRecord.to_dict` blob as JSONB (``record``,
    lossless) alongside denormalised, indexed projection columns for query and
    pagination, plus the MinIO object location of the frame image. The image
    bytes live in MinIO (bucket ``cleanplate-frames``); this table is the
    always-on, queryable metadata index a gallery reads even while the MinIO
    host (maglor) sleeps.

    Added additively — it does not touch the existing ``captured_frames`` table
    (a map-bound, annotation-linked entity that does not fit the rich survey
    ``FrameRecord``).
    """

    __tablename__ = "clean_plate_frames"
    __table_args__ = (Index("ix_clean_plate_frames_camera_captured", "camera_id", "captured_at"),)

    # id = capture_id (stable across re-runs); upsert target.
    id: Mapped[str] = mapped_column(String, primary_key=True)

    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    phase: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    pose_id: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)

    commanded_pan: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    commanded_tilt: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    commanded_zoom: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    burst_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # MinIO object location of the frame image (bucket + key + content hash).
    minio_bucket: Mapped[str] = mapped_column(String, nullable=False, default="")
    minio_object_key: Mapped[str] = mapped_column(String, nullable=False, default="")
    checksum_sha256: Mapped[str | None] = mapped_column(String, nullable=True)

    # Full FrameRecord.to_dict() — lossless; projection columns are denormalised.
    record: Mapped[dict] = mapped_column(JSONB, nullable=False)


__all__ = ["CleanPlateFrameModel"]
