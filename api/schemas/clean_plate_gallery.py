"""Pydantic response schemas for the clean-plate gallery endpoints."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — needed at runtime for Pydantic
from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Frame schemas
# ---------------------------------------------------------------------------


class CleanPlateFrameOut(BaseModel):
    """A single clean-plate frame: metadata + presigned image + thumbnail URL."""

    id: str
    run_id: str
    camera_id: str
    phase: str
    pose_id: str
    commanded_pan: float
    commanded_tilt: float
    commanded_zoom: float
    burst_id: str | None
    frame_index: int
    captured_at: datetime
    # Presigned MinIO GET URL for the full-resolution image (short TTL).
    image_url: str
    # imgproxy-served resized thumbnail (falls back to ``image_url`` when
    # imgproxy is not configured).
    thumbnail_url: str
    # Full lossless ``FrameRecord.to_dict()`` blob (optics, pose, etc.).
    record: dict[str, Any]


class CleanPlateFrameListResponse(BaseModel):
    """Envelope for ``GET /clean-plate/frames``."""

    frames: list[CleanPlateFrameOut]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Run schemas
# ---------------------------------------------------------------------------


class RunOut(BaseModel):
    """A distinct capture run with its frame count and time span."""

    run_id: str
    frame_count: int
    first_captured_at: datetime
    last_captured_at: datetime


class RunListResponse(BaseModel):
    """Envelope for ``GET /clean-plate/runs``."""

    runs: list[RunOut]


__all__ = [
    "CleanPlateFrameListResponse",
    "CleanPlateFrameOut",
    "RunListResponse",
    "RunOut",
]
