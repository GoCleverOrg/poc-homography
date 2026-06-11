"""FastAPI router for the clean-plate gallery (frames + runs).

Reads survey frame metadata from the Neon ``clean_plate_frames`` table and mints
short-TTL presigned MinIO URLs for the full images plus signed imgproxy URLs for
thumbnails, so the SPA gallery can browse maglor's captured floor images without
downloading them.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — needed at runtime for Query typing
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Query

from api.deps import get_current_user, get_db_session
from api.schemas.clean_plate_gallery import (
    CleanPlateFrameListResponse,
    CleanPlateFrameOut,
    RunListResponse,
    RunOut,
)
from api.utils.imgproxy import ImgproxySigner
from poc_homography.infrastructure.clients.minio_frame_store import MinioFrameStore
from poc_homography.infrastructure.repositories.repo_postgres_clean_plate_frame import (
    RepoPostgresCleanPlateFrame,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from poc_homography.infrastructure.models.clean_plate_frame import CleanPlateFrameModel
    from poc_homography.infrastructure.models.user import UserModel

# Presigned-URL TTL (seconds). Short-lived: the gallery re-fetches on load.
_PRESIGN_TTL = 3600

router = APIRouter(prefix="/clean-plate", tags=["clean-plate"])


def _to_frame_out(
    row: CleanPlateFrameModel,
    store: MinioFrameStore,
    signer: ImgproxySigner | None,
) -> CleanPlateFrameOut:
    """Map a DB row to the API schema, minting image + thumbnail URLs."""
    image_url = store.presign_get(row.minio_object_key, expires_in=_PRESIGN_TTL)
    if signer is not None and row.minio_bucket and row.minio_object_key:
        thumbnail_url = signer.thumbnail_url(f"s3://{row.minio_bucket}/{row.minio_object_key}")
    else:
        # imgproxy not configured — fall back to the full presigned image.
        thumbnail_url = image_url
    return CleanPlateFrameOut(
        id=row.id,
        run_id=row.run_id,
        camera_id=row.camera_id,
        phase=row.phase,
        pose_id=row.pose_id,
        commanded_pan=row.commanded_pan,
        commanded_tilt=row.commanded_tilt,
        commanded_zoom=row.commanded_zoom,
        burst_id=row.burst_id,
        frame_index=row.frame_index,
        captured_at=row.captured_at,
        image_url=image_url,
        thumbnail_url=thumbnail_url,
        record=row.record,
    )


@router.get("/frames", response_model=CleanPlateFrameListResponse)
def get_frames(
    run_id: str | None = Query(default=None),
    pose_id: str | None = Query(default=None),
    camera_id: str | None = Query(default=None),
    phase: str | None = Query(default=None),
    captured_after: datetime | None = Query(default=None),
    captured_before: datetime | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> CleanPlateFrameListResponse:
    """Query clean-plate frames with filters + pagination.

    Returns metadata plus a presigned MinIO image URL and an imgproxy thumbnail
    URL per frame, newest first.
    """
    repo = RepoPostgresCleanPlateFrame(session)
    rows, total = repo.query_frames(
        run_id=run_id,
        pose_id=pose_id,
        camera_id=camera_id,
        phase=phase,
        captured_after=captured_after,
        captured_before=captured_before,
        limit=limit,
        offset=offset,
    )

    if rows:
        store = MinioFrameStore.from_env()
        signer = ImgproxySigner.from_env()
        frames = [_to_frame_out(row, store, signer) for row in rows]
    else:
        frames = []

    return CleanPlateFrameListResponse(frames=frames, total=total, limit=limit, offset=offset)


@router.get("/runs", response_model=RunListResponse)
def get_runs(
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> RunListResponse:
    """List distinct capture runs (with frame counts and time spans)."""
    repo = RepoPostgresCleanPlateFrame(session)
    runs = repo.list_runs()
    return RunListResponse(runs=[RunOut(**run) for run in runs])
