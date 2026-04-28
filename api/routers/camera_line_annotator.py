"""FastAPI router for camera-line-annotator endpoints."""

from __future__ import annotations

import mimetypes
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from line_picker.state import from_line_repo_pg
from sqlalchemy.orm import Session

from api.deps import get_current_user, get_db_session
from api.schemas.camera_line_annotator import (
    CameraStatusOut,
    CreateAnnotationRequest,
    CreateAnnotationResponse,
    DeleteAnnotationResponse,
    LineAnnotationOut,
    LineIdsResponse,
    SwitchImageRequest,
    SwitchImageResponse,
    UpdateAnnotationRequest,
    UpdateAnnotationResponse,
)
from api.utils.frame_helpers import (
    get_frame_image_path,
    get_map_for_tenant,
    image_filename_to_frame,
    list_image_filenames,
    load_line_annotations_for_frame,
    validate_image_filename,
)
from poc_homography.domain.entities.line_annotation import LineAnnotation
from poc_homography.domain.vo import PixelPoint
from poc_homography.infrastructure.models.user import UserModel
from poc_homography.infrastructure.repositories import RepoPostgresLineAnnotation

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/camera-line-annotator", tags=["camera-line-annotator"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LINE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_map_id(tenant_id: str, session: Session) -> str:
    """Return the map_id for *tenant_id*, raising 404 when missing."""
    map_entity = get_map_for_tenant(tenant_id, session)
    if map_entity is None:
        raise HTTPException(status_code=404, detail=f"No map found for tenant: {tenant_id}")
    return map_entity.id


def _validate_line_id(line_id: str) -> None:
    """Raise 400 if *line_id* does not match the expected format."""
    if not line_id or not _LINE_ID_RE.fullmatch(line_id):
        raise HTTPException(status_code=400, detail="Invalid line_id format")


def _validate_and_get_frame(image_filename: str, session: Session):
    """Validate *image_filename* and return the CapturedFrame, or raise."""
    if not validate_image_filename(image_filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    frame = image_filename_to_frame(image_filename, session)
    if frame is None:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_filename}")
    return frame


def _annotation_dict_to_out(ann: dict) -> LineAnnotationOut:
    """Convert a legacy annotation dict to the response schema."""
    return LineAnnotationOut(
        line_id=ann["line_id"],
        start_pixel_x=ann["start_pixel_x"],
        start_pixel_y=ann["start_pixel_y"],
        end_pixel_x=ann["end_pixel_x"],
        end_pixel_y=ann["end_pixel_y"],
        points=ann.get("points"),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/api/images/", response_model=list[str])
def list_images(
    tenant_id: str = Query(...),
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> list[str]:
    """List available images for a tenant."""
    map_id = _resolve_map_id(tenant_id, session)
    return list_image_filenames(session, map_id)


@router.post("/api/switch-image/", response_model=SwitchImageResponse)
def switch_image(
    body: SwitchImageRequest,
    tenant_id: str = Query(...),
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> SwitchImageResponse:
    """Validate an image and return its line annotations and camera status (stateless)."""
    frame = _validate_and_get_frame(body.filename, session)

    # Ensure the frame belongs to the tenant's map
    map_entity = get_map_for_tenant(tenant_id, session)
    if map_entity and frame.map_id != map_entity.id:
        raise HTTPException(status_code=404, detail=f"Image not found: {body.filename}")

    annotations = load_line_annotations_for_frame(frame.id, session)

    camera_status = CameraStatusOut(
        pan=float(frame.ptz_state.pan_raw),
        tilt=float(frame.ptz_state.tilt_deg),
        zoom=float(frame.ptz_state.zoom),
    )

    return SwitchImageResponse(
        success=True,
        filename=body.filename,
        annotations=[_annotation_dict_to_out(ann) for ann in annotations],
        camera_status=camera_status,
    )


@router.get("/image/")
def serve_image(
    tenant_id: str = Query(...),
    image_filename: str = Query(...),
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> FileResponse:
    """Serve a camera image file from disk."""
    frame = _validate_and_get_frame(image_filename, session)

    resolved_path = get_frame_image_path(frame)
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    mime_type, _ = mimetypes.guess_type(str(resolved_path))
    if not mime_type:
        mime_type = "image/jpeg"

    return FileResponse(
        path=resolved_path,
        media_type=mime_type,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/api/line-ids/", response_model=LineIdsResponse)
def list_line_ids(
    tenant_id: str = Query(...),
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> LineIdsResponse:
    """Get available line IDs from the lines registry."""
    map_entity = get_map_for_tenant(tenant_id, session)
    if map_entity is None:
        raise HTTPException(status_code=404, detail=f"No map found for tenant: {tenant_id}")

    repo_lines = from_line_repo_pg(session, map_entity.id)

    line_ids = [line.line_id for line in repo_lines if line.line_id]
    return LineIdsResponse(map_id=map_entity.id, line_ids=line_ids)


@router.get("/api/annotations/", response_model=list[LineAnnotationOut])
def get_annotations(
    tenant_id: str = Query(...),
    image_filename: str = Query(...),
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> list[LineAnnotationOut]:
    """Get line annotations for a specific image."""
    frame = _validate_and_get_frame(image_filename, session)
    annotations = load_line_annotations_for_frame(frame.id, session)
    return [_annotation_dict_to_out(ann) for ann in annotations]


@router.post("/api/annotations/create/", response_model=CreateAnnotationResponse)
def create_annotation(
    body: CreateAnnotationRequest,
    tenant_id: str = Query(...),
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> CreateAnnotationResponse:
    """Create a new line annotation."""
    _validate_line_id(body.line_id)

    frame = _validate_and_get_frame(body.image_filename, session)

    repo = RepoPostgresLineAnnotation(session)

    points_tuple: tuple[PixelPoint, ...] | None = None
    if body.points and len(body.points) >= 2:
        points_tuple = tuple(PixelPoint.create(float(p[0]), float(p[1])) for p in body.points)

    line_ann = LineAnnotation(
        line_id=body.line_id,
        frame_id=frame.id,
        camera_pose=frame.ptz_state,
        start_pixel=PixelPoint.create(body.start_pixel_x, body.start_pixel_y),
        end_pixel=PixelPoint.create(body.end_pixel_x, body.end_pixel_y),
        points=points_tuple,
    )
    repo.save(line_ann)

    annotation_out = LineAnnotationOut(
        line_id=body.line_id,
        start_pixel_x=body.start_pixel_x,
        start_pixel_y=body.start_pixel_y,
        end_pixel_x=body.end_pixel_x,
        end_pixel_y=body.end_pixel_y,
        points=[[float(p[0]), float(p[1])] for p in body.points] if body.points and len(body.points) >= 2 else None,
    )

    return CreateAnnotationResponse(success=True, annotation=annotation_out)


@router.put("/api/annotations/{line_id}/", response_model=UpdateAnnotationResponse)
def update_annotation(
    line_id: str,
    body: UpdateAnnotationRequest,
    tenant_id: str = Query(...),
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> UpdateAnnotationResponse:
    """Update an existing line annotation."""
    _validate_line_id(line_id)

    frame = _validate_and_get_frame(body.image_filename, session)

    repo = RepoPostgresLineAnnotation(session)
    entity_id = f"{frame.id}/{line_id}"

    if not repo.exists(entity_id):
        raise HTTPException(status_code=404, detail=f"Annotation not found: {line_id}")

    points_tuple: tuple[PixelPoint, ...] | None = None
    if body.points and len(body.points) >= 2:
        points_tuple = tuple(PixelPoint.create(float(p[0]), float(p[1])) for p in body.points)

    line_ann = LineAnnotation(
        line_id=line_id,
        frame_id=frame.id,
        camera_pose=frame.ptz_state,
        start_pixel=PixelPoint.create(body.start_pixel_x, body.start_pixel_y),
        end_pixel=PixelPoint.create(body.end_pixel_x, body.end_pixel_y),
        points=points_tuple,
    )
    repo.save(line_ann)

    annotation_out = LineAnnotationOut(
        line_id=line_id,
        start_pixel_x=body.start_pixel_x,
        start_pixel_y=body.start_pixel_y,
        end_pixel_x=body.end_pixel_x,
        end_pixel_y=body.end_pixel_y,
        points=[[float(p[0]), float(p[1])] for p in body.points] if body.points and len(body.points) >= 2 else None,
    )

    return UpdateAnnotationResponse(success=True, annotation=annotation_out)


@router.delete("/api/annotations/{line_id}/", response_model=DeleteAnnotationResponse)
def delete_annotation(
    line_id: str,
    tenant_id: str = Query(...),
    image_filename: str = Query(...),
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> DeleteAnnotationResponse:
    """Delete a line annotation."""
    _validate_line_id(line_id)

    frame = _validate_and_get_frame(image_filename, session)

    repo = RepoPostgresLineAnnotation(session)
    entity_id = f"{frame.id}/{line_id}"

    deleted = repo.delete(entity_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Annotation not found: {line_id}")

    return DeleteAnnotationResponse(success=True, deleted_line_id=line_id)


@router.get("/api/camera-status/", response_model=CameraStatusOut)
def camera_status(
    tenant_id: str = Query(...),
    image_filename: str = Query(...),
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> CameraStatusOut:
    """Get camera PTZ status for a specific image."""
    frame = _validate_and_get_frame(image_filename, session)

    return CameraStatusOut(
        pan=float(frame.ptz_state.pan_raw),
        tilt=float(frame.ptz_state.tilt_deg),
        zoom=float(frame.ptz_state.zoom),
    )
