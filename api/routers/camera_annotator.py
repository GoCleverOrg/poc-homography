"""FastAPI router for camera-annotator endpoints."""

from __future__ import annotations

import mimetypes

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api.deps import get_current_user, get_db_session
from api.schemas.camera_annotator import (
    AnnotationOut,
    GcpOut,
    SaveAnnotationsRequest,
    SaveAnnotationsResponse,
    SwitchImageRequest,
    SwitchImageResponse,
)
from api.utils.frame_helpers import (
    FRAMES_DIR,
    get_frame_image_path,
    get_map_for_tenant,
    image_filename_to_frame,
    list_image_filenames,
    load_annotations_for_frame,
    validate_image_filename,
)
from poc_homography.domain.entities.annotation import Annotation
from poc_homography.domain.vo import PixelPoint
from poc_homography.infrastructure.models.user import UserModel
from poc_homography.infrastructure.repositories import RepoPostgresCapturedFrame
from poc_homography.map_points.gcp_registry import from_gcp_repo_pg

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/camera-annotator", tags=["camera-annotator"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_map_id(tenant_id: str, session: Session) -> str:
    """Return the map_id for *tenant_id*, raising 404 when missing."""
    map_entity = get_map_for_tenant(tenant_id, session)
    if map_entity is None:
        raise HTTPException(status_code=404, detail=f"No map found for tenant: {tenant_id}")
    return map_entity.id


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/api/gcps/", response_model=list[GcpOut])
def list_gcps(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> list[GcpOut]:
    """List GCPs for a tenant."""
    map_entity = get_map_for_tenant(tenant_id, session)
    if map_entity is None:
        return []
    try:
        registry = from_gcp_repo_pg(session, map_entity.id)
    except (KeyError, ValueError, OSError):
        return []

    return [
        GcpOut(id=pid, pixel_x=p.pixel_x, pixel_y=p.pixel_y)
        for pid, p in registry.points.items()
    ]


@router.get("/api/annotations/", response_model=list[AnnotationOut])
def get_annotations(
    tenant_id: str = Query(...),
    image_filename: str = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> list[AnnotationOut]:
    """Get annotations for a specific image."""
    frame = image_filename_to_frame(image_filename, session)
    if frame is None:
        return []
    annotations = load_annotations_for_frame(frame.id, session)
    return [AnnotationOut(**ann) for ann in annotations]


@router.get("/api/images/", response_model=list[str])
def list_images(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> list[str]:
    """List available images for a tenant."""
    map_id = _resolve_map_id(tenant_id, session)
    return list_image_filenames(session, map_id)


@router.post("/api/switch-image/", response_model=SwitchImageResponse)
def switch_image(
    body: SwitchImageRequest,
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> SwitchImageResponse:
    """Validate an image and return its annotations (stateless)."""
    if not validate_image_filename(body.filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    frame = image_filename_to_frame(body.filename, session)
    if frame is None:
        raise HTTPException(status_code=404, detail=f"Image not found: {body.filename}")

    # Ensure the frame belongs to the tenant's map
    map_entity = get_map_for_tenant(tenant_id, session)
    if map_entity and frame.map_id != map_entity.id:
        raise HTTPException(status_code=404, detail=f"Image not found: {body.filename}")

    annotations = load_annotations_for_frame(frame.id, session)
    return SwitchImageResponse(
        success=True,
        filename=body.filename,
        annotations=[AnnotationOut(**ann) for ann in annotations],
    )


@router.post("/api/save-annotations/", response_model=SaveAnnotationsResponse)
def save_annotations(
    body: SaveAnnotationsRequest,
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> SaveAnnotationsResponse:
    """Save annotations for an image."""
    frame = image_filename_to_frame(body.image_filename, session)
    if frame is None:
        raise HTTPException(status_code=400, detail="No image selected")

    if not body.annotations:
        raise HTTPException(status_code=400, detail="No annotations to save")

    # Build domain entities with rounded coordinates
    ann_entities = [
        Annotation(
            gcp_id=ann.gcp_id,
            frame_id=frame.id,
            camera_pose=frame.ptz_state,
            pixel=PixelPoint.create(
                round(float(ann.pixel_x), 1),
                round(float(ann.pixel_y), 1),
            ),
        )
        for ann in body.annotations
    ]

    RepoPostgresCapturedFrame(session, frames_dir=FRAMES_DIR).save_annotations(
        frame.id, ann_entities
    )

    return SaveAnnotationsResponse(success=True, saved=len(ann_entities))


@router.get("/image/")
def serve_image(
    tenant_id: str = Query(...),
    image_filename: str = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> FileResponse:
    """Serve a camera image file from disk."""
    if not validate_image_filename(image_filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    frame = image_filename_to_frame(image_filename, session)
    if frame is None:
        raise HTTPException(status_code=404, detail="Image not found")

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
