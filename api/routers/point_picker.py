"""FastAPI router for point-picker endpoints (image tiles, GCPs, geo coords)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from api.deps import get_current_user, get_db_session
from api.schemas.point_picker import (
    AddPointRequest,
    DeletePointResponse,
    GeoCoordsResponse,
    ImageInfoResponse,
    NextIdResponse,
    PointListResponse,
    PointOut,
    UpdatePointRequest,
)
from api.utils.tiles import render_full_image, render_tile
from poc_homography.infrastructure.models.user import UserModel
from webapp.point_picker.state import (
    delete_gcp_from_repo_pg,
    get_state,
    get_tag_from_id,
    save_gcp_to_repo_pg,
)
from webapp.point_picker.validation import (
    validate_add_point_request,
    validate_update_point_request,
)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/point-picker", tags=["point-picker"])

# ---------------------------------------------------------------------------
# Image endpoints
# ---------------------------------------------------------------------------


@router.get("/api/image/info/", response_model=ImageInfoResponse)
def image_info(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> ImageInfoResponse:
    """Return image metadata (dimensions, geotransform, CRS, filename)."""
    state = get_state(tenant_id, session)
    return ImageInfoResponse(
        width=state.width,
        height=state.height,
        geotransform=state.geotiff.geotransform.to_list() if state.geotiff else None,
        crs=state.geotiff.crs if state.geotiff else None,
        filename=state.geotiff_path.name,
    )


@router.get("/api/image/tile/")
def image_tile(
    tenant_id: str = Query(...),
    x: int = Query(0),
    y: int = Query(0),
    z: int = Query(0),
    size: int = Query(256, ge=1, le=4096),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> Response:
    """Return an OpenSeadragon tile as a PNG image.

    At level *z* the image appears at resolution ``original / 2^(max_level - z)``.
    """
    state = get_state(tenant_id, session)
    png_bytes = render_tile(
        image_path=state.geotiff_path,
        width=state.width,
        height=state.height,
        x=x, y=y, z=z, size=size,
    )
    return Response(content=png_bytes, media_type="image/png")


@router.get("/api/image/full/")
def image_full(
    tenant_id: str = Query(...),
    max_size: int = Query(2048, ge=1, le=8192),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> Response:
    """Return the full image scaled to *max_size* as a PNG."""
    state = get_state(tenant_id, session)
    png_bytes = render_full_image(image_path=state.geotiff_path, max_size=max_size)
    return Response(content=png_bytes, media_type="image/png")


# ---------------------------------------------------------------------------
# GCP (point) endpoints
# ---------------------------------------------------------------------------


@router.get("/api/points/", response_model=PointListResponse)
def list_points(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> PointListResponse:
    """List all GCPs for the tenant's map."""
    state = get_state(tenant_id, session)
    return PointListResponse(
        map_id=state.registry.map_id,
        points=[
            PointOut(
                id=pid,
                pixel_x=p.pixel_x,
                pixel_y=p.pixel_y,
                tag=get_tag_from_id(pid),
            )
            for pid, p in state.registry.points.items()
        ],
    )


@router.post("/api/points/", response_model=PointOut)
def add_point(
    body: AddPointRequest,
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> PointOut:
    """Add a new GCP.

    Persists to the database before mutating in-memory state so a failed
    write never leaves state and DB out of sync.
    """
    data = body.model_dump()
    error = validate_add_point_request(data)
    if error:
        raise HTTPException(status_code=422, detail=error)

    state = get_state(tenant_id, session)

    tag = data.get("tag", "extra")
    pixel_x = float(data["pixel_x"])
    pixel_y = float(data["pixel_y"])
    point_id = data.get("id")

    resolved_id = point_id if point_id is not None else state.get_next_id(tag)
    save_gcp_to_repo_pg(resolved_id, pixel_x, pixel_y, state.map_id, session)
    point_id = state.add_point(tag, pixel_x, pixel_y, point_id=resolved_id)
    point = state.registry.points[point_id]

    return PointOut(
        id=point_id,
        pixel_x=point.pixel_x,
        pixel_y=point.pixel_y,
        tag=get_tag_from_id(point_id),
    )


@router.put("/api/points/{point_id}/", response_model=PointOut)
def update_point(
    point_id: str,
    body: UpdatePointRequest,
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> PointOut:
    """Update the pixel coordinates of an existing GCP."""
    data = body.model_dump()
    error = validate_update_point_request(data)
    if error:
        raise HTTPException(status_code=422, detail=error)

    state = get_state(tenant_id, session)

    px = float(data["pixel_x"])
    py = float(data["pixel_y"])

    try:
        save_gcp_to_repo_pg(point_id, px, py, state.map_id, session)
        state.update_point(point_id, px, py)
        point = state.registry.points[point_id]
        return PointOut(
            id=point_id,
            pixel_x=point.pixel_x,
            pixel_y=point.pixel_y,
            tag=get_tag_from_id(point_id),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Point not found: {point_id}")


@router.delete("/api/points/{point_id}/", response_model=DeletePointResponse)
def delete_point(
    point_id: str,
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> DeletePointResponse:
    """Delete a GCP."""
    state = get_state(tenant_id, session)

    if point_id not in state.registry.points:
        raise HTTPException(status_code=404, detail=f"Point not found: {point_id}")

    delete_gcp_from_repo_pg(point_id, state.map_id, session)
    state.delete_point(point_id)
    return DeletePointResponse(deleted=point_id)


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------


@router.get("/api/points/next-id/", response_model=NextIdResponse)
def next_id(
    tenant_id: str = Query(...),
    tag: str = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> NextIdResponse:
    """Return the next auto-incremented ID for a tag category."""
    state = get_state(tenant_id, session)
    try:
        nid = state.get_next_id(tag)
        return NextIdResponse(tag=tag, next_id=nid)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/api/geo-coords/", response_model=GeoCoordsResponse)
def geo_coords(
    tenant_id: str = Query(...),
    pixel_x: float = Query(...),
    pixel_y: float = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> GeoCoordsResponse:
    """Convert pixel coordinates to geographic coordinates."""
    state = get_state(tenant_id, session)
    coords = state.get_geo_coords(pixel_x, pixel_y)
    if coords:
        return GeoCoordsResponse(
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            easting=coords[0],
            northing=coords[1],
            crs=state.geotiff.crs if state.geotiff else None,
        )
    return GeoCoordsResponse(
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        easting=None,
        northing=None,
        crs=None,
    )
