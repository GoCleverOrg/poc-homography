"""FastAPI router for line-picker endpoints (image tiles, lines, geo coords)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from homography_web.frame_utils import LINES_DIR
from webapp.line_picker.state import (
    delete_line_from_repo,
    get_state,
    save_line_to_repo,
)

from api.deps import get_current_user
from api.schemas.line_picker import (
    AddLineRequest,
    DeleteLineResponse,
    GeoCoordsResponse,
    ImageInfoResponse,
    LineListResponse,
    LineOut,
    NextLineIdResponse,
    UpdateLineRequest,
)
from api.utils.tiles import render_full_image, render_tile
from poc_homography.infrastructure.models.user import UserModel

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/line-picker", tags=["line-picker"])

# ---------------------------------------------------------------------------
# Image endpoints
# ---------------------------------------------------------------------------


@router.get("/api/image/info/", response_model=ImageInfoResponse)
def image_info(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> ImageInfoResponse:
    """Return image metadata (dimensions, geotransform, CRS, filename)."""
    state = get_state(tenant_id)
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
) -> Response:
    """Return an OpenSeadragon tile as a PNG image.

    At level *z* the image appears at resolution ``original / 2^(max_level - z)``.
    """
    state = get_state(tenant_id)
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
) -> Response:
    """Return the full image scaled to *max_size* as a PNG."""
    state = get_state(tenant_id)
    png_bytes = render_full_image(image_path=state.geotiff_path, max_size=max_size)
    return Response(content=png_bytes, media_type="image/png")


# ---------------------------------------------------------------------------
# Line endpoints
# ---------------------------------------------------------------------------


@router.get("/api/lines/", response_model=LineListResponse)
def list_lines(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> LineListResponse:
    """List all lines for the tenant's map."""
    state = get_state(tenant_id)
    return LineListResponse(
        map_id=state.map_id,
        lines=[
            LineOut(
                line_id=line.line_id,
                start_x=line.start_x,
                start_y=line.start_y,
                end_x=line.end_x,
                end_y=line.end_y,
            )
            for line in state.lines
        ],
    )


@router.post("/api/lines/", response_model=LineOut)
def add_line(
    body: AddLineRequest,
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> LineOut:
    """Add a new line.

    Persists to the YAML repo before mutating in-memory state so a failed
    write never leaves state and disk out of sync.
    """
    start_x = float(body.start_x)
    start_y = float(body.start_y)
    end_x = float(body.end_x)
    end_y = float(body.end_y)

    state = get_state(tenant_id)

    try:
        resolved_id = body.line_id if body.line_id is not None else state.get_next_id()
        save_line_to_repo(resolved_id, start_x, start_y, end_x, end_y, state.map_id, LINES_DIR)
        state.add_line(start_x, start_y, end_x, end_y, line_id=resolved_id)

        return LineOut(
            line_id=resolved_id,
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.put("/api/lines/{line_id}/", response_model=LineOut)
def update_line(
    line_id: str,
    body: UpdateLineRequest,
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> LineOut:
    """Update the coordinate endpoints of an existing line.

    Supports partial updates -- only provided fields are changed.
    """
    state = get_state(tenant_id)

    line = state.get_line(line_id)
    if line is None:
        raise HTTPException(status_code=404, detail=f"Line not found: {line_id}")

    # Keep existing coords when not provided
    new_start_x = float(body.start_x) if body.start_x is not None else line.start_x
    new_start_y = float(body.start_y) if body.start_y is not None else line.start_y
    new_end_x = float(body.end_x) if body.end_x is not None else line.end_x
    new_end_y = float(body.end_y) if body.end_y is not None else line.end_y

    # Validate different endpoints
    if new_start_x == new_end_x and new_start_y == new_end_y:
        raise HTTPException(status_code=422, detail="Start and end points must be different")

    # Persist to YAML repo before mutating in-memory state
    save_line_to_repo(line_id, new_start_x, new_start_y, new_end_x, new_end_y, state.map_id, LINES_DIR)
    line.start_x = new_start_x
    line.start_y = new_start_y
    line.end_x = new_end_x
    line.end_y = new_end_y

    return LineOut(
        line_id=line_id,
        start_x=new_start_x,
        start_y=new_start_y,
        end_x=new_end_x,
        end_y=new_end_y,
    )


@router.delete("/api/lines/{line_id}/", response_model=DeleteLineResponse)
def delete_line(
    line_id: str,
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> DeleteLineResponse:
    """Delete a line."""
    state = get_state(tenant_id)

    try:
        # Persist to YAML repo before mutating in-memory state
        delete_line_from_repo(line_id, state.map_id, LINES_DIR)
        state.delete_line(line_id)
        return DeleteLineResponse(deleted=line_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Line not found: {line_id}")


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------


@router.get("/api/lines/next-id/", response_model=NextLineIdResponse)
def next_line_id(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> NextLineIdResponse:
    """Return the next auto-incremented line ID."""
    state = get_state(tenant_id)
    return NextLineIdResponse(next_id=state.get_next_id())


@router.get("/api/geo-coords/", response_model=GeoCoordsResponse)
def geo_coords(
    tenant_id: str = Query(...),
    pixel_x: float = Query(...),
    pixel_y: float = Query(...),
    user: UserModel = Depends(get_current_user),
) -> GeoCoordsResponse:
    """Convert pixel coordinates to geographic coordinates."""
    state = get_state(tenant_id)

    if state.geotiff is None:
        return GeoCoordsResponse(easting=None, northing=None, crs=None)

    easting, northing = state.geotiff.pixel_to_geo(pixel_x, pixel_y)
    return GeoCoordsResponse(
        easting=float(easting),
        northing=float(northing),
        crs=state.geotiff.crs,
    )
