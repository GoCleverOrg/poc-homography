"""FastAPI router for point-picker endpoints (image tiles, GCPs, geo coords)."""

from __future__ import annotations

import io
import math

import numpy as np
import tifffile
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from homography_web.frame_utils import GCPS_DIR, normalize_array
from PIL import Image

from api.deps import get_current_user
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
from poc_homography.infrastructure.models.user import UserModel
from webapp.point_picker.state import (
    delete_gcp_from_repo,
    get_state,
    get_tag_from_id,
    save_gcp_to_repo,
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
    size: int = Query(256),
    user: UserModel = Depends(get_current_user),
) -> Response:
    """Return an OpenSeadragon tile as a PNG image.

    At level *z* the image appears at resolution ``original / 2^(max_level - z)``.
    """
    state = get_state(tenant_id)

    # Calculate max level for the pyramid
    max_level = math.ceil(math.log2(max(state.width, state.height)))

    # At level z each pixel in the tile grid corresponds to
    # 2^(max_level-z) original pixels.
    level_scale = 2 ** (max_level - z)

    # Bounds in original image coordinates
    x0 = x * size * level_scale
    y0 = y * size * level_scale
    x1 = (x + 1) * size * level_scale
    y1 = (y + 1) * size * level_scale

    # Clamp to image bounds
    x0 = max(0, min(x0, state.width))
    y0 = max(0, min(y0, state.height))
    x1 = max(0, min(x1, state.width))
    y1 = max(0, min(y1, state.height))

    if x1 <= x0 or y1 <= y0:
        # Return transparent tile
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

    # Read tile based on file type
    suffix = state.geotiff_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(state.geotiff_path) as tif:
            page = tif.pages[0]
            data = page.asarray()

            if data.ndim == 2:  # noqa: PLR2004
                tile_data = data[y0:y1, x0:x1]
                img = Image.fromarray(normalize_array(tile_data), mode="L")
                img = img.convert("RGB")
            elif data.ndim == 3:  # noqa: PLR2004
                if data.shape[2] >= 3:  # noqa: PLR2004
                    tile_data = data[y0:y1, x0:x1, :3]
                    img = Image.fromarray(normalize_array(tile_data), mode="RGB")
                else:
                    tile_data = data[y0:y1, x0:x1, 0]
                    img = Image.fromarray(normalize_array(tile_data), mode="L")
                    img = img.convert("RGB")
            else:
                if data.shape[0] in (1, 3, 4):
                    if data.shape[0] == 1:
                        tile_data = data[0, y0:y1, x0:x1]
                        img = Image.fromarray(normalize_array(tile_data), mode="L")
                        img = img.convert("RGB")
                    else:
                        tile_data = np.transpose(data[:3, y0:y1, x0:x1], (1, 2, 0))
                        img = Image.fromarray(normalize_array(tile_data), mode="RGB")
                else:
                    tile_data = (
                        data[y0:y1, x0:x1]
                        if data.ndim == 2  # noqa: PLR2004
                        else data[y0:y1, x0:x1, 0]
                    )
                    img = Image.fromarray(normalize_array(tile_data), mode="L")
                    img = img.convert("RGB")
    else:
        with Image.open(state.geotiff_path) as full_img:
            img = full_img.crop((x0, y0, x1, y1))
            if img.mode != "RGB":
                img = img.convert("RGB")

    # Resize to tile size
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


@router.get("/api/image/full/")
def image_full(
    tenant_id: str = Query(...),
    max_size: int = Query(2048),
    user: UserModel = Depends(get_current_user),
) -> Response:
    """Return the full image scaled to *max_size* as a PNG."""
    state = get_state(tenant_id)

    suffix = state.geotiff_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(state.geotiff_path) as tif:
            page = tif.pages[0]
            data = page.asarray()

            if data.ndim == 2:  # noqa: PLR2004
                img = Image.fromarray(normalize_array(data), mode="L")
                img = img.convert("RGB")
            elif data.ndim == 3:  # noqa: PLR2004
                if data.shape[2] >= 3:  # noqa: PLR2004
                    img = Image.fromarray(normalize_array(data[:, :, :3]), mode="RGB")
                else:
                    img = Image.fromarray(normalize_array(data[:, :, 0]), mode="L")
                    img = img.convert("RGB")
            else:
                if data.shape[0] in (1, 3, 4):
                    if data.shape[0] == 1:
                        img = Image.fromarray(normalize_array(data[0]), mode="L")
                        img = img.convert("RGB")
                    else:
                        img_array = np.transpose(data[:3], (1, 2, 0))
                        img = Image.fromarray(normalize_array(img_array), mode="RGB")
                else:
                    img = Image.fromarray(normalize_array(data), mode="L")
                    img = img.convert("RGB")
    else:
        with Image.open(state.geotiff_path) as pil_img:
            img = pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img.copy()

    # Scale preserving aspect ratio
    ratio = min(max_size / img.width, max_size / img.height)
    if ratio < 1:
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# GCP (point) endpoints
# ---------------------------------------------------------------------------


@router.get("/api/points/", response_model=PointListResponse)
def list_points(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> PointListResponse:
    """List all GCPs for the tenant's map."""
    state = get_state(tenant_id)
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
) -> PointOut:
    """Add a new GCP.

    Persists to the YAML repo before mutating in-memory state so a failed
    write never leaves state and disk out of sync.
    """
    data = body.model_dump()
    error = validate_add_point_request(data)
    if error:
        raise HTTPException(status_code=422, detail=error)

    state = get_state(tenant_id)

    tag = data.get("tag", "extra")
    pixel_x = float(data["pixel_x"])
    pixel_y = float(data["pixel_y"])
    point_id = data.get("id")

    resolved_id = point_id if point_id is not None else state.get_next_id(tag)
    save_gcp_to_repo(resolved_id, pixel_x, pixel_y, state.map_id, GCPS_DIR)
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
) -> PointOut:
    """Update the pixel coordinates of an existing GCP."""
    data = body.model_dump()
    error = validate_update_point_request(data)
    if error:
        raise HTTPException(status_code=422, detail=error)

    state = get_state(tenant_id)

    px = float(data["pixel_x"])
    py = float(data["pixel_y"])

    try:
        save_gcp_to_repo(point_id, px, py, state.map_id, GCPS_DIR)
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
) -> DeletePointResponse:
    """Delete a GCP."""
    state = get_state(tenant_id)

    if point_id not in state.registry.points:
        raise HTTPException(status_code=404, detail=f"Point not found: {point_id}")

    delete_gcp_from_repo(point_id, state.map_id, GCPS_DIR)
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
) -> NextIdResponse:
    """Return the next auto-incremented ID for a tag category."""
    state = get_state(tenant_id)
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
) -> GeoCoordsResponse:
    """Convert pixel coordinates to geographic coordinates."""
    state = get_state(tenant_id)
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
