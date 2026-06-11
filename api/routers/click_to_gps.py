"""FastAPI router for the interactive click-to-GPS tool (issue #33).

Translates the issue's "click on the live camera stream → GPS" intent onto the
live api/ + spa/ stack. The desktop ``VideoAnnotator`` described in the issue
never existed on this stack; instead the SPA serves a saved camera frame, the
user clicks a pixel, and this endpoint projects that pixel to a WGS84 GPS
coordinate by reusing the existing registration domain services:

    clicked camera pixel
      → MapPointHomography.camera_to_map()      (GCP homography → map pixel)
      → GeoTiff.pixel_to_geo()                  (map pixel → UTM easting/northing)
      → GeoTiff.pixel_to_latlon()               (UTM → WGS84 lat/lon, via pyproj)

The homography inlier ratio is surfaced as the projection confidence so the UI
can colour-code each clicked point.
"""

from __future__ import annotations

import math
import mimetypes
from typing import TYPE_CHECKING

import tifffile
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from api.deps import get_current_user, get_db_session
from api.schemas.click_to_gps import FrameSummary, ProjectRequest, ProjectResponse
from api.utils.frame_helpers import (
    DATA_MAPS_DIR,
    extract_geotiff,
    get_frame_image_path,
    get_map_for_tenant,
    image_filename_to_frame,
    list_frames,
    load_annotations_for_frame,
    validate_image_filename,
)
from poc_homography.domain.vo import PixelPoint
from poc_homography.homography.map_points import MapPointHomography
from poc_homography.map_points.gcp_registry import from_gcp_repo_pg

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.orm import Session

    from poc_homography.domain.vo.geotiff import GeoTiff
    from poc_homography.infrastructure.models.user import UserModel

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/click-to-gps", tags=["click-to-gps"])

# Minimum GCP annotations required to fit a homography for a frame.
_MIN_ANNOTATIONS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_map_id(tenant_id: str, session: Session) -> str | None:
    """Return the tenant's map_id, or ``None`` when no map is configured."""
    map_entity = get_map_for_tenant(tenant_id, session)
    return map_entity.id if map_entity else None


def _find_test_case(name: str, map_id: str | None, session: Session) -> dict | None:
    """Return ``{image, annotations}`` for the frame whose image stem is *name*."""
    for frame in list_frames(session, map_id):
        if frame.image_path.stem != name:
            continue
        annotations = load_annotations_for_frame(frame.id, session)
        return {"image": frame.image_path.name, "annotations": annotations}
    return None


def _load_geotiff(tenant_id: str, session: Session) -> tuple[GeoTiff, int, int] | None:
    """Return ``(geotiff, width, height)`` for the tenant's map GeoTIFF.

    Returns ``None`` when no map is configured, the file is missing, or the
    raster carries no georeferencing tags (without which GPS is undefined).
    """
    map_entity = get_map_for_tenant(tenant_id, session)
    if map_entity is None:
        return None
    geotiff_path: Path = DATA_MAPS_DIR / map_entity.photo.path
    if not geotiff_path.exists():
        return None

    with tifffile.TiffFile(geotiff_path) as tif:
        page = tif.pages[0]
        width: int = page.imagewidth  # type: ignore[union-attr]
        height: int = page.imagelength  # type: ignore[union-attr]
        geotiff = extract_geotiff(tif)

    if geotiff is None:
        return None
    return (geotiff, width, height)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/api/frames/", response_model=list[FrameSummary])
def list_projectable_frames(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> list[FrameSummary]:
    """List camera frames with enough GCP annotations to project from."""
    map_id = _resolve_map_id(tenant_id, session)
    frames: list[FrameSummary] = []
    for frame in list_frames(session, map_id):
        annotations = load_annotations_for_frame(frame.id, session)
        if len(annotations) < _MIN_ANNOTATIONS:
            continue
        frames.append(
            FrameSummary(
                name=frame.image_path.stem,
                image=frame.image_path.name,
                annotation_count=len(annotations),
            )
        )
    return frames


@router.post("/api/project/", response_model=ProjectResponse)
def project_pixel_to_gps(
    body: ProjectRequest,
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> ProjectResponse:
    """Project a clicked camera pixel to a WGS84 GPS coordinate."""
    map_id = _resolve_map_id(tenant_id, session)
    if map_id is None:
        raise HTTPException(
            status_code=422,
            detail="No map configured for the current tenant. Upload a GeoTIFF map first.",
        )

    test_case = _find_test_case(body.test_case_name, map_id, session)
    if test_case is None:
        raise HTTPException(status_code=404, detail=f"Frame not found: {body.test_case_name}")

    annotations = test_case["annotations"]
    if len(annotations) < _MIN_ANNOTATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {_MIN_ANNOTATIONS} annotations, got {len(annotations)}",
        )

    geo = _load_geotiff(tenant_id, session)
    if geo is None:
        raise HTTPException(
            status_code=422,
            detail="Map GeoTIFF is missing georeferencing; cannot derive GPS.",
        )
    geotiff, map_width, map_height = geo

    try:
        registry = from_gcp_repo_pg(session, map_id)
    except (KeyError, ValueError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load GCP registry: {exc}")

    # Fit the GCP homography for this frame. min_inlier_ratio=0 so we always
    # obtain a transform and report its (possibly low) confidence rather than
    # rejecting — the UI colour-codes low-confidence projections.
    try:
        homography = MapPointHomography(map_id=registry.map_id)
        result = homography.compute_from_gcps(
            gcps=annotations,
            map_registry=registry,
            ransac_threshold=50.0,
            min_inlier_ratio=0.0,
        )
    except (ValueError, RuntimeError) as exc:
        return ProjectResponse(success=False, error=f"Homography computation failed: {exc}")

    # Project the clicked pixel: camera pixel → map pixel → UTM → lat/lon.
    map_point = homography.camera_to_map(PixelPoint.create(body.pixel_x, body.pixel_y))
    map_x, map_y = map_point.pixel_x, map_point.pixel_y

    # A point near/above the horizon projects far outside the georeferenced map
    # plane (or to a non-finite coordinate). Reject those rather than emitting a
    # nonsensical GPS fix.
    margin_x = map_width
    margin_y = map_height
    if (
        not (math.isfinite(map_x) and math.isfinite(map_y))
        or map_x < -margin_x
        or map_x > map_width + margin_x
        or map_y < -margin_y
        or map_y > map_height + margin_y
    ):
        return ProjectResponse(
            success=False,
            on_horizon=True,
            confidence=round(result.inlier_ratio, 3),
            error="Point projects beyond the map (likely on or above the horizon).",
        )

    easting, northing = geotiff.pixel_to_geo(map_x, map_y)
    latitude, longitude = geotiff.pixel_to_latlon(map_x, map_y)

    return ProjectResponse(
        success=True,
        latitude=round(float(latitude), 7),
        longitude=round(float(longitude), 7),
        confidence=round(result.inlier_ratio, 3),
        easting=round(float(easting), 3),
        northing=round(float(northing), 3),
        crs=geotiff.crs,
        map_pixel_x=round(map_x, 2),
        map_pixel_y=round(map_y, 2),
    )


@router.get("/image/")
def serve_image(
    tenant_id: str = Query(...),
    image_filename: str = Query(...),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> FileResponse:
    """Serve a camera frame image file from disk (read-only)."""
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
