"""Pydantic request/response schemas for camera-line-annotator endpoints."""

from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Line annotation schemas
# ---------------------------------------------------------------------------


class LineAnnotationOut(BaseModel):
    """Single line annotation serialisation."""

    line_id: str
    start_pixel_x: float
    start_pixel_y: float
    end_pixel_x: float
    end_pixel_y: float
    points: list[list[float]] | None = None


class CameraStatusOut(BaseModel):
    """Camera PTZ status for a frame."""

    pan: float | None = None
    tilt: float | None = None
    zoom: float | None = None


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class SwitchImageRequest(BaseModel):
    """Body for ``POST /api/switch-image/``."""

    filename: str


class CreateAnnotationRequest(BaseModel):
    """Body for ``POST /api/annotations/create/``."""

    image_filename: str
    line_id: str
    start_pixel_x: float
    start_pixel_y: float
    end_pixel_x: float
    end_pixel_y: float
    points: list[list[float]] | None = None


class UpdateAnnotationRequest(BaseModel):
    """Body for ``PUT /api/annotations/{line_id}/``."""

    image_filename: str
    start_pixel_x: float
    start_pixel_y: float
    end_pixel_x: float
    end_pixel_y: float
    points: list[list[float]] | None = None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class SwitchImageResponse(BaseModel):
    """Response for ``POST /api/switch-image/``."""

    success: bool
    filename: str
    annotations: list[LineAnnotationOut]
    camera_status: CameraStatusOut


class LineIdsResponse(BaseModel):
    """Response for ``GET /api/line-ids/``."""

    map_id: str
    line_ids: list[str]


class CreateAnnotationResponse(BaseModel):
    """Response for ``POST /api/annotations/create/``."""

    success: bool
    annotation: LineAnnotationOut


class UpdateAnnotationResponse(BaseModel):
    """Response for ``PUT /api/annotations/{line_id}/``."""

    success: bool
    annotation: LineAnnotationOut


class DeleteAnnotationResponse(BaseModel):
    """Response for ``DELETE /api/annotations/{line_id}/``."""

    success: bool
    deleted_line_id: str
