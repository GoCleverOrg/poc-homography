"""Pydantic request/response schemas for line-picker endpoints."""

from __future__ import annotations

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Image info (re-uses the same shape as point-picker)
# ---------------------------------------------------------------------------


class ImageInfoResponse(BaseModel):
    """Response for ``GET /api/image/info/``."""

    width: int
    height: int
    geotransform: list[float] | None = None
    crs: str | None = None
    filename: str


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------


class LineOut(BaseModel):
    """Single line serialisation."""

    line_id: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float


class LineListResponse(BaseModel):
    """Envelope for ``GET /api/lines/``."""

    map_id: str
    lines: list[LineOut]


class AddLineRequest(BaseModel):
    """Body for ``POST /api/lines/``."""

    start_x: float
    start_y: float
    end_x: float
    end_y: float
    line_id: str | None = None


class UpdateLineRequest(BaseModel):
    """Body for ``PUT /api/lines/{line_id}/``."""

    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None


class DeleteLineResponse(BaseModel):
    """Response for ``DELETE /api/lines/{line_id}/``."""

    deleted: str


# ---------------------------------------------------------------------------
# Next ID
# ---------------------------------------------------------------------------


class NextLineIdResponse(BaseModel):
    """Response for ``GET /api/lines/next-id/``."""

    next_id: str


# ---------------------------------------------------------------------------
# Geo coordinates
# ---------------------------------------------------------------------------


class GeoCoordsResponse(BaseModel):
    """Response for ``GET /api/geo-coords/``."""

    easting: float | None = None
    northing: float | None = None
    crs: str | None = None
