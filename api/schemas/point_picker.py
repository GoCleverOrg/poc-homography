"""Pydantic request/response schemas for point-picker endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Image info
# ---------------------------------------------------------------------------


class ImageInfoResponse(BaseModel):
    """Response for ``GET /api/image/info/``."""

    width: int
    height: int
    geotransform: list[float] | None = None
    crs: str | None = None
    filename: str


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------


class PointOut(BaseModel):
    """Single GCP serialisation."""

    id: str
    pixel_x: float
    pixel_y: float
    tag: str


class PointListResponse(BaseModel):
    """Envelope for ``GET /api/points/``."""

    map_id: str
    points: list[PointOut]


class AddPointRequest(BaseModel):
    """Body for ``POST /api/points/``."""

    tag: Literal["parking_spot", "arrows", "crosswalk", "extra"] = "extra"
    pixel_x: float
    pixel_y: float
    id: str | None = None


class UpdatePointRequest(BaseModel):
    """Body for ``PUT /api/points/{point_id}/``."""

    pixel_x: float
    pixel_y: float


class DeletePointResponse(BaseModel):
    """Response for ``DELETE /api/points/{point_id}/``."""

    deleted: str


# ---------------------------------------------------------------------------
# Next ID
# ---------------------------------------------------------------------------


class NextIdResponse(BaseModel):
    """Response for ``GET /api/points/next-id/``."""

    tag: str
    next_id: str


# ---------------------------------------------------------------------------
# Geo coordinates
# ---------------------------------------------------------------------------


class GeoCoordsResponse(BaseModel):
    """Response for ``GET /api/geo-coords/``."""

    pixel_x: float
    pixel_y: float
    easting: float | None = None
    northing: float | None = None
    crs: str | None = None
