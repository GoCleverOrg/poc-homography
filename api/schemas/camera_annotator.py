"""Pydantic request/response schemas for camera-annotator endpoints."""

from __future__ import annotations

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# GCP schemas
# ---------------------------------------------------------------------------


class GcpOut(BaseModel):
    """Single GCP serialisation for the camera annotator."""

    id: str
    pixel_x: float
    pixel_y: float


# ---------------------------------------------------------------------------
# Annotation schemas
# ---------------------------------------------------------------------------


class AnnotationOut(BaseModel):
    """Single annotation serialisation."""

    gcp_id: str
    pixel_x: float
    pixel_y: float


class AnnotationIn(BaseModel):
    """Single annotation in a save request."""

    gcp_id: str
    pixel_x: float
    pixel_y: float


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class SwitchImageRequest(BaseModel):
    """Body for ``POST /api/switch-image/``."""

    filename: str


class SaveAnnotationsRequest(BaseModel):
    """Body for ``POST /api/save-annotations/``."""

    image_filename: str
    annotations: list[AnnotationIn]


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class SwitchImageResponse(BaseModel):
    """Response for ``POST /api/switch-image/``."""

    success: bool
    filename: str
    annotations: list[AnnotationOut]


class SaveAnnotationsResponse(BaseModel):
    """Response for ``POST /api/save-annotations/``."""

    success: bool
    saved: int
