"""Pydantic request/response schemas for camera-evaluation endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Survey request schemas
# ---------------------------------------------------------------------------


class SurveyStartRequest(BaseModel):
    """Body for ``POST /api/survey/start/``."""

    tenant_id: str
    camera_id: str
    axis: str
    start: float
    end: float
    step: float
    restore_ptz: bool = True
    retry_timeout: int = 60
    session_tags: list[str] | str = Field(default_factory=list)
    fixed_pan: float | None = None
    fixed_tilt: float | None = None
    fixed_zoom: float | None = None
