"""FastAPI application entry point."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SmartTerminal API",
    description="REST API for camera homography and coordinate transformations",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Health check (no auth required — used by Dockerfile HEALTHCHECK)
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------

_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

_extra = os.environ.get("CORS_ORIGINS", "")
_origins = _DEFAULT_ORIGINS + [o.strip() for o in _extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from api.routers import (  # noqa: E402 — routers imported after app + CORS setup (intentional)
    camera_annotator,
    camera_diagnostic,
    camera_evaluation,
    camera_line_annotator,
    clean_plate_gallery,
    click_to_gps,
    distortion_validator,
    gcp,
    homography_precision,
    lens_calibration,
    line_picker,
    point_picker,
)

app.include_router(camera_annotator.router)
app.include_router(camera_diagnostic.router)
app.include_router(camera_evaluation.router)
app.include_router(camera_line_annotator.router)
app.include_router(clean_plate_gallery.router)
app.include_router(click_to_gps.router)
app.include_router(distortion_validator.router)
app.include_router(gcp.router)
app.include_router(homography_precision.router)
app.include_router(lens_calibration.router)
app.include_router(line_picker.router)
app.include_router(point_picker.router)
