"""Pydantic request/response schemas for click-to-GPS endpoints (issue #33)."""

from __future__ import annotations

from pydantic import BaseModel


class FrameSummary(BaseModel):
    """A camera frame usable for click-to-GPS projection.

    Only frames with at least four GCP annotations can yield a homography, so
    ``annotation_count`` lets the UI flag (or hide) frames that cannot project.
    """

    name: str
    image: str
    annotation_count: int


class ProjectRequest(BaseModel):
    """Body for ``POST /click-to-gps/api/project/``.

    A single clicked pixel on a camera frame, identified by its test-case name
    (the image stem) plus the pixel coordinates in the camera image.
    """

    test_case_name: str
    pixel_x: float
    pixel_y: float


class ProjectResponse(BaseModel):
    """Result of projecting a clicked camera pixel to GPS.

    On success, ``latitude``/``longitude`` carry the WGS84 coordinate and
    ``confidence`` is the homography inlier ratio [0, 1]. On failure (e.g. a
    point that projects beyond the georeferenced map, the click-to-GPS analog
    of a point on/above the horizon), ``success`` is ``False`` and ``error``
    explains why; ``on_horizon`` marks that specific failure class.
    """

    success: bool
    latitude: float | None = None
    longitude: float | None = None
    confidence: float | None = None
    easting: float | None = None
    northing: float | None = None
    crs: str | None = None
    map_pixel_x: float | None = None
    map_pixel_y: float | None = None
    on_horizon: bool = False
    error: str | None = None
