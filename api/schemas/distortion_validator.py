"""Pydantic request/response schemas for distortion-validator endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class DistortionCoefficientsIn(BaseModel):
    """Lens distortion coefficients."""

    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0


class IntrinsicsIn(BaseModel):
    """Camera intrinsic parameters."""

    fx: float = 1000.0
    fy: float = 1000.0
    cx: float | None = None
    cy: float | None = None


# ---------------------------------------------------------------------------
# calibration-files
# ---------------------------------------------------------------------------


class CalibrationFilesResponse(BaseModel):
    """Response for ``GET /api/calibration-files/``."""

    camera_ids: list[str]


# ---------------------------------------------------------------------------
# load-calibration
# ---------------------------------------------------------------------------


class LoadCalibrationRequest(BaseModel):
    """Body for ``POST /api/load-calibration/``."""

    camera_id: str


class LoadCalibrationResponse(BaseModel):
    """Response for ``POST /api/load-calibration/``."""

    camera_id: str
    entries: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------


class ImageItem(BaseModel):
    """Single image entry."""

    name: str
    source: str


class ImagesResponse(BaseModel):
    """Response for ``GET /api/images/``."""

    images: list[ImageItem]


# ---------------------------------------------------------------------------
# undistort
# ---------------------------------------------------------------------------


class UndistortRequest(BaseModel):
    """Body for ``POST /api/undistort/``."""

    image_path: str
    coefficients: DistortionCoefficientsIn = Field(default_factory=DistortionCoefficientsIn)
    intrinsics: IntrinsicsIn = Field(default_factory=IntrinsicsIn)
    use_opencv: bool = False


class UndistortResponse(BaseModel):
    """Response for ``POST /api/undistort/``."""

    original: str
    width: int
    height: int
    method_used: str
    coefficients_used: DistortionCoefficientsIn
    intrinsics_used: dict[str, float]
    undistorted: str | None = None
    undistorted_url: str | None = None


# ---------------------------------------------------------------------------
# transform-points
# ---------------------------------------------------------------------------


class TransformPointsRequest(BaseModel):
    """Body for ``POST /api/transform-points/``."""

    points: list[list[float]]
    direction: str = "undistort"
    coefficients: DistortionCoefficientsIn = Field(default_factory=DistortionCoefficientsIn)
    intrinsics: IntrinsicsIn = Field(default_factory=IntrinsicsIn)


class TransformPointsResponse(BaseModel):
    """Response for ``POST /api/transform-points/``."""

    points: list[list[float]]
    direction: str


# ---------------------------------------------------------------------------
# measure-straightness
# ---------------------------------------------------------------------------


class MeasureStraightnessRequest(BaseModel):
    """Body for ``POST /api/measure-straightness/``."""

    points: list[list[float]]
    undistort: bool = False
    coefficients: DistortionCoefficientsIn = Field(default_factory=DistortionCoefficientsIn)
    intrinsics: IntrinsicsIn = Field(default_factory=IntrinsicsIn)


# ---------------------------------------------------------------------------
# compute-intrinsics
# ---------------------------------------------------------------------------


class ComputeIntrinsicsRequest(BaseModel):
    """Body for ``POST /api/compute-intrinsics/``."""

    zoom: float = 1.0
    image_width: int = 1920
    image_height: int = 1080
    sensor_width_mm: float | None = None
    base_focal_length_mm: float | None = None


class ComputeIntrinsicsResponse(BaseModel):
    """Response for ``POST /api/compute-intrinsics/``."""

    fx: float
    fy: float
    cx: float
    cy: float
    focal_length_mm: float
    sensor_width_mm: float
    base_focal_length_mm: float
    zoom: float
    image_width: int
    image_height: int
