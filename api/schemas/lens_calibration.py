"""Pydantic request/response schemas for lens-calibration endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class IntrinsicsIn(BaseModel):
    """Camera intrinsic parameters."""

    fx: float = 1000.0
    fy: float = 1000.0
    cx: float = 960.0
    cy: float = 540.0
    image_width: int = 1920
    image_height: int = 1080
    sensor_width_mm: float | None = None
    base_focal_length_mm: float | None = None
    zoom: float | None = None


class IntrinsicsOut(BaseModel):
    """Resolved intrinsic parameters returned by the server."""

    fx: float
    fy: float
    cx: float
    cy: float


class DistortionCoefficients(BaseModel):
    """Lens distortion coefficients."""

    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0


# ---------------------------------------------------------------------------
# calibrate-annotated-lines
# ---------------------------------------------------------------------------


class CameraLineAnnotationIn(BaseModel):
    """Single annotated line with N-point trace."""

    line_id: str
    points: list[list[float]]


class CalibrationConfigIn(BaseModel):
    """Optional solver configuration."""

    train_split_ratio: float = 0.7
    use_radial_only: bool = False


class CalibrateAnnotatedLinesRequest(BaseModel):
    """Body for ``POST /api/calibrate-annotated-lines/``."""

    camera_line_annotations: list[CameraLineAnnotationIn]
    intrinsics: IntrinsicsIn
    auto_intrinsics: bool = False
    zoom: float | None = None
    config: CalibrationConfigIn = Field(default_factory=CalibrationConfigIn)


class CalibrateAnnotatedLinesResponse(BaseModel):
    """Response for ``POST /api/calibrate-annotated-lines/``."""

    success: bool
    message: str
    iterations: int
    initial_error: float
    final_error: float
    overall_rmse: float
    coefficients: DistortionCoefficients
    intrinsics_used: IntrinsicsOut
    quality: str
    line_errors: list[float]
    improvement_percent: float
    intrinsics: dict | None = None


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class ValidationLineIn(BaseModel):
    """Single line for validation."""

    line_id: str | None = None
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    pan: float = 0.0
    tilt: float = 30.0
    zoom: float = 1.0
    image_path: str = ""
    points: list[list[float]] | None = None


class ValidateRequest(BaseModel):
    """Body for ``POST /api/validate/``."""

    intrinsics: IntrinsicsIn
    coefficients: DistortionCoefficients = Field(default_factory=DistortionCoefficients)
    lines: list[ValidationLineIn] = Field(default_factory=list)


class ValidateResponse(BaseModel):
    """Response for ``POST /api/validate/``."""

    baseline_rmse: float
    corrected_rmse: float
    improvement_percent: float
    num_lines: int


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


class ZoomEntryIn(BaseModel):
    """Single per-zoom calibration entry for a multi-zoom batch save."""

    zoom: float
    coefficients: DistortionCoefficients = Field(default_factory=DistortionCoefficients)
    intrinsics: IntrinsicsOut | None = None
    validation_rmse: float = 0.0
    num_lines: int = 0


class SaveCalibrationRequest(BaseModel):
    """Body for ``POST /api/save/``."""

    camera_id: str = "unknown_camera"
    zoom: float = 1.0
    coefficients: DistortionCoefficients = Field(default_factory=DistortionCoefficients)
    validation_rmse: float = 0.0
    intrinsics: IntrinsicsOut | None = None
    num_lines: int = 0
    zoom_entries: list[ZoomEntryIn] | None = None


class SaveCalibrationResponse(BaseModel):
    """Response for ``POST /api/save/``."""

    success: bool
    camera_id: str


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


class LoadCalibrationRequest(BaseModel):
    """Body for ``POST /api/load/``."""

    camera_id: str


class LoadCalibrationResponse(BaseModel):
    """Response for ``POST /api/load/``."""

    camera_id: str
    entries: list[dict]


# ---------------------------------------------------------------------------
# calibration-ids
# ---------------------------------------------------------------------------


class CalibrationIdsResponse(BaseModel):
    """Response for ``GET /api/calibration-ids/``."""

    camera_ids: list[str]


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


# ---------------------------------------------------------------------------
# line-trace-sets
# ---------------------------------------------------------------------------


class LineTraceSetsResponse(BaseModel):
    """Response for ``GET /api/line-trace-sets/``."""

    names: list[str]


class LineTraceSetDetailResponse(BaseModel):
    """Response for ``GET /api/line-trace-set-detail/``."""

    name: str
    line_traces: list[dict]
