"""Pydantic request/response schemas for homography-precision endpoints."""

from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestCaseSummary(BaseModel):
    """Single test case in the listing."""

    name: str
    image: str
    annotation_count: int


class TestCaseListResponse(BaseModel):
    """Response for ``GET /api/test-cases/``."""

    test_cases: list[TestCaseSummary]


class TestCaseDetailResponse(BaseModel):
    """Response for ``GET /api/test-cases/{name}/``."""

    name: str
    image: str
    annotations: list[dict]


# ---------------------------------------------------------------------------
# Line test cases
# ---------------------------------------------------------------------------


class LineTestCaseSummary(BaseModel):
    """Single line test case in the listing."""

    name: str
    image: str
    line_annotation_count: int


class LineTestCaseListResponse(BaseModel):
    """Response for ``GET /api/line-test-cases/``."""

    test_cases: list[LineTestCaseSummary]


class LineTestCaseDetailResponse(BaseModel):
    """Response for ``GET /api/line-test-cases/{name}/``."""

    name: str
    image: str
    point_annotations_ref: str
    line_annotations: list[dict]


# ---------------------------------------------------------------------------
# GCP registry
# ---------------------------------------------------------------------------


class GCPPointOut(BaseModel):
    """Single GCP point in the registry."""

    pixel_x: float
    pixel_y: float


class GCPRegistryResponse(BaseModel):
    """Response for ``GET /api/gcp-registry/``."""

    map_id: str
    points: dict[str, GCPPointOut]


# ---------------------------------------------------------------------------
# Line registry
# ---------------------------------------------------------------------------


class LineOut(BaseModel):
    """Single line in the registry."""

    line_id: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float


class LineRegistryResponse(BaseModel):
    """Response for ``GET /api/line-registry/``."""

    map_id: str | None
    lines: list[LineOut]


# ---------------------------------------------------------------------------
# Compute homography (point-based)
# ---------------------------------------------------------------------------


class ComputeHomographyRequest(BaseModel):
    """Body for ``POST /api/compute-homography/``."""

    test_case_name: str


class HomographyMetrics(BaseModel):
    """Metrics from point-based homography computation."""

    num_gcps: int
    num_inliers: int
    inlier_ratio: float
    mean_reproj_error: float
    max_reproj_error: float
    rmse: float


class PerPointError(BaseModel):
    """Per-point error detail."""

    gcp_id: str
    error_px: float
    camera_dx: float
    camera_dy: float
    map_dx: float
    map_dy: float
    camera_original: list[float]
    camera_reprojected: list[float]
    map_original: list[float]
    map_projected: list[float]


class ComputeHomographyResponse(BaseModel):
    """Response for ``POST /api/compute-homography/``."""

    success: bool
    metrics: HomographyMetrics | None = None
    per_point_errors: list[PerPointError] | None = None
    overlays: dict | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Compute homography from lines
# ---------------------------------------------------------------------------


class ComputeHomographyFromLinesRequest(BaseModel):
    """Body for ``POST /api/compute-homography-from-lines/``."""

    test_case_name: str | None = None
    line_annotations: list[dict] | None = None


class LineHomographyMetrics(BaseModel):
    """Metrics from line-based homography computation."""

    num_lines: int
    num_inliers: int
    inlier_ratio: float
    mean_perp_error: float
    max_perp_error: float
    rmse: float


class ComputeHomographyFromLinesResponse(BaseModel):
    """Response for ``POST /api/compute-homography-from-lines/``."""

    success: bool
    homography_source: str | None = None
    metrics: LineHomographyMetrics | None = None
    homography_matrix: list[list[float]] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Compute line errors
# ---------------------------------------------------------------------------


class ComputeLineErrorsRequest(BaseModel):
    """Body for ``POST /api/compute-line-errors/``."""

    test_case_name: str
    use_line_homography: bool = False


class LineErrorMetrics(BaseModel):
    """Metrics from line error computation."""

    num_lines: int
    mean_line_error: float
    max_line_error: float


class PerLineError(BaseModel):
    """Per-line error detail."""

    line_id: str
    error_px: float
    start_error: float
    end_error: float
    map_start_error: float
    map_end_error: float


class ComputeLineErrorsResponse(BaseModel):
    """Response for ``POST /api/compute-line-errors/``."""

    success: bool
    metrics: LineErrorMetrics | None = None
    per_line_errors: list[PerLineError] | None = None
    line_overlays: dict | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Image info
# ---------------------------------------------------------------------------


class CameraInfoResponse(BaseModel):
    """Response for ``GET /api/camera-info/``."""

    width: int
    height: int
    filename: str


class MapInfoResponse(BaseModel):
    """Response for ``GET /api/map-info/``."""

    width: int
    height: int
    filename: str
    geotransform: list[float] | None = None
    crs: str | None = None
