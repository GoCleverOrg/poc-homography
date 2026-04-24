"""FastAPI router for lens-calibration endpoints.

Ported from ``webapp/lens_calibration/views.py``.  Covers distortion
calibration from annotated lines, validation, save/load of calibration
tables, intrinsics computation, and line-trace-set listing/detail.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from homography_web.calibration_utils import (
    list_calibration_ids,
    load_calibration_from_repo,
    save_calibration_to_repo,
    serialize_calibration_entry,
)
from homography_web.frame_utils import (
    CALIBRATION_LINE_TRACES_DIR,
    CALIBRATIONS_DIR,
    get_line_annotation_repo,
    get_map_from_tenant_id,
)

from api.deps import get_current_user
from api.schemas.lens_calibration import (
    CalibrateAnnotatedLinesRequest,
    CalibrateAnnotatedLinesResponse,
    CalibrationIdsResponse,
    ComputeIntrinsicsRequest,
    ComputeIntrinsicsResponse,
    LineTraceSetDetailResponse,
    LineTraceSetsResponse,
    LoadCalibrationRequest,
    LoadCalibrationResponse,
    SaveCalibrationRequest,
    SaveCalibrationResponse,
    ValidateRequest,
    ValidateResponse,
)
from poc_homography.infrastructure.models.user import UserModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cached repo
# ---------------------------------------------------------------------------

_line_trace_set_repo = None


def _get_line_trace_set_repo():
    """Return a cached ``RepoYamlCalibrationLineTraceSet`` instance."""
    global _line_trace_set_repo
    if _line_trace_set_repo is None:
        from poc_homography.infrastructure.repositories import (
            RepoYamlCalibrationLineTraceSet,
        )

        _line_trace_set_repo = RepoYamlCalibrationLineTraceSet(CALIBRATION_LINE_TRACES_DIR)
    return _line_trace_set_repo


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/lens-calibration", tags=["lens-calibration"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_LINE_ANNOTATIONS = 500  # Prevent DoS via excessive input


def _build_intrinsic_matrix(
    data: CalibrateAnnotatedLinesRequest,
) -> tuple[Any, dict[str, float]]:
    """Build intrinsic matrix from request data, computing from specs if requested.

    Returns ``(intrinsic_matrix, intrinsics_dict)``.
    """
    import numpy as np

    intrinsics = data.intrinsics

    if data.auto_intrinsics:
        from poc_homography.camera.intrinsics import compute_intrinsics
        from poc_homography.camera_config import (
            DEFAULT_BASE_FOCAL_LENGTH_MM,
            DEFAULT_SENSOR_WIDTH_MM,
        )

        zoom = float(data.zoom if data.zoom is not None else (intrinsics.zoom or 1.0))
        result = compute_intrinsics(
            zoom=zoom,
            image_width=intrinsics.image_width,
            image_height=intrinsics.image_height,
            sensor_width_mm=float(
                intrinsics.sensor_width_mm
                if intrinsics.sensor_width_mm is not None
                else DEFAULT_SENSOR_WIDTH_MM
            ),
            base_focal_length_mm=float(
                intrinsics.base_focal_length_mm
                if intrinsics.base_focal_length_mm is not None
                else DEFAULT_BASE_FOCAL_LENGTH_MM
            ),
        )
        resolved = {
            "fx": float(result.focal_length_px),
            "fy": float(result.focal_length_px),
            "cx": float(result.cx),
            "cy": float(result.cy),
        }
    else:
        resolved = {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
        }

    intrinsic_matrix = np.array(
        [
            [resolved["fx"], 0.0, resolved["cx"]],
            [0.0, resolved["fy"], resolved["cy"]],
            [0.0, 0.0, 1.0],
        ]
    )

    return intrinsic_matrix, resolved


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/api/calibrate-annotated-lines/",
    response_model=CalibrateAnnotatedLinesResponse,
)
def calibrate_annotated_lines(
    body: CalibrateAnnotatedLinesRequest,
    user: UserModel = Depends(get_current_user),
) -> CalibrateAnnotatedLinesResponse:
    """Run distortion calibration using manually annotated N-point line traces."""
    if not body.camera_line_annotations:
        raise HTTPException(status_code=400, detail="Missing or empty camera_line_annotations")

    if len(body.camera_line_annotations) > MAX_LINE_ANNOTATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many line annotations (max {MAX_LINE_ANNOTATIONS})",
        )

    try:
        from poc_homography.calibration.lens_distortion.annotated_line_solver import (
            AnnotatedLineSolver,
            AnnotatedLineSolverConfig,
            build_camera_line_annotations,
        )

        intrinsic_matrix, intrinsics_used = _build_intrinsic_matrix(body)

        annotations_raw = [
            {"line_id": a.line_id, "points": a.points} for a in body.camera_line_annotations
        ]
        lines = build_camera_line_annotations(annotations_raw)

        solver_config = AnnotatedLineSolverConfig(
            train_split_ratio=body.config.train_split_ratio,
            use_radial_only=body.config.use_radial_only,
        )
        solver = AnnotatedLineSolver(config=solver_config)

        result = solver.solve(lines, intrinsic_matrix)

        improvement_percent = (
            (1 - result.improvement_ratio()) * 100 if result.success else 0.0
        )

        return CalibrateAnnotatedLinesResponse(
            success=result.success,
            message=result.message,
            iterations=result.iterations,
            initial_error=result.initial_error,
            final_error=result.final_error,
            overall_rmse=result.overall_rmse,
            coefficients={
                "k1": float(result.distortion.k1),
                "k2": float(result.distortion.k2),
                "k3": float(result.distortion.k3),
                "p1": float(result.distortion.p1),
                "p2": float(result.distortion.p2),
            },
            intrinsics_used=intrinsics_used,
            quality=(
                "good"
                if result.overall_rmse < 2.0
                else "acceptable"
                if result.overall_rmse < 5.0
                else "poor"
            ),
            line_errors=result.line_errors[:20],
            improvement_percent=improvement_percent,
            intrinsics=result.intrinsics if result.intrinsics else None,
        )

    except ImportError:
        logger.exception("Annotated line solver module not available")
        raise HTTPException(
            status_code=500, detail="Annotated line solver module not available"
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Annotated line calibration failed")
        raise HTTPException(status_code=500, detail="Annotated line calibration failed")


@router.post("/api/validate/", response_model=ValidateResponse)
def validate_calibration(
    body: ValidateRequest,
    user: UserModel = Depends(get_current_user),
) -> ValidateResponse:
    """Validate calibration by computing straightness RMSE on test lines."""
    try:
        import numpy as np

        from poc_homography.calibration.lens_distortion.distortion_solver import (
            straightness_rmse,
        )
        from poc_homography.calibration.lens_distortion.models import (
            CameraLine,
            PTZPosition,
        )
        from poc_homography.domain.vo import LensDistortion
        from poc_homography.types import Degrees, Unitless

        intrinsics = body.intrinsics
        intrinsic_matrix = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.cx],
                [0.0, intrinsics.fy, intrinsics.cy],
                [0.0, 0.0, 1.0],
            ]
        )

        camera_lines: list[CameraLine] = []
        for i, line in enumerate(body.lines):
            ptz = PTZPosition(
                pan_deg=Degrees(line.pan),
                tilt_deg=Degrees(line.tilt),
                zoom_factor=line.zoom,
            )
            edge_pixels = None
            if line.points and len(line.points) >= 2:
                edge_pixels = tuple(tuple(pt) for pt in line.points)
            camera_line = CameraLine(
                line_id=line.line_id or f"line_{i:04d}",
                image_path=line.image_path,
                start_pixel=(line.start_x, line.start_y),
                end_pixel=(line.end_x, line.end_y),
                ptz_position=ptz,
                edge_pixels=edge_pixels,
            )
            camera_lines.append(camera_line)

        if not camera_lines:
            raise HTTPException(status_code=400, detail="No lines provided")

        baseline_rmse = straightness_rmse(camera_lines, intrinsic_matrix)

        coeffs = body.coefficients
        distortion = LensDistortion(
            k1=Unitless(coeffs.k1),
            k2=Unitless(coeffs.k2),
            k3=Unitless(coeffs.k3),
            p1=Unitless(coeffs.p1),
            p2=Unitless(coeffs.p2),
        )
        corrected_rmse = straightness_rmse(camera_lines, intrinsic_matrix, distortion=distortion)

        improvement = (
            (baseline_rmse - corrected_rmse) / baseline_rmse * 100
            if baseline_rmse > 0
            else 0
        )

        return ValidateResponse(
            baseline_rmse=baseline_rmse,
            corrected_rmse=corrected_rmse,
            improvement_percent=improvement,
            num_lines=len(camera_lines),
        )

    except ImportError:
        logger.exception("Calibration module not available")
        raise HTTPException(status_code=500, detail="Calibration module not available")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Validation failed")
        raise HTTPException(status_code=500, detail="Validation failed")


@router.post("/api/save/", response_model=SaveCalibrationResponse)
def save_calibration(
    body: SaveCalibrationRequest,
    user: UserModel = Depends(get_current_user),
) -> SaveCalibrationResponse:
    """Save calibration results via the DDD repo."""
    try:
        from poc_homography.calibration.lens_distortion.calibration_table import (
            CameraCalibrationTable,
            ZoomCalibrationEntry,
        )
        from poc_homography.domain.vo import LensDistortion
        from poc_homography.types import Unitless

        coeffs = body.coefficients
        distortion = LensDistortion(
            k1=Unitless(coeffs.k1),
            k2=Unitless(coeffs.k2),
            k3=Unitless(coeffs.k3),
            p1=Unitless(coeffs.p1),
            p2=Unitless(coeffs.p2),
        )

        table = CameraCalibrationTable(camera_id=body.camera_id)
        entry = ZoomCalibrationEntry.from_solver_result(
            zoom_factor=body.zoom,
            distortion=distortion,
            validation_rmse=body.validation_rmse,
            source_images=[],
            num_lines_used=body.num_lines,
            fx=float(body.intrinsics.fx) if body.intrinsics else 0.0,
            fy=float(body.intrinsics.fy) if body.intrinsics else 0.0,
            cx=float(body.intrinsics.cx) if body.intrinsics else 0.0,
            cy=float(body.intrinsics.cy) if body.intrinsics else 0.0,
        )
        table.add_entry(entry)

        save_calibration_to_repo(table, CALIBRATIONS_DIR)

        return SaveCalibrationResponse(success=True, camera_id=body.camera_id)

    except ImportError:
        logger.exception("Calibration module not available")
        raise HTTPException(status_code=500, detail="Calibration module not available")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Save failed")
        raise HTTPException(status_code=500, detail="Save failed")


@router.post("/api/load/", response_model=LoadCalibrationResponse)
def load_calibration(
    body: LoadCalibrationRequest,
    user: UserModel = Depends(get_current_user),
) -> LoadCalibrationResponse:
    """Load calibration from DDD repo by camera_id."""
    if not body.camera_id:
        raise HTTPException(status_code=400, detail="Missing camera_id")

    try:
        entity = load_calibration_from_repo(body.camera_id, CALIBRATIONS_DIR)
        if entity is None:
            raise HTTPException(
                status_code=404,
                detail=f"No calibration found for {body.camera_id}",
            )

        entries = [serialize_calibration_entry(entry) for entry in entity.entries]

        return LoadCalibrationResponse(camera_id=entity.id, entries=entries)

    except ImportError:
        logger.exception("Calibration module not available")
        raise HTTPException(status_code=500, detail="Calibration module not available")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Load failed")
        raise HTTPException(status_code=500, detail="Load failed")


@router.get("/api/calibration-ids/", response_model=CalibrationIdsResponse)
def calibration_ids(
    user: UserModel = Depends(get_current_user),
) -> CalibrationIdsResponse:
    """List available camera_ids in the calibration repo."""
    try:
        camera_ids = list_calibration_ids(CALIBRATIONS_DIR)
        return CalibrationIdsResponse(camera_ids=camera_ids)
    except Exception:
        logger.exception("Failed to list calibration IDs")
        raise HTTPException(status_code=500, detail="Failed to list calibration IDs")


@router.post("/api/compute-intrinsics/", response_model=ComputeIntrinsicsResponse)
def compute_intrinsics(
    body: ComputeIntrinsicsRequest,
    user: UserModel = Depends(get_current_user),
) -> ComputeIntrinsicsResponse:
    """Compute camera intrinsics from sensor specs and zoom level."""
    from poc_homography.camera.intrinsics import compute_intrinsics as _compute_intrinsics
    from poc_homography.camera_config import (
        DEFAULT_BASE_FOCAL_LENGTH_MM,
        DEFAULT_SENSOR_WIDTH_MM,
    )

    zoom = body.zoom
    sensor_width_mm = (
        body.sensor_width_mm if body.sensor_width_mm is not None else DEFAULT_SENSOR_WIDTH_MM
    )
    base_focal_length_mm = (
        body.base_focal_length_mm
        if body.base_focal_length_mm is not None
        else DEFAULT_BASE_FOCAL_LENGTH_MM
    )

    if zoom <= 0 or body.image_width <= 0 or body.image_height <= 0:
        raise HTTPException(
            status_code=400,
            detail="zoom, image_width, and image_height must be positive",
        )

    try:
        result = _compute_intrinsics(
            zoom=zoom,
            image_width=body.image_width,
            image_height=body.image_height,
            sensor_width_mm=sensor_width_mm,
            base_focal_length_mm=base_focal_length_mm,
        )

        return ComputeIntrinsicsResponse(
            fx=float(result.focal_length_px),
            fy=float(result.focal_length_px),
            cx=float(result.cx),
            cy=float(result.cy),
            focal_length_mm=float(result.focal_length_mm),
            sensor_width_mm=sensor_width_mm,
            base_focal_length_mm=base_focal_length_mm,
            zoom=zoom,
            image_width=body.image_width,
            image_height=body.image_height,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {e}")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to compute intrinsics")
        raise HTTPException(status_code=500, detail="Failed to compute intrinsics")


@router.get("/api/line-trace-sets/", response_model=LineTraceSetsResponse)
def line_trace_sets(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> LineTraceSetsResponse:
    """List available line trace sources.

    Merges CalibrationLineTraceSet entity names with LineAnnotation frame
    groups so that annotations created via camera_line_annotator also appear.
    """
    try:
        repo = _get_line_trace_set_repo()
        entities = repo.get_all()
        names = sorted(e.name for e in entities)

        map_entity = get_map_from_tenant_id(tenant_id)
        map_id = map_entity.id if map_entity else None
        all_frame_ids = {ann.frame_id for ann in get_line_annotation_repo().get_all()}
        if map_id:
            all_frame_ids = {fid for fid in all_frame_ids if fid.startswith(map_id + "/")}
        frame_ids = sorted(all_frame_ids)

        return LineTraceSetsResponse(names=names + frame_ids)
    except Exception:
        logger.exception("Failed to list line trace sets")
        raise HTTPException(status_code=500, detail="Failed to list line trace sets")


@router.get("/api/line-trace-set-detail/", response_model=LineTraceSetDetailResponse)
def line_trace_set_detail(
    name: str = Query(...),
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> LineTraceSetDetailResponse:
    """Load line traces by name.

    First tries CalibrationLineTraceSet entities, then falls back to
    LineAnnotation entities grouped by frame_id.
    """
    try:
        # Try CalibrationLineTraceSet first
        repo = _get_line_trace_set_repo()
        entity = repo.get(name)
        if entity is not None:
            return LineTraceSetDetailResponse(
                name=entity.name,
                line_traces=[lt.to_dict() for lt in entity.line_traces],
            )

        # Fall back to LineAnnotation entities matching this frame_id (tenant-scoped)
        map_entity = get_map_from_tenant_id(tenant_id)
        map_id = map_entity.id if map_entity else None
        frame_anns = [
            ann
            for ann in get_line_annotation_repo().get_all()
            if ann.frame_id == name and (not map_id or name.startswith(map_id + "/"))
        ]
        if frame_anns:
            line_traces: list[dict] = []
            for ann in frame_anns:
                if ann.points is not None:
                    points = [[float(p.x), float(p.y)] for p in ann.points]
                else:
                    points = [
                        [float(ann.start_pixel.x), float(ann.start_pixel.y)],
                        [float(ann.end_pixel.x), float(ann.end_pixel.y)],
                    ]
                line_traces.append({"line_id": ann.line_id, "points": points})
            return LineTraceSetDetailResponse(name=name, line_traces=line_traces)

        raise HTTPException(status_code=404, detail=f"Not found: {name}")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to load line trace set %s", name)
        raise HTTPException(status_code=500, detail="Failed to load line trace set")
