"""FastAPI router for lens-calibration endpoints.

Ported from ``webapp/lens_calibration/views.py``.  Covers validation,
save/load of calibration tables, intrinsics computation, and
line-trace-set listing/detail.  Lens-distortion calibration itself is
automatic-only (scene self-calibration); there is no manual
annotated-line calibration endpoint.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query
from homography_web.calibration_utils import serialize_calibration_entry

from api.deps import get_current_user, get_db_session
from api.schemas.lens_calibration import (
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
    ZoomEntryIn,
)
from api.utils.frame_helpers import get_map_for_tenant
from poc_homography.calibration.lens_distortion.ddd_sync import sync_to_ddd_repo_pg
from poc_homography.infrastructure.repositories import (
    RepoPostgresCalibrationLineTraceSet,
    RepoPostgresLensCalibrationTable,
    RepoPostgresLineAnnotation,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from poc_homography.infrastructure.models.user import UserModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/lens-calibration", tags=["lens-calibration"])

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


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
            (baseline_rmse - corrected_rmse) / baseline_rmse * 100 if baseline_rmse > 0 else 0
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
    session: Session = Depends(get_db_session),
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

        # Normalize the two request shapes into a single list of entries: the
        # legacy single-entry body is just the 1-element case of a multi-zoom
        # batch. This keeps the save logic below at one altitude (one loop).
        entries = body.zoom_entries or [
            ZoomEntryIn(
                zoom=body.zoom,
                coefficients=body.coefficients,
                intrinsics=body.intrinsics,
                validation_rmse=body.validation_rmse,
                num_lines=body.num_lines,
            )
        ]

        table = CameraCalibrationTable(camera_id=body.camera_id)
        for e in entries:
            distortion = LensDistortion(
                k1=Unitless(e.coefficients.k1),
                k2=Unitless(e.coefficients.k2),
                k3=Unitless(e.coefficients.k3),
                p1=Unitless(e.coefficients.p1),
                p2=Unitless(e.coefficients.p2),
            )
            table.add_entry(
                ZoomCalibrationEntry.from_solver_result(
                    zoom_factor=e.zoom,
                    distortion=distortion,
                    validation_rmse=e.validation_rmse,
                    source_images=[],
                    num_lines_used=e.num_lines,
                    fx=float(e.intrinsics.fx) if e.intrinsics else 0.0,
                    fy=float(e.intrinsics.fy) if e.intrinsics else 0.0,
                    cx=float(e.intrinsics.cx) if e.intrinsics else 0.0,
                    cy=float(e.intrinsics.cy) if e.intrinsics else 0.0,
                )
            )

        sync_to_ddd_repo_pg(table, session)

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
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> LoadCalibrationResponse:
    """Load calibration from DDD repo by camera_id."""
    if not body.camera_id:
        raise HTTPException(status_code=400, detail="Missing camera_id")

    try:
        repo = RepoPostgresLensCalibrationTable(session)
        entity = repo.get(body.camera_id)
        if entity is None:
            raise HTTPException(
                status_code=404,
                detail=f"No calibration found for {body.camera_id}",
            )

        entries = [serialize_calibration_entry(entry) for entry in entity.entries]

        return LoadCalibrationResponse(camera_id=entity.id, entries=entries)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Load failed")
        raise HTTPException(status_code=500, detail="Load failed")


@router.get("/api/calibration-ids/", response_model=CalibrationIdsResponse)
def calibration_ids(
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> CalibrationIdsResponse:
    """List available camera_ids in the calibration repo."""
    try:
        repo = RepoPostgresLensCalibrationTable(session)
        camera_ids = sorted(entity.id for entity in repo.get_all())
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
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> LineTraceSetsResponse:
    """List available line trace sources.

    Merges CalibrationLineTraceSet entity names with LineAnnotation frame
    groups so that annotations created via camera_line_annotator also appear.
    """
    try:
        repo = RepoPostgresCalibrationLineTraceSet(session)
        entities = repo.get_all()
        names = sorted(e.name for e in entities)

        map_entity = get_map_for_tenant(tenant_id, session)
        map_id = map_entity.id if map_entity else None
        line_ann_repo = RepoPostgresLineAnnotation(session)
        all_frame_ids = {ann.frame_id for ann in line_ann_repo.get_all()}
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
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> LineTraceSetDetailResponse:
    """Load line traces by name.

    First tries CalibrationLineTraceSet entities, then falls back to
    LineAnnotation entities grouped by frame_id.
    """
    try:
        # Try CalibrationLineTraceSet first
        repo = RepoPostgresCalibrationLineTraceSet(session)
        entity = repo.get(name)
        if entity is not None:
            return LineTraceSetDetailResponse(
                name=entity.name,
                line_traces=[lt.to_dict() for lt in entity.line_traces],
            )

        # Fall back to LineAnnotation entities matching this frame_id (tenant-scoped)
        map_entity = get_map_for_tenant(tenant_id, session)
        map_id = map_entity.id if map_entity else None
        line_ann_repo = RepoPostgresLineAnnotation(session)
        frame_anns = [
            ann
            for ann in line_ann_repo.get_all()
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
