"""FastAPI router for distortion-validator endpoints.

Ported from ``webapp/distortion_validator/views.py``.  Provides endpoints for
evaluating and visualizing lens distortion calibration results: loading
calibrations, listing images, undistorting images, transforming points,
measuring line straightness, and computing intrinsics.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from homography_web.calibration_utils import serialize_calibration_entry
from sqlalchemy.orm import Session

from api.deps import get_current_user, get_db_session
from api.schemas.distortion_validator import (
    CalibrationFilesResponse,
    ComputeIntrinsicsRequest,
    ComputeIntrinsicsResponse,
    ImageItem,
    ImagesResponse,
    LoadCalibrationRequest,
    LoadCalibrationResponse,
    MeasureStraightnessRequest,
    TransformPointsRequest,
    TransformPointsResponse,
    UndistortRequest,
    UndistortResponse,
)
from api.utils.frame_helpers import (
    WEBAPP_DIR,
    get_frame_image_path,
    get_map_for_tenant,
    image_filename_to_frame,
    list_image_filenames,
    validate_image_filename,
)
from poc_homography.infrastructure.models.user import UserModel
from poc_homography.infrastructure.repositories import RepoPostgresLensCalibrationTable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SURVEY_DIR = WEBAPP_DIR / "survey"
RESULT_IMAGE_DIR = WEBAPP_DIR / "distortion_validator" / "_result_images"

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/distortion-validator", tags=["distortion-validator"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_POINTS = 10_000  # Prevent DoS via excessive input


def _resolve_image_path(image_path: str, session: Session) -> Path | None:
    """Resolve an image path safely against known directories."""
    # Try as filename in the CapturedFrame repo
    frame = image_filename_to_frame(image_path, session)
    if frame is not None:
        fp = get_frame_image_path(frame)
        if fp.exists():
            return fp

    # Try as relative path under SURVEY_DIR (e.g. "survey/2024-01-01/session/img.jpg")
    survey_rel = image_path
    if survey_rel.startswith("survey/"):
        survey_rel = survey_rel[len("survey/"):]
    try:
        candidate = (SURVEY_DIR / survey_rel).resolve()
        if candidate.is_relative_to(SURVEY_DIR.resolve()) and candidate.exists():
            return candidate
    except (ValueError, RuntimeError):
        pass

    return None


def _resolve_safe_path(filename: str, base_dir: Path) -> Path | None:
    """Resolve *filename* under *base_dir*, returning ``None`` on traversal."""
    if not validate_image_filename(filename):
        return None
    try:
        resolved = (base_dir / filename).resolve()
        if not resolved.is_relative_to(base_dir.resolve()):
            return None
        return resolved
    except (ValueError, RuntimeError):
        return None


def _save_result_image(image: np.ndarray, stem: str) -> str | None:
    """Save undistorted image to temp dir and return a relative URL."""
    try:
        RESULT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"{stem}_undistorted.jpg"
        out_path = RESULT_IMAGE_DIR / fname
        cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return f"api/result-image/{fname}"
    except Exception:
        logger.debug("Could not save result image to disk, falling back to base64")
        return None


def _encode_image(img: np.ndarray) -> str:
    """Encode an image as a base64-encoded JPEG string."""
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/api/calibration-files/", response_model=CalibrationFilesResponse)
def calibration_files(
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> CalibrationFilesResponse:
    """List available camera_ids in the calibration repo."""
    try:
        repo = RepoPostgresLensCalibrationTable(session)
        camera_ids = sorted(entity.id for entity in repo.get_all())
        return CalibrationFilesResponse(camera_ids=camera_ids)
    except Exception:
        logger.exception("Failed to list calibrations")
        raise HTTPException(status_code=500, detail="Failed to list calibrations")


@router.post("/api/load-calibration/", response_model=LoadCalibrationResponse)
def load_calibration(
    body: LoadCalibrationRequest,
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> LoadCalibrationResponse:
    """Load a calibration by camera_id from the DDD repo."""
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
        logger.exception("Failed to load calibration")
        raise HTTPException(status_code=500, detail="Failed to load calibration")


@router.get("/api/images/", response_model=ImagesResponse)
def images(
    tenant_id: str = Query(None),
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> ImagesResponse:
    """List available test images."""
    try:
        result: list[ImageItem] = []

        # Images from captured-frame repo (filtered by tenant)
        map_id: str | None = None
        if tenant_id:
            try:
                map_entity = get_map_for_tenant(tenant_id, session)
                map_id = map_entity.id if map_entity else None
            except ValueError:
                map_id = None

        for filename in list_image_filenames(session, map_id):
            result.append(ImageItem(name=filename, source="captured_frame"))

        # Survey images
        if SURVEY_DIR.exists():
            for date_dir in sorted(SURVEY_DIR.iterdir(), reverse=True):
                if not date_dir.is_dir():
                    continue
                for session_dir in sorted(date_dir.iterdir(), reverse=True):
                    if not session_dir.is_dir():
                        continue
                    for ext in ["*.jpg", "*.jpeg", "*.png"]:
                        for f in list(session_dir.glob(ext))[:5]:
                            result.append(
                                ImageItem(
                                    name=f"survey/{date_dir.name}/{session_dir.name}/{f.name}",
                                    source="survey",
                                )
                            )

        return ImagesResponse(images=result)
    except Exception:
        logger.exception("Failed to list images")
        raise HTTPException(status_code=500, detail="Failed to list images")


@router.post("/api/undistort/", response_model=UndistortResponse)
def undistort(
    body: UndistortRequest,
    user: UserModel = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> UndistortResponse:
    """Undistort an image using provided coefficients.

    Returns both original and undistorted images as base64-encoded JPEGs,
    or serves the undistorted image via a temp-file URL when available.
    """
    if not body.image_path:
        raise HTTPException(status_code=400, detail="Missing image_path")

    img_path = _resolve_image_path(body.image_path, session)
    if img_path is None:
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        image = cv2.imread(str(img_path))
        if image is None:
            raise HTTPException(status_code=400, detail="Could not load image")

        h, w = image.shape[:2]

        k1 = body.coefficients.k1
        k2 = body.coefficients.k2
        k3 = body.coefficients.k3
        p1 = body.coefficients.p1
        p2 = body.coefficients.p2

        fx = body.intrinsics.fx
        fy = body.intrinsics.fy
        cx = body.intrinsics.cx if body.intrinsics.cx is not None else w / 2
        cy = body.intrinsics.cy if body.intrinsics.cy is not None else h / 2

        if body.use_opencv:
            camera_matrix = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
            )
            dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
            undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)
            method_used = "opencv"
        else:
            from poc_homography.calibration.lens_distortion.apply_calibration import (
                undistort_image,
            )

            undistorted = undistort_image(image, k1, k2, k3, p1, p2, fx, fy, cx, cy)
            method_used = "solver"

        # Try to serve via temp-file URL for performance
        result_url = _save_result_image(undistorted, img_path.stem)

        response_data: dict[str, Any] = {
            "original": _encode_image(image),
            "width": w,
            "height": h,
            "method_used": method_used,
            "coefficients_used": {
                "k1": k1,
                "k2": k2,
                "k3": k3,
                "p1": p1,
                "p2": p2,
            },
            "intrinsics_used": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        }

        if result_url is not None:
            response_data["undistorted_url"] = result_url
        else:
            response_data["undistorted"] = _encode_image(undistorted)

        return UndistortResponse(**response_data)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to undistort image")
        raise HTTPException(status_code=500, detail="Failed to undistort image")


@router.get("/api/result-image/{filename}")
def serve_result_image(
    filename: str,
    user: UserModel = Depends(get_current_user),
) -> FileResponse:
    """Serve a previously generated undistorted result image."""
    resolved = _resolve_safe_path(filename, RESULT_IMAGE_DIR)
    if resolved is None or not resolved.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(resolved, media_type="image/jpeg")


@router.post("/api/transform-points/", response_model=TransformPointsResponse)
def transform_points(
    body: TransformPointsRequest,
    user: UserModel = Depends(get_current_user),
) -> TransformPointsResponse:
    """Transform points between distorted and undistorted coordinate spaces."""
    if not body.points:
        raise HTTPException(status_code=400, detail="Need at least 1 point")

    if len(body.points) > MAX_POINTS:
        raise HTTPException(
            status_code=400, detail=f"Too many points (max {MAX_POINTS})"
        )

    if body.direction not in ("distort", "undistort"):
        raise HTTPException(
            status_code=400, detail="direction must be 'distort' or 'undistort'"
        )

    try:
        pts = np.array(body.points, dtype=np.float64)

        k1 = body.coefficients.k1
        k2 = body.coefficients.k2
        k3 = body.coefficients.k3
        p1 = body.coefficients.p1
        p2 = body.coefficients.p2

        fx = body.intrinsics.fx
        fy = body.intrinsics.fy
        cx = body.intrinsics.cx if body.intrinsics.cx is not None else 960.0
        cy = body.intrinsics.cy if body.intrinsics.cy is not None else 540.0

        if body.direction == "undistort":
            from poc_homography.calibration.lens_distortion.apply_calibration import (
                undistort_points,
            )

            transformed = undistort_points(pts, k1, k2, k3, p1, p2, fx, fy, cx, cy)
        else:
            from poc_homography.calibration.lens_distortion.apply_calibration import (
                distort_points,
            )

            transformed = distort_points(pts, k1, k2, k3, p1, p2, fx, fy, cx, cy)

        return TransformPointsResponse(
            points=transformed.tolist(),
            direction=body.direction,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to transform points")
        raise HTTPException(status_code=500, detail="Failed to transform points")


@router.post("/api/measure-straightness/")
def measure_straightness(
    body: MeasureStraightnessRequest,
    user: UserModel = Depends(get_current_user),
) -> dict[str, Any]:
    """Measure the straightness of a set of points."""
    if len(body.points) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 points")

    try:
        pts = np.array(body.points, dtype=np.float64)

        if body.undistort:
            k1 = body.coefficients.k1
            k2 = body.coefficients.k2
            k3 = body.coefficients.k3
            p1 = body.coefficients.p1
            p2 = body.coefficients.p2

            fx = body.intrinsics.fx
            fy = body.intrinsics.fy
            cx = body.intrinsics.cx if body.intrinsics.cx is not None else 960.0
            cy = body.intrinsics.cy if body.intrinsics.cy is not None else 540.0

            from poc_homography.calibration.lens_distortion.apply_calibration import (
                undistort_points,
            )

            pts = undistort_points(pts, k1, k2, k3, p1, p2, fx, fy, cx, cy)

        from poc_homography.calibration.lens_distortion.apply_calibration import (
            measure_line_straightness,
        )

        result = measure_line_straightness(pts)

        rmse = result["rmse_pixels"]
        result["is_straight"] = rmse < 2.0
        result["quality"] = (
            "excellent"
            if rmse < 0.5
            else "good"
            if rmse < 2.0
            else "acceptable"
            if rmse < 5.0
            else "poor"
        )
        result["undistorted"] = body.undistort

        return result

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to measure straightness")
        raise HTTPException(status_code=500, detail="Failed to measure straightness")


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
