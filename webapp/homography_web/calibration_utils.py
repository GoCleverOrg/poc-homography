"""Shared utilities for calibration-related Django apps.

Provides filename validation, safe path resolution, mtime-based
calibration table caching, intrinsics computation, calibration
entry serialization, and DDD repo adapter functions used by both
``lens_calibration`` and ``distortion_validator``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_http_methods

from homography_web.frame_utils import validate_image_filename

if TYPE_CHECKING:
    from pathlib import Path

    from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
    from poc_homography.domain.vo.zoom_calibration_entry import ZoomCalibrationEntry

logger = logging.getLogger(__name__)


def resolve_safe_path(filename: str, base_dir: Path) -> Path | None:
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


# ---------------------------------------------------------------------------
# Shared intrinsics computation
# ---------------------------------------------------------------------------


@require_http_methods(["POST"])
def api_compute_intrinsics(request: HttpRequest) -> JsonResponse:
    """Compute camera intrinsics from sensor specs and zoom level.

    Shared endpoint used by both lens_calibration and distortion_validator.
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
        from poc_homography.camera.intrinsics import compute_intrinsics
        from poc_homography.camera_config import (
            DEFAULT_BASE_FOCAL_LENGTH_MM,
            DEFAULT_SENSOR_WIDTH_MM,
        )

        zoom = float(data.get("zoom", 1.0))
        image_width = int(data.get("image_width", 1920))
        image_height = int(data.get("image_height", 1080))
        sensor_width_mm = float(data.get("sensor_width_mm", DEFAULT_SENSOR_WIDTH_MM))
        base_focal_length_mm = float(data.get("base_focal_length_mm", DEFAULT_BASE_FOCAL_LENGTH_MM))

        if zoom <= 0 or image_width <= 0 or image_height <= 0:
            return JsonResponse(
                {"error": "zoom, image_width, and image_height must be positive"},
                status=400,
            )

        result = compute_intrinsics(
            zoom=zoom,
            image_width=image_width,
            image_height=image_height,
            sensor_width_mm=sensor_width_mm,
            base_focal_length_mm=base_focal_length_mm,
        )

        return JsonResponse(
            {
                "fx": float(result.focal_length_px),
                "fy": float(result.focal_length_px),
                "cx": float(result.cx),
                "cy": float(result.cy),
                "focal_length_mm": float(result.focal_length_mm),
                "sensor_width_mm": sensor_width_mm,
                "base_focal_length_mm": base_focal_length_mm,
                "zoom": zoom,
                "image_width": image_width,
                "image_height": image_height,
            }
        )

    except ValueError as e:
        return JsonResponse({"error": f"Invalid parameter: {e}"}, status=400)
    except Exception:
        logger.exception("Failed to compute intrinsics")
        return JsonResponse({"error": "Failed to compute intrinsics"}, status=500)


# ---------------------------------------------------------------------------
# Shared calibration entry serialization
# ---------------------------------------------------------------------------


def serialize_calibration_entry(entry: ZoomCalibrationEntry) -> dict[str, Any]:
    """Serialize a DDD ``ZoomCalibrationEntry`` VO to a JSON-safe dict.

    Includes intrinsics when stored (fx/fy non-zero).
    """
    entry_data: dict[str, Any] = {
        "zoom_factor": float(entry.zoom_factor),
        "coefficients": {
            "k1": float(entry.distortion.k1),
            "k2": float(entry.distortion.k2),
            "k3": float(entry.distortion.k3),
            "p1": float(entry.distortion.p1),
            "p2": float(entry.distortion.p2),
        },
        "calibration_date": entry.calibration_date,
        "validation_rmse": entry.validation_rmse,
        "num_lines_used": entry.num_lines_used,
    }
    if float(entry.fx) != 0.0 or float(entry.fy) != 0.0:
        entry_data["intrinsics"] = {
            "fx": float(entry.fx),
            "fy": float(entry.fy),
            "cx": float(entry.cx),
            "cy": float(entry.cy),
        }
    if entry.reprojection_error_px != 0.0:
        entry_data["reprojection_error_px"] = entry.reprojection_error_px
    return entry_data


# ---------------------------------------------------------------------------
# DDD repo adapter functions
# ---------------------------------------------------------------------------


def save_calibration_to_repo(table: Any, data_dir: Path) -> None:
    """Convert a legacy ``CameraCalibrationTable`` to ``LensCalibrationTable`` and persist."""
    from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
    from poc_homography.domain.vo.lens_distortion import LensDistortion
    from poc_homography.domain.vo.zoom_calibration_entry import (
        ZoomCalibrationEntry as DddEntry,
    )
    from poc_homography.infrastructure.repositories.repo_yaml_lens_calibration_table import (
        RepoYamlLensCalibrationTable,
    )
    from poc_homography.types import PixelsFloat, Unitless

    repo = RepoYamlLensCalibrationTable(data_dir)

    # Merge with any existing entries already persisted for this camera
    existing = repo.get(table.camera_id)
    existing_by_zoom: dict[float, DddEntry] = {}
    if existing:
        for e in existing.entries:
            existing_by_zoom[float(e.zoom_factor)] = e

    for legacy_entry in table.entries.values():
        ddd_entry = DddEntry(
            zoom_factor=Unitless(legacy_entry.zoom_factor),
            distortion=LensDistortion(
                k1=Unitless(legacy_entry.k1),
                k2=Unitless(legacy_entry.k2),
                p1=Unitless(legacy_entry.p1),
                p2=Unitless(legacy_entry.p2),
                k3=Unitless(legacy_entry.k3),
            ),
            calibration_date=legacy_entry.calibration_date,
            source_images=tuple(legacy_entry.source_images),
            validation_rmse=legacy_entry.validation_rmse,
            num_lines_used=legacy_entry.num_lines_used,
            fx=PixelsFloat(legacy_entry.fx),
            fy=PixelsFloat(legacy_entry.fy),
            cx=PixelsFloat(legacy_entry.cx),
            cy=PixelsFloat(legacy_entry.cy),
            reprojection_error_px=legacy_entry.reprojection_error_px,
        )
        existing_by_zoom[float(legacy_entry.zoom_factor)] = ddd_entry

    now = datetime.now().isoformat()
    entity = LensCalibrationTable(
        id=table.camera_id,
        entries=tuple(existing_by_zoom[z] for z in sorted(existing_by_zoom)),
        created_date=existing.created_date if existing else now,
        last_modified=now,
    )
    repo.save(entity)


def load_calibration_from_repo(camera_id: str, data_dir: Path) -> LensCalibrationTable | None:
    """Load a ``LensCalibrationTable`` entity from the DDD repo."""
    from poc_homography.infrastructure.repositories.repo_yaml_lens_calibration_table import (
        RepoYamlLensCalibrationTable,
    )

    repo = RepoYamlLensCalibrationTable(data_dir)
    return repo.get(camera_id)


def list_calibration_ids(data_dir: Path) -> list[str]:
    """List available camera_ids in the repo."""
    from poc_homography.infrastructure.repositories.repo_yaml_lens_calibration_table import (
        RepoYamlLensCalibrationTable,
    )

    repo = RepoYamlLensCalibrationTable(data_dir)
    return sorted(entity.id for entity in repo.get_all())
