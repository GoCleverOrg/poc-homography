"""Shared utilities for calibration-related Django apps.

Provides filename validation, safe path resolution, mtime-based
calibration table caching, intrinsics computation, and calibration
entry serialization used by both ``lens_calibration`` and
``distortion_validator``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_http_methods

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filename validation helpers
# ---------------------------------------------------------------------------

def validate_filename(filename: str) -> bool:
    """Validate filename to prevent path traversal attacks."""
    if not filename:
        return False
    if "/" in filename or ".." in filename or "\\" in filename:
        return False
    return True


def resolve_safe_path(filename: str, base_dir: Path) -> Path | None:
    """Resolve *filename* under *base_dir*, returning ``None`` on traversal."""
    if not validate_filename(filename):
        return None
    try:
        resolved = (base_dir / filename).resolve()
        if not resolved.is_relative_to(base_dir.resolve()):
            return None
        return resolved
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Calibration file cache
# ---------------------------------------------------------------------------

_calibration_cache: dict[tuple[str, float], Any] = {}


def get_cached_calibration_table(filepath: Path):
    """Return a cached CameraCalibrationTable, invalidated by mtime."""
    from poc_homography.calibration.lens_distortion.calibration_table import (
        CameraCalibrationTable,
    )
    key_path = str(filepath)
    mtime = filepath.stat().st_mtime
    cache_key = (key_path, mtime)
    if cache_key not in _calibration_cache:
        # Evict stale entries for this path
        _calibration_cache.pop(
            next((k for k in _calibration_cache if k[0] == key_path), None),  # type: ignore[arg-type]
            None,
        )
        _calibration_cache[cache_key] = CameraCalibrationTable.load(filepath)
    return _calibration_cache[cache_key]


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
        base_focal_length_mm = float(
            data.get("base_focal_length_mm", DEFAULT_BASE_FOCAL_LENGTH_MM)
        )

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

        return JsonResponse({
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
        })

    except ValueError as e:
        return JsonResponse({"error": f"Invalid parameter: {e}"}, status=400)
    except Exception:
        logger.exception("Failed to compute intrinsics")
        return JsonResponse({"error": "Failed to compute intrinsics"}, status=500)


# ---------------------------------------------------------------------------
# Shared calibration entry serialization
# ---------------------------------------------------------------------------

def serialize_calibration_entry(entry) -> dict[str, Any]:
    """Serialize a ZoomCalibrationEntry to a JSON-safe dict.

    Includes intrinsics when stored (fx/fy non-zero).
    """
    entry_data: dict[str, Any] = {
        "zoom_factor": entry.zoom_factor,
        "coefficients": {
            "k1": float(entry.k1),
            "k2": float(entry.k2),
            "k3": float(entry.k3),
            "p1": float(entry.p1),
            "p2": float(entry.p2),
        },
        "calibration_date": entry.calibration_date,
        "validation_rmse": entry.validation_rmse,
        "num_lines_used": entry.num_lines_used,
    }
    if entry.fx != 0.0 or entry.fy != 0.0:
        entry_data["intrinsics"] = {
            "fx": entry.fx,
            "fy": entry.fy,
            "cx": entry.cx,
            "cy": entry.cy,
        }
    if entry.reprojection_error_px != 0.0:
        entry_data["reprojection_error_px"] = entry.reprojection_error_px
    return entry_data
