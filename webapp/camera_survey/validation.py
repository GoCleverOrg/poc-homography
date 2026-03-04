"""Shared validation helpers for survey API endpoints."""

from __future__ import annotations

import logging
import math

from poc_homography.camera_config import get_camera_by_id

from .ptz import create_ptz_camera

logger = logging.getLogger(__name__)


def parse_optional_float(data: dict, field: str) -> tuple[float | None, str | None]:
    """Parse an optional float field from request data.

    Returns:
        (value, error_message). error_message is None on success.
    """
    if field not in data or data[field] is None:
        return None, None
    try:
        value = float(data[field])
    except (ValueError, TypeError):
        return None, f"{field} must be numeric"
    if not math.isfinite(value):
        return None, f"{field} must be a finite number"
    return value, None


def parse_fixed_axis_values(
    data: dict,
) -> tuple[float | None, float | None, float | None, str | None]:
    """Parse optional fixed axis values from request data.

    Returns:
        (fixed_pan, fixed_tilt, fixed_zoom, error_message).
        error_message is None on success.
    """
    fixed_pan, err = parse_optional_float(data, "fixed_pan")
    if err:
        return None, None, None, err

    fixed_tilt, err = parse_optional_float(data, "fixed_tilt")
    if err:
        return None, None, None, err

    fixed_zoom, err = parse_optional_float(data, "fixed_zoom")
    if err:
        return None, None, None, err

    return fixed_pan, fixed_tilt, fixed_zoom, None


def validate_fixed_axis_ranges(
    fixed_pan: float | None,
    fixed_tilt: float | None,
    fixed_zoom: float | None,
    camera_id: str,
    tenant_id: str,
) -> str | None:
    """Validate fixed axis values against camera capabilities.

    Returns error message if validation fails, None if all values are in range.
    """
    if fixed_pan is None and fixed_tilt is None and fixed_zoom is None:
        return None

    camera = get_camera_by_id(camera_id)
    if not camera:
        return f"Camera not found: {camera_id}"

    camera_ip = camera.ip_address
    if not camera_ip:
        return f"Camera {camera_id} has no IP address"

    try:
        ptz_camera = create_ptz_camera(
            camera_ip=camera_ip,
            camera_name=camera.name,
            camera_model=camera.spec.model_name,
            tenant_id=tenant_id,
        )
        capabilities = ptz_camera.get_capabilities()
    except ValueError as e:
        return str(e)
    except Exception as e:
        logger.exception(f"Failed to get camera capabilities for {camera_id}")
        return f"Failed to get camera capabilities: {e}"

    if fixed_pan is not None:
        if not (capabilities.pan_min <= fixed_pan <= capabilities.pan_max):
            return (
                f"Fixed pan value {fixed_pan} is outside valid range "
                f"[{capabilities.pan_min}, {capabilities.pan_max}]"
            )

    if fixed_tilt is not None:
        if not (capabilities.tilt_min <= fixed_tilt <= capabilities.tilt_max):
            return (
                f"Fixed tilt value {fixed_tilt} is outside valid range "
                f"[{capabilities.tilt_min}, {capabilities.tilt_max}]"
            )

    if fixed_zoom is not None:
        if not (capabilities.zoom_min <= fixed_zoom <= capabilities.zoom_max):
            return (
                f"Fixed zoom value {fixed_zoom} is outside valid range "
                f"[{capabilities.zoom_min}, {capabilities.zoom_max}]"
            )

    return None
