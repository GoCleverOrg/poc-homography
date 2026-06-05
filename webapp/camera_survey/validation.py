"""Shared validation helpers for survey API endpoints."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from poc_homography.camera_config import get_camera_by_id

from .models import SurveyAxis
from .ptz import create_ptz_camera

if TYPE_CHECKING:
    from poc_homography.domain.vo.camera_capabilities import CameraCapabilities

logger = logging.getLogger(__name__)


def validate_axis_range(
    capabilities: CameraCapabilities,
    axis: SurveyAxis,
    start: float,
    end: float,
) -> tuple[bool, str | None]:
    """Validate that ``start``/``end`` fall within the axis range.

    Replaces the former ``CameraCapabilities.validate_range`` method; the DDD
    capabilities VO is a pure data object, so the range check lives here.

    Args:
        capabilities: Real camera capabilities (degrees-based).
        axis: Survey axis being swept.
        start: Sweep start value.
        end: Sweep end value.

    Returns:
        ``(is_valid, error_message)``; ``error_message`` is ``None`` if valid.
    """
    if axis == SurveyAxis.PAN:
        min_val, max_val = float(capabilities.pan_min), float(capabilities.pan_max)
    elif axis == SurveyAxis.TILT:
        min_val, max_val = float(capabilities.tilt_min), float(capabilities.tilt_max)
    else:  # ZOOM
        min_val, max_val = float(capabilities.zoom_min), float(capabilities.zoom_max)

    if not (min_val <= start <= max_val):
        return (
            False,
            f"Start value {start} outside valid {axis.value} range [{min_val}, {max_val}]",
        )
    if not (min_val <= end <= max_val):
        return (
            False,
            f"End value {end} outside valid {axis.value} range [{min_val}, {max_val}]",
        )
    return True, None


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

    camera_ip = camera.get("ip")
    if not camera_ip:
        return f"Camera {camera_id} has no IP address"

    try:
        ptz_camera = create_ptz_camera(
            camera_ip=camera_ip,
            camera_name=camera.get("name", camera_id),
            camera_model=camera.get("model"),
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
