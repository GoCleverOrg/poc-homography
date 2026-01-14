"""Request validation for point picker API endpoints."""

from __future__ import annotations

from typing import Any

# Valid tag values
VALID_TAGS = {"parking_spot", "arrows", "crosswalk", "extra"}


def validate_add_point_request(data: dict[str, Any]) -> str | None:
    """Validate add point request data.

    Args:
        data: Request data dictionary.

    Returns:
        Error message string if validation fails, None if valid.
    """
    # Check required fields
    if "pixel_x" not in data:
        return "Missing required field: pixel_x"
    if "pixel_y" not in data:
        return "Missing required field: pixel_y"

    # Validate types
    try:
        float(data["pixel_x"])
    except (TypeError, ValueError):
        return "pixel_x must be a number"

    try:
        float(data["pixel_y"])
    except (TypeError, ValueError):
        return "pixel_y must be a number"

    # Validate tag if provided
    tag = data.get("tag", "extra")
    if tag not in VALID_TAGS:
        return f"Invalid tag: {tag}. Must be one of: {', '.join(sorted(VALID_TAGS))}"

    # Validate id if provided (optional string)
    if "id" in data and data["id"] is not None:
        if not isinstance(data["id"], str):
            return "id must be a string"

    return None


def validate_update_point_request(data: dict[str, Any]) -> str | None:
    """Validate update point request data.

    Args:
        data: Request data dictionary.

    Returns:
        Error message string if validation fails, None if valid.
    """
    # Check required fields
    if "pixel_x" not in data:
        return "Missing required field: pixel_x"
    if "pixel_y" not in data:
        return "Missing required field: pixel_y"

    # Validate types
    try:
        float(data["pixel_x"])
    except (TypeError, ValueError):
        return "pixel_x must be a number"

    try:
        float(data["pixel_y"])
    except (TypeError, ValueError):
        return "pixel_y must be a number"

    return None


def validate_export_request(data: dict[str, Any]) -> str | None:
    """Validate export request data.

    Args:
        data: Request data dictionary.

    Returns:
        Error message string if validation fails, None if valid.
    """
    # path is optional, defaults to empty string
    if "path" in data and data["path"] is not None:
        if not isinstance(data["path"], str):
            return "path must be a string"

    return None


def validate_import_request(data: dict[str, Any]) -> str | None:
    """Validate import request data.

    Args:
        data: Request data dictionary.

    Returns:
        Error message string if validation fails, None if valid.
    """
    # path is required
    if "path" not in data:
        return "Missing required field: path"

    if not isinstance(data["path"], str):
        return "path must be a string"

    if not data["path"]:
        return "path cannot be empty"

    return None
