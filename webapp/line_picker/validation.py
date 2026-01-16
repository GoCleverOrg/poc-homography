"""Request validation for line picker API endpoints."""

from __future__ import annotations

from typing import Any


def validate_add_line_request(data: dict[str, Any]) -> str | None:
    """Validate add line request data.

    Lines are defined by pixel coordinate endpoints (start_x, start_y, end_x, end_y).

    Args:
        data: Request data dictionary.

    Returns:
        Error message string if validation fails, None if valid.
    """
    # Check required coordinate fields
    required = ["start_x", "start_y", "end_x", "end_y"]
    for field in required:
        if field not in data:
            return f"Missing required field: {field}"

    # Validate coordinate types (must be numeric)
    for field in required:
        if not isinstance(data[field], (int, float)):
            return f"{field} must be a number"

    # Validate line_id if provided (optional string)
    if "line_id" in data and data["line_id"] is not None:
        if not isinstance(data["line_id"], str):
            return "line_id must be a string"
        if not data["line_id"]:
            return "line_id cannot be empty"

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
