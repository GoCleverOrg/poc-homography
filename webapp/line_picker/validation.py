"""Request validation for line picker API endpoints."""

from __future__ import annotations

from typing import Any


def validate_add_line_request(data: dict[str, Any]) -> str | None:
    """Validate add line request data.

    Args:
        data: Request data dictionary.

    Returns:
        Error message string if validation fails, None if valid.
    """
    # Check required fields
    if "start_gcp" not in data:
        return "Missing required field: start_gcp"
    if "end_gcp" not in data:
        return "Missing required field: end_gcp"

    # Validate types
    if not isinstance(data["start_gcp"], str):
        return "start_gcp must be a string"
    if not isinstance(data["end_gcp"], str):
        return "end_gcp must be a string"

    # Validate not empty
    if not data["start_gcp"]:
        return "start_gcp cannot be empty"
    if not data["end_gcp"]:
        return "end_gcp cannot be empty"

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
