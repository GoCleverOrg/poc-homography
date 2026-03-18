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


