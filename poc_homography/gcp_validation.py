"""
Ground Control Point (GCP) validation module.

Provides validation functions for GCP coordinates used in feature-based
homography and camera pose estimation. Validates map pixel coordinates,
image pixel coordinates, and detects duplicates.

GCPs use map pixel coordinates (map_id, map_pixel_x, map_pixel_y) to define
positions on a reference map, and image pixel coordinates (image_u, image_v)
to define corresponding positions in camera images.
"""

from __future__ import annotations

import logging
import math
import numbers
from typing import Any, SupportsFloat, cast

logger = logging.getLogger(__name__)


# GCP validation constants
MAP_PIXEL_EPSILON = 0.5  # Default epsilon for map pixel coordinate comparison (pixels)
IMAGE_PIXEL_EPSILON = 0.5  # Default epsilon for image pixel coordinate comparison (pixels)
MAX_GCP_COUNT = 1000  # Maximum number of GCPs to prevent O(n^2) performance issues
MAX_DESCRIPTION_LENGTH = 200  # Maximum length for description field in error messages


def _is_valid_finite_number(value: Any) -> bool:
    """Check if a value is a valid finite number (int, float, or numpy numeric).

    Args:
        value: Value to check

    Returns:
        True if value is a valid finite number, False otherwise
    """
    # Check for numeric types (includes numpy types via numbers.Number ABC)
    if not isinstance(value, numbers.Number):
        return False

    # Reject complex numbers
    if isinstance(value, complex):
        return False

    # Check for NaN and Infinity
    try:
        float_value = float(cast("SupportsFloat", value))
        if math.isnan(float_value) or math.isinf(float_value):
            return False
    except (TypeError, ValueError):
        return False

    return True


def _get_gcp_description(gcp: dict[str, Any], index: int) -> str:
    """Get a sanitized description for a GCP for use in error messages.

    Args:
        gcp: Ground control point dictionary
        index: Index of the GCP in the list

    Returns:
        Sanitized description string
    """
    raw_description = gcp.get("metadata", {}).get("description", f"index {index}")

    # Convert to string if not already
    if not isinstance(raw_description, str):
        raw_description = str(raw_description)

    # Sanitize: remove control characters and limit length
    sanitized = "".join(char for char in raw_description if char.isprintable() or char == " ")

    if len(sanitized) > MAX_DESCRIPTION_LENGTH:
        sanitized = sanitized[:MAX_DESCRIPTION_LENGTH] + "..."

    return sanitized


def _validate_numeric_field(
    value: Any,
    field_name: str,
    description: str,
    min_value: float | None = None,
    max_value: float | None = None,
    units: str = "",
) -> None:
    """Validate a numeric field with optional range checking.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        description: GCP description for error messages
        min_value: Optional minimum allowed value (inclusive)
        max_value: Optional maximum allowed value (inclusive)
        units: Optional units string for error messages (e.g., "pixels")

    Raises:
        ValueError: If value is invalid
    """
    if not _is_valid_finite_number(value):
        if isinstance(value, numbers.Number):
            # It's a number but NaN or Inf
            raise ValueError(
                f"GCP at {description}: {field_name} must be a finite number, "
                f"got {value} (NaN and Infinity are not allowed)"
            )
        else:
            raise ValueError(
                f"GCP at {description}: {field_name} must be a number, got {type(value).__name__}"
            )

    if min_value is not None and max_value is not None:
        if value < min_value or value > max_value:
            units_str = f" {units}" if units else ""
            raise ValueError(
                f"GCP at {description}: {field_name} {value}{units_str} outside valid range "
                f"[{min_value}, {max_value}]"
            )


def validate_gcp_map_coordinates(
    gcp: dict[str, Any],
    index: int,
    map_width: int | None = None,
    map_height: int | None = None,
) -> None:
    """Validate map pixel coordinates of a ground control point.

    Args:
        gcp: Ground control point dictionary containing map_id, map_pixel_x, map_pixel_y
        index: Index of the GCP in the list (for error messages)
        map_width: Optional map width in pixels for bounds checking
        map_height: Optional map height in pixels for bounds checking

    Raises:
        ValueError: If map coordinates are invalid
    """
    description = _get_gcp_description(gcp, index)

    # Validate map_id
    if "map_id" not in gcp:
        raise ValueError(f"GCP at {description} missing required 'map_id' field")

    map_id = gcp["map_id"]
    if not isinstance(map_id, str) or not map_id.strip():
        raise ValueError(
            f"GCP at {description}: map_id must be a non-empty string, got {type(map_id).__name__}"
        )

    # Validate map_pixel_x
    if "map_pixel_x" not in gcp:
        raise ValueError(f"GCP at {description} missing required 'map_pixel_x' field")

    map_pixel_x = gcp["map_pixel_x"]
    _validate_numeric_field(map_pixel_x, "map_pixel_x", description)

    # Validate map_pixel_y
    if "map_pixel_y" not in gcp:
        raise ValueError(f"GCP at {description} missing required 'map_pixel_y' field")

    map_pixel_y = gcp["map_pixel_y"]
    _validate_numeric_field(map_pixel_y, "map_pixel_y", description)

    # Validate bounds - check each dimension independently when provided
    if map_width is not None:
        if map_pixel_x < 0 or map_pixel_x >= map_width:
            raise ValueError(
                f"GCP at {description}: map_pixel_x {map_pixel_x} outside map width [0, {map_width})"
            )

    if map_height is not None:
        if map_pixel_y < 0 or map_pixel_y >= map_height:
            raise ValueError(
                f"GCP at {description}: map_pixel_y {map_pixel_y} outside map height [0, {map_height})"
            )


def _validate_image_dimension(dimension: Any, dimension_name: str) -> int | None:
    """Validate and normalize an image dimension parameter.

    Args:
        dimension: The dimension value to validate (or None)
        dimension_name: Name for error messages ('image_width' or 'image_height')

    Returns:
        Validated dimension as int, or None if not provided

    Raises:
        ValueError: If dimension is invalid
    """
    if dimension is None:
        return None

    # Check it's a valid finite number
    if not _is_valid_finite_number(dimension):
        if isinstance(dimension, numbers.Number):
            raise ValueError(
                f"{dimension_name} must be a finite positive integer, "
                f"got {dimension} (NaN and Infinity are not allowed)"
            )
        else:
            raise ValueError(
                f"{dimension_name} must be a positive integer, got {type(dimension).__name__}"
            )

    # Convert to int and validate range
    try:
        dim_int = int(dimension)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"{dimension_name} must be a positive integer, got {dimension}") from e

    if dim_int <= 0:
        raise ValueError(f"{dimension_name} must be positive, got {dim_int}")

    # Sanity check for reasonable image dimensions (up to 100 megapixels)
    if dim_int > 100000:
        raise ValueError(f"{dimension_name} {dim_int} exceeds maximum allowed value of 100000")

    return dim_int


def validate_gcp_image_coordinates(
    gcp: dict[str, Any],
    index: int,
    image_width: int | None = None,
    image_height: int | None = None,
) -> None:
    """Validate image pixel coordinates of a ground control point.

    Args:
        gcp: Ground control point dictionary containing image_u, image_v fields
        index: Index of the GCP in the list (for error messages)
        image_width: Optional image width in pixels for bounds checking
        image_height: Optional image height in pixels for bounds checking

    Raises:
        ValueError: If image coordinates are invalid
    """
    description = _get_gcp_description(gcp, index)

    # Validate image_u coordinate
    if "image_u" not in gcp:
        raise ValueError(f"GCP at {description} missing required 'image_u' field")

    image_u = gcp["image_u"]
    _validate_numeric_field(image_u, "image_u", description)

    # Validate image_v coordinate
    if "image_v" not in gcp:
        raise ValueError(f"GCP at {description} missing required 'image_v' field")

    image_v = gcp["image_v"]
    _validate_numeric_field(image_v, "image_v", description)

    # Validate bounds - check each dimension independently when provided
    if image_width is not None:
        if image_u < 0 or image_u >= image_width:
            raise ValueError(
                f"GCP at {description}: image_u {image_u} outside image width [0, {image_width})"
            )

    if image_height is not None:
        if image_v < 0 or image_v >= image_height:
            raise ValueError(
                f"GCP at {description}: image_v {image_v} outside image height [0, {image_height})"
            )


def detect_duplicate_gcps(
    gcps: list[dict[str, Any]],
    map_pixel_epsilon: float = MAP_PIXEL_EPSILON,
    image_pixel_epsilon: float = IMAGE_PIXEL_EPSILON,
) -> None:
    """Detect duplicate ground control points.

    Two GCPs are considered duplicates if BOTH their map pixel and image pixel
    coordinates are within epsilon thresholds of each other.

    Args:
        gcps: List of ground control points (must be pre-validated)
        map_pixel_epsilon: Epsilon threshold for map pixel coordinate comparison (pixels)
        image_pixel_epsilon: Epsilon threshold for image pixel coordinate comparison (pixels)

    Raises:
        ValueError: If duplicate GCPs are detected
    """
    # Extract coordinates upfront to avoid repeated dictionary lookups
    # At this point, GCPs have been validated so we know these fields exist
    coords: list[tuple[str, float, float, float, float, str]] = []
    for i, gcp in enumerate(gcps):
        desc = _get_gcp_description(gcp, i)
        coords.append(
            (
                gcp["map_id"],
                gcp["map_pixel_x"],
                gcp["map_pixel_y"],
                gcp["image_u"],
                gcp["image_v"],
                desc,
            )
        )

    for i in range(len(coords)):
        map_id_i, map_x_i, map_y_i, img_u_i, img_v_i, desc_i = coords[i]

        for j in range(i + 1, len(coords)):
            map_id_j, map_x_j, map_y_j, img_u_j, img_v_j, desc_j = coords[j]

            # Check if map pixel coordinates are within epsilon (same map_id required)
            map_duplicate = (
                map_id_i == map_id_j
                and abs(map_x_i - map_x_j) < map_pixel_epsilon
                and abs(map_y_i - map_y_j) < map_pixel_epsilon
            )

            # Check if image pixel coordinates are within epsilon
            image_duplicate = (
                abs(img_u_i - img_u_j) < image_pixel_epsilon
                and abs(img_v_i - img_v_j) < image_pixel_epsilon
            )

            # Only flag as duplicate if BOTH map and image coords match
            if map_duplicate and image_duplicate:
                raise ValueError(
                    f"Duplicate GCP detected at {desc_i} and {desc_j} "
                    f"(map pixel coordinates within {map_pixel_epsilon} pixels and "
                    f"image pixel coordinates within {image_pixel_epsilon} pixels)"
                )


def validate_ground_control_points(
    gcps: Any,
    image_width: int | None = None,
    image_height: int | None = None,
    map_width: int | None = None,
    map_height: int | None = None,
    min_gcp_count: int = 6,
) -> list[dict[str, Any]]:
    """Validate ground control points configuration.

    GCPs define correspondences between map pixel coordinates and image pixel
    coordinates. Each GCP must contain:
    - map_id: Identifier for the reference map
    - map_pixel_x: X coordinate on the reference map (pixels)
    - map_pixel_y: Y coordinate on the reference map (pixels)
    - image_u: Horizontal pixel coordinate in the camera image
    - image_v: Vertical pixel coordinate in the camera image

    Supports two formats:
    1. Single GCP set: gcps is a list of GCP dictionaries (default format)
    2. Multiple GCP sets: gcps is a dict where keys are set identifiers and
       values are lists of GCP dictionaries

    Args:
        gcps: Ground control points in either list or dict format
        image_width: Optional image width for image pixel coordinate validation
        image_height: Optional image height for image pixel coordinate validation
        map_width: Optional map width for map pixel coordinate validation
        map_height: Optional map height for map pixel coordinate validation
        min_gcp_count: Minimum recommended number of GCPs (default: 6)

    Returns:
        Validated list of GCP dictionaries (normalized to list format)

    Raises:
        ValueError: If GCPs fail validation with clear error messages

    Example:
        >>> # Single set format
        >>> gcps = [
        ...     {"map_id": "map1", "map_pixel_x": 100, "map_pixel_y": 200,
        ...      "image_u": 500, "image_v": 300},
        ...     ...
        ... ]
        >>> validated = validate_ground_control_points(gcps)

        >>> # Multiple sets format
        >>> gcps = {
        ...     "set_1": [{"map_id": "map1", ...}, ...],
        ...     "set_2": [{"map_id": "map1", ...}, ...]
        ... }
        >>> validated = validate_ground_control_points(gcps)
    """
    # Validate and normalize image dimensions first
    validated_image_width = _validate_image_dimension(image_width, "image_width")
    validated_image_height = _validate_image_dimension(image_height, "image_height")
    validated_map_width = _validate_image_dimension(map_width, "map_width")
    validated_map_height = _validate_image_dimension(map_height, "map_height")

    # Detect format and normalize to list
    if isinstance(gcps, list):
        # Single set format - use as-is
        gcp_list = gcps
        logger.debug(f"Processing single GCP set with {len(gcp_list)} points")
    elif isinstance(gcps, dict):
        # Multiple sets format - validate it's a dict of lists
        if not gcps:
            gcp_list = []
            logger.warning("Ground control points dict is empty")
        else:
            # Use the first set for now (could be extended to support multiple sets)
            first_key = next(iter(gcps))
            first_value = gcps[first_key]

            if not isinstance(first_value, list):
                raise ValueError(
                    f"When ground_control_points is a dict, values must be lists. "
                    f"Found {type(first_value).__name__} for key '{first_key}'"
                )

            gcp_list = first_value
            logger.info(f"Processing GCP set '{first_key}' with {len(gcp_list)} points")

            if len(gcps) > 1:
                logger.info(
                    f"Multiple GCP sets detected ({len(gcps)} sets). "
                    f"Using set '{first_key}' for validation"
                )
    else:
        raise ValueError(f"ground_control_points must be a list or dict, got {type(gcps).__name__}")

    # Check maximum count to prevent O(n^2) performance issues
    if len(gcp_list) > MAX_GCP_COUNT:
        raise ValueError(
            f"Too many GCPs provided: {len(gcp_list)}. "
            f"Maximum allowed is {MAX_GCP_COUNT} to prevent performance issues"
        )

    # Check minimum count (warning, not error)
    if len(gcp_list) < min_gcp_count:
        logger.warning(
            f"Only {len(gcp_list)} GCPs provided, recommended minimum is {min_gcp_count} "
            f"for robust RANSAC-based pose estimation"
        )

    # Log if image pixel validation will be skipped
    if validated_image_width is None and validated_image_height is None:
        logger.warning(
            "Image dimensions not provided, skipping image pixel coordinate bounds validation"
        )
    elif validated_image_width is None or validated_image_height is None:
        provided = "image_width" if validated_image_width is not None else "image_height"
        missing = "image_height" if validated_image_width is not None else "image_width"
        logger.warning(
            f"Only {provided} provided, {missing} missing. "
            f"Partial image pixel coordinate bounds validation will be performed"
        )

    # Log if map pixel validation will be skipped
    if validated_map_width is None and validated_map_height is None:
        logger.debug("Map dimensions not provided, skipping map pixel coordinate bounds validation")

    # Validate each GCP
    for i, gcp in enumerate(gcp_list):
        if not isinstance(gcp, dict):
            raise ValueError(f"GCP at index {i} must be a dictionary, got {type(gcp).__name__}")

        # Validate map pixel coordinates
        validate_gcp_map_coordinates(gcp, i, validated_map_width, validated_map_height)

        # Validate image pixel coordinates
        validate_gcp_image_coordinates(gcp, i, validated_image_width, validated_image_height)

    # Check for duplicates
    if len(gcp_list) > 1:
        detect_duplicate_gcps(gcp_list)

    logger.info(f"Successfully validated {len(gcp_list)} ground control points")

    return gcp_list
