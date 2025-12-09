"""
Ground Control Point (GCP) validation module.

Provides validation functions for GCP coordinates used in feature-based
homography and camera pose estimation. Validates GPS coordinates, elevation,
pixel coordinates, and detects duplicates.
"""

import logging
import math
import numbers
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# GCP validation constants
GPS_EPSILON = 1e-6  # Default epsilon for GPS coordinate comparison (degrees)
PIXEL_EPSILON = 0.5  # Default epsilon for pixel coordinate comparison (pixels)
MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0
MIN_ELEVATION = -500.0  # Dead Sea is ~-430m, allow some margin
MAX_ELEVATION = 9000.0  # Mount Everest is ~8849m, allow some margin
MAX_GCP_COUNT = 1000  # Maximum number of GCPs to prevent O(n^2) performance issues
MAX_DESCRIPTION_LENGTH = 200  # Maximum length for description field in error messages

# GPS precision constants
# Arc-second quantization (1/3600 degree) gives ~30m lat / ~23m lon precision at mid-latitudes
# We want sub-meter precision, which requires at least 6 decimal places
GPS_MIN_DECIMAL_PLACES = 6  # Minimum recommended decimal places for GPS
GPS_ARC_SECOND_THRESHOLD = 1.0 / 3600.0  # ~0.000278 degrees = 1 arc-second
METERS_PER_DEGREE_LAT = 111320  # Approximate meters per degree latitude
METERS_PER_ARC_SECOND_LAT = METERS_PER_DEGREE_LAT / 3600  # ~30.9 meters


def analyze_gps_precision(
    gcps: List[Dict[str, Any]],
    reference_lat: Optional[float] = None
) -> Dict[str, Any]:
    """Analyze GPS coordinate precision and detect potential quantization issues.

    This function detects common GPS precision issues:
    1. Arc-second quantization (coordinates entered as degrees-minutes-seconds)
    2. Insufficient decimal places (< 6 decimal places gives > 0.1m error)
    3. Inconsistent spacing between adjacent GCPs

    Args:
        gcps: List of validated GCP dictionaries
        reference_lat: Reference latitude for longitude precision calculation
                      (longitude precision varies with latitude)

    Returns:
        Dictionary with precision analysis:
            - estimated_precision_m: Estimated GPS precision in meters
            - decimal_places: Detected number of significant decimal places
            - quantization_detected: True if arc-second quantization detected
            - spacing_analysis: Statistics on GCP spacing
            - warnings: List of precision-related warnings
            - recommendations: List of recommendations to improve precision
    """
    if not gcps:
        return {
            'estimated_precision_m': None,
            'decimal_places': None,
            'quantization_detected': False,
            'spacing_analysis': None,
            'warnings': [],
            'recommendations': []
        }

    warnings = []
    recommendations = []

    # Extract GPS coordinates
    latitudes = [gcp['gps']['latitude'] for gcp in gcps]
    longitudes = [gcp['gps']['longitude'] for gcp in gcps]

    # Use centroid if no reference provided
    if reference_lat is None:
        reference_lat = sum(latitudes) / len(latitudes)

    # Analyze decimal places
    def count_decimal_places(value: float) -> int:
        """Count significant decimal places in a float."""
        s = f"{value:.15f}".rstrip('0')
        if '.' in s:
            return len(s.split('.')[1])
        return 0

    lat_decimals = [count_decimal_places(lat) for lat in latitudes]
    lon_decimals = [count_decimal_places(lon) for lon in longitudes]
    min_decimals = min(min(lat_decimals), min(lon_decimals))

    # Check for arc-second quantization
    # Arc-second values have specific patterns like X.XXXYYY where YYY is 000, 278, 556, 833
    arc_second_fractions = [0, 1/3600, 2/3600, 3/3600]  # 0, ~0.000278, etc.

    def is_arc_second_quantized(coords: List[float]) -> bool:
        """Check if coordinates appear to be arc-second quantized."""
        quantized_count = 0
        for coord in coords:
            # Get fractional part of coordinate
            frac = abs(coord) % (1/60)  # Fractional part within a minute
            # Check if close to arc-second boundary
            for arc_frac in [i/3600 for i in range(60)]:
                if abs(frac - arc_frac) < 1e-7:
                    quantized_count += 1
                    break
        return quantized_count > len(coords) * 0.8  # 80% threshold

    lat_quantized = is_arc_second_quantized(latitudes)
    lon_quantized = is_arc_second_quantized(longitudes)
    quantization_detected = lat_quantized or lon_quantized

    # Calculate estimated precision in meters
    # Based on decimal places: 6 decimals ≈ 0.1m, 5 decimals ≈ 1m, etc.
    precision_by_decimals = {
        8: 0.001,   # 8 decimals ≈ 1mm
        7: 0.01,    # 7 decimals ≈ 1cm
        6: 0.1,     # 6 decimals ≈ 10cm
        5: 1.0,     # 5 decimals ≈ 1m
        4: 11.0,    # 4 decimals ≈ 11m
        3: 111.0,   # 3 decimals ≈ 111m
    }
    estimated_precision = precision_by_decimals.get(min_decimals, 1000.0)

    # If arc-second quantized, precision is limited to ~30m
    if quantization_detected:
        estimated_precision = max(estimated_precision, METERS_PER_ARC_SECOND_LAT)

    # Analyze spacing between adjacent GCPs
    spacing_analysis = None
    if len(gcps) >= 2:
        spacings = []
        cos_lat = math.cos(math.radians(reference_lat))
        for i in range(1, len(gcps)):
            lat1, lon1 = latitudes[i-1], longitudes[i-1]
            lat2, lon2 = latitudes[i], longitudes[i]
            # Approximate distance in meters
            dlat = (lat2 - lat1) * METERS_PER_DEGREE_LAT
            dlon = (lon2 - lon1) * METERS_PER_DEGREE_LAT * cos_lat
            dist = math.sqrt(dlat**2 + dlon**2)
            if dist > 0.01:  # Ignore very close points
                spacings.append(dist)

        if spacings:
            import statistics
            spacing_analysis = {
                'min_m': min(spacings),
                'max_m': max(spacings),
                'mean_m': statistics.mean(spacings),
                'std_m': statistics.stdev(spacings) if len(spacings) > 1 else 0,
                'count': len(spacings)
            }

            # Check for suspiciously regular spacing (may indicate manual entry)
            if spacing_analysis['std_m'] > 0:
                cv = spacing_analysis['std_m'] / spacing_analysis['mean_m']
                if cv > 0.5:
                    warnings.append(
                        f"Inconsistent GCP spacing (CV={cv:.2f}). "
                        "Check GPS coordinate accuracy."
                    )

    # Generate warnings
    if min_decimals < GPS_MIN_DECIMAL_PLACES:
        warnings.append(
            f"GPS coordinates have only {min_decimals} decimal places. "
            f"Recommend at least {GPS_MIN_DECIMAL_PLACES} for sub-meter precision."
        )

    if quantization_detected:
        warnings.append(
            "GPS coordinates appear to be arc-second quantized (~30m precision). "
            "Use decimal degrees with more precision for accurate homography."
        )

    if estimated_precision > 1.0:
        warnings.append(
            f"Estimated GPS precision is {estimated_precision:.1f}m. "
            "This may cause significant homography errors."
        )

    # Generate recommendations
    if estimated_precision > 0.5:
        recommendations.append(
            "Use GPS coordinates with at least 7 decimal places for best results."
        )
    if quantization_detected:
        recommendations.append(
            "Avoid converting to degrees-minutes-seconds format. "
            "Use decimal degrees directly from GPS source."
        )
    if spacing_analysis and spacing_analysis['min_m'] < estimated_precision:
        recommendations.append(
            f"Some GCPs are closer ({spacing_analysis['min_m']:.2f}m) than GPS precision "
            f"({estimated_precision:.1f}m). Consider using more widely spaced points."
        )

    return {
        'estimated_precision_m': estimated_precision,
        'decimal_places': min_decimals,
        'quantization_detected': quantization_detected,
        'spacing_analysis': spacing_analysis,
        'warnings': warnings,
        'recommendations': recommendations
    }


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
        if math.isnan(value) or math.isinf(value):
            return False
    except (TypeError, ValueError):
        return False

    return True


def _get_gcp_description(gcp: Dict[str, Any], index: int) -> str:
    """Get a sanitized description for a GCP for use in error messages.

    Args:
        gcp: Ground control point dictionary
        index: Index of the GCP in the list

    Returns:
        Sanitized description string
    """
    raw_description = gcp.get('metadata', {}).get('description', f'index {index}')

    # Convert to string if not already
    if not isinstance(raw_description, str):
        raw_description = str(raw_description)

    # Sanitize: remove control characters and limit length
    sanitized = ''.join(
        char for char in raw_description
        if char.isprintable() or char == ' '
    )

    if len(sanitized) > MAX_DESCRIPTION_LENGTH:
        sanitized = sanitized[:MAX_DESCRIPTION_LENGTH] + '...'

    return sanitized


def _validate_numeric_field(
    value: Any,
    field_name: str,
    description: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    units: str = ""
) -> None:
    """Validate a numeric field with optional range checking.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        description: GCP description for error messages
        min_value: Optional minimum allowed value (inclusive)
        max_value: Optional maximum allowed value (inclusive)
        units: Optional units string for error messages (e.g., "meters")

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
                f"GCP at {description}: {field_name} must be a number, "
                f"got {type(value).__name__}"
            )

    if min_value is not None and max_value is not None:
        if value < min_value or value > max_value:
            units_str = f" {units}" if units else ""
            raise ValueError(
                f"GCP at {description}: {field_name} {value}{units_str} outside valid range "
                f"[{min_value}, {max_value}]"
            )


def validate_gcp_gps_coordinates(gcp: Dict[str, Any], index: int) -> None:
    """Validate GPS coordinates of a ground control point.

    Args:
        gcp: Ground control point dictionary containing 'gps' section
        index: Index of the GCP in the list (for error messages)

    Raises:
        ValueError: If GPS coordinates are invalid
    """
    if 'gps' not in gcp:
        raise ValueError(
            f"GCP at index {index} missing required 'gps' section"
        )

    gps = gcp['gps']
    description = _get_gcp_description(gcp, index)

    # Validate latitude
    if 'latitude' not in gps:
        raise ValueError(
            f"GCP at {description} missing required 'latitude' field"
        )

    _validate_numeric_field(
        gps['latitude'],
        'latitude',
        description,
        min_value=MIN_LATITUDE,
        max_value=MAX_LATITUDE
    )

    # Validate longitude
    if 'longitude' not in gps:
        raise ValueError(
            f"GCP at {description} missing required 'longitude' field"
        )

    _validate_numeric_field(
        gps['longitude'],
        'longitude',
        description,
        min_value=MIN_LONGITUDE,
        max_value=MAX_LONGITUDE
    )


def validate_gcp_elevation(gcp: Dict[str, Any], index: int) -> None:
    """Validate elevation of a ground control point.

    Args:
        gcp: Ground control point dictionary containing 'gps' section
        index: Index of the GCP in the list (for error messages)

    Raises:
        ValueError: If elevation is invalid
    """
    if 'gps' not in gcp:
        return  # Will be caught by GPS validation

    gps = gcp['gps']

    # Elevation is optional, only validate if present
    if 'elevation' not in gps:
        return

    description = _get_gcp_description(gcp, index)

    _validate_numeric_field(
        gps['elevation'],
        'elevation',
        description,
        min_value=MIN_ELEVATION,
        max_value=MAX_ELEVATION,
        units='meters'
    )


def _validate_image_dimension(
    dimension: Any,
    dimension_name: str
) -> Optional[int]:
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
                f"{dimension_name} must be a positive integer, "
                f"got {type(dimension).__name__}"
            )

    # Convert to int and validate range
    try:
        dim_int = int(dimension)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(
            f"{dimension_name} must be a positive integer, got {dimension}"
        ) from e

    if dim_int <= 0:
        raise ValueError(
            f"{dimension_name} must be positive, got {dim_int}"
        )

    # Sanity check for reasonable image dimensions (up to 100 megapixels)
    if dim_int > 100000:
        raise ValueError(
            f"{dimension_name} {dim_int} exceeds maximum allowed value of 100000"
        )

    return dim_int


def validate_gcp_pixel_coordinates(
    gcp: Dict[str, Any],
    index: int,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None
) -> None:
    """Validate pixel coordinates of a ground control point.

    Args:
        gcp: Ground control point dictionary containing 'image' section
        index: Index of the GCP in the list (for error messages)
        image_width: Optional image width in pixels for bounds checking
        image_height: Optional image height in pixels for bounds checking

    Raises:
        ValueError: If pixel coordinates are invalid
    """
    if 'image' not in gcp:
        raise ValueError(
            f"GCP at index {index} missing required 'image' section"
        )

    image = gcp['image']
    description = _get_gcp_description(gcp, index)

    # Validate u coordinate
    if 'u' not in image:
        raise ValueError(
            f"GCP at {description} missing required 'u' (horizontal pixel) field"
        )

    u = image['u']
    _validate_numeric_field(u, 'u coordinate', description)

    # Validate v coordinate
    if 'v' not in image:
        raise ValueError(
            f"GCP at {description} missing required 'v' (vertical pixel) field"
        )

    v = image['v']
    _validate_numeric_field(v, 'v coordinate', description)

    # Validate bounds - check each dimension independently when provided
    if image_width is not None:
        if u < 0 or u >= image_width:
            raise ValueError(
                f"GCP at {description}: u coordinate {u} outside image width "
                f"[0, {image_width})"
            )

    if image_height is not None:
        if v < 0 or v >= image_height:
            raise ValueError(
                f"GCP at {description}: v coordinate {v} outside image height "
                f"[0, {image_height})"
            )


def detect_duplicate_gcps(
    gcps: List[Dict[str, Any]],
    gps_epsilon: float = GPS_EPSILON,
    pixel_epsilon: float = PIXEL_EPSILON
) -> None:
    """Detect duplicate ground control points.

    Two GCPs are considered duplicates if BOTH their GPS and pixel coordinates
    are within epsilon thresholds of each other.

    Args:
        gcps: List of ground control points (must be pre-validated)
        gps_epsilon: Epsilon threshold for GPS coordinate comparison (degrees)
        pixel_epsilon: Epsilon threshold for pixel coordinate comparison (pixels)

    Raises:
        ValueError: If duplicate GCPs are detected
    """
    # Extract coordinates upfront to avoid repeated dictionary lookups
    # At this point, GCPs have been validated so we know these fields exist
    coords: List[Tuple[float, float, float, float, str]] = []
    for i, gcp in enumerate(gcps):
        desc = _get_gcp_description(gcp, i)
        gps = gcp['gps']
        image = gcp['image']
        coords.append((
            gps['latitude'],
            gps['longitude'],
            image['u'],
            image['v'],
            desc
        ))

    for i in range(len(coords)):
        lat_i, lon_i, u_i, v_i, desc_i = coords[i]

        for j in range(i + 1, len(coords)):
            lat_j, lon_j, u_j, v_j, desc_j = coords[j]

            # Check if GPS coordinates are within epsilon
            gps_duplicate = (
                abs(lat_i - lat_j) < gps_epsilon and
                abs(lon_i - lon_j) < gps_epsilon
            )

            # Check if pixel coordinates are within epsilon
            pixel_duplicate = (
                abs(u_i - u_j) < pixel_epsilon and
                abs(v_i - v_j) < pixel_epsilon
            )

            # Only flag as duplicate if BOTH GPS and pixel coords match
            if gps_duplicate and pixel_duplicate:
                raise ValueError(
                    f"Duplicate GCP detected at {desc_i} and {desc_j} "
                    f"(GPS coordinates within {gps_epsilon} degrees and "
                    f"pixel coordinates within {pixel_epsilon} pixels)"
                )


def validate_ground_control_points(
    gcps: Any,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    min_gcp_count: int = 6
) -> List[Dict[str, Any]]:
    """Validate ground control points configuration.

    Supports two formats:
    1. Single GCP set: gcps is a list of GCP dictionaries (default format)
    2. Multiple GCP sets: gcps is a dict where keys are set identifiers and
       values are lists of GCP dictionaries

    Args:
        gcps: Ground control points in either list or dict format
        image_width: Optional image width for pixel coordinate validation
        image_height: Optional image height for pixel coordinate validation
        min_gcp_count: Minimum recommended number of GCPs (default: 6)

    Returns:
        Validated list of GCP dictionaries (normalized to list format)

    Raises:
        ValueError: If GCPs fail validation with clear error messages

    Example:
        >>> # Single set format
        >>> gcps = [{"gps": {...}, "image": {...}}, ...]
        >>> validated = validate_ground_control_points(gcps)

        >>> # Multiple sets format
        >>> gcps = {
        ...     "set_1": [{"gps": {...}, "image": {...}}, ...],
        ...     "set_2": [{"gps": {...}, "image": {...}}, ...]
        ... }
        >>> validated = validate_ground_control_points(gcps)
    """
    # Validate and normalize image dimensions first
    validated_width = _validate_image_dimension(image_width, 'image_width')
    validated_height = _validate_image_dimension(image_height, 'image_height')

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
        raise ValueError(
            f"ground_control_points must be a list or dict, got {type(gcps).__name__}"
        )

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

    # Log if pixel validation will be skipped (WARNING level since validation is reduced)
    if validated_width is None and validated_height is None:
        logger.warning(
            "Image dimensions not provided, skipping pixel coordinate bounds validation"
        )
    elif validated_width is None or validated_height is None:
        provided = 'image_width' if validated_width is not None else 'image_height'
        missing = 'image_height' if validated_width is not None else 'image_width'
        logger.warning(
            f"Only {provided} provided, {missing} missing. "
            f"Partial pixel coordinate bounds validation will be performed"
        )

    # Validate each GCP
    for i, gcp in enumerate(gcp_list):
        if not isinstance(gcp, dict):
            raise ValueError(
                f"GCP at index {i} must be a dictionary, got {type(gcp).__name__}"
            )

        # Validate GPS coordinates
        validate_gcp_gps_coordinates(gcp, i)

        # Validate elevation if present
        validate_gcp_elevation(gcp, i)

        # Validate pixel coordinates
        validate_gcp_pixel_coordinates(gcp, i, validated_width, validated_height)

    # Check for duplicates
    if len(gcp_list) > 1:
        detect_duplicate_gcps(gcp_list)

    logger.info(f"Successfully validated {len(gcp_list)} ground control points")

    return gcp_list
