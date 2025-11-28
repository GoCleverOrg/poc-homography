"""
Configuration for homography approach selection.

Supports runtime selection and fallback chains for robust operation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
import logging

from homography_interface import HomographyApproach, CoordinateSystemMode

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
    description = gcp.get('metadata', {}).get('description', f'index {index}')

    # Validate latitude
    if 'latitude' not in gps:
        raise ValueError(
            f"GCP at {description} missing required 'latitude' field"
        )

    latitude = gps['latitude']
    if not isinstance(latitude, (int, float)):
        raise ValueError(
            f"GCP at {description}: latitude must be a number, got {type(latitude).__name__}"
        )

    if latitude < MIN_LATITUDE or latitude > MAX_LATITUDE:
        raise ValueError(
            f"GCP at {description}: latitude {latitude} outside valid range "
            f"[{MIN_LATITUDE}, {MAX_LATITUDE}]"
        )

    # Validate longitude
    if 'longitude' not in gps:
        raise ValueError(
            f"GCP at {description} missing required 'longitude' field"
        )

    longitude = gps['longitude']
    if not isinstance(longitude, (int, float)):
        raise ValueError(
            f"GCP at {description}: longitude must be a number, got {type(longitude).__name__}"
        )

    if longitude < MIN_LONGITUDE or longitude > MAX_LONGITUDE:
        raise ValueError(
            f"GCP at {description}: longitude {longitude} outside valid range "
            f"[{MIN_LONGITUDE}, {MAX_LONGITUDE}]"
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

    elevation = gps['elevation']
    description = gcp.get('metadata', {}).get('description', f'index {index}')

    if not isinstance(elevation, (int, float)):
        raise ValueError(
            f"GCP at {description}: elevation must be a number, got {type(elevation).__name__}"
        )

    if elevation < MIN_ELEVATION or elevation > MAX_ELEVATION:
        raise ValueError(
            f"GCP at {description}: elevation {elevation} meters outside valid range "
            f"[{MIN_ELEVATION}, {MAX_ELEVATION}]"
        )


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
    description = gcp.get('metadata', {}).get('description', f'index {index}')

    # Validate u coordinate
    if 'u' not in image:
        raise ValueError(
            f"GCP at {description} missing required 'u' (horizontal pixel) field"
        )

    u = image['u']
    if not isinstance(u, (int, float)):
        raise ValueError(
            f"GCP at {description}: u coordinate must be a number, got {type(u).__name__}"
        )

    # Validate v coordinate
    if 'v' not in image:
        raise ValueError(
            f"GCP at {description} missing required 'v' (vertical pixel) field"
        )

    v = image['v']
    if not isinstance(v, (int, float)):
        raise ValueError(
            f"GCP at {description}: v coordinate must be a number, got {type(v).__name__}"
        )

    # Validate bounds if image dimensions provided
    if image_width is not None and image_height is not None:
        if u < 0 or u >= image_width:
            raise ValueError(
                f"GCP at {description}: u coordinate {u} outside image width "
                f"[0, {image_width})"
            )

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
        gcps: List of ground control points
        gps_epsilon: Epsilon threshold for GPS coordinate comparison (degrees)
        pixel_epsilon: Epsilon threshold for pixel coordinate comparison (pixels)

    Raises:
        ValueError: If duplicate GCPs are detected
    """
    for i in range(len(gcps)):
        gcp_i = gcps[i]
        desc_i = gcp_i.get('metadata', {}).get('description', f'index {i}')

        gps_i = gcp_i.get('gps', {})
        lat_i = gps_i.get('latitude', 0)
        lon_i = gps_i.get('longitude', 0)

        image_i = gcp_i.get('image', {})
        u_i = image_i.get('u', 0)
        v_i = image_i.get('v', 0)

        for j in range(i + 1, len(gcps)):
            gcp_j = gcps[j]
            desc_j = gcp_j.get('metadata', {}).get('description', f'index {j}')

            gps_j = gcp_j.get('gps', {})
            lat_j = gps_j.get('latitude', 0)
            lon_j = gps_j.get('longitude', 0)

            image_j = gcp_j.get('image', {})
            u_j = image_j.get('u', 0)
            v_j = image_j.get('v', 0)

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

    # Check minimum count (warning, not error)
    if len(gcp_list) < min_gcp_count:
        logger.warning(
            f"Only {len(gcp_list)} GCPs provided, recommended minimum is {min_gcp_count} "
            f"for robust RANSAC-based pose estimation"
        )

    # Log if pixel validation will be skipped
    if image_width is None or image_height is None:
        logger.info(
            "Image dimensions not provided, skipping pixel coordinate bounds validation"
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
        validate_gcp_pixel_coordinates(gcp, i, image_width, image_height)

    # Check for duplicates
    if len(gcp_list) > 1:
        detect_duplicate_gcps(gcp_list)

    logger.info(f"Successfully validated {len(gcp_list)} ground control points")

    return gcp_list


@dataclass
class HomographyConfig:
    """Configuration for homography provider selection.

    Attributes:
        approach: Primary homography approach to use
        fallback_approaches: Ordered list of fallback approaches if primary fails
        approach_specific_config: Dict of approach-specific configuration
        coordinate_system_mode: Mode for setting world coordinate system origin.
            Controls whether camera position is at origin (ORIGIN_AT_CAMERA)
            or derived from GPS coordinates (GPS_BASED_ORIGIN). Default is
            ORIGIN_AT_CAMERA for backward compatibility and single-camera use.
    """
    approach: HomographyApproach = HomographyApproach.INTRINSIC_EXTRINSIC
    fallback_approaches: List[HomographyApproach] = field(default_factory=list)
    approach_specific_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    coordinate_system_mode: CoordinateSystemMode = CoordinateSystemMode.ORIGIN_AT_CAMERA

    @classmethod
    def from_yaml(cls, path: str) -> 'HomographyConfig':
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            HomographyConfig instance loaded from file

        Raises:
            FileNotFoundError: If configuration file does not exist
            ValueError: If configuration file is malformed or contains invalid values
            yaml.YAMLError: If YAML parsing fails

        Example:
            >>> config = HomographyConfig.from_yaml('config/homography_config.yaml')
            >>> print(config.approach)
            HomographyApproach.INTRINSIC_EXTRINSIC
        """
        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}\n"
                f"Please create a configuration file or use get_default_config()"
            )

        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML configuration file: {e}") from e

        if not data:
            raise ValueError(
                f"Configuration file is empty: {path}\n"
                f"Expected a 'homography' section with approach configuration"
            )

        # Extract homography configuration section
        if 'homography' not in data:
            raise ValueError(
                f"Configuration file missing 'homography' section: {path}\n"
                f"Expected structure: homography:\n  approach: ...\n  ..."
            )

        homography_config = data['homography']

        return cls.from_dict(homography_config)

    @staticmethod
    def _parse_approach(approach_str: str) -> HomographyApproach:
        """Parse an approach string into HomographyApproach enum.

        Args:
            approach_str: String representation of the approach

        Returns:
            HomographyApproach enum value

        Raises:
            ValueError: If approach_str is not a valid approach
        """
        try:
            return HomographyApproach(approach_str)
        except ValueError:
            valid_approaches = [a.value for a in HomographyApproach]
            raise ValueError(
                f"Invalid approach '{approach_str}'. "
                f"Must be one of: {', '.join(valid_approaches)}"
            ) from None

    @staticmethod
    def _parse_coordinate_system_mode(mode_str: str) -> CoordinateSystemMode:
        """Parse a coordinate system mode string into CoordinateSystemMode enum.

        Args:
            mode_str: String representation of the coordinate system mode

        Returns:
            CoordinateSystemMode enum value

        Raises:
            ValueError: If mode_str is not a valid coordinate system mode
        """
        try:
            return CoordinateSystemMode(mode_str)
        except ValueError:
            valid_modes = [m.value for m in CoordinateSystemMode]
            raise ValueError(
                f"Invalid coordinate_system_mode '{mode_str}'. "
                f"Must be one of: {', '.join(valid_modes)}"
            ) from None

    @classmethod
    def from_dict(cls, config: dict) -> 'HomographyConfig':
        """Create configuration from dictionary.

        Args:
            config: Dictionary containing configuration data with keys:
                - 'approach': Primary approach name (string)
                - 'fallback_approaches': List of fallback approach names (optional)
                - Approach-specific keys: e.g., 'intrinsic_extrinsic', 'feature_match', etc.

        Returns:
            HomographyConfig instance

        Raises:
            ValueError: If configuration is invalid or missing required fields

        Example:
            >>> config_dict = {
            ...     'approach': 'intrinsic_extrinsic',
            ...     'fallback_approaches': ['feature_match'],
            ...     'intrinsic_extrinsic': {'sensor_width_mm': 7.18}
            ... }
            >>> config = HomographyConfig.from_dict(config_dict)
        """
        if not isinstance(config, dict):
            raise ValueError(f"Configuration must be a dictionary, got {type(config)}")

        # Parse primary approach
        if 'approach' not in config:
            raise ValueError(
                "Configuration missing required 'approach' field. "
                "Must specify one of: 'intrinsic_extrinsic', 'feature_match', 'learned'"
            )

        approach_str = config['approach']
        approach = cls._parse_approach(approach_str)

        # Parse fallback approaches
        fallback_approaches = []
        if 'fallback_approaches' in config:
            fallback_list = config['fallback_approaches']
            if not isinstance(fallback_list, list):
                raise ValueError(
                    f"'fallback_approaches' must be a list, got {type(fallback_list)}"
                )

            for fallback_str in fallback_list:
                fallback_approach = cls._parse_approach(fallback_str)
                fallback_approaches.append(fallback_approach)

        # Parse coordinate system mode (optional, defaults to ORIGIN_AT_CAMERA)
        coordinate_system_mode = CoordinateSystemMode.ORIGIN_AT_CAMERA
        if 'coordinate_system_mode' in config:
            mode_str = config['coordinate_system_mode']
            coordinate_system_mode = cls._parse_coordinate_system_mode(mode_str)

        # Extract approach-specific configuration
        approach_specific_config = {}

        # Look for configuration keys matching approach names
        approach_keys = [a.value for a in HomographyApproach]
        for key in config.keys():
            if key in approach_keys and isinstance(config[key], dict):
                approach_specific_config[key] = config[key]

        # Validate ground control points if present in feature_match config
        if 'feature_match' in approach_specific_config:
            feature_match_config = approach_specific_config['feature_match']

            if 'ground_control_points' in feature_match_config:
                gcps = feature_match_config['ground_control_points']

                # Extract optional image dimensions for pixel validation
                image_width = feature_match_config.get('image_width')
                image_height = feature_match_config.get('image_height')

                # Extract optional minimum GCP count
                min_gcp_count = feature_match_config.get('min_gcp_count', 6)

                # Validate and normalize GCPs
                try:
                    validated_gcps = validate_ground_control_points(
                        gcps,
                        image_width=image_width,
                        image_height=image_height,
                        min_gcp_count=min_gcp_count
                    )
                    # Update config with validated GCPs (normalized to list format)
                    feature_match_config['ground_control_points'] = validated_gcps
                except ValueError as e:
                    raise ValueError(
                        f"Ground control points validation failed: {e}"
                    ) from e

        return cls(
            approach=approach,
            fallback_approaches=fallback_approaches,
            approach_specific_config=approach_specific_config,
            coordinate_system_mode=coordinate_system_mode
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation suitable for YAML serialization

        Example:
            >>> config = get_default_config()
            >>> config_dict = config.to_dict()
            >>> print(config_dict['approach'])
            'intrinsic_extrinsic'
        """
        result = {
            'approach': self.approach.value,
            'coordinate_system_mode': self.coordinate_system_mode.value,
        }

        if self.fallback_approaches:
            result['fallback_approaches'] = [
                approach.value for approach in self.fallback_approaches
            ]

        # Add approach-specific configurations
        for approach_key, approach_config in self.approach_specific_config.items():
            result[approach_key] = approach_config

        return result

    def get_approach_config(self, approach: HomographyApproach) -> Dict[str, Any]:
        """Get configuration for a specific approach.

        Args:
            approach: The homography approach to get configuration for

        Returns:
            Dictionary of approach-specific configuration parameters.
            Returns empty dict if no configuration exists for this approach.

        Example:
            >>> config = get_default_config()
            >>> intrinsic_config = config.get_approach_config(
            ...     HomographyApproach.INTRINSIC_EXTRINSIC
            ... )
            >>> print(intrinsic_config.get('sensor_width_mm'))
            7.18
        """
        return self.approach_specific_config.get(approach.value, {})

    def save_to_yaml(self, path: str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path where configuration file should be written.
                Should be a relative path within the project directory
                or an absolute path to a trusted location.

        Raises:
            IOError: If file cannot be written
            ValueError: If path contains suspicious patterns or escapes project directory

        Example:
            >>> config = get_default_config()
            >>> config.save_to_yaml('my_config.yaml')
        """
        config_path = Path(path).resolve()

        # Ensure path doesn't escape current directory or project root
        # This prevents directory traversal attacks
        try:
            config_path.relative_to(Path.cwd())
        except ValueError:
            raise ValueError(
                f"Path '{path}' must be within the project directory. "
                f"Resolved path: {config_path}, Current directory: {Path.cwd()}"
            ) from None

        # Create parent directories if needed
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Wrap in 'homography' section for consistency with from_yaml
        output = {'homography': self.to_dict()}

        try:
            with open(config_path, 'w') as f:
                yaml.safe_dump(output, f, default_flow_style=False, sort_keys=False)
        except IOError as e:
            raise IOError(f"Failed to write configuration file: {e}") from e


def get_default_config() -> HomographyConfig:
    """Return default configuration using intrinsic/extrinsic approach.

    The default configuration uses the intrinsic/extrinsic approach with
    feature matching as a fallback. This is suitable for most PTZ camera
    applications where camera parameters are known.

    Returns:
        HomographyConfig with sensible defaults

    Example:
        >>> config = get_default_config()
        >>> print(config.approach)
        HomographyApproach.INTRINSIC_EXTRINSIC
        >>> print(config.fallback_approaches)
        [HomographyApproach.FEATURE_MATCH]
    """
    return HomographyConfig(
        approach=HomographyApproach.INTRINSIC_EXTRINSIC,
        fallback_approaches=[HomographyApproach.FEATURE_MATCH],
        approach_specific_config={
            'intrinsic_extrinsic': {
                'sensor_width_mm': 7.18,
                'base_focal_length_mm': 5.9,
                'pixels_per_meter': 100.0,
            },
            'feature_match': {
                'detector': 'sift',
                'min_matches': 4,
                'ransac_threshold': 5.0,
            },
            'learned': {
                'model_path': None,
                'confidence_threshold': 0.5,
            }
        }
    )
