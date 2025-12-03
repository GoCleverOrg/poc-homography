"""
Configuration for homography approach selection.

Supports runtime selection and fallback chains for robust operation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
import logging

from poc_homography.homography_interface import HomographyApproach, CoordinateSystemMode

# Import GCP validation from dedicated module
from poc_homography.gcp_validation import (
    validate_ground_control_points,
    validate_gcp_gps_coordinates,
    validate_gcp_elevation,
    validate_gcp_pixel_coordinates,
    detect_duplicate_gcps,
    GPS_EPSILON,
    PIXEL_EPSILON,
    MIN_LATITUDE,
    MAX_LATITUDE,
    MIN_LONGITUDE,
    MAX_LONGITUDE,
    MIN_ELEVATION,
    MAX_ELEVATION,
    MAX_GCP_COUNT,
)

logger = logging.getLogger(__name__)


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
