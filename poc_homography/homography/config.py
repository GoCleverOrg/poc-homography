"""
Configuration for homography approach selection.

Supports runtime selection and fallback chains for robust operation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

# Import GCP validation from dedicated module
from poc_homography.gcp_validation import (
    validate_ground_control_points,
)
from poc_homography.homography.interface import CoordinateSystemMode, HomographyApproach

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
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
    fallback_approaches: list[HomographyApproach] = field(default_factory=list)
    approach_specific_config: dict[str, dict[str, Any]] = field(default_factory=dict)
    coordinate_system_mode: CoordinateSystemMode = CoordinateSystemMode.ORIGIN_AT_CAMERA

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
                f"Invalid approach '{approach_str}'. Must be one of: {', '.join(valid_approaches)}"
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
    def from_dict(cls, config: dict) -> "HomographyConfig":
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
        if "approach" not in config:
            raise ValueError(
                "Configuration missing required 'approach' field. "
                "Must specify one of: 'intrinsic_extrinsic', 'feature_match', 'learned'"
            )

        approach_str = config["approach"]
        approach = cls._parse_approach(approach_str)

        # Parse fallback approaches
        fallback_approaches = []
        if "fallback_approaches" in config:
            fallback_list = config["fallback_approaches"]
            if not isinstance(fallback_list, list):
                raise ValueError(f"'fallback_approaches' must be a list, got {type(fallback_list)}")

            for fallback_str in fallback_list:
                fallback_approach = cls._parse_approach(fallback_str)
                fallback_approaches.append(fallback_approach)

        # Parse coordinate system mode (optional, defaults to ORIGIN_AT_CAMERA)
        coordinate_system_mode = CoordinateSystemMode.ORIGIN_AT_CAMERA
        if "coordinate_system_mode" in config:
            mode_str = config["coordinate_system_mode"]
            coordinate_system_mode = cls._parse_coordinate_system_mode(mode_str)

        # Extract approach-specific configuration
        approach_specific_config = {}

        # Look for configuration keys matching approach names
        approach_keys = [a.value for a in HomographyApproach]
        for key in config:
            if key in approach_keys and isinstance(config[key], dict):
                approach_specific_config[key] = config[key]

        # Validate ground control points if present in feature_match config
        if "feature_match" in approach_specific_config:
            feature_match_config = approach_specific_config["feature_match"]

            if "ground_control_points" in feature_match_config:
                gcps = feature_match_config["ground_control_points"]

                # Extract optional image dimensions for pixel validation
                # Check directly under feature_match first, then in camera_capture_context
                image_width = feature_match_config.get("image_width")
                image_height = feature_match_config.get("image_height")

                # Fall back to camera_capture_context if dimensions not found directly
                if image_width is None or image_height is None:
                    camera_ctx = feature_match_config.get("camera_capture_context", {})
                    if image_width is None:
                        image_width = camera_ctx.get("image_width")
                    if image_height is None:
                        image_height = camera_ctx.get("image_height")

                # Extract optional minimum GCP count
                min_gcp_count = feature_match_config.get("min_gcp_count", 6)

                # Validate and normalize GCPs
                try:
                    validated_gcps = validate_ground_control_points(
                        gcps,
                        image_width=image_width,
                        image_height=image_height,
                        min_gcp_count=min_gcp_count,
                    )
                    # Update config with validated GCPs (normalized to list format)
                    feature_match_config["ground_control_points"] = validated_gcps
                except ValueError as e:
                    raise ValueError(f"Ground control points validation failed: {e}") from e

        return cls(
            approach=approach,
            fallback_approaches=fallback_approaches,
            approach_specific_config=approach_specific_config,
            coordinate_system_mode=coordinate_system_mode,
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
        result: dict[str, Any] = {
            "approach": self.approach.value,
            "coordinate_system_mode": self.coordinate_system_mode.value,
        }

        if self.fallback_approaches:
            result["fallback_approaches"] = [
                approach.value for approach in self.fallback_approaches
            ]

        # Add approach-specific configurations
        for approach_key, approach_config in self.approach_specific_config.items():
            result[approach_key] = approach_config

        return result


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
            "intrinsic_extrinsic": {
                "sensor_width_mm": 7.18,
                "base_focal_length_mm": 5.9,
                "pixels_per_meter": 100.0,
            },
            "feature_match": {
                "detector": "sift",
                "min_matches": 4,
                "ransac_threshold": 5.0,
            },
            "learned": {
                "model_path": None,
                "confidence_threshold": 0.5,
            },
        },
    )
