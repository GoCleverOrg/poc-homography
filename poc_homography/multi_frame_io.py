#!/usr/bin/env python3
"""
YAML I/O for multi-frame calibration data and results.

This module provides functions to load and save multi-frame calibration data
in YAML format, including:
- MultiFrameCalibrationData (frames, GCPs, camera config)
- MultiFrameCalibrationResult (optimized parameters, per-frame errors)

YAML Schema:
    The expected YAML structure follows this format:

    multi_frame_calibration:
      frames:
        - frame_id: "frame_001"
          image_path: "data/calibration/frame_001.jpg"
          ptz_position:
            pan: 31.0
            tilt: 13.0
            zoom: 1.0
          timestamp: "2025-01-05T10:00:00Z"  # ISO 8601 format with timezone
        - frame_id: "frame_002"
          image_path: "data/calibration/frame_002.jpg"
          ptz_position:
            pan: 45.2
            tilt: 15.5
            zoom: 1.0
          timestamp: "2025-01-05T10:05:00Z"

      gcps:
        - gcp_id: "gcp_001"
          gps:
            latitude: 39.640583
            longitude: -0.230194
            elevation: 12.5  # optional
          utm:  # optional
            easting: 729345.67
            northing: 4389234.12
          metadata:  # optional
            description: "Building corner NW"
            accuracy: "high"
          frame_observations:
            - frame_id: "frame_001"
              image:
                u: 1250.5
                v: 680.0
            - frame_id: "frame_002"
              image:
                u: 1180.2
                v: 720.5

      camera_config:
        reference_lat: 39.641000
        reference_lon: -0.230500
        utm_crs: "EPSG:25830"  # optional
        K:  # 3x3 intrinsic matrix (optional, can be computed from zoom)
          - [2500.0, 0.0, 960.0]
          - [0.0, 2500.0, 540.0]
          - [0.0, 0.0, 1.0]
        w_pos: [0.0, 0.0, 5.0]  # Camera position in meters [X, Y, Z]

      # Optional: calibration result (added after calibration completes)
      calibration_result:
        optimized_params:
          delta_pan_deg: 0.523
          delta_tilt_deg: -0.412
          delta_roll_deg: 0.089
          delta_X_m: 0.234
          delta_Y_m: -0.156
          delta_Z_m: 0.078
        diagnostics:
          initial_error_px: 12.4
          final_error_px: 3.2
          num_inliers: 27
          num_outliers: 3
          inlier_ratio: 0.900
          converged: true
          iterations: 45
          per_frame_residuals:
            - frame_id: "frame_001"
              rms_error_px: 2.8
              num_inliers: 15
              num_outliers: 1
            - frame_id: "frame_002"
              rms_error_px: 3.5
              num_inliers: 12
              num_outliers: 2

Usage Example:
    >>> from poc_homography.multi_frame_io import (
    ...     load_multi_frame_calibration_data,
    ...     save_multi_frame_calibration_data,
    ...     create_example_multi_frame_config
    ... )
    >>>
    >>> # Load calibration data from YAML
    >>> calib_data = load_multi_frame_calibration_data("config/multi_frame.yaml")
    >>> print(f"Loaded {len(calib_data.frames)} frames and {len(calib_data.gcps)} GCPs")
    >>>
    >>> # Save calibration data to YAML
    >>> save_multi_frame_calibration_data(calib_data, "output/calibration_data.yaml")
    >>>
    >>> # Create example configuration file
    >>> example_yaml = create_example_multi_frame_config()
    >>> with open("config/example.yaml", "w") as f:
    ...     f.write(example_yaml)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import numpy as np
import logging

from poc_homography.multi_frame_calibrator import (
    PTZPosition,
    FrameObservation,
    MultiFrameGCP,
    MultiFrameCalibrationData,
    MultiFrameCalibrationResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Serialization Helper Functions
# ============================================================================

def _serialize_ptz_position(ptz: PTZPosition) -> Dict[str, float]:
    """
    Convert PTZPosition to dictionary for YAML serialization.

    Args:
        ptz: PTZPosition instance

    Returns:
        Dictionary with keys 'pan', 'tilt', 'zoom'
    """
    return {
        'pan': float(ptz.pan),
        'tilt': float(ptz.tilt),
        'zoom': float(ptz.zoom)
    }


def _deserialize_ptz_position(data: Dict[str, Any]) -> PTZPosition:
    """
    Convert dictionary to PTZPosition instance.

    Args:
        data: Dictionary with keys 'pan', 'tilt', 'zoom'

    Returns:
        PTZPosition instance

    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = ['pan', 'tilt', 'zoom']
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        raise ValueError(
            f"PTZ position missing required fields: {', '.join(missing_fields)}"
        )

    try:
        return PTZPosition(
            pan=float(data['pan']),
            tilt=float(data['tilt']),
            zoom=float(data['zoom'])
        )
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid PTZ position data: {e}") from e


def _serialize_frame_observation(frame: FrameObservation) -> Dict[str, Any]:
    """
    Convert FrameObservation to dictionary for YAML serialization.

    Args:
        frame: FrameObservation instance

    Returns:
        Dictionary with frame data including serialized PTZ position and timestamp
    """
    return {
        'frame_id': frame.frame_id,
        'image_path': frame.image_path,
        'ptz_position': _serialize_ptz_position(frame.ptz_position),
        'timestamp': frame.timestamp.isoformat()
    }


def _deserialize_frame_observation(data: Dict[str, Any]) -> FrameObservation:
    """
    Convert dictionary to FrameObservation instance.

    Args:
        data: Dictionary with frame data

    Returns:
        FrameObservation instance

    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = ['frame_id', 'image_path', 'ptz_position', 'timestamp']
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        raise ValueError(
            f"Frame observation missing required fields: {', '.join(missing_fields)}"
        )

    try:
        # Parse timestamp (ISO 8601 format)
        timestamp_str = data['timestamp']
        if isinstance(timestamp_str, datetime):
            timestamp = timestamp_str
        else:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

        # Ensure timezone awareness
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        return FrameObservation(
            frame_id=str(data['frame_id']),
            image_path=str(data['image_path']),
            ptz_position=_deserialize_ptz_position(data['ptz_position']),
            timestamp=timestamp
        )
    except Exception as e:
        raise ValueError(f"Invalid frame observation data: {e}") from e


def _serialize_multi_frame_gcp(gcp: MultiFrameGCP) -> Dict[str, Any]:
    """
    Convert MultiFrameGCP to dictionary for YAML serialization.

    Args:
        gcp: MultiFrameGCP instance

    Returns:
        Dictionary with GCP data
    """
    result = {
        'gcp_id': gcp.gcp_id,
        'gps': {
            'latitude': float(gcp.gps_lat),
            'longitude': float(gcp.gps_lon)
        },
        'frame_observations': []
    }

    # Add optional UTM coordinates
    if gcp.utm_easting is not None and gcp.utm_northing is not None:
        result['utm'] = {
            'easting': float(gcp.utm_easting),
            'northing': float(gcp.utm_northing)
        }

    # Serialize frame observations
    for frame_id, pixel_coords in gcp.frame_observations.items():
        result['frame_observations'].append({
            'frame_id': frame_id,
            'image': {
                'u': float(pixel_coords['u']),
                'v': float(pixel_coords['v'])
            }
        })

    return result


def _deserialize_multi_frame_gcp(data: Dict[str, Any]) -> MultiFrameGCP:
    """
    Convert dictionary to MultiFrameGCP instance.

    Args:
        data: Dictionary with GCP data

    Returns:
        MultiFrameGCP instance

    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = ['gcp_id', 'gps', 'frame_observations']
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        raise ValueError(
            f"GCP missing required fields: {', '.join(missing_fields)}"
        )

    # Validate GPS coordinates
    gps = data['gps']
    if 'latitude' not in gps or 'longitude' not in gps:
        raise ValueError("GCP gps section missing 'latitude' or 'longitude'")

    try:
        gps_lat = float(gps['latitude'])
        gps_lon = float(gps['longitude'])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid GPS coordinates: {e}") from e

    # Parse UTM coordinates (optional)
    utm_easting = None
    utm_northing = None
    if 'utm' in data:
        utm = data['utm']
        if 'easting' in utm and 'northing' in utm:
            try:
                utm_easting = float(utm['easting'])
                utm_northing = float(utm['northing'])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid UTM coordinates: {e}") from e

    # Parse frame observations
    frame_observations = {}
    if not isinstance(data['frame_observations'], list):
        raise ValueError("frame_observations must be a list")

    for obs in data['frame_observations']:
        if 'frame_id' not in obs or 'image' not in obs:
            raise ValueError("Frame observation missing 'frame_id' or 'image'")

        image = obs['image']
        if 'u' not in image or 'v' not in image:
            raise ValueError("Image coordinates missing 'u' or 'v'")

        try:
            frame_id = str(obs['frame_id'])
            frame_observations[frame_id] = {
                'u': float(image['u']),
                'v': float(image['v'])
            }
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid frame observation data: {e}") from e

    if not frame_observations:
        raise ValueError(f"GCP '{data['gcp_id']}' has no valid frame observations")

    return MultiFrameGCP(
        gcp_id=str(data['gcp_id']),
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        frame_observations=frame_observations,
        utm_easting=utm_easting,
        utm_northing=utm_northing
    )


# ============================================================================
# Main Load/Save Functions
# ============================================================================

def save_multi_frame_calibration_data(
    data: MultiFrameCalibrationData,
    yaml_path: str
) -> None:
    """
    Save MultiFrameCalibrationData to YAML file.

    Args:
        data: MultiFrameCalibrationData instance to save
        yaml_path: Path where YAML file should be written

    Raises:
        IOError: If file cannot be written
        ValueError: If data is invalid

    Example:
        >>> calib_data = MultiFrameCalibrationData(frames=[...], gcps=[...], camera_config={...})
        >>> save_multi_frame_calibration_data(calib_data, "output/calibration.yaml")
    """
    # Validate input
    if not isinstance(data, MultiFrameCalibrationData):
        raise ValueError(
            f"data must be MultiFrameCalibrationData, got {type(data)}"
        )

    # Build YAML structure
    yaml_data = {
        'multi_frame_calibration': {
            'frames': [_serialize_frame_observation(f) for f in data.frames],
            'gcps': [_serialize_multi_frame_gcp(g) for g in data.gcps],
            'camera_config': {}
        }
    }

    # Serialize camera_config
    camera_config = data.camera_config.copy()

    # Convert numpy arrays to lists
    if 'K' in camera_config and isinstance(camera_config['K'], np.ndarray):
        camera_config['K'] = camera_config['K'].tolist()
    if 'w_pos' in camera_config and isinstance(camera_config['w_pos'], np.ndarray):
        camera_config['w_pos'] = camera_config['w_pos'].tolist()

    yaml_data['multi_frame_calibration']['camera_config'] = camera_config

    # Write YAML file
    yaml_path_obj = Path(yaml_path)
    yaml_path_obj.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(yaml_path_obj, 'w') as f:
            yaml.safe_dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved multi-frame calibration data to {yaml_path}")
    except IOError as e:
        raise IOError(f"Failed to write YAML file '{yaml_path}': {e}") from e


def load_multi_frame_calibration_data(yaml_path: str) -> MultiFrameCalibrationData:
    """
    Load MultiFrameCalibrationData from YAML file.

    Args:
        yaml_path: Path to YAML file containing calibration data

    Returns:
        MultiFrameCalibrationData instance

    Raises:
        FileNotFoundError: If YAML file does not exist
        ValueError: If YAML structure is invalid or missing required fields
        yaml.YAMLError: If YAML parsing fails

    Example:
        >>> calib_data = load_multi_frame_calibration_data("config/multi_frame.yaml")
        >>> print(f"Loaded {len(calib_data.frames)} frames")
    """
    yaml_path_obj = Path(yaml_path)

    if not yaml_path_obj.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    # Load YAML file
    try:
        with open(yaml_path_obj, 'r') as f:
            yaml_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file '{yaml_path}': {e}") from e

    if not yaml_data:
        raise ValueError(f"YAML file is empty: {yaml_path}")

    # Validate structure
    if 'multi_frame_calibration' not in yaml_data:
        raise ValueError(
            f"YAML file missing 'multi_frame_calibration' section. "
            f"Expected structure: multi_frame_calibration:\n  frames: [...]\n  gcps: [...]"
        )

    calib_section = yaml_data['multi_frame_calibration']

    # Validate required sections
    required_sections = ['frames', 'gcps', 'camera_config']
    missing_sections = [s for s in required_sections if s not in calib_section]
    if missing_sections:
        raise ValueError(
            f"Calibration section missing required fields: {', '.join(missing_sections)}"
        )

    # Deserialize frames
    frames_data = calib_section['frames']
    if not isinstance(frames_data, list):
        raise ValueError("'frames' section must be a list")
    if not frames_data:
        raise ValueError("'frames' list cannot be empty")

    try:
        frames = [_deserialize_frame_observation(f) for f in frames_data]
    except ValueError as e:
        raise ValueError(f"Failed to parse frames: {e}") from e

    # Deserialize GCPs
    gcps_data = calib_section['gcps']
    if not isinstance(gcps_data, list):
        raise ValueError("'gcps' section must be a list")
    if not gcps_data:
        raise ValueError("'gcps' list cannot be empty")

    try:
        gcps = [_deserialize_multi_frame_gcp(g) for g in gcps_data]
    except ValueError as e:
        raise ValueError(f"Failed to parse GCPs: {e}") from e

    # Deserialize camera_config
    camera_config = calib_section['camera_config'].copy()

    # Convert lists to numpy arrays if present
    if 'K' in camera_config and isinstance(camera_config['K'], list):
        camera_config['K'] = np.array(camera_config['K'], dtype=np.float64)
    if 'w_pos' in camera_config and isinstance(camera_config['w_pos'], list):
        camera_config['w_pos'] = np.array(camera_config['w_pos'], dtype=np.float64)

    logger.info(
        f"Loaded multi-frame calibration data from {yaml_path}: "
        f"{len(frames)} frames, {len(gcps)} GCPs"
    )

    return MultiFrameCalibrationData(
        frames=frames,
        gcps=gcps,
        camera_config=camera_config
    )


def save_multi_frame_calibration_result(
    result: MultiFrameCalibrationResult,
    yaml_path: str
) -> None:
    """
    Save MultiFrameCalibrationResult to YAML file.

    Args:
        result: MultiFrameCalibrationResult instance to save
        yaml_path: Path where YAML file should be written

    Raises:
        IOError: If file cannot be written
        ValueError: If result is invalid

    Example:
        >>> result = calibrator.calibrate()
        >>> save_multi_frame_calibration_result(result, "output/result.yaml")
    """
    if not isinstance(result, MultiFrameCalibrationResult):
        raise ValueError(
            f"result must be MultiFrameCalibrationResult, got {type(result)}"
        )

    # Build YAML structure
    yaml_data = {
        'calibration_result': {
            'optimized_params': {
                'delta_pan_deg': float(result.optimized_params[0]),
                'delta_tilt_deg': float(result.optimized_params[1]),
                'delta_roll_deg': float(result.optimized_params[2]),
                'delta_X_m': float(result.optimized_params[3]),
                'delta_Y_m': float(result.optimized_params[4]),
                'delta_Z_m': float(result.optimized_params[5])
            },
            'diagnostics': {
                'initial_error_px': float(result.initial_error),
                'final_error_px': float(result.final_error),
                'num_inliers': int(result.num_inliers),
                'num_outliers': int(result.num_outliers),
                'inlier_ratio': float(result.inlier_ratio),
                'converged': bool(result.convergence_info.get('success', False)),
                'iterations': int(result.convergence_info.get('iterations', 0)),
                'per_frame_residuals': []
            }
        }
    }

    # Add per-frame residuals
    for frame_id, rms_error in result.per_frame_errors.items():
        frame_residual = {
            'frame_id': frame_id,
            'rms_error_px': float(rms_error),
            'num_inliers': int(result.per_frame_inliers.get(frame_id, 0)),
            'num_outliers': int(result.per_frame_outliers.get(frame_id, 0))
        }
        yaml_data['calibration_result']['diagnostics']['per_frame_residuals'].append(
            frame_residual
        )

    # Write YAML file
    yaml_path_obj = Path(yaml_path)
    yaml_path_obj.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(yaml_path_obj, 'w') as f:
            yaml.safe_dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved multi-frame calibration result to {yaml_path}")
    except IOError as e:
        raise IOError(f"Failed to write YAML file '{yaml_path}': {e}") from e


def load_multi_frame_calibration_result(yaml_path: str) -> MultiFrameCalibrationResult:
    """
    Load MultiFrameCalibrationResult from YAML file.

    Args:
        yaml_path: Path to YAML file containing calibration result

    Returns:
        MultiFrameCalibrationResult instance

    Raises:
        FileNotFoundError: If YAML file does not exist
        ValueError: If YAML structure is invalid or missing required fields
        yaml.YAMLError: If YAML parsing fails

    Example:
        >>> result = load_multi_frame_calibration_result("output/result.yaml")
        >>> print(f"Final error: {result.final_error:.2f}px")
    """
    yaml_path_obj = Path(yaml_path)

    if not yaml_path_obj.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    # Load YAML file
    try:
        with open(yaml_path_obj, 'r') as f:
            yaml_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file '{yaml_path}': {e}") from e

    if not yaml_data:
        raise ValueError(f"YAML file is empty: {yaml_path}")

    # Validate structure
    if 'calibration_result' not in yaml_data:
        raise ValueError(
            f"YAML file missing 'calibration_result' section. "
            f"Expected structure: calibration_result:\n  optimized_params: {{...}}\n  diagnostics: {{...}}"
        )

    result_section = yaml_data['calibration_result']

    # Validate required sections
    required_sections = ['optimized_params', 'diagnostics']
    missing_sections = [s for s in required_sections if s not in result_section]
    if missing_sections:
        raise ValueError(
            f"Result section missing required fields: {', '.join(missing_sections)}"
        )

    # Parse optimized parameters
    params_data = result_section['optimized_params']
    required_params = [
        'delta_pan_deg', 'delta_tilt_deg', 'delta_roll_deg',
        'delta_X_m', 'delta_Y_m', 'delta_Z_m'
    ]
    missing_params = [p for p in required_params if p not in params_data]
    if missing_params:
        raise ValueError(
            f"Optimized params missing required fields: {', '.join(missing_params)}"
        )

    try:
        optimized_params = np.array([
            float(params_data['delta_pan_deg']),
            float(params_data['delta_tilt_deg']),
            float(params_data['delta_roll_deg']),
            float(params_data['delta_X_m']),
            float(params_data['delta_Y_m']),
            float(params_data['delta_Z_m'])
        ], dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid optimized parameters: {e}") from e

    # Parse diagnostics
    diagnostics = result_section['diagnostics']
    required_diag_fields = [
        'initial_error_px', 'final_error_px', 'num_inliers',
        'num_outliers', 'inlier_ratio'
    ]
    missing_diag = [f for f in required_diag_fields if f not in diagnostics]
    if missing_diag:
        raise ValueError(
            f"Diagnostics missing required fields: {', '.join(missing_diag)}"
        )

    try:
        initial_error = float(diagnostics['initial_error_px'])
        final_error = float(diagnostics['final_error_px'])
        num_inliers = int(diagnostics['num_inliers'])
        num_outliers = int(diagnostics['num_outliers'])
        inlier_ratio = float(diagnostics['inlier_ratio'])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid diagnostics data: {e}") from e

    # Parse convergence info
    convergence_info = {
        'success': bool(diagnostics.get('converged', False)),
        'iterations': int(diagnostics.get('iterations', 0))
    }

    # Parse per-frame errors
    per_frame_errors = {}
    per_frame_inliers = {}
    per_frame_outliers = {}

    if 'per_frame_residuals' in diagnostics:
        for frame_data in diagnostics['per_frame_residuals']:
            if 'frame_id' not in frame_data or 'rms_error_px' not in frame_data:
                continue

            frame_id = str(frame_data['frame_id'])
            per_frame_errors[frame_id] = float(frame_data['rms_error_px'])
            per_frame_inliers[frame_id] = int(frame_data.get('num_inliers', 0))
            per_frame_outliers[frame_id] = int(frame_data.get('num_outliers', 0))

    # Calculate total observations
    total_observations = num_inliers + num_outliers

    logger.info(f"Loaded multi-frame calibration result from {yaml_path}")

    return MultiFrameCalibrationResult(
        optimized_params=optimized_params,
        initial_error=initial_error,
        final_error=final_error,
        num_inliers=num_inliers,
        num_outliers=num_outliers,
        inlier_ratio=inlier_ratio,
        per_gcp_errors=[],  # Not stored in YAML
        convergence_info=convergence_info,
        per_frame_errors=per_frame_errors,
        per_frame_inliers=per_frame_inliers,
        per_frame_outliers=per_frame_outliers,
        total_observations=total_observations
    )


def create_example_multi_frame_config() -> str:
    """
    Create example multi-frame calibration YAML configuration.

    Returns a complete YAML configuration string with inline documentation
    explaining each field. Suitable for creating template configuration files.

    Returns:
        String containing example YAML configuration with comments

    Example:
        >>> example_yaml = create_example_multi_frame_config()
        >>> with open("config/example.yaml", "w") as f:
        ...     f.write(example_yaml)
    """
    return """# Multi-Frame Calibration Configuration
# This file defines calibration data for PTZ camera parameter optimization
# using Ground Control Points (GCPs) observed across multiple frames.

multi_frame_calibration:
  # ========================================================================
  # Frames: Camera captures at different PTZ positions
  # ========================================================================
  # Each frame represents a camera view at a known PTZ position.
  # The PTZ encoder readings (pan, tilt, zoom) are known from the camera API.
  # Calibration optimizes SHARED parameter deltas that apply to all frames.
  frames:
    - frame_id: "frame_001"               # Unique identifier for this frame
      image_path: "data/calibration/frame_001.jpg"  # Path to captured image
      ptz_position:
        pan: 31.0                         # Pan angle in degrees (from PTZ encoder)
        tilt: 13.0                        # Tilt angle in degrees (from PTZ encoder)
        zoom: 1.0                         # Zoom factor (unitless, typically 1.0-25.0)
      timestamp: "2025-01-05T10:00:00Z"   # Capture time (ISO 8601 with timezone)

    - frame_id: "frame_002"
      image_path: "data/calibration/frame_002.jpg"
      ptz_position:
        pan: 45.2
        tilt: 15.5
        zoom: 1.0
      timestamp: "2025-01-05T10:05:00Z"

    # Add more frames as needed (typically 3-10 frames recommended)

  # ========================================================================
  # Ground Control Points (GCPs): Known world-to-image correspondences
  # ========================================================================
  # Each GCP has a known GPS/UTM position and observed pixel coordinates
  # in one or more frames. GCPs visible in multiple frames provide strong
  # constraints for calibration.
  gcps:
    - gcp_id: "gcp_001"                   # Unique identifier for this GCP
      gps:
        latitude: 39.640583               # GPS latitude in decimal degrees
        longitude: -0.230194              # GPS longitude in decimal degrees
        elevation: 12.5                   # Optional: elevation in meters above sea level
      utm:                                # Optional: UTM coordinates (if available)
        easting: 729345.67                # UTM easting in meters
        northing: 4389234.12              # UTM northing in meters
      metadata:                           # Optional: additional information
        description: "Building corner NW" # Human-readable description
        accuracy: "high"                  # GPS accuracy level: high | medium | low
      frame_observations:                 # List of frames where this GCP is visible
        - frame_id: "frame_001"           # Must match a frame_id from frames list
          image:
            u: 1250.5                     # Horizontal pixel coordinate (0 = left edge)
            v: 680.0                      # Vertical pixel coordinate (0 = top edge)
        - frame_id: "frame_002"
          image:
            u: 1180.2
            v: 720.5

    - gcp_id: "gcp_002"
      gps:
        latitude: 39.640612
        longitude: -0.229856
        elevation: 12.8
      metadata:
        description: "Building corner NE"
        accuracy: "high"
      frame_observations:
        - frame_id: "frame_001"
          image:
            u: 2456.2
            v: 695.5
        - frame_id: "frame_002"
          image:
            u: 2380.0
            v: 710.3

    # Add more GCPs as needed (minimum 6 recommended, 10-20 optimal)
    # Distribute GCPs across the entire field of view for best results

  # ========================================================================
  # Camera Configuration: Initial camera parameters
  # ========================================================================
  camera_config:
    # Reference GPS position (typically the camera location)
    # Used for GPS-to-local coordinate conversion
    reference_lat: 39.641000              # Camera GPS latitude
    reference_lon: -0.230500              # Camera GPS longitude

    # Optional: UTM coordinate reference system
    utm_crs: "EPSG:25830"                 # EPSG code for UTM zone

    # Optional: Camera intrinsic matrix K (3x3)
    # If not provided, computed from zoom factor and sensor parameters
    # Format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    K:
      - [2500.0, 0.0, 960.0]              # Row 1: [fx, 0, cx]
      - [0.0, 2500.0, 540.0]              # Row 2: [0, fy, cy]
      - [0.0, 0.0, 1.0]                   # Row 3: [0, 0, 1]

    # Camera position in world coordinates (meters)
    # Format: [X, Y, Z] where Z is height above ground
    w_pos: [0.0, 0.0, 5.0]

  # ========================================================================
  # Calibration Result (optional, added after calibration completes)
  # ========================================================================
  # This section is populated by save_multi_frame_calibration_result()
  # after running the calibration optimization.
  calibration_result:
    # Optimized parameter deltas (shared across all frames)
    optimized_params:
      delta_pan_deg: 0.523                # Pan angle correction (degrees)
      delta_tilt_deg: -0.412              # Tilt angle correction (degrees)
      delta_roll_deg: 0.089               # Roll angle correction (degrees)
      delta_X_m: 0.234                    # X position correction (meters)
      delta_Y_m: -0.156                   # Y position correction (meters)
      delta_Z_m: 0.078                    # Z position correction (meters)

    # Calibration diagnostics and quality metrics
    diagnostics:
      initial_error_px: 12.4              # RMS error before optimization (pixels)
      final_error_px: 3.2                 # RMS error after optimization (pixels)
      num_inliers: 27                     # Number of inlier observations
      num_outliers: 3                     # Number of outlier observations
      inlier_ratio: 0.900                 # Ratio of inliers to total observations
      converged: true                     # Whether optimization converged
      iterations: 45                      # Number of optimization iterations

      # Per-frame residual errors
      per_frame_residuals:
        - frame_id: "frame_001"
          rms_error_px: 2.8               # RMS error for this frame (pixels)
          num_inliers: 15                 # Number of inliers in this frame
          num_outliers: 1                 # Number of outliers in this frame
        - frame_id: "frame_002"
          rms_error_px: 3.5
          num_inliers: 12
          num_outliers: 2
"""


if __name__ == '__main__':
    # Example usage
    print("Multi-Frame I/O Module")
    print("=" * 70)
    print()
    print("Example YAML configuration:")
    print()
    print(create_example_multi_frame_config())
