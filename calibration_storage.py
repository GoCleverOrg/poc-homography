#!/usr/bin/env python3
"""
Calibration Storage Module

Provides persistent storage and retrieval of camera calibration data.
Calibration results are stored per camera in JSON format in the calibrations/ directory.

Storage Structure:
  - calibrations/{camera_name}_calibration.json

File Format:
  JSON files containing camera matrix, distortion coefficients, image size,
  RMS error, and metadata (matching output from calibrate_camera.py)

Usage:
  # Save calibration data
  save_calibration(
      camera_name="Valte",
      camera_matrix=K,
      distortion_coeffs=dist,
      image_size=(1920, 1080),
      rms_error=0.345,
      num_images=15,
      pattern_size=(9, 6),
      square_size_mm=25.0
  )

  # Load calibration data
  calib_data = load_calibration("Valte")
  if calib_data:
      camera_matrix = calib_data['camera_matrix']
      distortion_coeffs = calib_data['distortion_coeffs']

  # Check if calibration exists
  if has_calibration("Valte"):
      print("Calibration found!")

  # List all calibrated cameras
  calibrated_cameras = list_calibrations()
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np


# Directory for storing calibration files
CALIBRATIONS_DIR = Path(__file__).parent / "calibrations"


def _ensure_calibrations_dir() -> Path:
    """
    Ensure the calibrations directory exists.

    Returns:
        Path to calibrations directory
    """
    CALIBRATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return CALIBRATIONS_DIR


def get_calibration_path(camera_name: str) -> Path:
    """
    Get the path where calibration would be stored for a camera.

    Args:
        camera_name: Name of the camera (e.g., "Valte", "Setram")

    Returns:
        Path object for the calibration file

    Examples:
        >>> path = get_calibration_path("Valte")
        >>> print(path)
        /path/to/calibrations/Valte_calibration.json
    """
    return CALIBRATIONS_DIR / f"{camera_name}_calibration.json"


def save_calibration(
    camera_name: str,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    image_size: Tuple[int, int],
    rms_error: float,
    **metadata
) -> str:
    """
    Save calibration data for a camera.

    Args:
        camera_name: Name of the camera
        camera_matrix: 3x3 camera matrix (K)
        distortion_coeffs: Distortion coefficients array [k1, k2, p1, p2, k3]
        image_size: Image dimensions as (width, height)
        rms_error: RMS reprojection error from calibration
        **metadata: Additional metadata fields such as:
            - num_images: Number of calibration images used
            - pattern_size: Checkerboard pattern size (width, height)
            - square_size_mm: Size of checkerboard squares in mm
            - calibration_date: ISO timestamp (auto-added if not provided)

    Returns:
        Path to saved calibration file as string

    Raises:
        ValueError: If camera_matrix or distortion_coeffs have invalid shapes
        OSError: If file cannot be written

    Examples:
        >>> path = save_calibration(
        ...     camera_name="Valte",
        ...     camera_matrix=np.eye(3),
        ...     distortion_coeffs=np.zeros(5),
        ...     image_size=(1920, 1080),
        ...     rms_error=0.345,
        ...     num_images=15,
        ...     pattern_size=(9, 6),
        ...     square_size_mm=25.0
        ... )
        >>> print(f"Saved to: {path}")
    """
    # Validate inputs
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3, got shape {camera_matrix.shape}")

    dist_flat = distortion_coeffs.ravel()
    if dist_flat.shape[0] not in [4, 5, 8, 12, 14]:
        raise ValueError(
            f"distortion_coeffs must have 4, 5, 8, 12, or 14 elements, got {dist_flat.shape[0]}"
        )

    # Ensure directory exists
    _ensure_calibrations_dir()

    # Build calibration data structure
    calibration_data = {
        "camera_name": camera_name,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coeffs": dist_flat.tolist(),
        "image_size": list(image_size),
        "rms_error": float(rms_error),
    }

    # Add optional metadata
    if "num_images" in metadata:
        calibration_data["num_images"] = int(metadata["num_images"])
    if "pattern_size" in metadata:
        calibration_data["pattern_size"] = list(metadata["pattern_size"])
    if "square_size_mm" in metadata:
        calibration_data["square_size_mm"] = float(metadata["square_size_mm"])

    # Add or update calibration date
    if "calibration_date" in metadata:
        calibration_data["calibration_date"] = metadata["calibration_date"]
    else:
        calibration_data["calibration_date"] = datetime.now().isoformat()

    # Add any other metadata fields
    for key, value in metadata.items():
        if key not in calibration_data:
            calibration_data[key] = value

    # Get output path and save
    output_path = get_calibration_path(camera_name)

    try:
        with open(output_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"Calibration data saved for camera '{camera_name}': {output_path}")
        return str(output_path)
    except Exception as e:
        raise OSError(f"Failed to save calibration data: {e}") from e


def load_calibration(camera_name: str) -> Optional[Dict[str, Any]]:
    """
    Load calibration data for a camera.

    Args:
        camera_name: Name of the camera

    Returns:
        Dictionary containing calibration data with numpy arrays for
        camera_matrix and distortion_coeffs, or None if not found

    Dictionary structure:
        {
            "camera_name": str,
            "camera_matrix": np.ndarray (3x3),
            "distortion_coeffs": np.ndarray (5,),
            "image_size": [width, height],
            "rms_error": float,
            "num_images": int (optional),
            "pattern_size": [width, height] (optional),
            "square_size_mm": float (optional),
            "calibration_date": str (ISO format, optional),
            ... (any other metadata)
        }

    Examples:
        >>> calib = load_calibration("Valte")
        >>> if calib:
        ...     K = calib['camera_matrix']
        ...     dist = calib['distortion_coeffs']
        ...     print(f"Loaded calibration with RMS error: {calib['rms_error']}")
        ... else:
        ...     print("Calibration not found")
    """
    calibration_path = get_calibration_path(camera_name)

    if not calibration_path.exists():
        return None

    try:
        with open(calibration_path, 'r') as f:
            data = json.load(f)

        # Convert lists back to numpy arrays
        data['camera_matrix'] = np.array(data['camera_matrix'], dtype=np.float64)
        data['distortion_coeffs'] = np.array(data['distortion_coeffs'], dtype=np.float64)

        return data
    except json.JSONDecodeError as e:
        print(f"Error: Corrupted calibration file for '{camera_name}': {e}")
        return None
    except KeyError as e:
        print(f"Error: Invalid calibration file format for '{camera_name}': missing {e}")
        return None
    except Exception as e:
        print(f"Error loading calibration for '{camera_name}': {e}")
        return None


def has_calibration(camera_name: str) -> bool:
    """
    Check if a camera has stored calibration data.

    Args:
        camera_name: Name of the camera

    Returns:
        True if calibration file exists and is readable, False otherwise

    Examples:
        >>> if has_calibration("Valte"):
        ...     print("Camera is calibrated")
        ... else:
        ...     print("Camera needs calibration")
    """
    calibration_path = get_calibration_path(camera_name)
    return calibration_path.exists() and calibration_path.is_file()


def list_calibrations() -> List[str]:
    """
    List all camera names with stored calibrations.

    Returns:
        List of camera names (sorted alphabetically)

    Examples:
        >>> cameras = list_calibrations()
        >>> print(f"Calibrated cameras: {', '.join(cameras)}")
        Calibrated cameras: Setram, Valte
    """
    if not CALIBRATIONS_DIR.exists():
        return []

    calibrations = []
    for file_path in CALIBRATIONS_DIR.glob("*_calibration.json"):
        # Extract camera name from filename (remove "_calibration.json" suffix)
        camera_name = file_path.stem.replace("_calibration", "")
        calibrations.append(camera_name)

    return sorted(calibrations)


def delete_calibration(camera_name: str) -> bool:
    """
    Delete calibration data for a camera.

    Args:
        camera_name: Name of the camera

    Returns:
        True if calibration was deleted, False if it didn't exist

    Examples:
        >>> if delete_calibration("OldCamera"):
        ...     print("Calibration deleted")
        ... else:
        ...     print("No calibration found to delete")
    """
    calibration_path = get_calibration_path(camera_name)

    if not calibration_path.exists():
        return False

    try:
        calibration_path.unlink()
        print(f"Deleted calibration for camera '{camera_name}'")
        return True
    except Exception as e:
        print(f"Error deleting calibration for '{camera_name}': {e}")
        return False


def get_calibration_info(camera_name: str) -> Optional[Dict[str, Any]]:
    """
    Get summary information about a camera's calibration without loading arrays.

    Useful for displaying calibration status without loading full numpy arrays.

    Args:
        camera_name: Name of the camera

    Returns:
        Dictionary with calibration summary info, or None if not found

    Examples:
        >>> info = get_calibration_info("Valte")
        >>> if info:
        ...     print(f"RMS Error: {info['rms_error']}")
        ...     print(f"Calibrated on: {info['calibration_date']}")
        ...     print(f"Images used: {info['num_images']}")
    """
    calibration_path = get_calibration_path(camera_name)

    if not calibration_path.exists():
        return None

    try:
        with open(calibration_path, 'r') as f:
            data = json.load(f)

        # Return only metadata (not the large arrays)
        info = {
            "camera_name": data.get("camera_name"),
            "image_size": data.get("image_size"),
            "rms_error": data.get("rms_error"),
            "num_images": data.get("num_images"),
            "pattern_size": data.get("pattern_size"),
            "square_size_mm": data.get("square_size_mm"),
            "calibration_date": data.get("calibration_date"),
        }

        return info
    except Exception as e:
        print(f"Error reading calibration info for '{camera_name}': {e}")
        return None


# Validation and CLI
if __name__ == "__main__":
    import sys

    print("Calibration Storage Module")
    print("=" * 70)
    print(f"\nCalibrations directory: {CALIBRATIONS_DIR}")
    print(f"Directory exists: {CALIBRATIONS_DIR.exists()}")

    # List all calibrations
    calibrations = list_calibrations()
    print(f"\nCalibrated cameras: {len(calibrations)}")

    if calibrations:
        print("\nCamera Calibration Details:")
        print("-" * 70)
        for camera_name in calibrations:
            info = get_calibration_info(camera_name)
            if info:
                print(f"\n{camera_name}:")
                print(f"  Image size: {info.get('image_size')}")
                print(f"  RMS error: {info.get('rms_error'):.4f}" if info.get('rms_error') else "  RMS error: N/A")
                print(f"  Images used: {info.get('num_images')}" if info.get('num_images') else "  Images used: N/A")
                print(f"  Pattern: {info.get('pattern_size')}" if info.get('pattern_size') else "  Pattern: N/A")
                print(f"  Square size: {info.get('square_size_mm')}mm" if info.get('square_size_mm') else "  Square size: N/A")
                print(f"  Date: {info.get('calibration_date')}" if info.get('calibration_date') else "  Date: N/A")

                # Load and verify full data
                calib = load_calibration(camera_name)
                if calib:
                    print(f"  Camera matrix shape: {calib['camera_matrix'].shape}")
                    print(f"  Distortion coeffs: {len(calib['distortion_coeffs'])} elements")
    else:
        print("\nNo calibrations found.")
        print("\nTo create a calibration, run:")
        print("  python calibrate_camera.py --camera <camera_name> --mode capture --images ./calibration_images")

    sys.exit(0)
