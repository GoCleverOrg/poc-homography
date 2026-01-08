#!/usr/bin/env python3
"""
Multi-frame session management for PTZ calibration workflow.

This module provides session management for capturing and organizing multiple frames
at different PTZ positions for multi-frame calibration. The MultiFrameCaptureSession
class manages the workflow of:
1. Capturing frames from camera at different PTZ positions
2. Adding and managing GCPs across multiple frames
3. Persisting session state to disk
4. Exporting data for MultiFrameCalibrator

Key Features:
    - Frame capture with automatic PTZ status retrieval
    - GCP observation management across multiple frames
    - Session persistence (save/load)
    - Validation checks before calibration
    - Frame image storage on disk to avoid memory issues

Usage Example:
    >>> from poc_homography.multi_frame_session import MultiFrameCaptureSession
    >>> from poc_homography.camera_geometry import CameraGeometry
    >>> from poc_homography.multi_frame_calibrator import MultiFrameCalibrator
    >>>
    >>> # Initialize session
    >>> session = MultiFrameCaptureSession(
    ...     camera_name="Valte",
    ...     output_dir="data/calibration_session_001"
    ... )
    >>>
    >>> # Capture frames at different PTZ positions
    >>> frame1 = session.capture_frame()
    >>> print(f"Captured {frame1.frame_id} at pan={frame1.ptz_position.pan}")
    >>>
    >>> # Add GCPs with observations across frames
    >>> gcp1 = session.add_gcp("building_corner_nw", gps_lat=39.640583, gps_lon=-0.230194)
    >>> session.add_gcp_observation("building_corner_nw", frame1.frame_id, u=1250.5, v=680.0)
    >>>
    >>> # Validate session is ready for calibration
    >>> validation = session.validate_session()
    >>> if validation['is_valid']:
    ...     # Export for calibration
    ...     calib_data = session.export_calibration_data(
    ...         camera_config={'K': K, 'w_pos': [0, 0, 5.0]}
    ...     )
    ...
    ...     # Run calibration
    ...     geo = CameraGeometry(1920, 1080)
    ...     calibrator = MultiFrameCalibrator(geo, calib_data)
    ...     result = calibrator.calibrate()
    >>>
    >>> # Save session for later
    >>> session.save_session("session.yaml")
    >>>
    >>> # Load existing session
    >>> loaded_session = MultiFrameCaptureSession.load_session("session.yaml")

Design Principles:
    - Frame images stored on disk to avoid memory issues with 5-30 frames
    - Immutable frame IDs and GCP IDs for consistency
    - Validation before calibration to catch issues early
    - Clean separation between session management and calibration logic
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import logging
import uuid

from poc_homography.multi_frame_calibrator import (
    PTZPosition,
    FrameObservation,
    MultiFrameGCP,
    MultiFrameCalibrationData,
)
from poc_homography.multi_frame_io import (
    save_multi_frame_calibration_data,
    load_multi_frame_calibration_data,
)

logger = logging.getLogger(__name__)


class MultiFrameCaptureSession:
    """
    Session manager for multi-frame PTZ calibration capture workflow.

    This class manages the complete workflow of capturing multiple frames at
    different PTZ positions and marking GCPs across those frames. It handles:
    - Frame capture from camera with PTZ status
    - Frame image persistence to disk
    - GCP creation and observation management
    - Session persistence (save/load)
    - Validation before calibration
    - Export to MultiFrameCalibrationData format

    The session stores frame images on disk rather than in memory to support
    capturing 5-30 frames without excessive memory usage.

    Attributes:
        camera_name: Name of the camera (e.g., "Valte", "Setram")
        output_dir: Directory for storing session data and frame images
        frames: List of captured FrameObservation instances
        gcps: Dictionary mapping gcp_id to MultiFrameGCP instance
        camera_config: Camera configuration dictionary (optional)
        _frame_images: Internal cache mapping frame_id to image path
        _session_metadata: Metadata about the session (creation time, etc.)
    """

    def __init__(
        self,
        camera_name: str,
        output_dir: str,
        camera_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-frame capture session.

        Args:
            camera_name: Name of the camera (must match camera_config.py)
            output_dir: Directory for storing session data and frame images
            camera_config: Optional camera configuration dictionary with keys:
                - 'K': Camera intrinsic matrix (3x3 numpy array)
                - 'w_pos': Camera world position [X, Y, Z] (meters)
                - 'reference_lat': Reference latitude (camera position)
                - 'reference_lon': Reference longitude (camera position)
                - 'utm_crs': UTM coordinate reference system (e.g., "EPSG:25830")

        Example:
            >>> session = MultiFrameCaptureSession(
            ...     camera_name="Valte",
            ...     output_dir="data/calibration_session_001",
            ...     camera_config={
            ...         'K': np.array([[2500, 0, 960], [0, 2500, 540], [0, 0, 1]]),
            ...         'w_pos': [0.0, 0.0, 5.0],
            ...         'reference_lat': 39.641000,
            ...         'reference_lon': -0.230500
            ...     }
            ... )
        """
        self.camera_name = camera_name
        self.output_dir = Path(output_dir)
        self.frames: List[FrameObservation] = []
        self.gcps: Dict[str, MultiFrameGCP] = {}
        self.camera_config: Dict[str, Any] = camera_config or {}
        self._frame_images: Dict[str, Path] = {}  # frame_id -> image path
        self._session_metadata: Dict[str, Any] = {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'camera_name': camera_name,
            'version': '1.0'
        }

        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._frames_dir = self.output_dir / "frames"
        self._frames_dir.mkdir(exist_ok=True)

        logger.info(
            f"Initialized MultiFrameCaptureSession for camera '{camera_name}' "
            f"in directory: {self.output_dir}"
        )

    # =========================================================================
    # Frame Capture Methods
    # =========================================================================

    def capture_frame(self, wait_time: float = 2.0) -> FrameObservation:
        """
        Capture a frame from the camera with current PTZ status.

        This method connects to the camera, captures a frame, retrieves the current
        PTZ position from the camera API, and stores both the frame image and
        observation metadata.

        Args:
            wait_time: Seconds to wait for camera stabilization before capture

        Returns:
            FrameObservation instance with captured frame metadata

        Raises:
            RuntimeError: If camera connection fails or capture fails
            ImportError: If required camera modules are not available

        Example:
            >>> frame = session.capture_frame()
            >>> print(f"Captured {frame.frame_id} at PTZ: pan={frame.ptz_position.pan}, "
            ...       f"tilt={frame.ptz_position.tilt}, zoom={frame.ptz_position.zoom}")
        """
        try:
            from tools.capture_gcps_web import grab_frame_from_camera
        except ImportError as e:
            raise ImportError(
                f"Failed to import grab_frame_from_camera: {e}. "
                "Ensure tools.capture_gcps_web is available."
            ) from e

        logger.info(f"Capturing frame from camera '{self.camera_name}'...")

        # Grab frame and PTZ status from camera
        try:
            frame_image, ptz_status = grab_frame_from_camera(self.camera_name, wait_time)
        except Exception as e:
            raise RuntimeError(f"Failed to capture frame from camera: {e}") from e

        # Extract PTZ position
        ptz_position = PTZPosition(
            pan=float(ptz_status['pan']),
            tilt=float(ptz_status['tilt']),
            zoom=float(ptz_status['zoom'])
        )

        # Add frame to session
        frame_obs = self.add_frame_from_image(frame_image, ptz_position)

        logger.info(
            f"Captured frame {frame_obs.frame_id}: "
            f"pan={ptz_position.pan:.2f}°, tilt={ptz_position.tilt:.2f}°, zoom={ptz_position.zoom:.1f}"
        )

        return frame_obs

    def add_frame_from_image(
        self,
        image: np.ndarray,
        ptz: PTZPosition,
        frame_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> FrameObservation:
        """
        Add a frame to the session from an existing image array.

        This method is useful for adding frames from pre-captured images or
        test scenarios where live camera capture is not available.

        Args:
            image: Image array (numpy array, typically HxWx3 BGR format from OpenCV)
            ptz: PTZ position for this frame
            frame_id: Optional custom frame ID (auto-generated if None)
            timestamp: Optional custom timestamp (current time if None)

        Returns:
            FrameObservation instance

        Raises:
            ValueError: If frame_id already exists or image is invalid

        Example:
            >>> import cv2
            >>> image = cv2.imread("frame_001.jpg")
            >>> ptz = PTZPosition(pan=31.0, tilt=13.0, zoom=1.0)
            >>> frame = session.add_frame_from_image(image, ptz, frame_id="frame_001")
        """
        # Validate image
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image must be a non-empty numpy array")

        # Generate frame ID if not provided
        if frame_id is None:
            frame_id = self.generate_frame_id()

        # Check for duplicate frame_id
        if any(f.frame_id == frame_id for f in self.frames):
            raise ValueError(f"Frame with ID '{frame_id}' already exists")

        # Use provided timestamp or current time
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        elif timestamp.tzinfo is None:
            # Ensure timezone awareness
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Save frame image to disk
        image_filename = f"{frame_id}.jpg"
        image_path = self._frames_dir / image_filename
        try:
            cv2.imwrite(str(image_path), image)
        except Exception as e:
            raise IOError(f"Failed to save frame image to {image_path}: {e}") from e

        # Create FrameObservation
        frame_obs = FrameObservation(
            frame_id=frame_id,
            timestamp=timestamp,
            ptz_position=ptz,
            image_path=str(image_path)
        )

        # Add to session
        self.frames.append(frame_obs)
        self._frame_images[frame_id] = image_path

        logger.info(f"Added frame {frame_id} to session (saved to {image_path})")

        return frame_obs

    def remove_frame(self, frame_id: str) -> bool:
        """
        Remove a frame from the session.

        This removes the frame metadata and deletes any GCP observations
        referencing this frame. The frame image file is NOT deleted from disk
        to prevent accidental data loss.

        Args:
            frame_id: ID of the frame to remove

        Returns:
            True if frame was removed, False if frame_id not found

        Example:
            >>> success = session.remove_frame("frame_003")
            >>> if success:
            ...     print("Frame removed successfully")
        """
        # Find and remove frame
        original_count = len(self.frames)
        self.frames = [f for f in self.frames if f.frame_id != frame_id]

        if len(self.frames) == original_count:
            logger.warning(f"Frame {frame_id} not found in session")
            return False

        # Remove from frame images cache
        if frame_id in self._frame_images:
            del self._frame_images[frame_id]

        # Remove GCP observations for this frame
        for gcp in self.gcps.values():
            if frame_id in gcp.frame_observations:
                del gcp.frame_observations[frame_id]
                logger.debug(f"Removed observation of GCP {gcp.gcp_id} from frame {frame_id}")

        logger.info(f"Removed frame {frame_id} from session")
        return True

    def get_frame_image(self, frame_id: str) -> np.ndarray:
        """
        Load and return the image for a specific frame.

        Args:
            frame_id: ID of the frame

        Returns:
            Image array (numpy array, HxWx3 BGR format)

        Raises:
            ValueError: If frame_id not found
            IOError: If image file cannot be loaded

        Example:
            >>> image = session.get_frame_image("frame_001")
            >>> print(f"Image shape: {image.shape}")
        """
        # Find frame
        frame = next((f for f in self.frames if f.frame_id == frame_id), None)
        if frame is None:
            raise ValueError(f"Frame '{frame_id}' not found in session")

        # Load image from disk
        image_path = Path(frame.image_path)
        if not image_path.exists():
            raise IOError(f"Frame image file not found: {image_path}")

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise IOError(f"Failed to decode image: {image_path}")
        except Exception as e:
            raise IOError(f"Failed to load frame image {image_path}: {e}") from e

        return image

    # =========================================================================
    # GCP Management Methods
    # =========================================================================

    def add_gcp(
        self,
        gcp_id: str,
        gps_lat: float,
        gps_lon: float,
        utm_easting: Optional[float] = None,
        utm_northing: Optional[float] = None
    ) -> MultiFrameGCP:
        """
        Add a new Ground Control Point to the session.

        Creates a GCP with GPS coordinates (and optionally UTM coordinates)
        but no frame observations. Use add_gcp_observation() to add observations
        of this GCP in specific frames.

        Args:
            gcp_id: Unique identifier for this GCP (e.g., "building_corner_nw")
            gps_lat: GPS latitude in decimal degrees
            gps_lon: GPS longitude in decimal degrees
            utm_easting: Optional UTM easting coordinate (meters)
            utm_northing: Optional UTM northing coordinate (meters)

        Returns:
            MultiFrameGCP instance

        Raises:
            ValueError: If gcp_id already exists or coordinates are invalid

        Example:
            >>> gcp = session.add_gcp(
            ...     gcp_id="building_corner_nw",
            ...     gps_lat=39.640583,
            ...     gps_lon=-0.230194,
            ...     utm_easting=729345.67,
            ...     utm_northing=4389234.12
            ... )
        """
        # Validate gcp_id
        if gcp_id in self.gcps:
            raise ValueError(f"GCP with ID '{gcp_id}' already exists")

        # Validate GPS coordinates
        if not (-90 <= gps_lat <= 90):
            raise ValueError(f"Invalid GPS latitude: {gps_lat} (must be in range [-90, 90])")
        if not (-180 <= gps_lon <= 180):
            raise ValueError(f"Invalid GPS longitude: {gps_lon} (must be in range [-180, 180])")

        # Create GCP (initially no frame observations)
        gcp = MultiFrameGCP(
            gcp_id=gcp_id,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            frame_observations={},
            utm_easting=utm_easting,
            utm_northing=utm_northing
        )

        self.gcps[gcp_id] = gcp

        logger.info(f"Added GCP {gcp_id} at GPS ({gps_lat:.6f}, {gps_lon:.6f})")

        return gcp

    def add_gcp_observation(
        self,
        gcp_id: str,
        frame_id: str,
        u: float,
        v: float
    ) -> None:
        """
        Add an observation of a GCP in a specific frame.

        Records the pixel coordinates (u, v) where the GCP is visible in the
        specified frame. The same GCP can be observed in multiple frames.

        Args:
            gcp_id: ID of the GCP
            frame_id: ID of the frame where GCP is observed
            u: Horizontal pixel coordinate (0 = left edge)
            v: Vertical pixel coordinate (0 = top edge)

        Raises:
            ValueError: If gcp_id or frame_id not found, or if observation already exists

        Example:
            >>> session.add_gcp_observation("building_corner_nw", "frame_001", u=1250.5, v=680.0)
            >>> session.add_gcp_observation("building_corner_nw", "frame_002", u=1180.2, v=720.5)
        """
        # Validate GCP exists
        if gcp_id not in self.gcps:
            raise ValueError(
                f"GCP '{gcp_id}' not found. Use add_gcp() to create it first."
            )

        # Validate frame exists
        if not any(f.frame_id == frame_id for f in self.frames):
            raise ValueError(
                f"Frame '{frame_id}' not found. Capture or add frame first."
            )

        # Check for duplicate observation
        gcp = self.gcps[gcp_id]
        if frame_id in gcp.frame_observations:
            raise ValueError(
                f"GCP '{gcp_id}' already has an observation in frame '{frame_id}'. "
                "Remove the frame or use a different frame."
            )

        # Add observation
        gcp.frame_observations[frame_id] = {
            'u': float(u),
            'v': float(v)
        }

        logger.info(f"Added observation of GCP {gcp_id} in frame {frame_id} at ({u:.1f}, {v:.1f})")

    def remove_gcp(self, gcp_id: str) -> bool:
        """
        Remove a GCP and all its observations from the session.

        Args:
            gcp_id: ID of the GCP to remove

        Returns:
            True if GCP was removed, False if gcp_id not found

        Example:
            >>> success = session.remove_gcp("building_corner_ne")
            >>> if success:
            ...     print("GCP removed successfully")
        """
        if gcp_id not in self.gcps:
            logger.warning(f"GCP {gcp_id} not found in session")
            return False

        del self.gcps[gcp_id]
        logger.info(f"Removed GCP {gcp_id} from session")
        return True

    def get_gcp_observations_for_frame(self, frame_id: str) -> List[Dict[str, Any]]:
        """
        Get all GCP observations for a specific frame.

        Args:
            frame_id: ID of the frame

        Returns:
            List of dictionaries with structure:
            [
                {
                    'gcp_id': str,
                    'gps_lat': float,
                    'gps_lon': float,
                    'u': float,
                    'v': float
                },
                ...
            ]

        Example:
            >>> observations = session.get_gcp_observations_for_frame("frame_001")
            >>> for obs in observations:
            ...     print(f"GCP {obs['gcp_id']} at pixel ({obs['u']}, {obs['v']})")
        """
        observations = []

        for gcp in self.gcps.values():
            if frame_id in gcp.frame_observations:
                pixel_coords = gcp.frame_observations[frame_id]
                observations.append({
                    'gcp_id': gcp.gcp_id,
                    'gps_lat': gcp.gps_lat,
                    'gps_lon': gcp.gps_lon,
                    'u': pixel_coords['u'],
                    'v': pixel_coords['v']
                })

        return observations

    def get_frames_for_gcp(self, gcp_id: str) -> List[str]:
        """
        Get list of frame IDs where a GCP is visible.

        Args:
            gcp_id: ID of the GCP

        Returns:
            List of frame IDs

        Raises:
            ValueError: If gcp_id not found

        Example:
            >>> frame_ids = session.get_frames_for_gcp("building_corner_nw")
            >>> print(f"GCP visible in frames: {', '.join(frame_ids)}")
        """
        if gcp_id not in self.gcps:
            raise ValueError(f"GCP '{gcp_id}' not found in session")

        gcp = self.gcps[gcp_id]
        return list(gcp.frame_observations.keys())

    # =========================================================================
    # Session Persistence Methods
    # =========================================================================

    def save_session(self, yaml_path: Optional[str] = None) -> None:
        """
        Save session to YAML file.

        Saves all frame metadata, GCP data, and camera configuration to a YAML
        file using the multi_frame_io module. Frame images are already stored
        on disk and referenced by path in the YAML file.

        Args:
            yaml_path: Path for YAML file (defaults to output_dir/session.yaml)

        Raises:
            IOError: If file cannot be written

        Example:
            >>> session.save_session("calibration_session_001.yaml")
        """
        if yaml_path is None:
            yaml_path = str(self.output_dir / "session.yaml")

        # Build MultiFrameCalibrationData for serialization
        calib_data = MultiFrameCalibrationData(
            frames=self.frames,
            gcps=list(self.gcps.values()),
            camera_config=self.camera_config
        )

        # Save using multi_frame_io
        save_multi_frame_calibration_data(calib_data, yaml_path)

        logger.info(f"Saved session to {yaml_path}")

    @classmethod
    def load_session(cls, yaml_path: str) -> 'MultiFrameCaptureSession':
        """
        Load session from YAML file.

        Creates a new MultiFrameCaptureSession instance populated with data
        from a saved session file.

        Args:
            yaml_path: Path to YAML session file

        Returns:
            MultiFrameCaptureSession instance

        Raises:
            FileNotFoundError: If YAML file does not exist
            ValueError: If YAML structure is invalid

        Example:
            >>> session = MultiFrameCaptureSession.load_session("session.yaml")
            >>> print(f"Loaded session with {len(session.frames)} frames")
        """
        # Load using multi_frame_io
        calib_data = load_multi_frame_calibration_data(yaml_path)

        # Determine output_dir from YAML path
        yaml_path_obj = Path(yaml_path)
        output_dir = yaml_path_obj.parent

        # Extract camera name from first frame or camera_config
        camera_name = "unknown"
        if calib_data.camera_config and 'camera_name' in calib_data.camera_config:
            camera_name = calib_data.camera_config['camera_name']
        elif calib_data.frames:
            # Try to infer from image paths or metadata
            # For now, use a default
            camera_name = "loaded_session"

        # Create session instance
        session = cls(
            camera_name=camera_name,
            output_dir=str(output_dir),
            camera_config=calib_data.camera_config
        )

        # Populate frames
        session.frames = calib_data.frames

        # Populate GCPs
        session.gcps = {gcp.gcp_id: gcp for gcp in calib_data.gcps}

        # Rebuild frame images cache
        for frame in session.frames:
            image_path = Path(frame.image_path)
            session._frame_images[frame.frame_id] = image_path

        logger.info(
            f"Loaded session from {yaml_path}: "
            f"{len(session.frames)} frames, {len(session.gcps)} GCPs"
        )

        return session

    # =========================================================================
    # Export and Validation Methods
    # =========================================================================

    def export_calibration_data(
        self,
        camera_config: Optional[Dict[str, Any]] = None
    ) -> MultiFrameCalibrationData:
        """
        Export session data for MultiFrameCalibrator.

        Converts the session data into a MultiFrameCalibrationData instance
        suitable for passing to MultiFrameCalibrator.calibrate().

        Args:
            camera_config: Optional camera configuration to use (overrides session config).
                          Should include keys: 'K', 'w_pos', 'reference_lat', 'reference_lon'

        Returns:
            MultiFrameCalibrationData instance

        Raises:
            ValueError: If session is not valid (use validate_session() first)

        Example:
            >>> validation = session.validate_session()
            >>> if validation['is_valid']:
            ...     calib_data = session.export_calibration_data(
            ...         camera_config={
            ...             'K': np.array([[2500, 0, 960], [0, 2500, 540], [0, 0, 1]]),
            ...             'w_pos': [0.0, 0.0, 5.0],
            ...             'reference_lat': 39.641000,
            ...             'reference_lon': -0.230500
            ...         }
            ...     )
            ...     calibrator = MultiFrameCalibrator(geo, calib_data)
        """
        # Validate session before export
        validation = self.validate_session()
        if not validation['is_valid']:
            raise ValueError(
                f"Session is not valid for calibration: {validation['errors']}"
            )

        # Use provided camera_config or fall back to session config
        config_to_use = camera_config if camera_config is not None else self.camera_config

        return MultiFrameCalibrationData(
            frames=self.frames,
            gcps=list(self.gcps.values()),
            camera_config=config_to_use
        )

    def validate_session(self) -> Dict[str, Any]:
        """
        Validate that session is ready for calibration.

        Checks various requirements for successful multi-frame calibration:
        - Minimum number of frames (3+)
        - Minimum number of GCPs (6+)
        - Each GCP observed in at least 2 frames
        - Each frame has at least 3 GCP observations
        - Frame images exist on disk

        Returns:
            Dictionary with validation results:
            {
                'is_valid': bool,
                'errors': List[str],      # Critical issues preventing calibration
                'warnings': List[str],    # Non-critical issues
                'stats': Dict[str, Any]   # Session statistics
            }

        Example:
            >>> validation = session.validate_session()
            >>> if not validation['is_valid']:
            ...     print("Errors:")
            ...     for error in validation['errors']:
            ...         print(f"  - {error}")
            >>> if validation['warnings']:
            ...     print("Warnings:")
            ...     for warning in validation['warnings']:
            ...         print(f"  - {warning}")
        """
        errors = []
        warnings = []

        # Check minimum frame count
        MIN_FRAMES = 3
        if len(self.frames) < MIN_FRAMES:
            errors.append(
                f"Insufficient frames: {len(self.frames)} frames "
                f"(minimum {MIN_FRAMES} required)"
            )

        # Check minimum GCP count
        MIN_GCPS = 6
        if len(self.gcps) < MIN_GCPS:
            errors.append(
                f"Insufficient GCPs: {len(self.gcps)} GCPs "
                f"(minimum {MIN_GCPS} required)"
            )

        # Check GCP observations
        gcps_with_insufficient_observations = []
        for gcp in self.gcps.values():
            num_observations = len(gcp.frame_observations)
            if num_observations < 2:
                gcps_with_insufficient_observations.append(
                    f"{gcp.gcp_id} ({num_observations} frame)"
                )

        if gcps_with_insufficient_observations:
            errors.append(
                f"GCPs with insufficient observations (need 2+ frames): "
                f"{', '.join(gcps_with_insufficient_observations)}"
            )

        # Check frame coverage
        frames_with_insufficient_gcps = []
        for frame in self.frames:
            num_gcps = sum(
                1 for gcp in self.gcps.values()
                if frame.frame_id in gcp.frame_observations
            )
            if num_gcps < 3:
                frames_with_insufficient_gcps.append(
                    f"{frame.frame_id} ({num_gcps} GCPs)"
                )

        if frames_with_insufficient_gcps:
            warnings.append(
                f"Frames with few GCPs (3+ recommended): "
                f"{', '.join(frames_with_insufficient_gcps)}"
            )

        # Check frame images exist
        missing_images = []
        for frame in self.frames:
            image_path = Path(frame.image_path)
            if not image_path.exists():
                missing_images.append(f"{frame.frame_id} ({frame.image_path})")

        if missing_images:
            errors.append(
                f"Frame images not found: {', '.join(missing_images)}"
            )

        # Compute statistics
        total_observations = sum(
            len(gcp.frame_observations) for gcp in self.gcps.values()
        )
        avg_observations_per_gcp = (
            total_observations / len(self.gcps) if self.gcps else 0
        )
        avg_gcps_per_frame = (
            total_observations / len(self.frames) if self.frames else 0
        )

        stats = {
            'num_frames': len(self.frames),
            'num_gcps': len(self.gcps),
            'total_observations': total_observations,
            'avg_observations_per_gcp': avg_observations_per_gcp,
            'avg_gcps_per_frame': avg_gcps_per_frame
        }

        # Overall validity
        is_valid = len(errors) == 0

        return {
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'stats': stats
        }

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive session statistics.

        Returns detailed statistics about frames, GCPs, and coverage useful
        for monitoring capture progress and session quality.

        Returns:
            Dictionary with statistics:
            {
                'num_frames': int,
                'num_gcps': int,
                'total_observations': int,
                'avg_observations_per_gcp': float,
                'avg_gcps_per_frame': float,
                'frame_details': List[Dict],
                'gcp_details': List[Dict],
                'ptz_range': Dict[str, Tuple[float, float]]
            }

        Example:
            >>> stats = session.get_session_stats()
            >>> print(f"Session has {stats['num_frames']} frames, "
            ...       f"{stats['num_gcps']} GCPs, "
            ...       f"{stats['total_observations']} total observations")
            >>> print(f"Pan range: {stats['ptz_range']['pan']}")
        """
        # Basic counts
        total_observations = sum(
            len(gcp.frame_observations) for gcp in self.gcps.values()
        )
        avg_observations_per_gcp = (
            total_observations / len(self.gcps) if self.gcps else 0.0
        )
        avg_gcps_per_frame = (
            total_observations / len(self.frames) if self.frames else 0.0
        )

        # Frame details
        frame_details = []
        for frame in self.frames:
            num_gcps = sum(
                1 for gcp in self.gcps.values()
                if frame.frame_id in gcp.frame_observations
            )
            frame_details.append({
                'frame_id': frame.frame_id,
                'timestamp': frame.timestamp.isoformat(),
                'pan': frame.ptz_position.pan,
                'tilt': frame.ptz_position.tilt,
                'zoom': frame.ptz_position.zoom,
                'num_gcps': num_gcps
            })

        # GCP details
        gcp_details = []
        for gcp in self.gcps.values():
            gcp_details.append({
                'gcp_id': gcp.gcp_id,
                'gps_lat': gcp.gps_lat,
                'gps_lon': gcp.gps_lon,
                'num_observations': len(gcp.frame_observations),
                'frame_ids': list(gcp.frame_observations.keys())
            })

        # PTZ range
        ptz_range = {}
        if self.frames:
            pans = [f.ptz_position.pan for f in self.frames]
            tilts = [f.ptz_position.tilt for f in self.frames]
            zooms = [f.ptz_position.zoom for f in self.frames]
            ptz_range = {
                'pan': (min(pans), max(pans)),
                'tilt': (min(tilts), max(tilts)),
                'zoom': (min(zooms), max(zooms))
            }

        return {
            'num_frames': len(self.frames),
            'num_gcps': len(self.gcps),
            'total_observations': total_observations,
            'avg_observations_per_gcp': avg_observations_per_gcp,
            'avg_gcps_per_frame': avg_gcps_per_frame,
            'frame_details': frame_details,
            'gcp_details': gcp_details,
            'ptz_range': ptz_range
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def generate_frame_id() -> str:
        """
        Generate a unique frame ID.

        Uses a combination of timestamp and UUID for uniqueness.

        Returns:
            Unique frame ID string (e.g., "frame_20250108_150234_a3b2c1")

        Example:
            >>> frame_id = MultiFrameCaptureSession.generate_frame_id()
            >>> print(frame_id)
            frame_20250108_150234_a3b2c1
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_suffix = str(uuid.uuid4())[:6]
        return f"frame_{timestamp}_{unique_suffix}"

    def __repr__(self) -> str:
        """String representation of session."""
        return (
            f"MultiFrameCaptureSession(camera='{self.camera_name}', "
            f"frames={len(self.frames)}, gcps={len(self.gcps)}, "
            f"output_dir='{self.output_dir}')"
        )


if __name__ == '__main__':
    # Example usage and module test
    import sys

    print("Multi-Frame Capture Session Module")
    print("=" * 70)
    print()
    print("This module provides session management for multi-frame PTZ calibration.")
    print()
    print("Example usage:")
    print()
    print("    >>> from poc_homography.multi_frame_session import MultiFrameCaptureSession")
    print("    >>> ")
    print("    >>> # Initialize session")
    print("    >>> session = MultiFrameCaptureSession(")
    print("    ...     camera_name='Valte',")
    print("    ...     output_dir='data/calibration_session_001'")
    print("    ... )")
    print("    >>> ")
    print("    >>> # Capture frames")
    print("    >>> frame1 = session.capture_frame()")
    print("    >>> frame2 = session.capture_frame()")
    print("    >>> ")
    print("    >>> # Add GCPs")
    print("    >>> gcp1 = session.add_gcp('building_corner_nw', 39.640583, -0.230194)")
    print("    >>> session.add_gcp_observation('building_corner_nw', frame1.frame_id, 1250.5, 680.0)")
    print("    >>> session.add_gcp_observation('building_corner_nw', frame2.frame_id, 1180.2, 720.5)")
    print("    >>> ")
    print("    >>> # Validate and export")
    print("    >>> validation = session.validate_session()")
    print("    >>> if validation['is_valid']:")
    print("    ...     calib_data = session.export_calibration_data(camera_config={...})")
    print("    ...     # Use calib_data with MultiFrameCalibrator")
    print("    >>> ")
    print("    >>> # Save session")
    print("    >>> session.save_session('session.yaml')")
    print()
