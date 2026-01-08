#!/usr/bin/env python3
"""
Unit and integration tests for MultiFrameCaptureSession class.

Tests cover:
- Initialization with camera name and output directory
- Frame management (add, remove, get)
- GCP management (add, remove, observations)
- Session validation
- Session persistence (save/load)
- Export to calibration data
- Full integration workflows
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import shutil
import cv2

from poc_homography.multi_frame_session import MultiFrameCaptureSession
from poc_homography.multi_frame_calibrator import (
    PTZPosition,
    FrameObservation,
    MultiFrameGCP,
    MultiFrameCalibrationData,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image():
    """Create a sample image for testing frame capture."""
    # Create a simple 1920x1080 BGR image
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Add some content to make it non-empty
    cv2.rectangle(image, (100, 100), (500, 500), (0, 255, 0), -1)
    cv2.circle(image, (960, 540), 50, (255, 0, 0), -1)
    return image


@pytest.fixture
def sample_ptz_positions():
    """Create sample PTZ positions for multiple frames."""
    return [
        PTZPosition(pan=31.5, tilt=13.2, zoom=1.5),
        PTZPosition(pan=45.0, tilt=15.0, zoom=1.5),
        PTZPosition(pan=60.5, tilt=18.5, zoom=2.0),
    ]


@pytest.fixture
def sample_camera_config():
    """Create a sample camera configuration."""
    return {
        'K': np.array([[2500.0, 0.0, 960.0],
                       [0.0, 2500.0, 540.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64),
        'w_pos': np.array([0.0, 0.0, 5.0], dtype=np.float64),
        'reference_lat': 39.641000,
        'reference_lon': -0.230500,
        'utm_crs': 'EPSG:25830'
    }


# ============================================================================
# Test: Initialization
# ============================================================================

class TestInitialization:
    """Tests for MultiFrameCaptureSession initialization."""

    def test_init_with_camera_name_and_output_dir(self, temp_dir):
        """Test initialization with valid camera name and output directory."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session_001")
        )

        assert session.camera_name == "TestCamera"
        assert session.output_dir == Path(temp_dir / "session_001")
        assert len(session.frames) == 0
        assert len(session.gcps) == 0
        assert isinstance(session.camera_config, dict)

    def test_init_creates_output_directory(self, temp_dir):
        """Test that initialization creates output directory structure."""
        output_path = temp_dir / "session_002"
        assert not output_path.exists()

        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(output_path)
        )

        assert output_path.exists()
        assert (output_path / "frames").exists()

    def test_init_with_camera_config(self, temp_dir, sample_camera_config):
        """Test initialization with camera configuration."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session_003"),
            camera_config=sample_camera_config
        )

        assert 'K' in session.camera_config
        assert 'w_pos' in session.camera_config
        assert 'reference_lat' in session.camera_config
        np.testing.assert_array_equal(
            session.camera_config['K'],
            sample_camera_config['K']
        )

    def test_init_metadata_created(self, temp_dir):
        """Test that session metadata is initialized."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session_004")
        )

        assert 'created_at' in session._session_metadata
        assert session._session_metadata['camera_name'] == "TestCamera"
        assert 'version' in session._session_metadata

    def test_init_with_nested_output_dir(self, temp_dir):
        """Test initialization creates nested directories."""
        nested_path = temp_dir / "level1" / "level2" / "session"

        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(nested_path)
        )

        assert nested_path.exists()
        assert (nested_path / "frames").exists()


# ============================================================================
# Test: Frame Management
# ============================================================================

class TestFrameManagement:
    """Tests for frame addition, removal, and retrieval."""

    def test_add_frame_from_image_basic(self, temp_dir, sample_image):
        """Test adding a frame from an image array."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        frame = session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")

        assert frame.frame_id == "frame_001"
        assert frame.ptz_position.pan == 31.5
        assert frame.ptz_position.tilt == 13.2
        assert frame.ptz_position.zoom == 1.5
        assert len(session.frames) == 1
        assert "frame_001" in session._frame_images

    def test_add_frame_from_image_saves_to_disk(self, temp_dir, sample_image):
        """Test that frame image is saved to disk."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        frame = session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")

        image_path = Path(frame.image_path)
        assert image_path.exists()
        assert image_path.name == "frame_001.jpg"

        # Verify image can be loaded
        loaded_image = cv2.imread(str(image_path))
        assert loaded_image is not None
        assert loaded_image.shape == sample_image.shape

    def test_add_frame_from_image_with_custom_frame_id(self, temp_dir, sample_image):
        """Test adding frame with custom frame ID."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        frame = session.add_frame_from_image(
            sample_image, ptz, frame_id="custom_frame_id_123"
        )

        assert frame.frame_id == "custom_frame_id_123"
        assert len(session.frames) == 1

    def test_add_frame_from_image_auto_generates_frame_id(self, temp_dir, sample_image):
        """Test that frame ID is auto-generated if not provided."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        frame = session.add_frame_from_image(sample_image, ptz)

        assert frame.frame_id is not None
        assert frame.frame_id.startswith("frame_")
        assert len(session.frames) == 1

    def test_add_frame_from_image_with_custom_timestamp(self, temp_dir, sample_image):
        """Test adding frame with custom timestamp."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        timestamp = datetime(2025, 1, 5, 10, 30, 0, tzinfo=timezone.utc)

        frame = session.add_frame_from_image(
            sample_image, ptz, frame_id="frame_001", timestamp=timestamp
        )

        assert frame.timestamp == timestamp

    def test_add_frame_from_image_ensures_timezone(self, temp_dir, sample_image):
        """Test that naive timestamps are given UTC timezone."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        naive_timestamp = datetime(2025, 1, 5, 10, 30, 0)  # No timezone

        frame = session.add_frame_from_image(
            sample_image, ptz, frame_id="frame_001", timestamp=naive_timestamp
        )

        assert frame.timestamp.tzinfo == timezone.utc

    def test_add_frame_from_image_duplicate_frame_id_raises_error(self, temp_dir, sample_image):
        """Test that adding frame with duplicate frame_id raises error."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")

        with pytest.raises(ValueError, match="already exists"):
            session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")

    def test_add_frame_from_image_invalid_image_raises_error(self, temp_dir):
        """Test that invalid image raises error."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        with pytest.raises(ValueError, match="non-empty numpy array"):
            session.add_frame_from_image(np.array([]), ptz, frame_id="frame_001")

    def test_add_multiple_frames(self, temp_dir, sample_image, sample_ptz_positions):
        """Test adding multiple frames to session."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )

        for i, ptz in enumerate(sample_ptz_positions):
            frame_id = f"frame_{i+1:03d}"
            session.add_frame_from_image(sample_image, ptz, frame_id=frame_id)

        assert len(session.frames) == 3
        assert session.frames[0].frame_id == "frame_001"
        assert session.frames[1].frame_id == "frame_002"
        assert session.frames[2].frame_id == "frame_003"

    def test_remove_frame_existing(self, temp_dir, sample_image):
        """Test removing an existing frame."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_002")

        result = session.remove_frame("frame_001")

        assert result is True
        assert len(session.frames) == 1
        assert session.frames[0].frame_id == "frame_002"
        assert "frame_001" not in session._frame_images

    def test_remove_frame_nonexistent(self, temp_dir):
        """Test removing a non-existent frame returns False."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )

        result = session.remove_frame("nonexistent_frame")

        assert result is False

    def test_remove_frame_removes_gcp_observations(self, temp_dir, sample_image):
        """Test that removing frame also removes GCP observations for that frame."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_002")

        session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)
        session.add_gcp_observation("gcp_001", "frame_001", u=100.0, v=200.0)
        session.add_gcp_observation("gcp_001", "frame_002", u=150.0, v=250.0)

        session.remove_frame("frame_001")

        gcp = session.gcps["gcp_001"]
        assert "frame_001" not in gcp.frame_observations
        assert "frame_002" in gcp.frame_observations

    def test_get_frame_image_existing(self, temp_dir, sample_image):
        """Test retrieving image for an existing frame."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")

        loaded_image = session.get_frame_image("frame_001")

        assert loaded_image is not None
        assert loaded_image.shape == sample_image.shape
        # Check image can be loaded successfully (don't check exact pixels due to JPEG compression)
        assert loaded_image.dtype == sample_image.dtype

    def test_get_frame_image_nonexistent_frame_raises_error(self, temp_dir):
        """Test that getting image for non-existent frame raises error."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )

        with pytest.raises(ValueError, match="not found"):
            session.get_frame_image("nonexistent_frame")

    def test_get_frame_image_missing_file_raises_error(self, temp_dir, sample_image):
        """Test that getting image with missing file raises IOError."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        frame = session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")

        # Delete the image file
        Path(frame.image_path).unlink()

        with pytest.raises(IOError, match="not found"):
            session.get_frame_image("frame_001")

    def test_generate_frame_id_uniqueness(self):
        """Test that generate_frame_id creates unique IDs."""
        frame_ids = set()
        for _ in range(100):
            frame_id = MultiFrameCaptureSession.generate_frame_id()
            assert frame_id not in frame_ids  # Should be unique
            frame_ids.add(frame_id)
            assert frame_id.startswith("frame_")

    def test_generate_frame_id_format(self):
        """Test that generate_frame_id follows expected format."""
        frame_id = MultiFrameCaptureSession.generate_frame_id()

        assert frame_id.startswith("frame_")
        # Format: frame_YYYYMMDD_HHMMSS_xxxxxx
        parts = frame_id.split("_")
        assert len(parts) == 4
        assert parts[0] == "frame"
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
        assert len(parts[3]) == 6  # UUID suffix


# ============================================================================
# Test: GCP Management
# ============================================================================

class TestGCPManagement:
    """Tests for GCP creation and observation management."""

    def test_add_gcp_with_gps_coordinates(self, temp_dir):
        """Test adding a GCP with GPS coordinates."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )

        gcp = session.add_gcp(
            gcp_id="gcp_001",
            gps_lat=39.640583,
            gps_lon=-0.230194
        )

        assert gcp.gcp_id == "gcp_001"
        assert gcp.gps_lat == 39.640583
        assert gcp.gps_lon == -0.230194
        assert gcp.utm_easting is None
        assert gcp.utm_northing is None
        assert len(gcp.frame_observations) == 0
        assert "gcp_001" in session.gcps

    def test_add_gcp_with_utm_coordinates(self, temp_dir):
        """Test adding a GCP with both GPS and UTM coordinates."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )

        gcp = session.add_gcp(
            gcp_id="gcp_002",
            gps_lat=39.640583,
            gps_lon=-0.230194,
            utm_easting=729345.67,
            utm_northing=4389234.12
        )

        assert gcp.utm_easting == 729345.67
        assert gcp.utm_northing == 4389234.12

    def test_add_gcp_duplicate_id_raises_error(self, temp_dir):
        """Test that adding GCP with duplicate ID raises error."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )

        session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)

        with pytest.raises(ValueError, match="already exists"):
            session.add_gcp("gcp_001", gps_lat=39.641000, gps_lon=-0.230500)

    def test_add_gcp_invalid_latitude_raises_error(self, temp_dir):
        """Test that invalid latitude raises error."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )

        with pytest.raises(ValueError, match="Invalid GPS latitude"):
            session.add_gcp("gcp_001", gps_lat=95.0, gps_lon=-0.230194)

        with pytest.raises(ValueError, match="Invalid GPS latitude"):
            session.add_gcp("gcp_002", gps_lat=-95.0, gps_lon=-0.230194)

    def test_add_gcp_invalid_longitude_raises_error(self, temp_dir):
        """Test that invalid longitude raises error."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )

        with pytest.raises(ValueError, match="Invalid GPS longitude"):
            session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=185.0)

        with pytest.raises(ValueError, match="Invalid GPS longitude"):
            session.add_gcp("gcp_002", gps_lat=39.640583, gps_lon=-185.0)

    def test_add_gcp_observation_basic(self, temp_dir, sample_image):
        """Test adding a GCP observation to a frame."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")
        session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)

        session.add_gcp_observation("gcp_001", "frame_001", u=1250.5, v=680.0)

        gcp = session.gcps["gcp_001"]
        assert "frame_001" in gcp.frame_observations
        assert gcp.frame_observations["frame_001"]["u"] == 1250.5
        assert gcp.frame_observations["frame_001"]["v"] == 680.0

    def test_add_gcp_observation_multiple_frames(self, temp_dir, sample_image):
        """Test adding observations of same GCP across multiple frames."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_002")
        session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)

        session.add_gcp_observation("gcp_001", "frame_001", u=1250.5, v=680.0)
        session.add_gcp_observation("gcp_001", "frame_002", u=1180.2, v=720.5)

        gcp = session.gcps["gcp_001"]
        assert len(gcp.frame_observations) == 2
        assert "frame_001" in gcp.frame_observations
        assert "frame_002" in gcp.frame_observations

    def test_add_gcp_observation_validates_frame_exists(self, temp_dir):
        """Test that adding observation validates frame exists."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)

        with pytest.raises(ValueError, match="Frame.*not found"):
            session.add_gcp_observation("gcp_001", "nonexistent_frame", u=100.0, v=200.0)

    def test_add_gcp_observation_validates_gcp_exists(self, temp_dir, sample_image):
        """Test that adding observation validates GCP exists."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")

        with pytest.raises(ValueError, match="GCP.*not found"):
            session.add_gcp_observation("nonexistent_gcp", "frame_001", u=100.0, v=200.0)

    def test_add_gcp_observation_duplicate_raises_error(self, temp_dir, sample_image):
        """Test that adding duplicate observation raises error."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")
        session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)
        session.add_gcp_observation("gcp_001", "frame_001", u=1250.5, v=680.0)

        with pytest.raises(ValueError, match="already has an observation"):
            session.add_gcp_observation("gcp_001", "frame_001", u=1300.0, v=700.0)

    def test_remove_gcp_existing(self, temp_dir, sample_image):
        """Test removing an existing GCP."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")
        session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)
        session.add_gcp("gcp_002", gps_lat=39.641000, gps_lon=-0.230500)
        session.add_gcp_observation("gcp_001", "frame_001", u=100.0, v=200.0)

        result = session.remove_gcp("gcp_001")

        assert result is True
        assert "gcp_001" not in session.gcps
        assert "gcp_002" in session.gcps

    def test_remove_gcp_nonexistent(self, temp_dir):
        """Test removing a non-existent GCP returns False."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )

        result = session.remove_gcp("nonexistent_gcp")

        assert result is False

    def test_get_gcp_observations_for_frame_basic(self, temp_dir, sample_image):
        """Test getting all GCP observations for a frame."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")

        session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)
        session.add_gcp("gcp_002", gps_lat=39.641000, gps_lon=-0.230500)
        session.add_gcp_observation("gcp_001", "frame_001", u=1250.5, v=680.0)
        session.add_gcp_observation("gcp_002", "frame_001", u=1400.0, v=700.0)

        observations = session.get_gcp_observations_for_frame("frame_001")

        assert len(observations) == 2
        assert observations[0]['gcp_id'] == "gcp_001"
        assert observations[0]['u'] == 1250.5
        assert observations[0]['v'] == 680.0
        assert observations[0]['gps_lat'] == 39.640583
        assert observations[1]['gcp_id'] == "gcp_002"

    def test_get_gcp_observations_for_frame_no_observations(self, temp_dir, sample_image):
        """Test getting observations for frame with no GCP observations."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")

        observations = session.get_gcp_observations_for_frame("frame_001")

        assert len(observations) == 0

    def test_get_frames_for_gcp_basic(self, temp_dir, sample_image):
        """Test getting list of frames where GCP is visible."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_002")
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_003")

        session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)
        session.add_gcp_observation("gcp_001", "frame_001", u=100.0, v=200.0)
        session.add_gcp_observation("gcp_001", "frame_003", u=150.0, v=250.0)

        frame_ids = session.get_frames_for_gcp("gcp_001")

        assert len(frame_ids) == 2
        assert "frame_001" in frame_ids
        assert "frame_003" in frame_ids
        assert "frame_002" not in frame_ids

    def test_get_frames_for_gcp_nonexistent_raises_error(self, temp_dir):
        """Test that getting frames for non-existent GCP raises error."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )

        with pytest.raises(ValueError, match="not found"):
            session.get_frames_for_gcp("nonexistent_gcp")


# ============================================================================
# Test: Session Validation
# ============================================================================

class TestSessionValidation:
    """Tests for session validation before calibration."""

    def test_validate_session_empty_fails(self, temp_dir):
        """Test that validation fails with empty session."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )

        validation = session.validate_session()

        assert validation['is_valid'] is False
        assert len(validation['errors']) > 0
        assert any("frames" in error.lower() for error in validation['errors'])

    def test_validate_session_insufficient_frames_fails(self, temp_dir, sample_image):
        """Test that validation fails with insufficient frames."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        # Add only 2 frames (need minimum 3)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_002")

        validation = session.validate_session()

        assert validation['is_valid'] is False
        assert any("Insufficient frames" in error for error in validation['errors'])

    def test_validate_session_insufficient_gcps_fails(self, temp_dir, sample_image):
        """Test that validation fails with insufficient GCPs."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        # Add enough frames but insufficient GCPs (need minimum 6)
        for i in range(3):
            session.add_frame_from_image(sample_image, ptz, frame_id=f"frame_{i+1:03d}")

        # Add only 3 GCPs
        for i in range(3):
            session.add_gcp(f"gcp_{i+1:03d}", gps_lat=39.64 + i * 0.001, gps_lon=-0.23)

        validation = session.validate_session()

        assert validation['is_valid'] is False
        assert any("Insufficient GCPs" in error for error in validation['errors'])

    def test_validate_session_gcp_insufficient_observations_fails(self, temp_dir, sample_image):
        """Test that validation fails when GCPs lack sufficient observations."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        # Add frames
        for i in range(3):
            session.add_frame_from_image(sample_image, ptz, frame_id=f"frame_{i+1:03d}")

        # Add GCPs but with insufficient observations (need 2+ frames per GCP)
        for i in range(6):
            session.add_gcp(f"gcp_{i+1:03d}", gps_lat=39.64 + i * 0.001, gps_lon=-0.23)
            # Only add observation to first frame (need 2+ frames)
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_001", u=100.0 + i * 10, v=200.0)

        validation = session.validate_session()

        assert validation['is_valid'] is False
        assert any("insufficient observations" in error for error in validation['errors'])

    def test_validate_session_valid_passes(self, temp_dir, sample_image):
        """Test that validation passes with valid session."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        # Add sufficient frames (3+)
        for i in range(3):
            session.add_frame_from_image(sample_image, ptz, frame_id=f"frame_{i+1:03d}")

        # Add sufficient GCPs (6+) with observations in 2+ frames
        for i in range(6):
            session.add_gcp(f"gcp_{i+1:03d}", gps_lat=39.64 + i * 0.001, gps_lon=-0.23)
            # Add observations in first two frames
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_001", u=100.0 + i * 10, v=200.0)
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_002", u=150.0 + i * 10, v=250.0)

        validation = session.validate_session()

        assert validation['is_valid'] is True
        assert len(validation['errors']) == 0

    def test_validate_session_warnings_for_low_coverage(self, temp_dir, sample_image):
        """Test that validation generates warnings for low GCP coverage per frame."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        # Add frames
        for i in range(3):
            session.add_frame_from_image(sample_image, ptz, frame_id=f"frame_{i+1:03d}")

        # Add sufficient GCPs but distribute unevenly (frame_003 has only 2 GCPs)
        for i in range(6):
            session.add_gcp(f"gcp_{i+1:03d}", gps_lat=39.64 + i * 0.001, gps_lon=-0.23)
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_001", u=100.0, v=200.0)
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_002", u=150.0, v=250.0)

        # Add only 2 observations to frame_003
        session.add_gcp_observation("gcp_001", "frame_003", u=200.0, v=300.0)
        session.add_gcp_observation("gcp_002", "frame_003", u=210.0, v=310.0)

        validation = session.validate_session()

        # Should be valid but have warnings
        assert validation['is_valid'] is True
        assert len(validation['warnings']) > 0
        assert any("few GCPs" in warning for warning in validation['warnings'])

    def test_validate_session_statistics(self, temp_dir, sample_image):
        """Test that validation includes comprehensive statistics."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        # Add 3 frames and 6 GCPs
        for i in range(3):
            session.add_frame_from_image(sample_image, ptz, frame_id=f"frame_{i+1:03d}")

        for i in range(6):
            session.add_gcp(f"gcp_{i+1:03d}", gps_lat=39.64 + i * 0.001, gps_lon=-0.23)
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_001", u=100.0, v=200.0)
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_002", u=150.0, v=250.0)

        validation = session.validate_session()

        stats = validation['stats']
        assert stats['num_frames'] == 3
        assert stats['num_gcps'] == 6
        assert stats['total_observations'] == 12
        assert stats['avg_observations_per_gcp'] == 2.0
        assert stats['avg_gcps_per_frame'] == 4.0

    def test_validate_session_missing_image_file_fails(self, temp_dir, sample_image):
        """Test that validation fails if frame image file is missing."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        frame = session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")

        # Delete the image file
        Path(frame.image_path).unlink()

        validation = session.validate_session()

        assert validation['is_valid'] is False
        assert any("not found" in error for error in validation['errors'])


# ============================================================================
# Test: Session Persistence
# ============================================================================

class TestSessionPersistence:
    """Tests for saving and loading session state."""

    def test_save_session_creates_yaml_file(self, temp_dir, sample_image):
        """Test that save_session creates a YAML file."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")
        session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)
        session.add_gcp_observation("gcp_001", "frame_001", u=100.0, v=200.0)

        yaml_path = temp_dir / "session" / "test_session.yaml"
        session.save_session(str(yaml_path))

        assert yaml_path.exists()

    def test_save_session_default_location(self, temp_dir, sample_image):
        """Test that save_session uses default location when path not provided."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")
        # Need at least one GCP for save to work
        session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)
        session.add_gcp_observation("gcp_001", "frame_001", u=100.0, v=200.0)

        session.save_session()

        default_path = temp_dir / "session" / "session.yaml"
        assert default_path.exists()

    def test_load_session_restores_state(self, temp_dir, sample_image):
        """Test that load_session restores session state."""
        # Create and save session
        original_session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz1 = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        ptz2 = PTZPosition(pan=45.0, tilt=15.0, zoom=1.5)
        original_session.add_frame_from_image(sample_image, ptz1, frame_id="frame_001")
        original_session.add_frame_from_image(sample_image, ptz2, frame_id="frame_002")
        original_session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)
        original_session.add_gcp("gcp_002", gps_lat=39.641000, gps_lon=-0.230500)
        original_session.add_gcp_observation("gcp_001", "frame_001", u=100.0, v=200.0)
        original_session.add_gcp_observation("gcp_002", "frame_002", u=150.0, v=250.0)

        yaml_path = temp_dir / "session" / "test_session.yaml"
        original_session.save_session(str(yaml_path))

        # Load session
        loaded_session = MultiFrameCaptureSession.load_session(str(yaml_path))

        assert len(loaded_session.frames) == 2
        assert len(loaded_session.gcps) == 2
        assert loaded_session.frames[0].frame_id == "frame_001"
        assert loaded_session.frames[1].frame_id == "frame_002"
        assert "gcp_001" in loaded_session.gcps
        assert "gcp_002" in loaded_session.gcps

    def test_load_session_preserves_frame_images(self, temp_dir, sample_image):
        """Test that load_session preserves frame image references."""
        # Create and save session
        original_session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        original_session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")
        # Need at least one GCP for save to work
        original_session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)
        original_session.add_gcp_observation("gcp_001", "frame_001", u=100.0, v=200.0)

        yaml_path = temp_dir / "session" / "test_session.yaml"
        original_session.save_session(str(yaml_path))

        # Load session
        loaded_session = MultiFrameCaptureSession.load_session(str(yaml_path))

        # Verify frame image can be loaded
        loaded_image = loaded_session.get_frame_image("frame_001")
        assert loaded_image is not None
        assert loaded_image.shape == sample_image.shape

    def test_round_trip_save_load_preserves_data(self, temp_dir, sample_image, sample_camera_config):
        """Test that save/load round-trip preserves all data."""
        # Create comprehensive session
        original_session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session"),
            camera_config=sample_camera_config
        )

        # Add frames
        ptz_positions = [
            PTZPosition(pan=31.5, tilt=13.2, zoom=1.5),
            PTZPosition(pan=45.0, tilt=15.0, zoom=1.5),
            PTZPosition(pan=60.5, tilt=18.5, zoom=2.0),
        ]
        for i, ptz in enumerate(ptz_positions):
            original_session.add_frame_from_image(sample_image, ptz, frame_id=f"frame_{i+1:03d}")

        # Add GCPs with observations
        for i in range(4):
            gcp_id = f"gcp_{i+1:03d}"
            original_session.add_gcp(
                gcp_id,
                gps_lat=39.64 + i * 0.001,
                gps_lon=-0.23 + i * 0.001,
                utm_easting=729345.67 + i * 10,
                utm_northing=4389234.12 + i * 10
            )
            # Add observations to multiple frames
            original_session.add_gcp_observation(gcp_id, "frame_001", u=100.0 + i * 10, v=200.0)
            original_session.add_gcp_observation(gcp_id, "frame_002", u=150.0 + i * 10, v=250.0)

        # Save session
        yaml_path = temp_dir / "session" / "test_session.yaml"
        original_session.save_session(str(yaml_path))

        # Load session
        loaded_session = MultiFrameCaptureSession.load_session(str(yaml_path))

        # Verify frames
        assert len(loaded_session.frames) == 3
        for i, frame in enumerate(loaded_session.frames):
            assert frame.frame_id == f"frame_{i+1:03d}"
            assert frame.ptz_position.pan == ptz_positions[i].pan
            assert frame.ptz_position.tilt == ptz_positions[i].tilt
            assert frame.ptz_position.zoom == ptz_positions[i].zoom

        # Verify GCPs
        assert len(loaded_session.gcps) == 4
        for i in range(4):
            gcp_id = f"gcp_{i+1:03d}"
            assert gcp_id in loaded_session.gcps
            gcp = loaded_session.gcps[gcp_id]
            assert gcp.gps_lat == pytest.approx(39.64 + i * 0.001)
            assert gcp.gps_lon == pytest.approx(-0.23 + i * 0.001)
            assert gcp.utm_easting == pytest.approx(729345.67 + i * 10)
            assert len(gcp.frame_observations) == 2

        # Verify camera config
        assert 'K' in loaded_session.camera_config
        np.testing.assert_array_almost_equal(
            loaded_session.camera_config['K'],
            sample_camera_config['K']
        )

    def test_load_session_nonexistent_file_raises_error(self, temp_dir):
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            MultiFrameCaptureSession.load_session(str(temp_dir / "nonexistent.yaml"))


# ============================================================================
# Test: Export to Calibration Data
# ============================================================================

class TestExportCalibrationData:
    """Tests for exporting session to MultiFrameCalibrationData."""

    def test_export_calibration_data_basic(self, temp_dir, sample_image, sample_camera_config):
        """Test exporting valid session to calibration data."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session"),
            camera_config=sample_camera_config
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        # Create valid session
        for i in range(3):
            session.add_frame_from_image(sample_image, ptz, frame_id=f"frame_{i+1:03d}")

        for i in range(6):
            session.add_gcp(f"gcp_{i+1:03d}", gps_lat=39.64 + i * 0.001, gps_lon=-0.23)
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_001", u=100.0, v=200.0)
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_002", u=150.0, v=250.0)

        calib_data = session.export_calibration_data()

        assert isinstance(calib_data, MultiFrameCalibrationData)
        assert len(calib_data.frames) == 3
        assert len(calib_data.gcps) == 6
        assert 'K' in calib_data.camera_config

    def test_export_calibration_data_validates_session_first(self, temp_dir, sample_image):
        """Test that export validates session before exporting."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )

        # Create invalid session (insufficient frames)
        session.add_frame_from_image(sample_image, PTZPosition(31.5, 13.2, 1.5), frame_id="frame_001")

        with pytest.raises(ValueError, match="not valid"):
            session.export_calibration_data()

    def test_export_calibration_data_with_custom_camera_config(self, temp_dir, sample_image):
        """Test exporting with custom camera config overriding session config."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session"),
            camera_config={'K': np.eye(3), 'w_pos': [0, 0, 0]}
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        # Create valid session
        for i in range(3):
            session.add_frame_from_image(sample_image, ptz, frame_id=f"frame_{i+1:03d}")

        for i in range(6):
            session.add_gcp(f"gcp_{i+1:03d}", gps_lat=39.64 + i * 0.001, gps_lon=-0.23)
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_001", u=100.0, v=200.0)
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_002", u=150.0, v=250.0)

        # Export with custom config
        custom_config = {
            'K': np.array([[3000.0, 0.0, 960.0],
                          [0.0, 3000.0, 540.0],
                          [0.0, 0.0, 1.0]]),
            'w_pos': [1.0, 2.0, 3.0]
        }
        calib_data = session.export_calibration_data(camera_config=custom_config)

        np.testing.assert_array_equal(calib_data.camera_config['K'], custom_config['K'])
        assert calib_data.camera_config['w_pos'] == custom_config['w_pos']

    def test_export_calibration_data_includes_all_frames(self, temp_dir, sample_image):
        """Test that export includes all frames."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )

        ptz_positions = [
            PTZPosition(pan=31.5, tilt=13.2, zoom=1.5),
            PTZPosition(pan=45.0, tilt=15.0, zoom=1.5),
            PTZPosition(pan=60.5, tilt=18.5, zoom=2.0),
        ]
        for i, ptz in enumerate(ptz_positions):
            session.add_frame_from_image(sample_image, ptz, frame_id=f"frame_{i+1:03d}")

        for i in range(6):
            session.add_gcp(f"gcp_{i+1:03d}", gps_lat=39.64 + i * 0.001, gps_lon=-0.23)
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_001", u=100.0, v=200.0)
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_002", u=150.0, v=250.0)

        calib_data = session.export_calibration_data()

        assert len(calib_data.frames) == 3
        for i, frame in enumerate(calib_data.frames):
            assert frame.frame_id == f"frame_{i+1:03d}"
            assert frame.ptz_position.pan == ptz_positions[i].pan

    def test_export_calibration_data_includes_all_gcps(self, temp_dir, sample_image):
        """Test that export includes all GCPs with observations."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        for i in range(3):
            session.add_frame_from_image(sample_image, ptz, frame_id=f"frame_{i+1:03d}")

        for i in range(8):
            session.add_gcp(f"gcp_{i+1:03d}", gps_lat=39.64 + i * 0.001, gps_lon=-0.23)
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_001", u=100.0, v=200.0)
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_002", u=150.0, v=250.0)

        calib_data = session.export_calibration_data()

        assert len(calib_data.gcps) == 8
        for i, gcp in enumerate(calib_data.gcps):
            assert gcp.gcp_id == f"gcp_{i+1:03d}"
            assert len(gcp.frame_observations) >= 2


# ============================================================================
# Test: Session Statistics
# ============================================================================

class TestSessionStatistics:
    """Tests for session statistics and reporting."""

    def test_get_session_stats_basic(self, temp_dir, sample_image):
        """Test getting basic session statistics."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")
        session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)
        session.add_gcp_observation("gcp_001", "frame_001", u=100.0, v=200.0)

        stats = session.get_session_stats()

        assert stats['num_frames'] == 1
        assert stats['num_gcps'] == 1
        assert stats['total_observations'] == 1
        assert stats['avg_observations_per_gcp'] == 1.0
        assert stats['avg_gcps_per_frame'] == 1.0

    def test_get_session_stats_includes_frame_details(self, temp_dir, sample_image):
        """Test that statistics include detailed frame information."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz_positions = [
            PTZPosition(pan=31.5, tilt=13.2, zoom=1.5),
            PTZPosition(pan=45.0, tilt=15.0, zoom=2.0),
        ]
        for i, ptz in enumerate(ptz_positions):
            session.add_frame_from_image(sample_image, ptz, frame_id=f"frame_{i+1:03d}")

        stats = session.get_session_stats()

        assert len(stats['frame_details']) == 2
        assert stats['frame_details'][0]['frame_id'] == "frame_001"
        assert stats['frame_details'][0]['pan'] == 31.5
        assert stats['frame_details'][0]['tilt'] == 13.2
        assert stats['frame_details'][1]['zoom'] == 2.0

    def test_get_session_stats_includes_gcp_details(self, temp_dir, sample_image):
        """Test that statistics include detailed GCP information."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_001")
        session.add_frame_from_image(sample_image, ptz, frame_id="frame_002")

        session.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)
        session.add_gcp_observation("gcp_001", "frame_001", u=100.0, v=200.0)
        session.add_gcp_observation("gcp_001", "frame_002", u=150.0, v=250.0)

        stats = session.get_session_stats()

        assert len(stats['gcp_details']) == 1
        assert stats['gcp_details'][0]['gcp_id'] == "gcp_001"
        assert stats['gcp_details'][0]['gps_lat'] == 39.640583
        assert stats['gcp_details'][0]['num_observations'] == 2
        assert len(stats['gcp_details'][0]['frame_ids']) == 2

    def test_get_session_stats_includes_ptz_range(self, temp_dir, sample_image):
        """Test that statistics include PTZ range information."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )
        ptz_positions = [
            PTZPosition(pan=31.5, tilt=13.2, zoom=1.5),
            PTZPosition(pan=45.0, tilt=15.0, zoom=1.5),
            PTZPosition(pan=60.5, tilt=18.5, zoom=2.5),
        ]
        for i, ptz in enumerate(ptz_positions):
            session.add_frame_from_image(sample_image, ptz, frame_id=f"frame_{i+1:03d}")

        stats = session.get_session_stats()

        assert 'ptz_range' in stats
        assert stats['ptz_range']['pan'] == (31.5, 60.5)
        assert stats['ptz_range']['tilt'] == (13.2, 18.5)
        assert stats['ptz_range']['zoom'] == (1.5, 2.5)


# ============================================================================
# Test: Integration Workflows
# ============================================================================

class TestIntegrationWorkflows:
    """Integration tests for complete workflows."""

    def test_full_workflow_init_to_export(self, temp_dir, sample_image, sample_camera_config):
        """Test full workflow from initialization to calibration export."""
        # 1. Initialize session
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session"),
            camera_config=sample_camera_config
        )

        # 2. Add frames at different PTZ positions
        ptz_positions = [
            PTZPosition(pan=31.5, tilt=13.2, zoom=1.5),
            PTZPosition(pan=45.0, tilt=15.0, zoom=1.5),
            PTZPosition(pan=60.5, tilt=18.5, zoom=2.0),
        ]
        for i, ptz in enumerate(ptz_positions):
            session.add_frame_from_image(sample_image, ptz, frame_id=f"frame_{i+1:03d}")

        # 3. Add GCPs
        for i in range(6):
            session.add_gcp(
                f"gcp_{i+1:03d}",
                gps_lat=39.64 + i * 0.001,
                gps_lon=-0.23 + i * 0.001
            )

        # 4. Mark GCP observations in frames
        for i in range(6):
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_001", u=100.0 + i * 10, v=200.0)
            session.add_gcp_observation(f"gcp_{i+1:03d}", "frame_002", u=150.0 + i * 10, v=250.0)

        # 5. Validate session
        validation = session.validate_session()
        assert validation['is_valid'] is True

        # 6. Export for calibration
        calib_data = session.export_calibration_data()
        assert isinstance(calib_data, MultiFrameCalibrationData)
        assert len(calib_data.frames) == 3
        assert len(calib_data.gcps) == 6

        # 7. Save session
        yaml_path = temp_dir / "session" / "final_session.yaml"
        session.save_session(str(yaml_path))
        assert yaml_path.exists()

    def test_workflow_save_and_resume(self, temp_dir, sample_image, sample_camera_config):
        """Test workflow of saving session and resuming work later."""
        # Create initial session
        session1 = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session"),
            camera_config=sample_camera_config
        )

        # Add some initial frames and GCPs
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)
        session1.add_frame_from_image(sample_image, ptz, frame_id="frame_001")
        session1.add_gcp("gcp_001", gps_lat=39.640583, gps_lon=-0.230194)
        session1.add_gcp_observation("gcp_001", "frame_001", u=100.0, v=200.0)

        # Save session
        yaml_path = temp_dir / "session" / "work_in_progress.yaml"
        session1.save_session(str(yaml_path))

        # Load session and continue work
        session2 = MultiFrameCaptureSession.load_session(str(yaml_path))

        # Add more frames and observations
        session2.add_frame_from_image(sample_image, ptz, frame_id="frame_002")
        session2.add_gcp_observation("gcp_001", "frame_002", u=150.0, v=250.0)

        # Verify continued work
        assert len(session2.frames) == 2
        gcp = session2.gcps["gcp_001"]
        assert len(gcp.frame_observations) == 2

    def test_workflow_multiple_gcps_across_frames(self, temp_dir, sample_image):
        """Test realistic workflow with multiple GCPs visible across different frames."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )

        # Add 5 frames at different PTZ positions
        ptz_positions = [
            PTZPosition(pan=20.0, tilt=10.0, zoom=1.0),
            PTZPosition(pan=30.0, tilt=12.0, zoom=1.0),
            PTZPosition(pan=40.0, tilt=14.0, zoom=1.5),
            PTZPosition(pan=50.0, tilt=16.0, zoom=1.5),
            PTZPosition(pan=60.0, tilt=18.0, zoom=2.0),
        ]
        for i, ptz in enumerate(ptz_positions):
            session.add_frame_from_image(sample_image, ptz, frame_id=f"frame_{i+1:03d}")

        # Add 10 GCPs with varying visibility across frames
        for i in range(10):
            session.add_gcp(f"gcp_{i+1:03d}", gps_lat=39.64 + i * 0.001, gps_lon=-0.23)

        # Simulate GCP visibility patterns:
        # - Some GCPs visible in all frames
        # - Some visible in subset of frames
        visibility_patterns = [
            ["frame_001", "frame_002", "frame_003", "frame_004", "frame_005"],  # gcp_001
            ["frame_001", "frame_002", "frame_003"],  # gcp_002
            ["frame_003", "frame_004", "frame_005"],  # gcp_003
            ["frame_001", "frame_005"],  # gcp_004
            ["frame_002", "frame_003", "frame_004"],  # gcp_005
            ["frame_001", "frame_002"],  # gcp_006
            ["frame_004", "frame_005"],  # gcp_007
            ["frame_001", "frame_003", "frame_005"],  # gcp_008
            ["frame_002", "frame_004"],  # gcp_009
            ["frame_001", "frame_002", "frame_003", "frame_004"],  # gcp_010
        ]

        for i, frame_ids in enumerate(visibility_patterns):
            for frame_id in frame_ids:
                u = 100.0 + i * 50 + float(int(frame_id[-3:]) * 10)
                v = 200.0 + i * 30
                session.add_gcp_observation(f"gcp_{i+1:03d}", frame_id, u=u, v=v)

        # Validate session
        validation = session.validate_session()
        assert validation['is_valid'] is True
        assert validation['stats']['total_observations'] == sum(len(p) for p in visibility_patterns)

        # Export and verify
        calib_data = session.export_calibration_data()
        assert len(calib_data.gcps) == 10
        assert len(calib_data.frames) == 5

    def test_session_repr(self, temp_dir):
        """Test string representation of session."""
        session = MultiFrameCaptureSession(
            camera_name="TestCamera",
            output_dir=str(temp_dir / "session")
        )

        repr_str = repr(session)

        assert "MultiFrameCaptureSession" in repr_str
        assert "TestCamera" in repr_str
        assert "frames=0" in repr_str
        assert "gcps=0" in repr_str


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
