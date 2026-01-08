#!/usr/bin/env python3
"""
Unit and integration tests for MultiFrameCalibrator.

Tests cover:
- PTZPosition dataclass creation and validation
- FrameObservation dataclass with timestamp handling
- MultiFrameGCP dataclass with multiple frame observations
- MultiFrameCalibrationData assembly
- MultiFrameCalibrator initialization and validation
- Parameter validation (loss function, regularization weight)
- Calibration with synthetic multi-frame data
- Residual computation across all frames
- Per-frame error tracking
- Inlier/outlier classification
- Integration tests with CameraGeometry
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock
import copy

from poc_homography.multi_frame_calibrator import (
    PTZPosition,
    FrameObservation,
    MultiFrameGCP,
    MultiFrameCalibrationData,
    MultiFrameCalibrationResult,
    MultiFrameCalibrator,
)
from poc_homography.camera_geometry import CameraGeometry


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_ptz_position():
    """Create a sample PTZPosition for testing."""
    return PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)


@pytest.fixture
def sample_frame_observation():
    """Create a sample FrameObservation for testing."""
    return FrameObservation(
        frame_id="frame_001",
        timestamp=datetime(2025, 1, 5, 10, 0, 0, tzinfo=timezone.utc),
        ptz_position=PTZPosition(pan=31.5, tilt=13.2, zoom=1.5),
        image_path="data/calibration/frame_001.jpg"
    )


@pytest.fixture
def sample_multi_frame_gcp():
    """Create a sample MultiFrameGCP for testing."""
    return MultiFrameGCP(
        gcp_id="gcp_001",
        gps_lat=39.640583,
        gps_lon=-0.230194,
        frame_observations={
            "frame_001": {"u": 1250.5, "v": 680.0},
            "frame_002": {"u": 1180.2, "v": 720.5}
        },
        utm_easting=729345.67,
        utm_northing=4389234.12
    )


@pytest.fixture
def real_camera_geometry():
    """Create a real CameraGeometry instance for testing.

    Uses low zoom (1.0) for wide FOV to ensure synthetic world points
    project within image bounds during integration tests.
    """
    geo = CameraGeometry(1920, 1080)
    # Use zoom=1.0 for wide FOV (~60°) so synthetic points project into image
    K = CameraGeometry.get_intrinsics(zoom_factor=1.0, W_px=1920, H_px=1080)
    geo.set_camera_parameters(
        K=K,
        w_pos=np.array([0.0, 0.0, 5.0]),
        pan_deg=0.0,
        tilt_deg=45.0,
        map_width=640,
        map_height=640
    )
    return geo


@pytest.fixture
def sample_frames():
    """Create sample frame observations at different PTZ positions.

    Uses zoom=1.0 to match real_camera_geometry fixture for wide FOV.
    """
    return [
        FrameObservation(
            frame_id="frame_001",
            timestamp=datetime(2025, 1, 5, 10, 0, 0, tzinfo=timezone.utc),
            ptz_position=PTZPosition(pan=0.0, tilt=45.0, zoom=1.0),
            image_path="data/calibration/frame_001.jpg"
        ),
        FrameObservation(
            frame_id="frame_002",
            timestamp=datetime(2025, 1, 5, 10, 5, 0, tzinfo=timezone.utc),
            ptz_position=PTZPosition(pan=15.0, tilt=50.0, zoom=1.0),
            image_path="data/calibration/frame_002.jpg"
        ),
        FrameObservation(
            frame_id="frame_003",
            timestamp=datetime(2025, 1, 5, 10, 10, 0, tzinfo=timezone.utc),
            ptz_position=PTZPosition(pan=-10.0, tilt=40.0, zoom=1.0),
            image_path="data/calibration/frame_003.jpg"
        )
    ]


@pytest.fixture
def sample_gcps_multi_frame():
    """Create sample GCPs with observations in multiple frames."""
    return [
        MultiFrameGCP(
            gcp_id="gcp_001",
            gps_lat=39.640583,
            gps_lon=-0.230194,
            frame_observations={
                "frame_001": {"u": 1250.5, "v": 680.0},
                "frame_002": {"u": 1180.2, "v": 720.5}
            }
        ),
        MultiFrameGCP(
            gcp_id="gcp_002",
            gps_lat=39.640612,
            gps_lon=-0.229856,
            frame_observations={
                "frame_001": {"u": 2456.2, "v": 695.5},
                "frame_003": {"u": 1800.0, "v": 550.0}
            }
        ),
        MultiFrameGCP(
            gcp_id="gcp_003",
            gps_lat=39.640300,
            gps_lon=-0.230000,
            frame_observations={
                "frame_002": {"u": 920.0, "v": 800.0},
                "frame_003": {"u": 1100.0, "v": 750.0}
            }
        )
    ]


@pytest.fixture
def sample_calibration_data(sample_frames, sample_gcps_multi_frame):
    """Create sample MultiFrameCalibrationData."""
    camera_config = {
        'reference_lat': 39.641000,
        'reference_lon': -0.230500,
        'utm_crs': 'EPSG:25830',
        'K': np.array([[2500.0, 0.0, 960.0],
                       [0.0, 2500.0, 540.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64),
        'w_pos': np.array([0.0, 0.0, 5.0], dtype=np.float64)
    }

    return MultiFrameCalibrationData(
        frames=sample_frames,
        gcps=sample_gcps_multi_frame,
        camera_config=camera_config
    )


# ============================================================================
# Test: PTZPosition Dataclass
# ============================================================================

class TestPTZPosition:
    """Tests for PTZPosition dataclass."""

    def test_creation_with_valid_values(self):
        """Test PTZPosition creation with valid values."""
        ptz = PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)

        assert ptz.pan == 31.5
        assert ptz.tilt == 13.2
        assert ptz.zoom == 1.5

    def test_creation_with_negative_angles(self):
        """Test PTZPosition can have negative pan/tilt angles."""
        ptz = PTZPosition(pan=-45.0, tilt=-10.0, zoom=2.0)

        assert ptz.pan == -45.0
        assert ptz.tilt == -10.0

    def test_creation_with_zero_values(self):
        """Test PTZPosition with zero values."""
        ptz = PTZPosition(pan=0.0, tilt=0.0, zoom=1.0)

        assert ptz.pan == 0.0
        assert ptz.tilt == 0.0
        assert ptz.zoom == 1.0

    def test_creation_with_high_zoom(self):
        """Test PTZPosition with high zoom value."""
        ptz = PTZPosition(pan=0.0, tilt=45.0, zoom=25.0)

        assert ptz.zoom == 25.0

    def test_attributes_are_accessible(self):
        """Test that all PTZPosition attributes are accessible."""
        ptz = PTZPosition(pan=10.0, tilt=20.0, zoom=5.0)

        assert hasattr(ptz, 'pan')
        assert hasattr(ptz, 'tilt')
        assert hasattr(ptz, 'zoom')


# ============================================================================
# Test: FrameObservation Dataclass
# ============================================================================

class TestFrameObservation:
    """Tests for FrameObservation dataclass."""

    def test_creation_with_all_fields(self):
        """Test FrameObservation creation with all required fields."""
        ptz = PTZPosition(pan=10.0, tilt=45.0, zoom=10.0)
        timestamp = datetime(2025, 1, 5, 10, 0, 0, tzinfo=timezone.utc)

        frame = FrameObservation(
            frame_id="frame_001",
            timestamp=timestamp,
            ptz_position=ptz,
            image_path="/path/to/image.jpg"
        )

        assert frame.frame_id == "frame_001"
        assert frame.timestamp == timestamp
        assert frame.ptz_position == ptz
        assert frame.image_path == "/path/to/image.jpg"

    def test_timestamp_with_utc_timezone(self):
        """Test that timestamp can be created with UTC timezone."""
        timestamp = datetime(2025, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
        frame = FrameObservation(
            frame_id="frame_001",
            timestamp=timestamp,
            ptz_position=PTZPosition(pan=0, tilt=45, zoom=10),
            image_path="test.jpg"
        )

        assert frame.timestamp.tzinfo == timezone.utc

    def test_timestamp_with_custom_timezone(self):
        """Test that timestamp can have custom timezone."""
        tz = timezone(timedelta(hours=2))
        timestamp = datetime(2025, 1, 5, 10, 0, 0, tzinfo=tz)
        frame = FrameObservation(
            frame_id="frame_001",
            timestamp=timestamp,
            ptz_position=PTZPosition(pan=0, tilt=45, zoom=10),
            image_path="test.jpg"
        )

        assert frame.timestamp.tzinfo == tz

    def test_ptz_position_is_stored(self):
        """Test that PTZPosition is properly stored in FrameObservation."""
        ptz = PTZPosition(pan=25.5, tilt=50.2, zoom=15.0)
        frame = FrameObservation(
            frame_id="frame_001",
            timestamp=datetime.now(timezone.utc),
            ptz_position=ptz,
            image_path="test.jpg"
        )

        assert frame.ptz_position.pan == 25.5
        assert frame.ptz_position.tilt == 50.2
        assert frame.ptz_position.zoom == 15.0

    def test_frame_id_is_string(self):
        """Test that frame_id is stored as string."""
        frame = FrameObservation(
            frame_id="frame_001",
            timestamp=datetime.now(timezone.utc),
            ptz_position=PTZPosition(pan=0, tilt=45, zoom=10),
            image_path="test.jpg"
        )

        assert isinstance(frame.frame_id, str)


# ============================================================================
# Test: MultiFrameGCP Dataclass
# ============================================================================

class TestMultiFrameGCP:
    """Tests for MultiFrameGCP dataclass."""

    def test_creation_with_required_fields(self):
        """Test MultiFrameGCP creation with required fields only."""
        gcp = MultiFrameGCP(
            gcp_id="gcp_001",
            gps_lat=39.640583,
            gps_lon=-0.230194,
            frame_observations={
                "frame_001": {"u": 1250.5, "v": 680.0}
            }
        )

        assert gcp.gcp_id == "gcp_001"
        assert gcp.gps_lat == 39.640583
        assert gcp.gps_lon == -0.230194
        assert len(gcp.frame_observations) == 1
        assert gcp.utm_easting is None
        assert gcp.utm_northing is None

    def test_creation_with_utm_coordinates(self):
        """Test MultiFrameGCP creation with UTM coordinates."""
        gcp = MultiFrameGCP(
            gcp_id="gcp_001",
            gps_lat=39.640583,
            gps_lon=-0.230194,
            frame_observations={
                "frame_001": {"u": 1250.5, "v": 680.0}
            },
            utm_easting=729345.67,
            utm_northing=4389234.12
        )

        assert gcp.utm_easting == 729345.67
        assert gcp.utm_northing == 4389234.12

    def test_multiple_frame_observations(self):
        """Test MultiFrameGCP with observations in multiple frames."""
        gcp = MultiFrameGCP(
            gcp_id="gcp_001",
            gps_lat=39.640583,
            gps_lon=-0.230194,
            frame_observations={
                "frame_001": {"u": 1250.5, "v": 680.0},
                "frame_002": {"u": 1180.2, "v": 720.5},
                "frame_003": {"u": 1350.0, "v": 650.0}
            }
        )

        assert len(gcp.frame_observations) == 3
        assert "frame_001" in gcp.frame_observations
        assert "frame_002" in gcp.frame_observations
        assert "frame_003" in gcp.frame_observations

    def test_frame_observation_format(self):
        """Test that frame observations have correct format."""
        gcp = MultiFrameGCP(
            gcp_id="gcp_001",
            gps_lat=39.640583,
            gps_lon=-0.230194,
            frame_observations={
                "frame_001": {"u": 1250.5, "v": 680.0}
            }
        )

        obs = gcp.frame_observations["frame_001"]
        assert "u" in obs
        assert "v" in obs
        assert obs["u"] == 1250.5
        assert obs["v"] == 680.0

    def test_gcp_id_is_string(self):
        """Test that gcp_id is stored as string."""
        gcp = MultiFrameGCP(
            gcp_id="gcp_001",
            gps_lat=39.640583,
            gps_lon=-0.230194,
            frame_observations={"frame_001": {"u": 100, "v": 200}}
        )

        assert isinstance(gcp.gcp_id, str)


# ============================================================================
# Test: MultiFrameCalibrationData Dataclass
# ============================================================================

class TestMultiFrameCalibrationData:
    """Tests for MultiFrameCalibrationData dataclass."""

    def test_creation_with_all_fields(self, sample_frames, sample_gcps_multi_frame):
        """Test MultiFrameCalibrationData creation with all fields."""
        camera_config = {
            'K': np.eye(3),
            'w_pos': np.array([0, 0, 5])
        }

        data = MultiFrameCalibrationData(
            frames=sample_frames,
            gcps=sample_gcps_multi_frame,
            camera_config=camera_config
        )

        assert len(data.frames) == 3
        assert len(data.gcps) == 3
        assert 'K' in data.camera_config
        assert 'w_pos' in data.camera_config

    def test_frames_list_is_stored(self, sample_frames, sample_gcps_multi_frame):
        """Test that frames list is properly stored."""
        data = MultiFrameCalibrationData(
            frames=sample_frames,
            gcps=sample_gcps_multi_frame,
            camera_config={}
        )

        assert isinstance(data.frames, list)
        assert all(isinstance(f, FrameObservation) for f in data.frames)

    def test_gcps_list_is_stored(self, sample_frames, sample_gcps_multi_frame):
        """Test that GCPs list is properly stored."""
        data = MultiFrameCalibrationData(
            frames=sample_frames,
            gcps=sample_gcps_multi_frame,
            camera_config={}
        )

        assert isinstance(data.gcps, list)
        assert all(isinstance(g, MultiFrameGCP) for g in data.gcps)

    def test_camera_config_is_dict(self, sample_frames, sample_gcps_multi_frame):
        """Test that camera_config is stored as dictionary."""
        config = {'K': np.eye(3), 'w_pos': np.zeros(3)}
        data = MultiFrameCalibrationData(
            frames=sample_frames,
            gcps=sample_gcps_multi_frame,
            camera_config=config
        )

        assert isinstance(data.camera_config, dict)


# ============================================================================
# Test: MultiFrameCalibrationResult Dataclass
# ============================================================================

class TestMultiFrameCalibrationResult:
    """Tests for MultiFrameCalibrationResult dataclass."""

    def test_creation_with_all_fields(self):
        """Test MultiFrameCalibrationResult creation with all fields."""
        result = MultiFrameCalibrationResult(
            optimized_params=np.array([1.0, 2.0, 0.5, 0.1, 0.2, -0.3]),
            initial_error=12.4,
            final_error=3.2,
            num_inliers=27,
            num_outliers=3,
            inlier_ratio=0.900,
            per_gcp_errors=[2.5, 3.1, 2.8, 3.5],
            convergence_info={'success': True},
            per_frame_errors={'frame_001': 2.8, 'frame_002': 3.5},
            per_frame_inliers={'frame_001': 15, 'frame_002': 12},
            per_frame_outliers={'frame_001': 1, 'frame_002': 2},
            total_observations=30
        )

        assert result.total_observations == 30
        assert len(result.per_frame_errors) == 2
        assert len(result.per_frame_inliers) == 2
        assert len(result.per_frame_outliers) == 2

    def test_per_frame_errors_dict(self):
        """Test that per_frame_errors is a dictionary."""
        result = MultiFrameCalibrationResult(
            optimized_params=np.zeros(6),
            initial_error=10.0,
            final_error=5.0,
            num_inliers=20,
            num_outliers=5,
            inlier_ratio=0.8,
            per_gcp_errors=[],
            convergence_info={},
            per_frame_errors={'frame_001': 4.5, 'frame_002': 5.5},
            per_frame_inliers={'frame_001': 10, 'frame_002': 10},
            per_frame_outliers={'frame_001': 2, 'frame_002': 3},
            total_observations=25
        )

        assert isinstance(result.per_frame_errors, dict)
        assert result.per_frame_errors['frame_001'] == 4.5
        assert result.per_frame_errors['frame_002'] == 5.5

    def test_inherits_from_calibration_result(self):
        """Test that MultiFrameCalibrationResult extends CalibrationResult."""
        from poc_homography.gcp_calibrator import CalibrationResult

        result = MultiFrameCalibrationResult(
            optimized_params=np.zeros(6),
            initial_error=10.0,
            final_error=5.0,
            num_inliers=20,
            num_outliers=5,
            inlier_ratio=0.8,
            per_gcp_errors=[],
            convergence_info={},
            per_frame_errors={},
            per_frame_inliers={},
            per_frame_outliers={},
            total_observations=25
        )

        # Should have all CalibrationResult attributes
        assert hasattr(result, 'optimized_params')
        assert hasattr(result, 'initial_error')
        assert hasattr(result, 'final_error')
        assert hasattr(result, 'num_inliers')
        assert hasattr(result, 'num_outliers')


# ============================================================================
# Test: MultiFrameCalibrator Initialization
# ============================================================================

class TestMultiFrameCalibratorInit:
    """Tests for MultiFrameCalibrator initialization and validation."""

    def test_valid_initialization_huber(self, real_camera_geometry, sample_calibration_data):
        """Test successful initialization with huber loss."""
        calibrator = MultiFrameCalibrator(
            camera_geometry=real_camera_geometry,
            calibration_data=sample_calibration_data,
            loss_function='huber',
            loss_scale=1.0
        )

        assert calibrator.camera_geometry == real_camera_geometry
        assert calibrator.calibration_data == sample_calibration_data
        assert calibrator.loss_function == 'huber'
        assert calibrator.loss_scale == 1.0

    def test_valid_initialization_cauchy(self, real_camera_geometry, sample_calibration_data):
        """Test successful initialization with cauchy loss."""
        calibrator = MultiFrameCalibrator(
            camera_geometry=real_camera_geometry,
            calibration_data=sample_calibration_data,
            loss_function='cauchy',
            loss_scale=2.0
        )

        assert calibrator.loss_function == 'cauchy'
        assert calibrator.loss_scale == 2.0

    def test_loss_function_case_insensitive(self, real_camera_geometry, sample_calibration_data):
        """Test that loss_function parameter is case-insensitive."""
        calibrator1 = MultiFrameCalibrator(
            real_camera_geometry,
            sample_calibration_data,
            loss_function='HUBER'
        )
        calibrator2 = MultiFrameCalibrator(
            real_camera_geometry,
            sample_calibration_data,
            loss_function='Cauchy'
        )

        assert calibrator1.loss_function == 'huber'
        assert calibrator2.loss_function == 'cauchy'

    def test_invalid_loss_function(self, real_camera_geometry, sample_calibration_data):
        """Test that invalid loss_function raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MultiFrameCalibrator(
                real_camera_geometry,
                sample_calibration_data,
                loss_function='invalid_loss'
            )

        assert "Invalid loss_function" in str(exc_info.value)
        assert "invalid_loss" in str(exc_info.value)

    def test_empty_frames_list(self, real_camera_geometry, sample_gcps_multi_frame):
        """Test that empty frames list raises ValueError."""
        data = MultiFrameCalibrationData(
            frames=[],
            gcps=sample_gcps_multi_frame,
            camera_config={}
        )

        with pytest.raises(ValueError) as exc_info:
            MultiFrameCalibrator(real_camera_geometry, data)

        assert "frames list cannot be empty" in str(exc_info.value)

    def test_empty_gcps_list(self, real_camera_geometry, sample_frames):
        """Test that empty GCPs list raises ValueError."""
        data = MultiFrameCalibrationData(
            frames=sample_frames,
            gcps=[],
            camera_config={}
        )

        with pytest.raises(ValueError) as exc_info:
            MultiFrameCalibrator(real_camera_geometry, data)

        assert "gcps list cannot be empty" in str(exc_info.value)

    def test_duplicate_frame_ids(self, real_camera_geometry, sample_gcps_multi_frame):
        """Test that duplicate frame_ids raise ValueError."""
        frames = [
            FrameObservation(
                frame_id="frame_001",
                timestamp=datetime.now(timezone.utc),
                ptz_position=PTZPosition(pan=0, tilt=45, zoom=10),
                image_path="test1.jpg"
            ),
            FrameObservation(
                frame_id="frame_001",  # Duplicate
                timestamp=datetime.now(timezone.utc),
                ptz_position=PTZPosition(pan=10, tilt=50, zoom=10),
                image_path="test2.jpg"
            )
        ]

        data = MultiFrameCalibrationData(
            frames=frames,
            gcps=sample_gcps_multi_frame,
            camera_config={}
        )

        with pytest.raises(ValueError) as exc_info:
            MultiFrameCalibrator(real_camera_geometry, data)

        assert "Duplicate frame_id" in str(exc_info.value)
        assert "frame_001" in str(exc_info.value)

    def test_gcp_references_unknown_frame_id(self, real_camera_geometry, sample_frames):
        """Test that GCP referencing unknown frame_id raises ValueError."""
        gcps = [
            MultiFrameGCP(
                gcp_id="gcp_001",
                gps_lat=39.640583,
                gps_lon=-0.230194,
                frame_observations={
                    "frame_999": {"u": 100, "v": 200}  # Unknown frame_id
                }
            )
        ]

        data = MultiFrameCalibrationData(
            frames=sample_frames,
            gcps=gcps,
            camera_config={}
        )

        with pytest.raises(ValueError) as exc_info:
            MultiFrameCalibrator(real_camera_geometry, data)

        assert "unknown frame_id" in str(exc_info.value)
        assert "frame_999" in str(exc_info.value)

    def test_gcp_with_no_frame_observations(self, real_camera_geometry, sample_frames):
        """Test that GCP with empty frame_observations raises ValueError."""
        gcps = [
            MultiFrameGCP(
                gcp_id="gcp_001",
                gps_lat=39.640583,
                gps_lon=-0.230194,
                frame_observations={}  # Empty
            )
        ]

        data = MultiFrameCalibrationData(
            frames=sample_frames,
            gcps=gcps,
            camera_config={}
        )

        with pytest.raises(ValueError) as exc_info:
            MultiFrameCalibrator(real_camera_geometry, data)

        assert "no frame observations" in str(exc_info.value)

    def test_negative_loss_scale(self, real_camera_geometry, sample_calibration_data):
        """Test that negative loss_scale raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MultiFrameCalibrator(
                real_camera_geometry,
                sample_calibration_data,
                loss_scale=-1.0
            )

        assert "loss_scale must be positive" in str(exc_info.value)

    def test_zero_loss_scale(self, real_camera_geometry, sample_calibration_data):
        """Test that zero loss_scale raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MultiFrameCalibrator(
                real_camera_geometry,
                sample_calibration_data,
                loss_scale=0.0
            )

        assert "loss_scale must be positive" in str(exc_info.value)

    def test_negative_regularization_weight(self, real_camera_geometry, sample_calibration_data):
        """Test that negative regularization_weight raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MultiFrameCalibrator(
                real_camera_geometry,
                sample_calibration_data,
                regularization_weight=-1.0
            )

        assert "regularization_weight must be >= 0.0" in str(exc_info.value)

    def test_invalid_regularization_weight_nan(self, real_camera_geometry, sample_calibration_data):
        """Test that NaN regularization_weight raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MultiFrameCalibrator(
                real_camera_geometry,
                sample_calibration_data,
                regularization_weight=np.nan
            )

        assert "regularization_weight must be >= 0.0 and finite" in str(exc_info.value)

    def test_invalid_regularization_weight_inf(self, real_camera_geometry, sample_calibration_data):
        """Test that infinite regularization_weight raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MultiFrameCalibrator(
                real_camera_geometry,
                sample_calibration_data,
                regularization_weight=np.inf
            )

        assert "regularization_weight must be >= 0.0 and finite" in str(exc_info.value)

    def test_invalid_prior_sigma_negative(self, real_camera_geometry, sample_calibration_data):
        """Test that negative prior_sigma raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MultiFrameCalibrator(
                real_camera_geometry,
                sample_calibration_data,
                prior_sigmas={'pan_deg': -1.0}
            )

        assert "prior_sigmas" in str(exc_info.value)
        assert "positive and finite" in str(exc_info.value)

    def test_invalid_prior_sigma_zero(self, real_camera_geometry, sample_calibration_data):
        """Test that zero prior_sigma raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MultiFrameCalibrator(
                real_camera_geometry,
                sample_calibration_data,
                prior_sigmas={'tilt_deg': 0.0}
            )

        assert "prior_sigmas" in str(exc_info.value)
        assert "positive and finite" in str(exc_info.value)

    def test_custom_prior_sigmas(self, real_camera_geometry, sample_calibration_data):
        """Test initialization with custom prior_sigmas."""
        custom_sigmas = {
            'pan_deg': 5.0,
            'tilt_deg': 5.0,
            'roll_deg': 2.0
        }

        calibrator = MultiFrameCalibrator(
            real_camera_geometry,
            sample_calibration_data,
            prior_sigmas=custom_sigmas,
            regularization_weight=2.0
        )

        # Should merge with defaults
        assert calibrator._prior_sigmas['pan_deg'] == 5.0
        assert calibrator._prior_sigmas['tilt_deg'] == 5.0
        assert calibrator._prior_sigmas['roll_deg'] == 2.0
        # Should keep defaults for unspecified keys
        assert 'gps_position_m' in calibrator._prior_sigmas

    def test_reference_coordinates_from_params(self, real_camera_geometry, sample_calibration_data):
        """Test that reference coordinates can be set via parameters."""
        calibrator = MultiFrameCalibrator(
            real_camera_geometry,
            sample_calibration_data,
            reference_lat=39.641000,
            reference_lon=-0.230500
        )

        assert calibrator._reference_lat == 39.641000
        assert calibrator._reference_lon == -0.230500

    def test_reference_coordinates_from_config(self, real_camera_geometry, sample_frames, sample_gcps_multi_frame):
        """Test that reference coordinates can be read from config."""
        config = {
            'reference_lat': 39.641000,
            'reference_lon': -0.230500
        }

        data = MultiFrameCalibrationData(
            frames=sample_frames,
            gcps=sample_gcps_multi_frame,
            camera_config=config
        )

        calibrator = MultiFrameCalibrator(real_camera_geometry, data)

        assert calibrator._reference_lat == 39.641000
        assert calibrator._reference_lon == -0.230500

    def test_reference_coordinates_fallback_to_centroid(self, real_camera_geometry, sample_frames, sample_gcps_multi_frame):
        """Test that reference coordinates fall back to GCP centroid."""
        data = MultiFrameCalibrationData(
            frames=sample_frames,
            gcps=sample_gcps_multi_frame,
            camera_config={}
        )

        calibrator = MultiFrameCalibrator(real_camera_geometry, data)

        # Should compute centroid of GCP coordinates
        expected_lat = np.mean([gcp.gps_lat for gcp in sample_gcps_multi_frame])
        expected_lon = np.mean([gcp.gps_lon for gcp in sample_gcps_multi_frame])

        assert abs(calibrator._reference_lat - expected_lat) < 1e-6
        assert abs(calibrator._reference_lon - expected_lon) < 1e-6


# ============================================================================
# Test: Residual Computation
# ============================================================================

class TestComputeResiduals:
    """Tests for _compute_residuals method."""

    def test_residual_count_matches_observations(self, real_camera_geometry, sample_calibration_data):
        """Test that residual count matches total observations."""
        calibrator = MultiFrameCalibrator(real_camera_geometry, sample_calibration_data)

        params = np.zeros(6)
        residuals = calibrator._compute_residuals(params)

        # Each observation produces 2 residuals (u, v)
        expected_count = sum(len(gcp.frame_observations) for gcp in sample_calibration_data.gcps) * 2
        assert len(residuals) == expected_count

    def test_residuals_are_finite(self, real_camera_geometry, sample_calibration_data):
        """Test that residuals are finite for reasonable parameters."""
        calibrator = MultiFrameCalibrator(real_camera_geometry, sample_calibration_data)

        params = np.array([1.0, -1.0, 0.5, 0.5, -0.5, 0.2])
        residuals = calibrator._compute_residuals(params)

        # Filter out INFINITY_RESIDUAL values (they might appear for bad parameters)
        finite_residuals = residuals[residuals < calibrator.INFINITY_RESIDUAL]
        assert len(finite_residuals) > 0
        assert np.all(np.isfinite(finite_residuals))

    def test_zero_params_computes_residuals(self, real_camera_geometry, sample_calibration_data):
        """Test that zero parameters produce residuals."""
        calibrator = MultiFrameCalibrator(real_camera_geometry, sample_calibration_data)

        params = np.zeros(6)
        residuals = calibrator._compute_residuals(params)

        # Should produce residuals based on initial camera geometry
        assert len(residuals) > 0
        assert isinstance(residuals, np.ndarray)


# ============================================================================
# Test: Per-Frame Error Computation
# ============================================================================

class TestComputePerFrameErrors:
    """Tests for _compute_per_frame_errors method."""

    def test_per_frame_errors_dict_has_all_frames(self, real_camera_geometry, sample_calibration_data):
        """Test that per_frame_errors dict contains all frame_ids."""
        calibrator = MultiFrameCalibrator(real_camera_geometry, sample_calibration_data)

        params = np.zeros(6)
        per_frame_errors = calibrator._compute_per_frame_errors(params)

        # Should have entry for each frame
        for frame in sample_calibration_data.frames:
            assert frame.frame_id in per_frame_errors

    def test_per_frame_errors_are_positive(self, real_camera_geometry, sample_calibration_data):
        """Test that all per-frame errors are non-negative."""
        calibrator = MultiFrameCalibrator(real_camera_geometry, sample_calibration_data)

        params = np.array([1.0, -1.0, 0.0, 0.5, -0.5, 0.2])
        per_frame_errors = calibrator._compute_per_frame_errors(params)

        for frame_id, error in per_frame_errors.items():
            assert error >= 0.0

    def test_per_frame_errors_are_floats(self, real_camera_geometry, sample_calibration_data):
        """Test that all per-frame errors are float values."""
        calibrator = MultiFrameCalibrator(real_camera_geometry, sample_calibration_data)

        params = np.zeros(6)
        per_frame_errors = calibrator._compute_per_frame_errors(params)

        for frame_id, error in per_frame_errors.items():
            assert isinstance(error, (float, np.floating))


# ============================================================================
# Test: Regularization Residuals
# ============================================================================

class TestComputeRegularizationResiduals:
    """Tests for _compute_regularization_residuals method."""

    def test_regularization_residuals_zero_weight(self, real_camera_geometry, sample_calibration_data):
        """Test that regularization residuals are zero when weight is zero."""
        calibrator = MultiFrameCalibrator(
            real_camera_geometry,
            sample_calibration_data,
            regularization_weight=0.0
        )

        params = np.array([1.0, 2.0, 0.5, 0.1, 0.2, -0.3])
        reg_residuals = calibrator._compute_regularization_residuals(params)

        assert len(reg_residuals) == 6
        np.testing.assert_array_equal(reg_residuals, np.zeros(6))

    def test_regularization_residuals_nonzero_weight(self, real_camera_geometry, sample_calibration_data):
        """Test that regularization residuals are nonzero when weight > 0."""
        calibrator = MultiFrameCalibrator(
            real_camera_geometry,
            sample_calibration_data,
            regularization_weight=1.0
        )

        params = np.array([1.0, 2.0, 0.5, 0.1, 0.2, -0.3])
        reg_residuals = calibrator._compute_regularization_residuals(params)

        assert len(reg_residuals) == 6
        # Should have nonzero values
        assert np.any(reg_residuals != 0)

    def test_regularization_residuals_scale_with_weight(self, real_camera_geometry, sample_calibration_data):
        """Test that regularization residuals scale with sqrt(weight)."""
        params = np.array([1.0, 2.0, 0.5, 0.1, 0.2, -0.3])

        calibrator1 = MultiFrameCalibrator(
            real_camera_geometry,
            sample_calibration_data,
            regularization_weight=1.0
        )
        calibrator2 = MultiFrameCalibrator(
            real_camera_geometry,
            sample_calibration_data,
            regularization_weight=4.0
        )

        reg_residuals1 = calibrator1._compute_regularization_residuals(params)
        reg_residuals2 = calibrator2._compute_regularization_residuals(params)

        # Should scale by sqrt(4) = 2
        np.testing.assert_allclose(reg_residuals2, reg_residuals1 * 2.0, rtol=1e-10)


# ============================================================================
# Integration Tests: Multi-Frame Calibration
# ============================================================================

class TestCalibrateIntegration:
    """Integration tests for full calibrate() method with multi-frame data."""

    @pytest.fixture
    def create_synthetic_multi_frame_data(self, real_camera_geometry):
        """Helper to create synthetic multi-frame GCP data."""
        def _create(num_frames=3, gcps_per_frame=5, perturbation=None):
            """
            Create synthetic multi-frame calibration data.

            Args:
                num_frames: Number of frames at different PTZ positions
                gcps_per_frame: Number of GCPs visible in each frame
                perturbation: Parameter vector [Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ] or None

            Returns:
                Tuple of (frames, gcps, camera_config, perturbation)
            """
            np.random.seed(42)

            # Create frames at different PTZ positions
            # Use zoom=1.0 for wide FOV to ensure GCPs project within image bounds
            frames = []
            for i in range(num_frames):
                pan = -20.0 + i * 15.0  # Spread PTZ positions
                tilt = 40.0 + i * 5.0
                frames.append(
                    FrameObservation(
                        frame_id=f"frame_{i:03d}",
                        timestamp=datetime(2025, 1, 5, 10, i * 5, 0, tzinfo=timezone.utc),
                        ptz_position=PTZPosition(pan=pan, tilt=tilt, zoom=1.0),
                        image_path=f"data/frame_{i:03d}.jpg"
                    )
                )

            # Generate world coordinates for GCPs
            world_coords = np.random.randn(gcps_per_frame, 2) * 5.0  # 5m std dev

            # Create GCPs with observations in multiple frames
            gcps = []
            for j in range(gcps_per_frame):
                x_world, y_world = world_coords[j]

                # Convert to GPS (simple approximation)
                ref_lat = 39.640444
                ref_lon = -0.230111
                lat = ref_lat + (y_world / 111000.0)
                lon = ref_lon + (x_world / (111000.0 * np.cos(np.radians(ref_lat))))

                # Project into each frame
                frame_observations = {}
                for frame in frames:
                    # Apply perturbation if provided
                    if perturbation is not None:
                        delta_pan, delta_tilt, delta_roll, delta_x, delta_y, delta_z = perturbation
                        temp_geo = copy.copy(real_camera_geometry)
                        temp_geo.set_camera_parameters(
                            K=real_camera_geometry.K,
                            w_pos=real_camera_geometry.w_pos + np.array([delta_x, delta_y, delta_z]),
                            pan_deg=frame.ptz_position.pan + delta_pan,
                            tilt_deg=frame.ptz_position.tilt + delta_tilt,
                            map_width=real_camera_geometry.map_width,
                            map_height=real_camera_geometry.map_height,
                            roll_deg=delta_roll
                        )
                        H = temp_geo.H
                    else:
                        # Use frame's PTZ position without perturbation
                        temp_geo = copy.copy(real_camera_geometry)
                        temp_geo.set_camera_parameters(
                            K=real_camera_geometry.K,
                            w_pos=real_camera_geometry.w_pos,
                            pan_deg=frame.ptz_position.pan,
                            tilt_deg=frame.ptz_position.tilt,
                            map_width=real_camera_geometry.map_width,
                            map_height=real_camera_geometry.map_height
                        )
                        H = temp_geo.H

                    # Project world point to image
                    world_pt = np.array([x_world, y_world, 1.0])
                    image_pt_hom = H @ world_pt
                    if abs(image_pt_hom[2]) > 1e-10:
                        u = image_pt_hom[0] / image_pt_hom[2]
                        v = image_pt_hom[1] / image_pt_hom[2]

                        # Only add observation if it's within image bounds
                        if 0 <= u < real_camera_geometry.w and 0 <= v < real_camera_geometry.h:
                            frame_observations[frame.frame_id] = {"u": u, "v": v}

                # Only create GCP if it's visible in at least one frame
                if frame_observations:
                    gcps.append(
                        MultiFrameGCP(
                            gcp_id=f"gcp_{j:03d}",
                            gps_lat=lat,
                            gps_lon=lon,
                            frame_observations=frame_observations
                        )
                    )

            camera_config = {
                'reference_lat': ref_lat,
                'reference_lon': ref_lon,
                'K': real_camera_geometry.K,
                'w_pos': real_camera_geometry.w_pos
            }

            return frames, gcps, camera_config, perturbation

        return _create

    def test_calibrate_with_multiple_frames(self, real_camera_geometry, create_synthetic_multi_frame_data):
        """Test calibration with multiple frames at different PTZ positions."""
        frames, gcps, camera_config, _ = create_synthetic_multi_frame_data(
            num_frames=3,
            gcps_per_frame=10
        )

        data = MultiFrameCalibrationData(
            frames=frames,
            gcps=gcps,
            camera_config=camera_config
        )

        calibrator = MultiFrameCalibrator(real_camera_geometry, data, loss_function='huber')
        result = calibrator.calibrate()

        # Should complete successfully
        assert isinstance(result, MultiFrameCalibrationResult)
        assert result.convergence_info['success']
        assert result.total_observations > 0

    def test_calibrate_recovers_pan_offset(self, real_camera_geometry, create_synthetic_multi_frame_data):
        """Test that calibration recovers pan angle offset across frames."""
        # Create data with +2 degree pan offset
        perturbation = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        frames, gcps, camera_config, _ = create_synthetic_multi_frame_data(
            num_frames=3,
            gcps_per_frame=10,
            perturbation=perturbation
        )

        data = MultiFrameCalibrationData(
            frames=frames,
            gcps=gcps,
            camera_config=camera_config
        )

        calibrator = MultiFrameCalibrator(real_camera_geometry, data, loss_function='huber')
        result = calibrator.calibrate()

        # Should recover approximately +2° pan
        recovered_pan = result.optimized_params[0]
        assert abs(recovered_pan - 2.0) < 1.0

    def test_calibrate_final_error_less_than_initial(self, real_camera_geometry, create_synthetic_multi_frame_data):
        """Test that calibration reduces error."""
        perturbation = np.array([1.5, -1.0, 0.0, 0.5, 0.0, 0.0])
        frames, gcps, camera_config, _ = create_synthetic_multi_frame_data(
            num_frames=3,
            gcps_per_frame=10,
            perturbation=perturbation
        )

        data = MultiFrameCalibrationData(
            frames=frames,
            gcps=gcps,
            camera_config=camera_config
        )

        calibrator = MultiFrameCalibrator(real_camera_geometry, data)
        result = calibrator.calibrate()

        # Final error should be less than initial
        assert result.final_error < result.initial_error

    def test_calibrate_per_frame_errors_populated(self, real_camera_geometry, create_synthetic_multi_frame_data):
        """Test that per-frame errors are computed for all frames."""
        frames, gcps, camera_config, _ = create_synthetic_multi_frame_data(
            num_frames=3,
            gcps_per_frame=8
        )

        data = MultiFrameCalibrationData(
            frames=frames,
            gcps=gcps,
            camera_config=camera_config
        )

        calibrator = MultiFrameCalibrator(real_camera_geometry, data)
        result = calibrator.calibrate()

        # Should have error for each frame
        assert len(result.per_frame_errors) == len(frames)
        for frame in frames:
            assert frame.frame_id in result.per_frame_errors
            assert result.per_frame_errors[frame.frame_id] >= 0

    def test_calibrate_inlier_outlier_tracking(self, real_camera_geometry, create_synthetic_multi_frame_data):
        """Test that inliers and outliers are tracked per frame."""
        frames, gcps, camera_config, _ = create_synthetic_multi_frame_data(
            num_frames=3,
            gcps_per_frame=10
        )

        data = MultiFrameCalibrationData(
            frames=frames,
            gcps=gcps,
            camera_config=camera_config
        )

        calibrator = MultiFrameCalibrator(real_camera_geometry, data)
        result = calibrator.calibrate()

        # Should have inlier/outlier counts for each frame
        assert len(result.per_frame_inliers) == len(frames)
        assert len(result.per_frame_outliers) == len(frames)

        # Counts should be non-negative integers
        for frame in frames:
            assert result.per_frame_inliers[frame.frame_id] >= 0
            assert result.per_frame_outliers[frame.frame_id] >= 0

    def test_calibrate_with_custom_bounds(self, real_camera_geometry, create_synthetic_multi_frame_data):
        """Test calibration with custom parameter bounds."""
        perturbation = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        frames, gcps, camera_config, _ = create_synthetic_multi_frame_data(
            num_frames=3,
            gcps_per_frame=10,
            perturbation=perturbation
        )

        data = MultiFrameCalibrationData(
            frames=frames,
            gcps=gcps,
            camera_config=camera_config
        )

        # Constrain pan to ±2°
        custom_bounds = {'pan': (-2.0, 2.0)}

        calibrator = MultiFrameCalibrator(real_camera_geometry, data)
        result = calibrator.calibrate(bounds=custom_bounds)

        # Pan should be at or near bound
        assert abs(result.optimized_params[0]) <= 2.1

    def test_calibrate_with_regularization(self, real_camera_geometry, create_synthetic_multi_frame_data):
        """Test calibration with regularization enabled."""
        perturbation = np.array([1.0, -0.5, 0.0, 0.2, 0.0, 0.0])
        frames, gcps, camera_config, _ = create_synthetic_multi_frame_data(
            num_frames=3,
            gcps_per_frame=10,
            perturbation=perturbation
        )

        data = MultiFrameCalibrationData(
            frames=frames,
            gcps=gcps,
            camera_config=camera_config
        )

        calibrator = MultiFrameCalibrator(
            real_camera_geometry,
            data,
            regularization_weight=2.0
        )
        result = calibrator.calibrate()

        # Should complete successfully
        assert result.convergence_info['success']
        # Should have regularization penalty
        assert hasattr(result, 'regularization_penalty')
        if result.regularization_penalty is not None:
            assert result.regularization_penalty >= 0

    def test_calibrate_timestamp_recorded(self, real_camera_geometry, create_synthetic_multi_frame_data):
        """Test that calibration result includes timestamp."""
        frames, gcps, camera_config, _ = create_synthetic_multi_frame_data(
            num_frames=2,
            gcps_per_frame=5
        )

        data = MultiFrameCalibrationData(
            frames=frames,
            gcps=gcps,
            camera_config=camera_config
        )

        before = datetime.now(timezone.utc)
        calibrator = MultiFrameCalibrator(real_camera_geometry, data)
        result = calibrator.calibrate()
        after = datetime.now(timezone.utc)

        # Timestamp should be within test execution time
        assert before <= result.timestamp <= after

    def test_calibrate_convergence_info_populated(self, real_camera_geometry, create_synthetic_multi_frame_data):
        """Test that convergence info is properly populated."""
        frames, gcps, camera_config, _ = create_synthetic_multi_frame_data(
            num_frames=2,
            gcps_per_frame=8
        )

        data = MultiFrameCalibrationData(
            frames=frames,
            gcps=gcps,
            camera_config=camera_config
        )

        calibrator = MultiFrameCalibrator(real_camera_geometry, data)
        result = calibrator.calibrate()

        # Check convergence info structure
        assert 'success' in result.convergence_info
        assert 'message' in result.convergence_info
        assert 'iterations' in result.convergence_info
        assert 'function_evals' in result.convergence_info
        assert 'optimality' in result.convergence_info

        # Should have done some iterations
        assert result.convergence_info['iterations'] > 0
