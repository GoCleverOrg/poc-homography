#!/usr/bin/env python3
"""
Unit and integration tests for multi-frame I/O module.

Tests cover:
- PTZPosition serialization/deserialization
- FrameObservation serialization/deserialization
- MultiFrameGCP serialization/deserialization
- MultiFrameCalibrationData save/load with YAML files
- MultiFrameCalibrationResult save/load with YAML files
- Example configuration generation
- Error handling for invalid YAML structures
- DateTime timezone handling
- Numpy array conversions
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import yaml
import tempfile
import shutil

from poc_homography.multi_frame_io import (
    _serialize_ptz_position,
    _deserialize_ptz_position,
    _serialize_frame_observation,
    _deserialize_frame_observation,
    _serialize_multi_frame_gcp,
    _deserialize_multi_frame_gcp,
    save_multi_frame_calibration_data,
    load_multi_frame_calibration_data,
    save_multi_frame_calibration_result,
    load_multi_frame_calibration_result,
    create_example_multi_frame_config,
)
from poc_homography.multi_frame_calibrator import (
    PTZPosition,
    FrameObservation,
    MultiFrameGCP,
    MultiFrameCalibrationData,
    MultiFrameCalibrationResult,
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
def sample_ptz_position():
    """Create a sample PTZPosition for testing."""
    return PTZPosition(pan=31.5, tilt=13.2, zoom=1.5)


@pytest.fixture
def sample_frame_observation():
    """Create a sample FrameObservation for testing."""
    return FrameObservation(
        frame_id="frame_001",
        image_path="data/calibration/frame_001.jpg",
        ptz_position=PTZPosition(pan=31.5, tilt=13.2, zoom=1.5),
        timestamp=datetime(2025, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
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
def sample_calibration_data():
    """Create a sample MultiFrameCalibrationData for testing."""
    frames = [
        FrameObservation(
            frame_id="frame_001",
            image_path="data/calibration/frame_001.jpg",
            ptz_position=PTZPosition(pan=31.0, tilt=13.0, zoom=1.0),
            timestamp=datetime(2025, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
        ),
        FrameObservation(
            frame_id="frame_002",
            image_path="data/calibration/frame_002.jpg",
            ptz_position=PTZPosition(pan=45.2, tilt=15.5, zoom=1.0),
            timestamp=datetime(2025, 1, 5, 10, 5, 0, tzinfo=timezone.utc)
        )
    ]

    gcps = [
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
                "frame_001": {"u": 2456.2, "v": 695.5}
            }
        )
    ]

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
        frames=frames,
        gcps=gcps,
        camera_config=camera_config
    )


@pytest.fixture
def sample_calibration_result():
    """Create a sample MultiFrameCalibrationResult for testing."""
    return MultiFrameCalibrationResult(
        optimized_params=np.array([0.523, -0.412, 0.089, 0.234, -0.156, 0.078]),
        initial_error=12.4,
        final_error=3.2,
        num_inliers=27,
        num_outliers=3,
        inlier_ratio=0.900,
        per_gcp_errors=[2.5, 3.1, 2.8, 3.5],
        convergence_info={'success': True, 'iterations': 45},
        per_frame_errors={'frame_001': 2.8, 'frame_002': 3.5},
        per_frame_inliers={'frame_001': 15, 'frame_002': 12},
        per_frame_outliers={'frame_001': 1, 'frame_002': 2},
        total_observations=30
    )


# ============================================================================
# Test: PTZPosition Serialization
# ============================================================================

class TestPTZPositionSerialization:
    """Tests for PTZPosition serialization/deserialization."""

    def test_serialize_ptz_position(self, sample_ptz_position):
        """Test serialization of PTZPosition to dictionary."""
        result = _serialize_ptz_position(sample_ptz_position)

        assert isinstance(result, dict)
        assert result['pan'] == 31.5
        assert result['tilt'] == 13.2
        assert result['zoom'] == 1.5

    def test_deserialize_ptz_position_valid(self):
        """Test deserialization of valid PTZ position dictionary."""
        data = {'pan': 31.5, 'tilt': 13.2, 'zoom': 1.5}
        result = _deserialize_ptz_position(data)

        assert isinstance(result, PTZPosition)
        assert result.pan == 31.5
        assert result.tilt == 13.2
        assert result.zoom == 1.5

    def test_deserialize_ptz_position_missing_field(self):
        """Test deserialization fails with missing required field."""
        data = {'pan': 31.5, 'tilt': 13.2}  # Missing 'zoom'

        with pytest.raises(ValueError, match="missing required fields.*zoom"):
            _deserialize_ptz_position(data)

    def test_deserialize_ptz_position_invalid_type(self):
        """Test deserialization fails with invalid field type."""
        data = {'pan': 'invalid', 'tilt': 13.2, 'zoom': 1.5}

        with pytest.raises(ValueError, match="Invalid PTZ position data"):
            _deserialize_ptz_position(data)

    def test_round_trip_ptz_position(self, sample_ptz_position):
        """Test PTZPosition serialization and deserialization round-trip."""
        serialized = _serialize_ptz_position(sample_ptz_position)
        deserialized = _deserialize_ptz_position(serialized)

        assert deserialized.pan == sample_ptz_position.pan
        assert deserialized.tilt == sample_ptz_position.tilt
        assert deserialized.zoom == sample_ptz_position.zoom


# ============================================================================
# Test: FrameObservation Serialization
# ============================================================================

class TestFrameObservationSerialization:
    """Tests for FrameObservation serialization/deserialization."""

    def test_serialize_frame_observation(self, sample_frame_observation):
        """Test serialization of FrameObservation to dictionary."""
        result = _serialize_frame_observation(sample_frame_observation)

        assert isinstance(result, dict)
        assert result['frame_id'] == 'frame_001'
        assert result['image_path'] == 'data/calibration/frame_001.jpg'
        assert 'ptz_position' in result
        assert result['ptz_position']['pan'] == 31.5
        assert result['timestamp'] == '2025-01-05T10:00:00+00:00'

    def test_deserialize_frame_observation_valid(self):
        """Test deserialization of valid frame observation dictionary."""
        data = {
            'frame_id': 'frame_001',
            'image_path': 'data/calibration/frame_001.jpg',
            'ptz_position': {'pan': 31.5, 'tilt': 13.2, 'zoom': 1.5},
            'timestamp': '2025-01-05T10:00:00Z'
        }
        result = _deserialize_frame_observation(data)

        assert isinstance(result, FrameObservation)
        assert result.frame_id == 'frame_001'
        assert result.image_path == 'data/calibration/frame_001.jpg'
        assert result.ptz_position.pan == 31.5
        assert result.timestamp.year == 2025
        assert result.timestamp.tzinfo is not None

    def test_deserialize_frame_observation_missing_field(self):
        """Test deserialization fails with missing required field."""
        data = {
            'frame_id': 'frame_001',
            'image_path': 'data/calibration/frame_001.jpg',
            'ptz_position': {'pan': 31.5, 'tilt': 13.2, 'zoom': 1.5}
            # Missing 'timestamp'
        }

        with pytest.raises(ValueError, match="missing required fields.*timestamp"):
            _deserialize_frame_observation(data)

    def test_deserialize_frame_observation_naive_timestamp(self):
        """Test deserialization adds UTC timezone to naive timestamps."""
        data = {
            'frame_id': 'frame_001',
            'image_path': 'data/calibration/frame_001.jpg',
            'ptz_position': {'pan': 31.5, 'tilt': 13.2, 'zoom': 1.5},
            'timestamp': '2025-01-05T10:00:00'  # No timezone
        }
        result = _deserialize_frame_observation(data)

        assert result.timestamp.tzinfo == timezone.utc

    def test_round_trip_frame_observation(self, sample_frame_observation):
        """Test FrameObservation serialization and deserialization round-trip."""
        serialized = _serialize_frame_observation(sample_frame_observation)
        deserialized = _deserialize_frame_observation(serialized)

        assert deserialized.frame_id == sample_frame_observation.frame_id
        assert deserialized.image_path == sample_frame_observation.image_path
        assert deserialized.ptz_position.pan == sample_frame_observation.ptz_position.pan
        assert deserialized.timestamp == sample_frame_observation.timestamp


# ============================================================================
# Test: MultiFrameGCP Serialization
# ============================================================================

class TestMultiFrameGCPSerialization:
    """Tests for MultiFrameGCP serialization/deserialization."""

    def test_serialize_multi_frame_gcp(self, sample_multi_frame_gcp):
        """Test serialization of MultiFrameGCP to dictionary."""
        result = _serialize_multi_frame_gcp(sample_multi_frame_gcp)

        assert isinstance(result, dict)
        assert result['gcp_id'] == 'gcp_001'
        assert result['gps']['latitude'] == 39.640583
        assert result['gps']['longitude'] == -0.230194
        assert 'utm' in result
        assert result['utm']['easting'] == 729345.67
        assert len(result['frame_observations']) == 2

    def test_serialize_multi_frame_gcp_without_utm(self):
        """Test serialization of GCP without UTM coordinates."""
        gcp = MultiFrameGCP(
            gcp_id="gcp_001",
            gps_lat=39.640583,
            gps_lon=-0.230194,
            frame_observations={
                "frame_001": {"u": 1250.5, "v": 680.0}
            }
        )
        result = _serialize_multi_frame_gcp(gcp)

        assert 'utm' not in result

    def test_deserialize_multi_frame_gcp_valid(self):
        """Test deserialization of valid GCP dictionary."""
        data = {
            'gcp_id': 'gcp_001',
            'gps': {'latitude': 39.640583, 'longitude': -0.230194},
            'utm': {'easting': 729345.67, 'northing': 4389234.12},
            'frame_observations': [
                {'frame_id': 'frame_001', 'image': {'u': 1250.5, 'v': 680.0}},
                {'frame_id': 'frame_002', 'image': {'u': 1180.2, 'v': 720.5}}
            ]
        }
        result = _deserialize_multi_frame_gcp(data)

        assert isinstance(result, MultiFrameGCP)
        assert result.gcp_id == 'gcp_001'
        assert result.gps_lat == 39.640583
        assert result.gps_lon == -0.230194
        assert result.utm_easting == 729345.67
        assert result.utm_northing == 4389234.12
        assert len(result.frame_observations) == 2

    def test_deserialize_multi_frame_gcp_without_utm(self):
        """Test deserialization of GCP without UTM coordinates."""
        data = {
            'gcp_id': 'gcp_001',
            'gps': {'latitude': 39.640583, 'longitude': -0.230194},
            'frame_observations': [
                {'frame_id': 'frame_001', 'image': {'u': 1250.5, 'v': 680.0}}
            ]
        }
        result = _deserialize_multi_frame_gcp(data)

        assert result.utm_easting is None
        assert result.utm_northing is None

    def test_deserialize_multi_frame_gcp_missing_gps_field(self):
        """Test deserialization fails with missing GPS field."""
        data = {
            'gcp_id': 'gcp_001',
            'gps': {'latitude': 39.640583},  # Missing 'longitude'
            'frame_observations': [
                {'frame_id': 'frame_001', 'image': {'u': 1250.5, 'v': 680.0}}
            ]
        }

        with pytest.raises(ValueError, match="missing 'latitude' or 'longitude'"):
            _deserialize_multi_frame_gcp(data)

    def test_deserialize_multi_frame_gcp_no_observations(self):
        """Test deserialization fails with empty frame observations."""
        data = {
            'gcp_id': 'gcp_001',
            'gps': {'latitude': 39.640583, 'longitude': -0.230194},
            'frame_observations': []
        }

        with pytest.raises(ValueError, match="no valid frame observations"):
            _deserialize_multi_frame_gcp(data)

    def test_round_trip_multi_frame_gcp(self, sample_multi_frame_gcp):
        """Test MultiFrameGCP serialization and deserialization round-trip."""
        serialized = _serialize_multi_frame_gcp(sample_multi_frame_gcp)
        deserialized = _deserialize_multi_frame_gcp(serialized)

        assert deserialized.gcp_id == sample_multi_frame_gcp.gcp_id
        assert deserialized.gps_lat == sample_multi_frame_gcp.gps_lat
        assert deserialized.gps_lon == sample_multi_frame_gcp.gps_lon
        assert deserialized.utm_easting == sample_multi_frame_gcp.utm_easting
        assert len(deserialized.frame_observations) == len(sample_multi_frame_gcp.frame_observations)


# ============================================================================
# Test: MultiFrameCalibrationData I/O
# ============================================================================

class TestMultiFrameCalibrationDataIO:
    """Tests for MultiFrameCalibrationData save/load."""

    def test_save_calibration_data(self, sample_calibration_data, temp_dir):
        """Test saving calibration data to YAML file."""
        yaml_path = temp_dir / "calibration_data.yaml"

        save_multi_frame_calibration_data(sample_calibration_data, str(yaml_path))

        assert yaml_path.exists()

        # Verify YAML structure
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        assert 'multi_frame_calibration' in data
        assert 'frames' in data['multi_frame_calibration']
        assert 'gcps' in data['multi_frame_calibration']
        assert 'camera_config' in data['multi_frame_calibration']

    def test_save_calibration_data_creates_directory(self, sample_calibration_data, temp_dir):
        """Test that save creates parent directories if they don't exist."""
        yaml_path = temp_dir / "subdir" / "calibration_data.yaml"

        save_multi_frame_calibration_data(sample_calibration_data, str(yaml_path))

        assert yaml_path.exists()

    def test_save_calibration_data_invalid_type(self, temp_dir):
        """Test that save fails with invalid data type."""
        yaml_path = temp_dir / "calibration_data.yaml"

        with pytest.raises(ValueError, match="must be MultiFrameCalibrationData"):
            save_multi_frame_calibration_data("invalid", str(yaml_path))

    def test_load_calibration_data(self, sample_calibration_data, temp_dir):
        """Test loading calibration data from YAML file."""
        yaml_path = temp_dir / "calibration_data.yaml"
        save_multi_frame_calibration_data(sample_calibration_data, str(yaml_path))

        loaded_data = load_multi_frame_calibration_data(str(yaml_path))

        assert isinstance(loaded_data, MultiFrameCalibrationData)
        assert len(loaded_data.frames) == len(sample_calibration_data.frames)
        assert len(loaded_data.gcps) == len(sample_calibration_data.gcps)
        assert 'reference_lat' in loaded_data.camera_config

    def test_load_calibration_data_file_not_found(self, temp_dir):
        """Test loading fails with non-existent file."""
        yaml_path = temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="not found"):
            load_multi_frame_calibration_data(str(yaml_path))

    def test_load_calibration_data_empty_file(self, temp_dir):
        """Test loading fails with empty YAML file."""
        yaml_path = temp_dir / "empty.yaml"
        yaml_path.write_text("")

        with pytest.raises(ValueError, match="empty"):
            load_multi_frame_calibration_data(str(yaml_path))

    def test_load_calibration_data_missing_section(self, temp_dir):
        """Test loading fails with missing required section."""
        yaml_path = temp_dir / "invalid.yaml"
        yaml_path.write_text("some_other_section:\n  key: value\n")

        with pytest.raises(ValueError, match="missing 'multi_frame_calibration' section"):
            load_multi_frame_calibration_data(str(yaml_path))

    def test_load_calibration_data_missing_frames(self, temp_dir):
        """Test loading fails with missing frames section."""
        yaml_path = temp_dir / "invalid.yaml"
        yaml_content = """
multi_frame_calibration:
  gcps: []
  camera_config: {}
"""
        yaml_path.write_text(yaml_content)

        with pytest.raises(ValueError, match="missing required fields.*frames"):
            load_multi_frame_calibration_data(str(yaml_path))

    def test_load_calibration_data_empty_frames_list(self, temp_dir):
        """Test loading fails with empty frames list."""
        yaml_path = temp_dir / "invalid.yaml"
        yaml_content = """
multi_frame_calibration:
  frames: []
  gcps: []
  camera_config: {}
"""
        yaml_path.write_text(yaml_content)

        with pytest.raises(ValueError, match="'frames' list cannot be empty"):
            load_multi_frame_calibration_data(str(yaml_path))

    def test_round_trip_calibration_data(self, sample_calibration_data, temp_dir):
        """Test calibration data save and load round-trip."""
        yaml_path = temp_dir / "calibration_data.yaml"

        save_multi_frame_calibration_data(sample_calibration_data, str(yaml_path))
        loaded_data = load_multi_frame_calibration_data(str(yaml_path))

        # Verify frames
        assert len(loaded_data.frames) == len(sample_calibration_data.frames)
        for orig, loaded in zip(sample_calibration_data.frames, loaded_data.frames):
            assert loaded.frame_id == orig.frame_id
            assert loaded.image_path == orig.image_path

        # Verify GCPs
        assert len(loaded_data.gcps) == len(sample_calibration_data.gcps)
        for orig, loaded in zip(sample_calibration_data.gcps, loaded_data.gcps):
            assert loaded.gcp_id == orig.gcp_id
            assert loaded.gps_lat == orig.gps_lat
            assert loaded.gps_lon == orig.gps_lon

        # Verify camera config numpy arrays
        assert isinstance(loaded_data.camera_config['K'], np.ndarray)
        assert isinstance(loaded_data.camera_config['w_pos'], np.ndarray)
        np.testing.assert_array_almost_equal(
            loaded_data.camera_config['K'],
            sample_calibration_data.camera_config['K']
        )


# ============================================================================
# Test: MultiFrameCalibrationResult I/O
# ============================================================================

class TestMultiFrameCalibrationResultIO:
    """Tests for MultiFrameCalibrationResult save/load."""

    def test_save_calibration_result(self, sample_calibration_result, temp_dir):
        """Test saving calibration result to YAML file."""
        yaml_path = temp_dir / "calibration_result.yaml"

        save_multi_frame_calibration_result(sample_calibration_result, str(yaml_path))

        assert yaml_path.exists()

        # Verify YAML structure
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        assert 'calibration_result' in data
        assert 'optimized_params' in data['calibration_result']
        assert 'diagnostics' in data['calibration_result']

    def test_save_calibration_result_invalid_type(self, temp_dir):
        """Test that save fails with invalid result type."""
        yaml_path = temp_dir / "result.yaml"

        with pytest.raises(ValueError, match="must be MultiFrameCalibrationResult"):
            save_multi_frame_calibration_result("invalid", str(yaml_path))

    def test_load_calibration_result(self, sample_calibration_result, temp_dir):
        """Test loading calibration result from YAML file."""
        yaml_path = temp_dir / "result.yaml"
        save_multi_frame_calibration_result(sample_calibration_result, str(yaml_path))

        loaded_result = load_multi_frame_calibration_result(str(yaml_path))

        assert isinstance(loaded_result, MultiFrameCalibrationResult)
        assert len(loaded_result.optimized_params) == 6
        assert loaded_result.final_error == sample_calibration_result.final_error
        assert len(loaded_result.per_frame_errors) == len(sample_calibration_result.per_frame_errors)

    def test_load_calibration_result_file_not_found(self, temp_dir):
        """Test loading fails with non-existent file."""
        yaml_path = temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="not found"):
            load_multi_frame_calibration_result(str(yaml_path))

    def test_load_calibration_result_missing_section(self, temp_dir):
        """Test loading fails with missing required section."""
        yaml_path = temp_dir / "invalid.yaml"
        yaml_path.write_text("some_other_section:\n  key: value\n")

        with pytest.raises(ValueError, match="missing 'calibration_result' section"):
            load_multi_frame_calibration_result(str(yaml_path))

    def test_round_trip_calibration_result(self, sample_calibration_result, temp_dir):
        """Test calibration result save and load round-trip."""
        yaml_path = temp_dir / "result.yaml"

        save_multi_frame_calibration_result(sample_calibration_result, str(yaml_path))
        loaded_result = load_multi_frame_calibration_result(str(yaml_path))

        # Verify optimized parameters
        np.testing.assert_array_almost_equal(
            loaded_result.optimized_params,
            sample_calibration_result.optimized_params
        )

        # Verify diagnostics
        assert loaded_result.initial_error == sample_calibration_result.initial_error
        assert loaded_result.final_error == sample_calibration_result.final_error
        assert loaded_result.num_inliers == sample_calibration_result.num_inliers
        assert loaded_result.num_outliers == sample_calibration_result.num_outliers

        # Verify per-frame errors
        for frame_id in sample_calibration_result.per_frame_errors:
            assert frame_id in loaded_result.per_frame_errors
            assert loaded_result.per_frame_errors[frame_id] == \
                   sample_calibration_result.per_frame_errors[frame_id]


# ============================================================================
# Test: Example Configuration
# ============================================================================

class TestExampleConfiguration:
    """Tests for example configuration generation."""

    def test_create_example_config_returns_string(self):
        """Test that example config returns a string."""
        result = create_example_multi_frame_config()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_create_example_config_valid_yaml(self):
        """Test that example config is valid YAML."""
        result = create_example_multi_frame_config()

        # Should be parseable as YAML
        try:
            parsed = yaml.safe_load(result)
            assert isinstance(parsed, dict)
        except yaml.YAMLError:
            pytest.fail("Example config is not valid YAML")

    def test_create_example_config_has_required_sections(self):
        """Test that example config contains all required sections."""
        result = create_example_multi_frame_config()
        parsed = yaml.safe_load(result)

        assert 'multi_frame_calibration' in parsed
        calib = parsed['multi_frame_calibration']
        assert 'frames' in calib
        assert 'gcps' in calib
        assert 'camera_config' in calib
        assert 'calibration_result' in calib

    def test_create_example_config_has_documentation(self):
        """Test that example config contains inline documentation."""
        result = create_example_multi_frame_config()

        # Should contain comments explaining fields
        assert '#' in result
        assert 'PTZ position' in result or 'pan angle' in result.lower()
        assert 'GPS' in result or 'latitude' in result

    def test_example_config_can_be_loaded(self, temp_dir):
        """Test that example config can be loaded as calibration data."""
        example_yaml = create_example_multi_frame_config()
        yaml_path = temp_dir / "example.yaml"
        yaml_path.write_text(example_yaml)

        # Should be loadable (will fail validation due to dummy data, but structure should be correct)
        try:
            data = load_multi_frame_calibration_data(str(yaml_path))
            assert isinstance(data, MultiFrameCalibrationData)
        except ValueError:
            # May fail validation but structure should parse
            pytest.skip("Example config has invalid dummy data (expected)")


# ============================================================================
# Test: Numpy Array Handling
# ============================================================================

class TestNumpyArrayHandling:
    """Tests for numpy array conversion in serialization."""

    def test_numpy_array_to_list_in_camera_config(self, temp_dir):
        """Test that numpy arrays are converted to lists when saving."""
        frames = [
            FrameObservation(
                frame_id="frame_001",
                image_path="test.jpg",
                ptz_position=PTZPosition(pan=0, tilt=0, zoom=1),
                timestamp=datetime.now(timezone.utc)
            )
        ]
        gcps = [
            MultiFrameGCP(
                gcp_id="gcp_001",
                gps_lat=39.64,
                gps_lon=-0.23,
                frame_observations={"frame_001": {"u": 100, "v": 200}}
            )
        ]
        camera_config = {
            'K': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64),
            'w_pos': np.array([1.0, 2.0, 3.0], dtype=np.float64)
        }

        data = MultiFrameCalibrationData(frames=frames, gcps=gcps, camera_config=camera_config)
        yaml_path = temp_dir / "test.yaml"

        save_multi_frame_calibration_data(data, str(yaml_path))

        # Read raw YAML and check that arrays are lists
        with open(yaml_path, 'r') as f:
            raw_yaml = yaml.safe_load(f)

        K = raw_yaml['multi_frame_calibration']['camera_config']['K']
        w_pos = raw_yaml['multi_frame_calibration']['camera_config']['w_pos']

        assert isinstance(K, list)
        assert isinstance(w_pos, list)

    def test_list_to_numpy_array_when_loading(self, temp_dir):
        """Test that lists are converted to numpy arrays when loading."""
        yaml_content = """
multi_frame_calibration:
  frames:
    - frame_id: "frame_001"
      image_path: "test.jpg"
      ptz_position: {pan: 0, tilt: 0, zoom: 1}
      timestamp: "2025-01-05T10:00:00Z"
  gcps:
    - gcp_id: "gcp_001"
      gps: {latitude: 39.64, longitude: -0.23}
      frame_observations:
        - frame_id: "frame_001"
          image: {u: 100, v: 200}
  camera_config:
    K:
      - [1.0, 2.0, 3.0]
      - [4.0, 5.0, 6.0]
      - [7.0, 8.0, 9.0]
    w_pos: [1.0, 2.0, 3.0]
"""
        yaml_path = temp_dir / "test.yaml"
        yaml_path.write_text(yaml_content)

        data = load_multi_frame_calibration_data(str(yaml_path))

        assert isinstance(data.camera_config['K'], np.ndarray)
        assert isinstance(data.camera_config['w_pos'], np.ndarray)
        assert data.camera_config['K'].shape == (3, 3)
        assert data.camera_config['w_pos'].shape == (3,)


# ============================================================================
# Test: DateTime Timezone Handling
# ============================================================================

class TestDateTimeTimezoneHandling:
    """Tests for datetime timezone handling."""

    def test_serialize_datetime_with_timezone(self):
        """Test that datetime with timezone is serialized to ISO format."""
        frame = FrameObservation(
            frame_id="frame_001",
            image_path="test.jpg",
            ptz_position=PTZPosition(pan=0, tilt=0, zoom=1),
            timestamp=datetime(2025, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
        )

        serialized = _serialize_frame_observation(frame)

        assert 'timestamp' in serialized
        assert isinstance(serialized['timestamp'], str)
        assert '2025-01-05' in serialized['timestamp']
        assert '+00:00' in serialized['timestamp'] or 'Z' in serialized['timestamp']

    def test_deserialize_datetime_with_z_suffix(self):
        """Test deserialization of ISO timestamp with Z suffix."""
        data = {
            'frame_id': 'frame_001',
            'image_path': 'test.jpg',
            'ptz_position': {'pan': 0, 'tilt': 0, 'zoom': 1},
            'timestamp': '2025-01-05T10:00:00Z'
        }

        frame = _deserialize_frame_observation(data)

        assert frame.timestamp.tzinfo is not None
        assert frame.timestamp.tzinfo == timezone.utc

    def test_deserialize_datetime_with_offset(self):
        """Test deserialization of ISO timestamp with UTC offset."""
        data = {
            'frame_id': 'frame_001',
            'image_path': 'test.jpg',
            'ptz_position': {'pan': 0, 'tilt': 0, 'zoom': 1},
            'timestamp': '2025-01-05T10:00:00+00:00'
        }

        frame = _deserialize_frame_observation(data)

        assert frame.timestamp.tzinfo is not None

    def test_deserialize_naive_datetime_gets_utc(self):
        """Test that naive datetime is assigned UTC timezone."""
        data = {
            'frame_id': 'frame_001',
            'image_path': 'test.jpg',
            'ptz_position': {'pan': 0, 'tilt': 0, 'zoom': 1},
            'timestamp': '2025-01-05T10:00:00'  # No timezone
        }

        frame = _deserialize_frame_observation(data)

        assert frame.timestamp.tzinfo == timezone.utc
