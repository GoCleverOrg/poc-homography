#!/usr/bin/env python3
"""
Unit tests for the height calibration module.

Tests cover:
- CalibrationPoint and CalibrationResult dataclasses
- HeightCalibrator initialization and point management
- Least-squares optimization
- MAD and RANSAC outlier detection
- CalibrationHistory persistence
"""

from datetime import datetime

import numpy as np
import pytest

from poc_homography.calibration_history import (
    CalibrationHistory,
    CalibrationHistoryEntry,
)
from poc_homography.height_calibration import (
    CalibrationPoint,
    CalibrationResult,
    HeightCalibrator,
)

# ============================================================================
# Mock Classes
# ============================================================================


class MockCameraGeometry:
    """Mock CameraGeometry that projects pixel (1, 1) to a given world coordinate."""

    def __init__(self, world_x: float, world_y: float):
        """
        Create mock with H_inv that projects pixel (1, 1, 1) to (world_x, world_y, 1).

        Args:
            world_x: World X coordinate result
            world_y: World Y coordinate result
        """
        # H_inv such that [1, 1, 1]^T @ H_inv gives [world_x, world_y, 1]
        # For homogeneous coords: H_inv @ [px, py, 1] = [Xw, Yw, W]
        # We want: [world_x, world_y, 1] for pixel [1, 1, 1]
        self.H_inv = np.array([[world_x, 0, 0], [0, world_y, 0], [0, 0, 1.0]])


class MockCameraGeometryHorizon:
    """Mock CameraGeometry that returns horizon (W=0) for projections."""

    def __init__(self):
        # H_inv that results in W=0 (horizon point)
        self.H_inv = np.array(
            [
                [1.0, 0, 0],
                [0, 1.0, 0],
                [0, 0, 0.0],  # This makes W = 0
            ]
        )


# ============================================================================
# Test: CalibrationPoint Dataclass
# ============================================================================


class TestCalibrationPoint:
    """Tests for CalibrationPoint dataclass."""

    def test_creation(self):
        """Test basic dataclass creation."""
        point = CalibrationPoint(
            pixel_x=100.0,
            pixel_y=200.0,
            gps_lat=39.64,
            gps_lon=-0.23,
            world_x=5.0,
            world_y=10.0,
            gps_distance=11.18,
            homography_distance=11.18,
            current_height=5.0,
        )

        assert point.pixel_x == 100.0
        assert point.pixel_y == 200.0
        assert point.gps_lat == 39.64
        assert point.gps_lon == -0.23
        assert point.world_x == 5.0
        assert point.world_y == 10.0
        assert point.gps_distance == 11.18
        assert point.homography_distance == 11.18
        assert point.current_height == 5.0

    def test_all_fields_exist(self):
        """Verify all required fields exist."""
        point = CalibrationPoint(
            pixel_x=0,
            pixel_y=0,
            gps_lat=0,
            gps_lon=0,
            world_x=0,
            world_y=0,
            gps_distance=0,
            homography_distance=0,
            current_height=0,
        )

        required_fields = [
            "pixel_x",
            "pixel_y",
            "gps_lat",
            "gps_lon",
            "world_x",
            "world_y",
            "gps_distance",
            "homography_distance",
            "current_height",
        ]

        for field in required_fields:
            assert hasattr(point, field), f"Missing field: {field}"


# ============================================================================
# Test: CalibrationResult Dataclass
# ============================================================================


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_creation(self):
        """Test basic dataclass creation."""
        result = CalibrationResult(
            estimated_height=5.5,
            confidence_interval=(5.2, 5.8),
            inlier_count=7,
            outlier_count=2,
            timestamp=datetime.now(),
        )

        assert result.estimated_height == 5.5
        assert result.confidence_interval == (5.2, 5.8)
        assert result.inlier_count == 7
        assert result.outlier_count == 2

    def test_default_factory_for_points(self):
        """Test that calibration_points has default factory."""
        result = CalibrationResult(
            estimated_height=5.0,
            confidence_interval=(4.8, 5.2),
            inlier_count=5,
            outlier_count=0,
            timestamp=datetime.now(),
        )

        # calibration_points should default to empty list
        assert result.calibration_points == []
        assert isinstance(result.calibration_points, list)


# ============================================================================
# Test: HeightCalibrator Initialization
# ============================================================================


class TestHeightCalibratorInit:
    """Tests for HeightCalibrator initialization."""

    def test_init_with_dms_coordinates(self):
        """Test initialization with DMS string GPS coordinates."""
        camera_gps = {"lat": "39°38'25.7\"N", "lon": "0°13'48.7\"W"}
        calibrator = HeightCalibrator(camera_gps, min_points=5)

        # Verify DMS was converted to decimal degrees
        assert abs(calibrator.camera_lat_dd - 39.640472) < 0.001
        assert abs(calibrator.camera_lon_dd - (-0.230194)) < 0.001
        assert calibrator.min_points == 5

    def test_init_with_decimal_coordinates(self):
        """Test initialization with decimal degree GPS coordinates."""
        camera_gps = {"lat": 39.640472, "lon": -0.230194}
        calibrator = HeightCalibrator(camera_gps, min_points=3)

        assert calibrator.camera_lat_dd == 39.640472
        assert calibrator.camera_lon_dd == -0.230194
        assert calibrator.min_points == 3

    def test_init_missing_lat(self):
        """Test that missing 'lat' key raises ValueError."""
        with pytest.raises(ValueError, match="must contain 'lat' and 'lon'"):
            HeightCalibrator({"lon": -0.23})

    def test_init_missing_lon(self):
        """Test that missing 'lon' key raises ValueError."""
        with pytest.raises(ValueError, match="must contain 'lat' and 'lon'"):
            HeightCalibrator({"lat": 39.64})

    def test_init_invalid_min_points(self):
        """Test that invalid min_points raises ValueError."""
        with pytest.raises(ValueError, match="min_points must be at least 1"):
            HeightCalibrator({"lat": 39.64, "lon": -0.23}, min_points=0)


# ============================================================================
# Test: HeightCalibrator Point Management
# ============================================================================


class TestHeightCalibratorPoints:
    """Tests for HeightCalibrator point management methods."""

    @pytest.fixture
    def calibrator(self):
        """Create a calibrator for testing."""
        return HeightCalibrator({"lat": 39.640472, "lon": -0.230194}, min_points=3)

    def test_add_point_valid(self, calibrator):
        """Test adding a valid point."""
        # Create mock geometry that projects to known world coords
        geo = MockCameraGeometry(world_x=3.0, world_y=4.0)

        # Add point with GPS coordinates slightly offset from camera
        # GPS distance will be calculated using haversine
        point = calibrator.add_point(
            pixel_x=1.0,
            pixel_y=1.0,
            gps_lat=39.640500,  # Slightly north of camera
            gps_lon=-0.230150,  # Slightly east of camera
            current_height=5.0,
            geo=geo,
        )

        assert point.pixel_x == 1.0
        assert point.pixel_y == 1.0
        assert point.world_x == 3.0
        assert point.world_y == 4.0
        assert point.homography_distance == 5.0  # sqrt(3^2 + 4^2)
        assert point.gps_distance > 0
        assert point.current_height == 5.0
        assert calibrator.get_point_count() == 1

    def test_add_point_horizon_raises(self, calibrator):
        """Test that adding a horizon point raises ValueError."""
        geo = MockCameraGeometryHorizon()

        with pytest.raises(ValueError, match="too close to horizon"):
            calibrator.add_point(
                pixel_x=1.0, pixel_y=1.0, gps_lat=39.64, gps_lon=-0.23, current_height=5.0, geo=geo
            )

    def test_clear_points(self, calibrator):
        """Test clearing all points."""
        geo = MockCameraGeometry(world_x=3.0, world_y=4.0)

        # Add some points
        calibrator.add_point(1, 1, 39.64, -0.23, 5.0, geo)
        calibrator.add_point(2, 2, 39.64, -0.23, 5.0, geo)
        assert calibrator.get_point_count() == 2

        # Clear points
        calibrator.clear_points()
        assert calibrator.get_point_count() == 0

    def test_is_ready_true(self, calibrator):
        """Test is_ready returns True with enough points."""
        geo = MockCameraGeometry(world_x=3.0, world_y=4.0)

        for i in range(3):
            calibrator.add_point(1, 1, 39.64 + i * 0.0001, -0.23, 5.0, geo)

        assert calibrator.is_ready() is True

    def test_is_ready_false(self, calibrator):
        """Test is_ready returns False with insufficient points."""
        geo = MockCameraGeometry(world_x=3.0, world_y=4.0)

        calibrator.add_point(1, 1, 39.64, -0.23, 5.0, geo)
        calibrator.add_point(2, 2, 39.64, -0.23, 5.0, geo)

        assert calibrator.is_ready() is False


# ============================================================================
# Test: Height Estimation
# ============================================================================


class TestHeightEstimation:
    """Tests for height estimation methods."""

    def test_estimate_height_from_point_scale_up(self):
        """Test height estimation when GPS > homography (need to increase height)."""
        calibrator = HeightCalibrator({"lat": 39.64, "lon": -0.23}, min_points=1)

        # Create point where GPS distance is 2x homography distance
        # This means height should be 2x current
        point = CalibrationPoint(
            pixel_x=1,
            pixel_y=1,
            gps_lat=39.64,
            gps_lon=-0.23,
            world_x=3.0,
            world_y=4.0,
            gps_distance=10.0,  # GPS says 10m
            homography_distance=5.0,  # Homography says 5m
            current_height=5.0,
        )

        estimated = calibrator.estimate_height_from_point(point)

        # scale_factor = 10/5 = 2, estimated = 5 * 2 = 10
        assert estimated == 10.0

    def test_estimate_height_from_point_scale_down(self):
        """Test height estimation when GPS < homography (need to decrease height)."""
        calibrator = HeightCalibrator({"lat": 39.64, "lon": -0.23}, min_points=1)

        # Create point where GPS distance is 0.5x homography distance
        point = CalibrationPoint(
            pixel_x=1,
            pixel_y=1,
            gps_lat=39.64,
            gps_lon=-0.23,
            world_x=3.0,
            world_y=4.0,
            gps_distance=5.0,  # GPS says 5m
            homography_distance=10.0,  # Homography says 10m
            current_height=10.0,
        )

        estimated = calibrator.estimate_height_from_point(point)

        # scale_factor = 5/10 = 0.5, estimated = 10 * 0.5 = 5
        assert estimated == 5.0

    def test_estimate_height_zero_homography_raises(self):
        """Test that zero homography distance raises ValueError."""
        calibrator = HeightCalibrator({"lat": 39.64, "lon": -0.23}, min_points=1)

        point = CalibrationPoint(
            pixel_x=1,
            pixel_y=1,
            gps_lat=39.64,
            gps_lon=-0.23,
            world_x=0,
            world_y=0,
            gps_distance=10.0,
            homography_distance=0.0,  # Invalid
            current_height=5.0,
        )

        with pytest.raises(ValueError, match="near-zero homography distance"):
            calibrator.estimate_height_from_point(point)

    def test_get_all_height_estimates(self):
        """Test getting all height estimates from multiple points."""
        calibrator = HeightCalibrator({"lat": 39.64, "lon": -0.23}, min_points=1)

        # Manually add points with known values
        calibrator.calibration_points = [
            CalibrationPoint(1, 1, 39.64, -0.23, 3, 4, 10.0, 5.0, 5.0),  # estimate: 10
            CalibrationPoint(2, 2, 39.64, -0.23, 3, 4, 7.5, 5.0, 5.0),  # estimate: 7.5
            CalibrationPoint(3, 3, 39.64, -0.23, 3, 4, 5.0, 5.0, 5.0),  # estimate: 5
        ]

        estimates = calibrator.get_all_height_estimates()

        assert len(estimates) == 3
        assert estimates[0] == 10.0
        assert estimates[1] == 7.5
        assert estimates[2] == 5.0

    def test_get_all_height_estimates_empty_raises(self):
        """Test that empty calibration points raises ValueError."""
        calibrator = HeightCalibrator({"lat": 39.64, "lon": -0.23}, min_points=1)

        with pytest.raises(ValueError, match="No calibration points"):
            calibrator.get_all_height_estimates()


# ============================================================================
# Test: Least-Squares Optimization
# ============================================================================


class TestOptimization:
    """Tests for least-squares optimization."""

    @pytest.fixture
    def calibrator_with_points(self):
        """Create calibrator with multiple calibration points."""
        calibrator = HeightCalibrator({"lat": 39.64, "lon": -0.23}, min_points=5)

        # Add points that all suggest height should be around 6.0
        # (GPS distance is 1.2x homography distance, so height should be 1.2x current)
        for i in range(6):
            calibrator.calibration_points.append(
                CalibrationPoint(
                    pixel_x=i,
                    pixel_y=i,
                    gps_lat=39.64,
                    gps_lon=-0.23,
                    world_x=3.0 + i * 0.1,
                    world_y=4.0 + i * 0.1,
                    gps_distance=6.0 + i * 0.1,  # GPS says ~6m
                    homography_distance=5.0 + i * 0.1,  # Homography says ~5m
                    current_height=5.0,
                )
            )

        return calibrator

    def test_optimize_height_least_squares(self, calibrator_with_points):
        """Test basic least-squares optimization."""
        result = calibrator_with_points.optimize_height_least_squares()

        # Height should be around 6.0 (5.0 * 1.2 scale factor)
        assert 5.5 < result.estimated_height < 6.5
        assert result.inlier_count == 6
        assert result.outlier_count == 0
        assert result.confidence_interval[0] < result.estimated_height
        assert result.confidence_interval[1] > result.estimated_height

    def test_optimize_height_not_ready_raises(self):
        """Test that optimization raises when not enough points."""
        calibrator = HeightCalibrator({"lat": 39.64, "lon": -0.23}, min_points=5)

        # Add only 3 points
        for i in range(3):
            calibrator.calibration_points.append(
                CalibrationPoint(i, i, 39.64, -0.23, 3, 4, 6.0, 5.0, 5.0)
            )

        with pytest.raises(ValueError, match="Not enough calibration points"):
            calibrator.optimize_height_least_squares()

    def test_confidence_interval_valid(self, calibrator_with_points):
        """Test that confidence interval is properly computed."""
        result = calibrator_with_points.optimize_height_least_squares()

        lower, upper = result.confidence_interval
        assert lower <= result.estimated_height
        assert upper >= result.estimated_height
        assert lower > 0  # Height should be positive
        assert upper - lower > 0  # Interval should have width


# ============================================================================
# Test: Outlier Detection
# ============================================================================


class TestOutlierDetection:
    """Tests for outlier detection methods."""

    @pytest.fixture
    def calibrator(self):
        """Create calibrator for testing."""
        return HeightCalibrator({"lat": 39.64, "lon": -0.23}, min_points=5)

    def test_detect_outliers_mad(self, calibrator):
        """Test MAD-based outlier detection."""
        # Most estimates around 5.0, with one outlier at 10.0
        estimates = [5.0, 5.1, 4.9, 5.0, 5.2, 10.0]

        inliers, indices = calibrator._detect_outliers_mad(estimates, threshold=2.5)

        # The outlier (10.0) should be excluded
        assert 10.0 not in inliers
        assert len(inliers) == 5
        assert 5 not in indices  # Index of outlier

    def test_detect_outliers_mad_all_same(self, calibrator):
        """Test MAD when all estimates are identical."""
        estimates = [5.0, 5.0, 5.0, 5.0, 5.0]

        inliers, indices = calibrator._detect_outliers_mad(estimates)

        # All should be inliers
        assert len(inliers) == 5
        assert indices == [0, 1, 2, 3, 4]

    def test_detect_outliers_ransac(self, calibrator):
        """Test RANSAC-based outlier detection."""
        # Most estimates around 5.0, with one outlier at 10.0
        np.random.seed(42)  # For reproducibility
        estimates = [5.0, 5.1, 4.9, 5.0, 5.2, 10.0]

        inliers, indices = calibrator._detect_outliers_ransac(estimates, threshold_ratio=0.1)

        # The outlier (10.0) should be excluded
        assert 10.0 not in inliers
        assert len(inliers) >= 4  # At least most inliers should be found

    def test_optimize_with_outliers_mad(self):
        """Test optimization with MAD outlier detection."""
        calibrator = HeightCalibrator({"lat": 39.64, "lon": -0.23}, min_points=5)

        # Add 7 consistent points (height estimates around 6.0)
        for i in range(7):
            calibrator.calibration_points.append(
                CalibrationPoint(i, i, 39.64, -0.23, 3, 4, 6.0 + i * 0.1, 5.0, 5.0)
            )

        # Add 1 extreme outlier (GPS distance very high = height estimate ~25)
        calibrator.calibration_points.append(
            CalibrationPoint(10, 10, 39.64, -0.23, 3, 4, 50.0, 5.0, 5.0)
        )

        result = calibrator.optimize_height_with_outliers(method="mad", threshold=2.0)

        # Height should be around 6.x, not pulled up by the single outlier
        assert result.estimated_height < 15.0  # Much less than outlier's 50
        assert result.outlier_count >= 1  # At least one outlier detected

    def test_optimize_with_outliers_ransac(self):
        """Test optimization with RANSAC outlier detection."""
        np.random.seed(42)
        calibrator = HeightCalibrator({"lat": 39.64, "lon": -0.23}, min_points=5)

        # Add 5 consistent points and 2 outlier points
        for i in range(5):
            calibrator.calibration_points.append(
                CalibrationPoint(i, i, 39.64, -0.23, 3, 4, 6.0, 5.0, 5.0)
            )

        # Add outliers
        for i in range(2):
            calibrator.calibration_points.append(
                CalibrationPoint(i + 5, i + 5, 39.64, -0.23, 3, 4, 25.0, 5.0, 5.0)
            )

        result = calibrator.optimize_height_with_outliers(method="ransac")

        # Height should be around 6.0
        assert 5.5 < result.estimated_height < 7.0

    def test_optimize_with_outliers_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        calibrator = HeightCalibrator({"lat": 39.64, "lon": -0.23}, min_points=5)

        for i in range(5):
            calibrator.calibration_points.append(
                CalibrationPoint(i, i, 39.64, -0.23, 3, 4, 6.0, 5.0, 5.0)
            )

        with pytest.raises(ValueError, match="Invalid outlier detection method"):
            calibrator.optimize_height_with_outliers(method="invalid")


# ============================================================================
# Test: CalibrationHistory
# ============================================================================


class TestCalibrationHistory:
    """Tests for CalibrationHistory persistence."""

    @pytest.fixture
    def sample_entry(self):
        """Create a sample calibration entry."""
        return CalibrationHistoryEntry(
            camera_name="TestCam",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            estimated_height=5.23,
            confidence_interval=(5.10, 5.36),
            inlier_count=7,
            outlier_count=2,
            method="mad",
            camera_gps_lat=39.640472,
            camera_gps_lon=-0.230194,
        )

    def test_entry_to_dict(self, sample_entry):
        """Test entry serialization to dict."""
        data = sample_entry.to_dict()

        assert data["camera_name"] == "TestCam"
        assert data["estimated_height"] == 5.23
        assert data["timestamp"] == "2024-01-15T10:30:00"
        assert data["confidence_interval"] == (5.10, 5.36)

    def test_entry_from_dict(self):
        """Test entry deserialization from dict."""
        data = {
            "camera_name": "TestCam",
            "timestamp": "2024-01-15T10:30:00",
            "estimated_height": 5.23,
            "confidence_interval": [5.10, 5.36],  # List from YAML
            "inlier_count": 7,
            "outlier_count": 2,
            "method": "mad",
            "camera_gps_lat": 39.640472,
            "camera_gps_lon": -0.230194,
        }

        entry = CalibrationHistoryEntry.from_dict(data)

        assert entry.camera_name == "TestCam"
        assert entry.timestamp == datetime(2024, 1, 15, 10, 30, 0)
        assert entry.confidence_interval == (5.10, 5.36)  # Converted to tuple

    def test_add_entry(self, tmp_path, sample_entry):
        """Test adding an entry to history."""
        storage_file = tmp_path / "history.yaml"
        history = CalibrationHistory(storage_path=str(storage_file))

        history.add_entry(sample_entry)

        assert len(history.entries) == 1
        assert history.entries[0].camera_name == "TestCam"

    def test_get_entries_all(self, tmp_path, sample_entry):
        """Test retrieving all entries."""
        storage_file = tmp_path / "history.yaml"
        history = CalibrationHistory(storage_path=str(storage_file))

        # Add multiple entries
        history.add_entry(sample_entry)
        entry2 = CalibrationHistoryEntry(
            camera_name="OtherCam",
            timestamp=datetime(2024, 1, 16, 10, 0, 0),
            estimated_height=6.0,
            confidence_interval=(5.8, 6.2),
            inlier_count=5,
            outlier_count=1,
            method="ransac",
            camera_gps_lat=40.0,
            camera_gps_lon=-1.0,
        )
        history.add_entry(entry2)

        entries = history.get_entries()
        assert len(entries) == 2

    def test_get_entries_filtered(self, tmp_path, sample_entry):
        """Test filtering entries by camera name."""
        storage_file = tmp_path / "history.yaml"
        history = CalibrationHistory(storage_path=str(storage_file))

        history.add_entry(sample_entry)
        history.add_entry(
            CalibrationHistoryEntry(
                camera_name="OtherCam",
                timestamp=datetime.now(),
                estimated_height=6.0,
                confidence_interval=(5.8, 6.2),
                inlier_count=5,
                outlier_count=1,
                method="mad",
                camera_gps_lat=40.0,
                camera_gps_lon=-1.0,
            )
        )

        entries = history.get_entries(camera_name="TestCam")
        assert len(entries) == 1
        assert entries[0].camera_name == "TestCam"

    def test_get_latest(self, tmp_path):
        """Test getting latest entry for camera."""
        storage_file = tmp_path / "history.yaml"
        history = CalibrationHistory(storage_path=str(storage_file))

        # Add older entry first
        history.add_entry(
            CalibrationHistoryEntry(
                camera_name="TestCam",
                timestamp=datetime(2024, 1, 10, 10, 0, 0),
                estimated_height=5.0,
                confidence_interval=(4.8, 5.2),
                inlier_count=5,
                outlier_count=0,
                method="mad",
                camera_gps_lat=39.64,
                camera_gps_lon=-0.23,
            )
        )

        # Add newer entry
        history.add_entry(
            CalibrationHistoryEntry(
                camera_name="TestCam",
                timestamp=datetime(2024, 1, 15, 10, 0, 0),
                estimated_height=6.0,
                confidence_interval=(5.8, 6.2),
                inlier_count=7,
                outlier_count=1,
                method="mad",
                camera_gps_lat=39.64,
                camera_gps_lon=-0.23,
            )
        )

        latest = history.get_latest("TestCam")

        assert latest is not None
        assert latest.estimated_height == 6.0
        assert latest.timestamp == datetime(2024, 1, 15, 10, 0, 0)

    def test_save_load_roundtrip(self, tmp_path, sample_entry):
        """Test that save/load preserves data."""
        storage_file = tmp_path / "history.yaml"

        # Save
        history = CalibrationHistory(storage_path=str(storage_file))
        history.add_entry(sample_entry)
        history.save()

        # Load in new instance
        history2 = CalibrationHistory(storage_path=str(storage_file))

        assert len(history2.entries) == 1
        loaded = history2.entries[0]
        assert loaded.camera_name == sample_entry.camera_name
        assert loaded.estimated_height == sample_entry.estimated_height
        assert loaded.confidence_interval == sample_entry.confidence_interval

    def test_clear_all(self, tmp_path, sample_entry):
        """Test clearing all entries."""
        storage_file = tmp_path / "history.yaml"
        history = CalibrationHistory(storage_path=str(storage_file))

        history.add_entry(sample_entry)
        assert len(history.entries) == 1

        history.clear()
        assert len(history.entries) == 0

    def test_clear_by_camera(self, tmp_path, sample_entry):
        """Test clearing entries for specific camera."""
        storage_file = tmp_path / "history.yaml"
        history = CalibrationHistory(storage_path=str(storage_file))

        history.add_entry(sample_entry)
        history.add_entry(
            CalibrationHistoryEntry(
                camera_name="OtherCam",
                timestamp=datetime.now(),
                estimated_height=6.0,
                confidence_interval=(5.8, 6.2),
                inlier_count=5,
                outlier_count=1,
                method="mad",
                camera_gps_lat=40.0,
                camera_gps_lon=-1.0,
            )
        )

        history.clear(camera_name="TestCam")

        assert len(history.entries) == 1
        assert history.entries[0].camera_name == "OtherCam"

    def test_nonexistent_file(self, tmp_path):
        """Test that nonexistent file doesn't cause error."""
        storage_file = tmp_path / "nonexistent.yaml"
        history = CalibrationHistory(storage_path=str(storage_file))

        # Should have empty entries
        assert len(history.entries) == 0


# ============================================================================
# Test: Confidence Interval Computation
# ============================================================================


class TestConfidenceInterval:
    """Tests for confidence interval computation."""

    def test_confidence_interval_with_variance(self):
        """Test CI with varied height estimates."""
        calibrator = HeightCalibrator({"lat": 39.64, "lon": -0.23}, min_points=3)

        estimates = [4.8, 5.0, 5.2, 5.1, 4.9]
        estimated_height = 5.0

        ci = calibrator._compute_confidence_interval(estimates, estimated_height)

        assert ci[0] < 5.0
        assert ci[1] > 5.0
        assert ci[0] > 4.0  # Should be reasonable
        assert ci[1] < 6.0

    def test_confidence_interval_single_point(self):
        """Test CI with single estimate returns zero-width interval."""
        calibrator = HeightCalibrator({"lat": 39.64, "lon": -0.23}, min_points=1)

        estimates = [5.0]
        estimated_height = 5.0

        ci = calibrator._compute_confidence_interval(estimates, estimated_height)

        # With n=1, cannot compute CI, should return zero-width
        assert ci == (5.0, 5.0)

    def test_confidence_interval_two_points(self):
        """Test CI with two estimates."""
        calibrator = HeightCalibrator({"lat": 39.64, "lon": -0.23}, min_points=2)

        estimates = [4.0, 6.0]
        estimated_height = 5.0

        ci = calibrator._compute_confidence_interval(estimates, estimated_height)

        # With only 2 points, CI will be wide
        assert ci[0] < 5.0
        assert ci[1] > 5.0
