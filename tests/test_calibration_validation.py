#!/usr/bin/env python3
"""
Unit tests for calibration validation metrics and visualization.

Tests cover:
- Train/test split functionality with validation_split parameter
- CalibrationResult extension with validation_metrics
- Systematic error detection (directional bias, radial growth)
- Residual visualization functions
- Validation with pass/fail thresholds
- Validation report generation
"""

# Check if matplotlib is available
import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

requires_matplotlib = pytest.mark.skipif(
    not HAS_MATPLOTLIB, reason="matplotlib is required for visualization tests"
)

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.gcp_calibrator import (
    CalibrationResult,
    GCPCalibrator,
    detect_systematic_errors,
    generate_residual_histogram,
    generate_residual_plot,
    generate_validation_report,
    validate_calibration,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def real_camera_geometry():
    """Create a real CameraGeometry instance for tests."""
    geo = CameraGeometry(1920, 1080)
    K = CameraGeometry.get_intrinsics(zoom_factor=10.0, W_px=1920, H_px=1080)
    geo.set_camera_parameters(
        K=K,
        w_pos=np.array([0.0, 0.0, 5.0]),
        pan_deg=0.0,
        tilt_deg=45.0,
        map_width=640,
        map_height=640,
    )
    return geo


@pytest.fixture
def valid_gcps_large():
    """Create a larger list of valid GCP dictionaries for train/test split testing."""
    # Generate 20 GCPs around a reference point
    ref_map_x = 320.0
    ref_map_y = 320.0

    gcps = []
    rng = np.random.default_rng(42)  # Reproducible without polluting global state
    for i in range(20):
        # Scatter GCPs around reference
        map_x_offset = rng.uniform(-50.0, 50.0)
        map_y_offset = rng.uniform(-50.0, 50.0)

        gcps.append(
            {
                "map_id": "test_map",
                "map_pixel_x": ref_map_x + map_x_offset,
                "map_pixel_y": ref_map_y + map_y_offset,
                "image_u": 960.0 + i * 20,
                "image_v": 540.0 + i * 10,
            }
        )

    return gcps


@pytest.fixture
def sample_calibration_result():
    """Create a sample CalibrationResult for testing visualization/validation functions."""
    params = np.array([1.5, -0.8, 0.0, 0.2, -0.1, 0.05])

    # Per-GCP errors for 20 GCPs (realistic distribution)
    per_gcp_errors = [
        1.2,
        2.3,
        0.8,
        1.5,
        3.1,  # GCPs 0-4
        1.8,
        2.0,
        1.1,
        2.8,
        1.4,  # GCPs 5-9
        0.9,
        1.6,
        2.2,
        1.3,
        2.5,  # GCPs 10-14
        1.7,
        0.7,
        2.9,
        1.0,
        2.1,  # GCPs 15-19
    ]

    validation_metrics = {
        "train": {"rms": 1.8, "p90": 2.7, "max": 3.1, "count": 16},
        "test": {"rms": 2.1, "p90": 2.8, "max": 2.9, "count": 4},
    }

    train_indices = list(range(16))
    test_indices = [16, 17, 18, 19]

    return CalibrationResult(
        optimized_params=params,
        initial_error=8.5,
        final_error=1.8,
        num_inliers=18,
        num_outliers=2,
        inlier_ratio=0.9,
        per_gcp_errors=per_gcp_errors,
        convergence_info={
            "success": True,
            "message": "Optimization converged",
            "iterations": 25,
            "function_evals": 25,
            "optimality": 1e-8,
        },
        validation_metrics=validation_metrics,
        train_indices=train_indices,
        test_indices=test_indices,
    )


@pytest.fixture
def sample_gcps_for_plot():
    """Create sample GCPs with image coordinates for plotting tests."""
    gcps = []
    ref_map_x = 320.0
    ref_map_y = 320.0

    # Create 20 GCPs in a grid pattern
    for i in range(20):
        row = i // 5
        col = i % 5
        gcps.append(
            {
                "map_id": "test_map",
                "map_pixel_x": ref_map_x + col * 30.0,
                "map_pixel_y": ref_map_y + row * 30.0,
                "image_u": 400.0 + col * 200,  # Spread across image width
                "image_v": 200.0 + row * 150,  # Spread across image height
            }
        )

    return gcps


# ============================================================================
# Test: CalibrationResult Extension with Validation Metrics
# ============================================================================


class TestCalibrationResultValidation:
    """Tests for CalibrationResult with validation_metrics, train_indices, test_indices."""

    def test_calibration_result_with_validation_metrics(self):
        """Test CalibrationResult creation with validation_metrics field."""
        params = np.array([1.0, 2.0, 0.5, 0.1, 0.2, -0.3])
        validation_metrics = {
            "train": {"rms": 2.5, "p90": 4.0, "max": 5.5, "count": 15},
            "test": {"rms": 3.0, "p90": 4.5, "max": 6.0, "count": 5},
        }
        train_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        test_indices = [15, 16, 17, 18, 19]

        result = CalibrationResult(
            optimized_params=params,
            initial_error=10.0,
            final_error=2.5,
            num_inliers=18,
            num_outliers=2,
            inlier_ratio=0.9,
            per_gcp_errors=[1.0, 2.0, 3.0],
            convergence_info={"success": True},
            validation_metrics=validation_metrics,
            train_indices=train_indices,
            test_indices=test_indices,
        )

        assert result.validation_metrics == validation_metrics
        assert result.train_indices == train_indices
        assert result.test_indices == test_indices

        # Check train metrics structure
        assert "train" in result.validation_metrics
        assert "test" in result.validation_metrics
        assert result.validation_metrics["train"]["rms"] == 2.5
        assert result.validation_metrics["train"]["p90"] == 4.0
        assert result.validation_metrics["train"]["max"] == 5.5
        assert result.validation_metrics["train"]["count"] == 15

        # Check test metrics structure
        assert result.validation_metrics["test"]["rms"] == 3.0
        assert result.validation_metrics["test"]["p90"] == 4.5
        assert result.validation_metrics["test"]["max"] == 6.0
        assert result.validation_metrics["test"]["count"] == 5

    def test_calibration_result_optional_validation_fields(self):
        """Test CalibrationResult with optional validation fields as None."""
        params = np.array([1.0, 2.0, 0.5, 0.1, 0.2, -0.3])

        result = CalibrationResult(
            optimized_params=params,
            initial_error=10.0,
            final_error=2.5,
            num_inliers=18,
            num_outliers=2,
            inlier_ratio=0.9,
            per_gcp_errors=[1.0, 2.0, 3.0],
            convergence_info={"success": True},
            validation_metrics=None,
            train_indices=None,
            test_indices=None,
        )

        assert result.validation_metrics is None
        assert result.train_indices is None
        assert result.test_indices is None


# ============================================================================
# Test: Train/Test Split Functionality
# ============================================================================


class TestTrainTestSplit:
    """Tests for validation_split parameter and train/test splitting."""

    def test_calibrator_with_validation_split_default(self, real_camera_geometry, valid_gcps_large):
        """Test GCPCalibrator initialization with default validation_split (0.2)."""
        calibrator = GCPCalibrator(
            camera_geometry=real_camera_geometry, gcps=valid_gcps_large, validation_split=0.2
        )

        # Should have validation_split attribute
        assert hasattr(calibrator, "validation_split")
        assert calibrator.validation_split == 0.2

    def test_calibrator_with_validation_split_custom(self, real_camera_geometry, valid_gcps_large):
        """Test GCPCalibrator with custom validation_split."""
        calibrator = GCPCalibrator(
            camera_geometry=real_camera_geometry, gcps=valid_gcps_large, validation_split=0.3
        )

        assert calibrator.validation_split == 0.3

    def test_calibrator_validation_split_out_of_range_high(
        self, real_camera_geometry, valid_gcps_large
    ):
        """Test that validation_split > 0.5 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            GCPCalibrator(
                camera_geometry=real_camera_geometry, gcps=valid_gcps_large, validation_split=0.6
            )

        assert "validation_split must be" in str(exc_info.value)

    def test_calibrator_validation_split_negative(self, real_camera_geometry, valid_gcps_large):
        """Test that negative validation_split raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            GCPCalibrator(
                camera_geometry=real_camera_geometry, gcps=valid_gcps_large, validation_split=-0.1
            )

        assert "validation_split must be" in str(exc_info.value)

    def test_train_test_split_reproducibility(self, real_camera_geometry, valid_gcps_large):
        """Test that train/test split is reproducible with same random seed."""
        # Use tight bounds to prevent camera height violations
        bounds = {
            "pan": (-2.0, 2.0),
            "tilt": (-2.0, 2.0),
            "roll": (-0.1, 0.1),
            "X": (-0.5, 0.5),
            "Y": (-0.5, 0.5),
            "Z": (-0.5, 0.5),
        }

        calibrator1 = GCPCalibrator(
            camera_geometry=real_camera_geometry,
            gcps=valid_gcps_large,
            validation_split=0.2,
            random_seed=42,
        )
        result1 = calibrator1.calibrate(bounds=bounds)

        calibrator2 = GCPCalibrator(
            camera_geometry=real_camera_geometry,
            gcps=valid_gcps_large,
            validation_split=0.2,
            random_seed=42,
        )
        result2 = calibrator2.calibrate(bounds=bounds)

        # Train/test indices should match
        assert result1.train_indices == result2.train_indices
        assert result1.test_indices == result2.test_indices

    def test_train_test_split_no_overlap(self, real_camera_geometry, valid_gcps_large):
        """Test that train and test sets have no overlap."""
        bounds = {
            "pan": (-2.0, 2.0),
            "tilt": (-2.0, 2.0),
            "roll": (-0.1, 0.1),
            "X": (-0.5, 0.5),
            "Y": (-0.5, 0.5),
            "Z": (-0.5, 0.5),
        }

        calibrator = GCPCalibrator(
            camera_geometry=real_camera_geometry,
            gcps=valid_gcps_large,
            validation_split=0.2,
            random_seed=42,
        )
        result = calibrator.calibrate(bounds=bounds)

        # Check no overlap
        train_set = set(result.train_indices)
        test_set = set(result.test_indices)

        assert len(train_set.intersection(test_set)) == 0

    def test_train_test_split_covers_all_gcps(self, real_camera_geometry, valid_gcps_large):
        """Test that train + test sets cover all GCPs."""
        bounds = {
            "pan": (-2.0, 2.0),
            "tilt": (-2.0, 2.0),
            "roll": (-0.1, 0.1),
            "X": (-0.5, 0.5),
            "Y": (-0.5, 0.5),
            "Z": (-0.5, 0.5),
        }

        calibrator = GCPCalibrator(
            camera_geometry=real_camera_geometry,
            gcps=valid_gcps_large,
            validation_split=0.2,
            random_seed=42,
        )
        result = calibrator.calibrate(bounds=bounds)

        # Check union covers all indices
        all_indices = set(result.train_indices).union(set(result.test_indices))
        expected_indices = set(range(len(valid_gcps_large)))

        assert all_indices == expected_indices

    def test_train_test_split_meets_minimum_requirements(
        self, real_camera_geometry, valid_gcps_large
    ):
        """Test that train set has >= 6 GCPs and test set has >= 3 GCPs."""
        bounds = {
            "pan": (-2.0, 2.0),
            "tilt": (-2.0, 2.0),
            "roll": (-0.1, 0.1),
            "X": (-0.5, 0.5),
            "Y": (-0.5, 0.5),
            "Z": (-0.5, 0.5),
        }

        calibrator = GCPCalibrator(
            camera_geometry=real_camera_geometry,
            gcps=valid_gcps_large,
            validation_split=0.2,
            random_seed=42,
        )
        result = calibrator.calibrate(bounds=bounds)

        assert len(result.train_indices) >= 6
        assert len(result.test_indices) >= 3

    def test_insufficient_gcps_skips_split_with_warning(self, real_camera_geometry):
        """Test that insufficient GCPs skips splitting and warns."""
        # Only 8 GCPs - not enough for 6 train + 3 test with 20% split
        few_gcps = [
            {
                "map_id": "test_map",
                "map_pixel_x": 320.0 + i * 10.0,
                "map_pixel_y": 320.0 + i * 10.0,
                "image_u": 960.0 + i * 10,
                "image_v": 540.0 + i * 10,
            }
            for i in range(8)
        ]

        bounds = {
            "pan": (-2.0, 2.0),
            "tilt": (-2.0, 2.0),
            "roll": (-0.1, 0.1),
            "X": (-0.5, 0.5),
            "Y": (-0.5, 0.5),
            "Z": (-0.5, 0.5),
        }

        calibrator = GCPCalibrator(
            camera_geometry=real_camera_geometry,
            gcps=few_gcps,
            validation_split=0.2,
            random_seed=42,
        )
        result = calibrator.calibrate(bounds=bounds)

        # Should validate on training set (all GCPs used for training)
        # Test set should be empty
        assert result.train_indices is not None
        assert len(result.test_indices) == 0

    def test_validation_metrics_computed_for_train_and_test(
        self, real_camera_geometry, valid_gcps_large
    ):
        """Test that validation_metrics contains correct train/test metrics."""
        bounds = {
            "pan": (-2.0, 2.0),
            "tilt": (-2.0, 2.0),
            "roll": (-0.1, 0.1),
            "X": (-0.5, 0.5),
            "Y": (-0.5, 0.5),
            "Z": (-0.5, 0.5),
        }

        calibrator = GCPCalibrator(
            camera_geometry=real_camera_geometry,
            gcps=valid_gcps_large,
            validation_split=0.2,
            random_seed=42,
        )
        result = calibrator.calibrate(bounds=bounds)

        # Should have validation_metrics
        assert result.validation_metrics is not None
        assert "train" in result.validation_metrics
        assert "test" in result.validation_metrics

        # Check train metrics
        train_metrics = result.validation_metrics["train"]
        assert "rms" in train_metrics
        assert "p90" in train_metrics
        assert "max" in train_metrics
        assert "count" in train_metrics
        assert train_metrics["count"] == len(result.train_indices)

        # Check test metrics
        test_metrics = result.validation_metrics["test"]
        assert "rms" in test_metrics
        assert "p90" in test_metrics
        assert "max" in test_metrics
        assert "count" in test_metrics
        assert test_metrics["count"] == len(result.test_indices)

        # All metrics should be non-negative
        for split_name in ["train", "test"]:
            metrics = result.validation_metrics[split_name]
            assert metrics["rms"] >= 0
            assert metrics["p90"] >= 0
            assert metrics["max"] >= 0
            assert metrics["count"] > 0


# ============================================================================
# Test: Metric Computation (RMS, P90, Max)
# ============================================================================


class TestMetricComputation:
    """Tests for metric computation functions."""

    def test_compute_validation_metrics_known_errors(self):
        """Test metric computation with known error values."""
        # Known errors: [1.0, 2.0, 3.0, 4.0, 5.0]
        # RMS = sqrt((1+4+9+16+25)/5) = sqrt(55/5) = sqrt(11) ≈ 3.317
        # P90 = 90th percentile ≈ 4.6
        # Max = 5.0
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        rms = np.sqrt(np.mean(errors**2))
        p90 = np.percentile(errors, 90)
        max_err = np.max(errors)

        assert np.isclose(rms, np.sqrt(11), rtol=1e-10)
        assert np.isclose(p90, 4.6, rtol=1e-10)
        assert max_err == 5.0

    def test_p90_percentile_calculation(self):
        """Test 90th percentile calculation for various distributions."""
        # Linear distribution [0, 1, 2, ..., 99]
        errors = np.arange(100, dtype=float)
        p90 = np.percentile(errors, 90)

        # 90th percentile of [0..99] should be around 89-90
        assert 89.0 <= p90 <= 90.5


# ============================================================================
# Test: Systematic Error Detection
# ============================================================================


class TestSystematicErrorDetection:
    """Tests for detect_systematic_errors() function."""

    def test_detect_systematic_errors_no_patterns_clean_data(self):
        """Test that clean random residuals produce no warnings."""
        # Random residuals with small magnitude, no directional bias
        np.random.seed(42)
        residuals_2d = np.random.randn(20, 2) * 0.5  # Small random errors
        image_coords = np.random.rand(20, 2) * 1000  # Random pixel positions
        image_center = (500.0, 500.0)

        warnings = detect_systematic_errors(residuals_2d, image_coords, image_center)

        # Should return empty list (no warnings)
        assert isinstance(warnings, list)
        assert len(warnings) == 0

    def test_detect_systematic_errors_directional_bias_detected(self):
        """Test detection of directional bias (consistent residual direction)."""
        # Create residuals with strong directional bias (all pointing right-up)
        residuals_2d = np.tile([3.0, 2.0], (20, 1))  # All residuals = [3.0, 2.0]
        image_coords = np.random.rand(20, 2) * 1000
        image_center = (500.0, 500.0)

        warnings = detect_systematic_errors(residuals_2d, image_coords, image_center)

        # Should detect directional bias
        assert len(warnings) > 0
        assert any("directional bias" in w.lower() for w in warnings)
        # Should mention the magnitude
        assert any(
            "3.6" in w or "3.61" in w for w in warnings
        )  # magnitude = sqrt(3^2 + 2^2) = 3.606

    def test_detect_systematic_errors_radial_growth_detected(self):
        """Test detection of radial error growth (errors increase from center)."""
        # Create residuals that grow with distance from center
        image_center = (500.0, 500.0)
        n_points = 30

        # Points arranged in a grid
        image_coords = []
        residuals_2d = []
        for i in range(n_points):
            # Point at varying distances from center
            distance = i * 20  # 0, 20, 40, ...
            u = 500.0 + distance
            v = 500.0 + distance
            image_coords.append([u, v])

            # Residual magnitude proportional to distance
            # Stronger correlation: error = 0.01 * distance
            error_mag = 0.01 * distance
            residuals_2d.append([error_mag, error_mag])

        image_coords = np.array(image_coords)
        residuals_2d = np.array(residuals_2d)

        warnings = detect_systematic_errors(residuals_2d, image_coords, image_center)

        # Should detect radial growth pattern
        assert len(warnings) > 0
        assert any("radial" in w.lower() for w in warnings)

    def test_detect_systematic_errors_both_patterns_detected(self):
        """Test detection when both directional bias and radial growth present."""
        # Combine directional bias and radial pattern
        image_center = (500.0, 500.0)
        n_points = 30

        image_coords = []
        residuals_2d = []
        for i in range(n_points):
            distance = i * 20
            u = 500.0 + distance
            v = 500.0 + distance
            image_coords.append([u, v])

            # Strong directional bias + radial component
            base_error = 2.5  # Directional bias
            radial_error = 0.01 * distance
            residuals_2d.append([base_error + radial_error, base_error + radial_error])

        image_coords = np.array(image_coords)
        residuals_2d = np.array(residuals_2d)

        warnings = detect_systematic_errors(residuals_2d, image_coords, image_center)

        # Should detect both patterns
        assert len(warnings) >= 2
        assert any("directional" in w.lower() for w in warnings)
        assert any("radial" in w.lower() for w in warnings)

    def test_detect_systematic_errors_default_image_center(self):
        """Test that image_center defaults to centroid of image_coords."""
        # Create simple directional bias pattern
        residuals_2d = np.tile([3.0, 2.0], (10, 1))
        image_coords = np.random.rand(10, 2) * 1000

        # Call without image_center (should use centroid)
        warnings = detect_systematic_errors(residuals_2d, image_coords, image_center=None)

        # Should still detect directional bias
        assert len(warnings) > 0
        assert any("directional bias" in w.lower() for w in warnings)

    def test_detect_systematic_errors_weak_patterns_not_flagged(self):
        """Test that weak patterns below thresholds are not flagged."""
        # Weak directional bias (magnitude 1.5 px, threshold is 2.0)
        residuals_2d = np.tile([1.0, 1.0], (20, 1))  # magnitude = sqrt(2) ≈ 1.41
        image_coords = np.random.rand(20, 2) * 1000
        image_center = (500.0, 500.0)

        warnings = detect_systematic_errors(residuals_2d, image_coords, image_center)

        # Should not flag weak bias
        assert len(warnings) == 0


# ============================================================================
# Test: Task 4 - generate_residual_plot()
# ============================================================================


@requires_matplotlib
class TestGenerateResidualPlot:
    """Tests for generate_residual_plot() function."""

    def test_generate_residual_plot_basic_call(
        self, sample_calibration_result, sample_gcps_for_plot
    ):
        """Test basic call to generate_residual_plot with mocked matplotlib."""
        with patch("poc_homography.gcp_calibrator.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            fig = generate_residual_plot(
                calibration_result=sample_calibration_result,
                gcps=sample_gcps_for_plot,
                image_width=1920,
                image_height=1080,
            )

            # Should create figure with subplots
            mock_plt.subplots.assert_called_once()

            # Should return figure
            assert fig == mock_fig

    def test_generate_residual_plot_with_validation_split(
        self, sample_calibration_result, sample_gcps_for_plot
    ):
        """Test generate_residual_plot distinguishes train/test GCPs."""
        with patch("poc_homography.gcp_calibrator.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            generate_residual_plot(
                calibration_result=sample_calibration_result,
                gcps=sample_gcps_for_plot,
                image_width=1920,
                image_height=1080,
            )

            # Should call scatter multiple times (train and test points)
            assert mock_ax.scatter.call_count >= 2

    def test_generate_residual_plot_saves_to_file(
        self, sample_calibration_result, sample_gcps_for_plot
    ):
        """Test that generate_residual_plot can save to file."""
        with patch("poc_homography.gcp_calibrator.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            output_path = "/tmp/residual_plot.png"
            generate_residual_plot(
                calibration_result=sample_calibration_result,
                gcps=sample_gcps_for_plot,
                image_width=1920,
                image_height=1080,
                output_path=output_path,
            )

            # Should save figure
            mock_fig.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")

    def test_generate_residual_plot_no_validation_split(self, sample_gcps_for_plot):
        """Test generate_residual_plot without train/test split (all training data)."""
        # Result without validation_metrics
        result_no_split = CalibrationResult(
            optimized_params=np.array([1.0, 0.5, 0.0, 0.1, 0.0, 0.0]),
            initial_error=5.0,
            final_error=1.5,
            num_inliers=18,
            num_outliers=2,
            inlier_ratio=0.9,
            per_gcp_errors=[1.0, 1.5] * 10,  # 20 errors
            convergence_info={"success": True},
            validation_metrics=None,
            train_indices=None,
            test_indices=None,
        )

        with patch("poc_homography.gcp_calibrator.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            generate_residual_plot(
                calibration_result=result_no_split,
                gcps=sample_gcps_for_plot,
                image_width=1920,
                image_height=1080,
            )

            # Should still create plot (all points as training)
            mock_plt.subplots.assert_called_once()


# ============================================================================
# Test: Task 5 - generate_residual_histogram()
# ============================================================================


@requires_matplotlib
class TestGenerateResidualHistogram:
    """Tests for generate_residual_histogram() function."""

    def test_generate_residual_histogram_basic_call(self, sample_calibration_result):
        """Test basic call to generate_residual_histogram with mocked matplotlib."""
        with patch("poc_homography.gcp_calibrator.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            fig = generate_residual_histogram(calibration_result=sample_calibration_result, bins=20)

            # Should create figure
            mock_plt.subplots.assert_called_once()

            # Should return figure
            assert fig == mock_fig

    def test_generate_residual_histogram_with_validation_split(self, sample_calibration_result):
        """Test generate_residual_histogram creates separate histograms for train/test."""
        with patch("poc_homography.gcp_calibrator.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            generate_residual_histogram(calibration_result=sample_calibration_result, bins=15)

            # Should call hist at least twice (train and test)
            assert mock_ax.hist.call_count >= 2

    def test_generate_residual_histogram_saves_to_file(self, sample_calibration_result):
        """Test that generate_residual_histogram can save to file."""
        with patch("poc_homography.gcp_calibrator.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            output_path = "/tmp/residual_histogram.png"
            generate_residual_histogram(
                calibration_result=sample_calibration_result, bins=20, output_path=output_path
            )

            # Should save figure
            mock_fig.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")

    def test_generate_residual_histogram_custom_bins(self, sample_calibration_result):
        """Test generate_residual_histogram with custom number of bins."""
        with patch("poc_homography.gcp_calibrator.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            generate_residual_histogram(calibration_result=sample_calibration_result, bins=30)

            # Should create histogram with specified bins
            mock_plt.subplots.assert_called_once()


# ============================================================================
# Test: Task 6 - validate_calibration()
# ============================================================================


class TestValidateCalibration:
    """Tests for validate_calibration() function."""

    def test_validate_calibration_pass_all_thresholds(self, sample_calibration_result):
        """Test validate_calibration with result that passes all thresholds."""
        # Set generous thresholds
        thresholds = {
            "train_rms_max": 3.0,  # sample has 1.8
            "test_rms_max": 3.0,  # sample has 2.1
            "train_p90_max": 4.0,  # sample has 2.7
            "test_p90_max": 4.0,  # sample has 2.8
            "min_inlier_ratio": 0.8,  # sample has 0.9
        }

        is_valid, failures = validate_calibration(sample_calibration_result, thresholds)

        # Should pass
        assert is_valid is True
        assert len(failures) == 0

    def test_validate_calibration_fail_train_rms(self, sample_calibration_result):
        """Test validate_calibration fails on train RMS threshold."""
        thresholds = {
            "train_rms_max": 1.0,  # sample has 1.8, should fail
            "test_rms_max": 3.0,
            "train_p90_max": 4.0,
            "test_p90_max": 4.0,
            "min_inlier_ratio": 0.8,
        }

        is_valid, failures = validate_calibration(sample_calibration_result, thresholds)

        # Should fail
        assert is_valid is False
        assert len(failures) > 0
        assert any("train" in f.lower() and "rms" in f.lower() for f in failures)

    def test_validate_calibration_fail_test_rms(self, sample_calibration_result):
        """Test validate_calibration fails on test RMS threshold."""
        thresholds = {
            "train_rms_max": 3.0,
            "test_rms_max": 1.5,  # sample has 2.1, should fail
            "train_p90_max": 4.0,
            "test_p90_max": 4.0,
            "min_inlier_ratio": 0.8,
        }

        is_valid, failures = validate_calibration(sample_calibration_result, thresholds)

        # Should fail
        assert is_valid is False
        assert len(failures) > 0
        assert any("test" in f.lower() and "rms" in f.lower() for f in failures)

    def test_validate_calibration_fail_inlier_ratio(self, sample_calibration_result):
        """Test validate_calibration fails on inlier ratio threshold."""
        thresholds = {
            "train_rms_max": 3.0,
            "test_rms_max": 3.0,
            "train_p90_max": 4.0,
            "test_p90_max": 4.0,
            "min_inlier_ratio": 0.95,  # sample has 0.9, should fail
        }

        is_valid, failures = validate_calibration(sample_calibration_result, thresholds)

        # Should fail
        assert is_valid is False
        assert len(failures) > 0
        assert any("inlier" in f.lower() and "ratio" in f.lower() for f in failures)

    def test_validate_calibration_multiple_failures(self, sample_calibration_result):
        """Test validate_calibration with multiple threshold failures."""
        thresholds = {
            "train_rms_max": 1.0,  # Fail
            "test_rms_max": 1.0,  # Fail
            "train_p90_max": 2.0,  # Fail
            "test_p90_max": 2.0,  # Fail
            "min_inlier_ratio": 0.95,  # Fail
        }

        is_valid, failures = validate_calibration(sample_calibration_result, thresholds)

        # Should fail with multiple reasons
        assert is_valid is False
        assert len(failures) >= 3  # At least 3 failures

    def test_validate_calibration_no_validation_metrics(self):
        """Test validate_calibration with result that has no validation_metrics (all training data)."""
        result_no_split = CalibrationResult(
            optimized_params=np.array([1.0, 0.5, 0.0, 0.1, 0.0, 0.0]),
            initial_error=5.0,
            final_error=1.5,
            num_inliers=18,
            num_outliers=2,
            inlier_ratio=0.9,
            per_gcp_errors=[1.0, 1.5] * 10,
            convergence_info={"success": True},
            validation_metrics=None,
            train_indices=None,
            test_indices=None,
        )

        thresholds = {"train_rms_max": 3.0, "test_rms_max": 3.0, "min_inlier_ratio": 0.8}

        # Should handle missing validation_metrics gracefully
        # (skip train/test checks, only check inlier_ratio and final_error)
        is_valid, failures = validate_calibration(result_no_split, thresholds)

        # Should pass inlier_ratio check
        assert is_valid is True or "train_rms" not in str(failures)

    def test_validate_calibration_default_thresholds(self, sample_calibration_result):
        """Test validate_calibration with default thresholds (None)."""
        # Should use reasonable defaults
        is_valid, failures = validate_calibration(sample_calibration_result, thresholds=None)

        # Should execute without error and return valid result
        assert isinstance(is_valid, bool)
        assert isinstance(failures, list)


# ============================================================================
# Test: Task 7 - generate_validation_report()
# ============================================================================


class TestGenerateValidationReport:
    """Tests for generate_validation_report() function."""

    def test_generate_validation_report_basic_structure(self, sample_calibration_result):
        """Test that generate_validation_report produces well-structured report."""
        report = generate_validation_report(sample_calibration_result)

        # Should return non-empty string
        assert isinstance(report, str)
        assert len(report) > 0

        # Should contain key sections
        assert "Calibration Validation Report" in report
        assert "Optimized Parameters" in report
        assert "Error Metrics" in report
        assert "Convergence" in report

    def test_generate_validation_report_with_validation_metrics(self, sample_calibration_result):
        """Test report includes train/test metrics when available."""
        report = generate_validation_report(sample_calibration_result)

        # Should include train/test sections
        assert "Train Set" in report or "Training" in report
        assert "Test Set" in report or "Testing" in report

        # Should include metric values
        assert "RMS" in report
        assert "P90" in report
        assert "Max" in report

    def test_generate_validation_report_without_validation_metrics(self):
        """Test report handles missing validation_metrics gracefully."""
        result_no_split = CalibrationResult(
            optimized_params=np.array([1.0, 0.5, 0.0, 0.1, 0.0, 0.0]),
            initial_error=5.0,
            final_error=1.5,
            num_inliers=18,
            num_outliers=2,
            inlier_ratio=0.9,
            per_gcp_errors=[1.0, 1.5] * 10,
            convergence_info={"success": True},
            validation_metrics=None,
            train_indices=None,
            test_indices=None,
        )

        report = generate_validation_report(result_no_split)

        # Should still generate valid report
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Calibration Validation Report" in report

    def test_generate_validation_report_includes_parameter_values(self, sample_calibration_result):
        """Test report includes optimized parameter values."""
        report = generate_validation_report(sample_calibration_result)

        # Should include parameter names
        assert "pan" in report.lower() or "Pan" in report
        assert "tilt" in report.lower() or "Tilt" in report

        # Should include numeric values (check for degrees symbol or 'deg')
        assert "°" in report or "deg" in report

    def test_generate_validation_report_includes_validation_status(self, sample_calibration_result):
        """Test report includes validation pass/fail status when thresholds provided."""
        thresholds = {"train_rms_max": 3.0, "test_rms_max": 3.0, "min_inlier_ratio": 0.8}

        report = generate_validation_report(
            sample_calibration_result, validation_thresholds=thresholds
        )

        # Should include validation status
        assert "PASS" in report or "FAIL" in report or "Status" in report


# ============================================================================
# Test: Task 8 - Integration Tests
# ============================================================================


@requires_matplotlib
class TestCalibrationValidationIntegration:
    """Integration tests for complete calibration validation workflow."""

    def test_complete_validation_workflow(self, real_camera_geometry, valid_gcps_large):
        """Test complete workflow: calibrate -> visualize -> validate -> report."""
        # Step 1: Calibrate with validation split
        bounds = {
            "pan": (-2.0, 2.0),
            "tilt": (-2.0, 2.0),
            "roll": (-0.1, 0.1),
            "X": (-0.5, 0.5),
            "Y": (-0.5, 0.5),
            "Z": (-0.5, 0.5),
        }

        calibrator = GCPCalibrator(
            camera_geometry=real_camera_geometry,
            gcps=valid_gcps_large,
            validation_split=0.2,
            random_seed=42,
        )
        result = calibrator.calibrate(bounds=bounds)

        # Step 2: Generate visualizations
        with patch("poc_homography.gcp_calibrator.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            fig_plot = generate_residual_plot(
                calibration_result=result,
                gcps=valid_gcps_large,
                image_width=1920,
                image_height=1080,
            )

            fig_hist = generate_residual_histogram(calibration_result=result, bins=20)

            assert fig_plot is not None
            assert fig_hist is not None

        # Step 3: Validate with thresholds
        thresholds = {
            "train_rms_max": 10.0,  # Generous for synthetic data
            "test_rms_max": 10.0,
            "min_inlier_ratio": 0.5,
        }
        is_valid, failures = validate_calibration(result, thresholds)

        assert isinstance(is_valid, bool)
        assert isinstance(failures, list)

        # Step 4: Generate report
        report = generate_validation_report(result, validation_thresholds=thresholds)

        assert isinstance(report, str)
        assert len(report) > 0

    def test_validation_workflow_without_split(self, real_camera_geometry, valid_gcps_large):
        """Test validation workflow without train/test split (validation_split=0)."""
        bounds = {
            "pan": (-2.0, 2.0),
            "tilt": (-2.0, 2.0),
            "roll": (-0.1, 0.1),
            "X": (-0.5, 0.5),
            "Y": (-0.5, 0.5),
            "Z": (-0.5, 0.5),
        }

        # Calibrate without validation split
        calibrator = GCPCalibrator(
            camera_geometry=real_camera_geometry,
            gcps=valid_gcps_large,
            validation_split=0.0,  # No split
        )
        result = calibrator.calibrate(bounds=bounds)

        # Should still be able to generate visualizations and reports
        with patch("poc_homography.gcp_calibrator.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            fig_plot = generate_residual_plot(
                calibration_result=result,
                gcps=valid_gcps_large,
                image_width=1920,
                image_height=1080,
            )

            fig_hist = generate_residual_histogram(result, bins=20)

            assert fig_plot is not None
            assert fig_hist is not None

        report = generate_validation_report(result)
        assert isinstance(report, str)
        assert len(report) > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    """
    Run tests with pytest.

    Usage:
        python test_calibration_validation.py
        pytest test_calibration_validation.py -v
    """
    pytest.main([__file__, "-v", "--tb=short"])
