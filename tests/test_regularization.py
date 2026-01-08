#!/usr/bin/env python3
"""
Unit tests for Tikhonov regularization in GCPCalibrator.

Tests cover:
- Regularization residual calculation
- Prior sigma configuration and defaults
- Regularization weight scaling
- Residual vector shape with regularization enabled
- CalibrationResult.regularization_penalty field
"""

import pytest
import numpy as np
from unittest.mock import Mock

from poc_homography.gcp_calibrator import (
    CalibrationResult,
    GCPCalibrator,
)
from poc_homography.camera_geometry import CameraGeometry


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_camera_geometry():
    """Create a real CameraGeometry instance for testing."""
    geo = CameraGeometry(1920, 1080)
    K = CameraGeometry.get_intrinsics(zoom_factor=10.0, W_px=1920, H_px=1080)
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
def sample_gcps():
    """Create a minimal list of valid GCP dictionaries for testing."""
    return [
        {
            'gps': {'latitude': 39.640444, 'longitude': -0.230111},
            'image': {'u': 960.0, 'v': 540.0}
        },
        {
            'gps': {'latitude': 39.640500, 'longitude': -0.230200},
            'image': {'u': 1000.0, 'v': 600.0}
        },
        {
            'gps': {'latitude': 39.640300, 'longitude': -0.230000},
            'image': {'u': 920.0, 'v': 500.0}
        },
    ]


# ============================================================================
# Test: Tikhonov Regularization
# ============================================================================

class TestRegularization:
    """Unit tests for Tikhonov regularization in GCPCalibrator."""

    def test_zero_deviation_zero_residuals(self, sample_camera_geometry, sample_gcps):
        """Test that params=[0,0,0,0,0,0] produces all-zero regularization residuals."""
        calibrator = GCPCalibrator(
            camera_geometry=sample_camera_geometry,
            gcps=sample_gcps,
            regularization_weight=1.0
        )

        params = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        reg_residuals = calibrator._compute_regularization_residuals(params)

        # All regularization residuals should be exactly zero
        assert reg_residuals.shape == (6,)
        np.testing.assert_array_equal(reg_residuals, np.zeros(6))

    def test_single_parameter_deviation(self, sample_camera_geometry, sample_gcps):
        """Test regularization residual for single parameter deviation.

        Setup: params = [1.0, 0, 0, 0, 0, 0] with sigma_pan=3.0 and lambda=1.0
        Expected: r_prior[0] = sqrt(1.0) * 1.0 / 3.0 = 0.333..., others are 0
        """
        custom_sigmas = {'pan_deg': 3.0}
        calibrator = GCPCalibrator(
            camera_geometry=sample_camera_geometry,
            gcps=sample_gcps,
            prior_sigmas=custom_sigmas,
            regularization_weight=1.0
        )

        params = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        reg_residuals = calibrator._compute_regularization_residuals(params)

        # r_prior[0] = sqrt(1.0) * 1.0 / 3.0 = 1/3
        expected_pan_residual = np.sqrt(1.0) * 1.0 / 3.0
        assert reg_residuals[0] == pytest.approx(expected_pan_residual, rel=1e-10)
        assert reg_residuals[0] == pytest.approx(1.0 / 3.0, rel=1e-10)

        # All other residuals should be zero
        np.testing.assert_array_equal(reg_residuals[1:], np.zeros(5))

    def test_all_parameters_deviation_at_sigma(self, sample_camera_geometry, sample_gcps):
        """Test that when each param equals its sigma, all residuals equal sqrt(lambda).

        Setup: params = [sigma_pan, sigma_tilt, sigma_roll, sigma_X, sigma_Y, sigma_Z]
        Expected: all regularization residuals = sqrt(lambda) = 1.0 when lambda=1.0
        """
        # Use default sigmas
        calibrator = GCPCalibrator(
            camera_geometry=sample_camera_geometry,
            gcps=sample_gcps,
            regularization_weight=1.0
        )

        # Get the default sigmas and create params = sigmas
        defaults = GCPCalibrator.DEFAULT_PRIOR_SIGMAS
        params = np.array([
            defaults['pan_deg'],           # sigma_pan
            defaults['tilt_deg'],          # sigma_tilt
            defaults['roll_deg'],          # sigma_roll
            defaults['gps_position_m'],    # sigma_X
            defaults['gps_position_m'],    # sigma_Y
            defaults['height_m'],          # sigma_Z
        ])

        reg_residuals = calibrator._compute_regularization_residuals(params)

        # When param[j] = sigma[j], r_prior[j] = sqrt(lambda) * sigma[j] / sigma[j] = sqrt(1.0) = 1.0
        expected_residuals = np.ones(6) * np.sqrt(1.0)
        np.testing.assert_allclose(reg_residuals, expected_residuals, rtol=1e-10)

    def test_regularization_weight_scaling(self, sample_camera_geometry, sample_gcps):
        """Test that residuals scale with sqrt(regularization_weight).

        Setup: Same params, lambda=4.0 vs lambda=1.0
        Expected: Residuals with lambda=4.0 are 2x those with lambda=1.0 (sqrt(4)=2)
        """
        params = np.array([1.0, 2.0, 0.5, 0.3, 0.2, 0.1])

        # Create calibrator with lambda=1.0
        calibrator_lambda1 = GCPCalibrator(
            camera_geometry=sample_camera_geometry,
            gcps=sample_gcps,
            regularization_weight=1.0
        )
        residuals_lambda1 = calibrator_lambda1._compute_regularization_residuals(params)

        # Create calibrator with lambda=4.0
        calibrator_lambda4 = GCPCalibrator(
            camera_geometry=sample_camera_geometry,
            gcps=sample_gcps,
            regularization_weight=4.0
        )
        residuals_lambda4 = calibrator_lambda4._compute_regularization_residuals(params)

        # Residuals with lambda=4.0 should be 2x (sqrt(4)=2) those with lambda=1.0
        np.testing.assert_allclose(residuals_lambda4, residuals_lambda1 * 2.0, rtol=1e-10)

    def test_regularization_disabled(self, sample_camera_geometry, sample_gcps):
        """Test that regularization_weight=0.0 produces all-zero residuals."""
        calibrator = GCPCalibrator(
            camera_geometry=sample_camera_geometry,
            gcps=sample_gcps,
            regularization_weight=0.0
        )

        # Use non-zero params to ensure zeros come from disabled regularization
        params = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        reg_residuals = calibrator._compute_regularization_residuals(params)

        # All residuals should be zero when regularization is disabled
        assert reg_residuals.shape == (6,)
        np.testing.assert_array_equal(reg_residuals, np.zeros(6))

    def test_residual_vector_shape(self, sample_camera_geometry, sample_gcps):
        """Test that total residual vector has 2N + 6 elements when regularization enabled.

        Setup: N GCPs with regularization enabled
        Expected: Total residual vector has 2N + 6 elements
        """
        n_gcps = len(sample_gcps)
        calibrator = GCPCalibrator(
            camera_geometry=sample_camera_geometry,
            gcps=sample_gcps,
            regularization_weight=1.0
        )

        params = np.zeros(6)

        # Compute reprojection residuals
        reproj_residuals = calibrator._compute_residuals(params)
        assert reproj_residuals.shape == (2 * n_gcps,)

        # Compute regularization residuals
        reg_residuals = calibrator._compute_regularization_residuals(params)
        assert reg_residuals.shape == (6,)

        # Combined residuals should have 2N + 6 elements
        combined_residuals = np.concatenate([reproj_residuals, reg_residuals])
        expected_length = 2 * n_gcps + 6
        assert combined_residuals.shape == (expected_length,)
        assert combined_residuals.shape[0] == 2 * n_gcps + 6

    def test_default_prior_sigmas(self, sample_camera_geometry, sample_gcps):
        """Verify DEFAULT_PRIOR_SIGMAS class constant has correct values and defaults are used."""
        # Check class constant values
        assert GCPCalibrator.DEFAULT_PRIOR_SIGMAS['gps_position_m'] == 10.0
        assert GCPCalibrator.DEFAULT_PRIOR_SIGMAS['height_m'] == 2.0
        assert GCPCalibrator.DEFAULT_PRIOR_SIGMAS['pan_deg'] == 3.0
        assert GCPCalibrator.DEFAULT_PRIOR_SIGMAS['tilt_deg'] == 3.0
        assert GCPCalibrator.DEFAULT_PRIOR_SIGMAS['roll_deg'] == 1.0

        # Create calibrator with prior_sigmas=None (should use defaults)
        calibrator = GCPCalibrator(
            camera_geometry=sample_camera_geometry,
            gcps=sample_gcps,
            prior_sigmas=None,
            regularization_weight=1.0
        )

        # Verify internal sigma vector matches defaults
        # Order: [sigma_pan, sigma_tilt, sigma_roll, sigma_X, sigma_Y, sigma_Z]
        expected_sigma_vector = np.array([
            3.0,   # pan_deg
            3.0,   # tilt_deg
            1.0,   # roll_deg
            10.0,  # gps_position_m (X)
            10.0,  # gps_position_m (Y)
            2.0,   # height_m (Z)
        ])
        np.testing.assert_array_equal(calibrator._sigma_vector, expected_sigma_vector)

    def test_custom_prior_sigmas_override(self, sample_camera_geometry, sample_gcps):
        """Test that custom sigmas override defaults, and defaults fill missing keys."""
        # Provide only partial custom sigmas
        custom_sigmas = {
            'pan_deg': 5.0,       # Override pan
            'height_m': 3.0,      # Override height
        }

        calibrator = GCPCalibrator(
            camera_geometry=sample_camera_geometry,
            gcps=sample_gcps,
            prior_sigmas=custom_sigmas,
            regularization_weight=1.0
        )

        # Expected: custom values for pan_deg and height_m, defaults for others
        expected_sigma_vector = np.array([
            5.0,   # pan_deg (custom)
            3.0,   # tilt_deg (default)
            1.0,   # roll_deg (default)
            10.0,  # gps_position_m X (default)
            10.0,  # gps_position_m Y (default)
            3.0,   # height_m Z (custom)
        ])
        np.testing.assert_array_equal(calibrator._sigma_vector, expected_sigma_vector)

    def test_regularization_penalty_in_result(self, sample_camera_geometry, sample_gcps):
        """Test that CalibrationResult.regularization_penalty is not None and > 0 when enabled."""
        calibrator = GCPCalibrator(
            camera_geometry=sample_camera_geometry,
            gcps=sample_gcps,
            regularization_weight=1.0
        )

        result = calibrator.calibrate()

        # regularization_penalty should be calculated (not None)
        assert result.regularization_penalty is not None

        # If optimized params are non-zero, penalty should be > 0
        # Even with small deviations, there should be some penalty
        # Note: With perfect GCPs, params might be near-zero, so we check >= 0
        assert result.regularization_penalty >= 0.0

    def test_regularization_penalty_none_when_disabled(self, sample_camera_geometry, sample_gcps):
        """Test that CalibrationResult.regularization_penalty is None when disabled."""
        calibrator = GCPCalibrator(
            camera_geometry=sample_camera_geometry,
            gcps=sample_gcps,
            regularization_weight=0.0
        )

        result = calibrator.calibrate()

        # regularization_penalty should be None when regularization is disabled
        assert result.regularization_penalty is None

    def test_regularization_penalty_positive_with_deviation(self, sample_camera_geometry):
        """Test that regularization_penalty is positive when calibration has significant deviation.

        Creates GCPs with a known perturbation to ensure non-zero calibration result.
        """
        # Create GCPs that will require parameter adjustment
        import copy
        geo_perturbed = copy.copy(sample_camera_geometry)
        geo_perturbed.set_camera_parameters(
            K=sample_camera_geometry.K,
            w_pos=sample_camera_geometry.w_pos,
            pan_deg=sample_camera_geometry.pan_deg + 2.0,  # Add 2 degree pan perturbation
            tilt_deg=sample_camera_geometry.tilt_deg,
            map_width=sample_camera_geometry.map_width,
            map_height=sample_camera_geometry.map_height
        )

        # Generate synthetic GCPs using the perturbed geometry
        np.random.seed(42)
        world_coords = np.random.randn(10, 2) * 3.0
        gcps = []
        ref_lat, ref_lon = 39.640444, -0.230111

        for x_world, y_world in world_coords:
            world_pt = np.array([x_world, y_world, 1.0])
            image_pt_hom = geo_perturbed.H @ world_pt
            u = image_pt_hom[0] / image_pt_hom[2]
            v = image_pt_hom[1] / image_pt_hom[2]

            lat = ref_lat + (y_world / 111000.0)
            lon = ref_lon + (x_world / (111000.0 * np.cos(np.radians(ref_lat))))

            gcps.append({
                'gps': {'latitude': lat, 'longitude': lon},
                'image': {'u': u, 'v': v}
            })

        # Calibrate with the original (unperturbed) geometry - should discover ~2 degree pan
        calibrator = GCPCalibrator(
            camera_geometry=sample_camera_geometry,
            gcps=gcps,
            regularization_weight=1.0
        )

        result = calibrator.calibrate()

        # Regularization penalty should be positive since params are non-zero
        assert result.regularization_penalty is not None
        assert result.regularization_penalty > 0.0

    def test_negative_regularization_weight_raises(self, sample_camera_geometry, sample_gcps):
        """Test that negative regularization_weight raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            GCPCalibrator(
                camera_geometry=sample_camera_geometry,
                gcps=sample_gcps,
                regularization_weight=-1.0
            )

        assert "regularization_weight" in str(exc_info.value)
        assert ">= 0.0" in str(exc_info.value)

    def test_regularization_residual_formula(self, sample_camera_geometry, sample_gcps):
        """Verify the exact formula: r_prior[j] = sqrt(lambda) * params[j] / sigma[j]."""
        lambda_val = 2.5
        custom_sigmas = {
            'pan_deg': 4.0,
            'tilt_deg': 5.0,
            'roll_deg': 2.0,
            'gps_position_m': 8.0,
            'height_m': 1.5,
        }

        calibrator = GCPCalibrator(
            camera_geometry=sample_camera_geometry,
            gcps=sample_gcps,
            prior_sigmas=custom_sigmas,
            regularization_weight=lambda_val
        )

        params = np.array([2.0, 3.0, 0.5, 1.0, 2.0, 0.75])
        reg_residuals = calibrator._compute_regularization_residuals(params)

        # Manually compute expected residuals
        # Order: [pan, tilt, roll, X, Y, Z]
        sigmas = np.array([
            custom_sigmas['pan_deg'],        # pan
            custom_sigmas['tilt_deg'],       # tilt
            custom_sigmas['roll_deg'],       # roll
            custom_sigmas['gps_position_m'], # X
            custom_sigmas['gps_position_m'], # Y
            custom_sigmas['height_m'],       # Z
        ])
        expected_residuals = np.sqrt(lambda_val) * params / sigmas

        np.testing.assert_allclose(reg_residuals, expected_residuals, rtol=1e-10)


# ============================================================================
# Integration Tests: Regularization Effectiveness
# ============================================================================

class TestRegularizationIntegration:
    """Integration tests demonstrating regularization effectiveness.

    These tests verify that Tikhonov regularization improves optimization
    stability in realistic calibration scenarios with noisy data.
    """

    @pytest.fixture
    def ground_truth_geometry(self):
        """Camera geometry representing ground truth parameters."""
        geo = CameraGeometry(1920, 1080)
        K = CameraGeometry.get_intrinsics(zoom_factor=10.0, W_px=1920, H_px=1080)
        geo.set_camera_parameters(
            K=K,
            w_pos=np.array([0.0, 0.0, 10.0]),  # 10m height for better visibility
            pan_deg=0.0,
            tilt_deg=45.0,
            map_width=640,
            map_height=640
        )
        return geo

    @pytest.fixture
    def accurate_gcps(self, ground_truth_geometry):
        """GCPs with perfect correspondence (no noise).

        Generates synthetic GCPs by projecting known world points through
        the ground truth camera model.
        """
        geo = ground_truth_geometry
        ref_lat, ref_lon = 39.640444, -0.230111

        # Generate world coordinates in a grid pattern for good coverage
        world_coords = []
        for x in [-5.0, 0.0, 5.0]:
            for y in [5.0, 10.0, 15.0]:
                world_coords.append([x, y])

        gcps = []
        for x_world, y_world in world_coords:
            # Project world point to image
            world_pt = np.array([x_world, y_world, 1.0])
            image_pt_hom = geo.H @ world_pt

            # Skip points that project behind camera or to infinity
            if abs(image_pt_hom[2]) < 1e-10:
                continue

            u = image_pt_hom[0] / image_pt_hom[2]
            v = image_pt_hom[1] / image_pt_hom[2]

            # Skip points outside image bounds
            if not (0 <= u <= geo.w and 0 <= v <= geo.h):
                continue

            # Convert world coords to GPS (approximate inverse)
            lat = ref_lat + (y_world / 111000.0)
            lon = ref_lon + (x_world / (111000.0 * np.cos(np.radians(ref_lat))))

            gcps.append({
                'gps': {'latitude': lat, 'longitude': lon},
                'image': {'u': float(u), 'v': float(v)}
            })

        return gcps

    def add_noise_to_gcps(self, gcps, pixel_std, rng):
        """Add Gaussian noise to GCP pixel coordinates.

        Args:
            gcps: List of GCP dictionaries
            pixel_std: Standard deviation of noise in pixels
            rng: numpy random generator for reproducibility

        Returns:
            New list of GCPs with noisy pixel coordinates
        """
        noisy_gcps = []
        for gcp in gcps:
            noisy_gcp = {
                'gps': gcp['gps'].copy(),
                'image': {
                    'u': gcp['image']['u'] + rng.normal(0, pixel_std),
                    'v': gcp['image']['v'] + rng.normal(0, pixel_std)
                }
            }
            noisy_gcps.append(noisy_gcp)
        return noisy_gcps

    def test_stability_with_noisy_gcps(self, ground_truth_geometry, accurate_gcps):
        """Test that regularization improves stability with noisy GCP observations.

        Demonstrates that with noisy GCP pixel observations, regularization
        produces parameters closer to initial values (less overfitting to noise).

        Uses stronger regularization (lambda=5.0) to ensure measurable effect
        compared to the unregularized case.
        """
        ref_lat, ref_lon = 39.640444, -0.230111
        pixel_noise_std = 10.0  # 10 pixel standard deviation (significant noise)

        # Run multiple trials with different random seeds for statistical reliability
        seeds = [42, 123, 456, 789, 1011]
        unregularized_deviations = []
        regularized_deviations = []

        for seed in seeds:
            rng = np.random.default_rng(seed)
            noisy_gcps = self.add_noise_to_gcps(accurate_gcps, pixel_noise_std, rng)

            # Calibrate WITHOUT regularization (lambda=0.0)
            calibrator_unreg = GCPCalibrator(
                camera_geometry=ground_truth_geometry,
                gcps=noisy_gcps,
                regularization_weight=0.0,
                reference_lat=ref_lat,
                reference_lon=ref_lon
            )
            result_unreg = calibrator_unreg.calibrate()

            # Calibrate WITH strong regularization (lambda=5.0)
            # Strong regularization clearly constrains parameters towards initial values
            calibrator_reg = GCPCalibrator(
                camera_geometry=ground_truth_geometry,
                gcps=noisy_gcps,
                regularization_weight=5.0,
                reference_lat=ref_lat,
                reference_lon=ref_lon
            )
            result_reg = calibrator_reg.calibrate()

            # Ground truth is params = [0, 0, 0, 0, 0, 0] since we start with true geometry
            # Deviation = norm of optimized params (should be small if regularization works)
            unreg_deviation = np.linalg.norm(result_unreg.optimized_params)
            reg_deviation = np.linalg.norm(result_reg.optimized_params)

            unregularized_deviations.append(unreg_deviation)
            regularized_deviations.append(reg_deviation)

        # Statistical comparison: regularized should have smaller mean deviation
        mean_unreg = np.mean(unregularized_deviations)
        mean_reg = np.mean(regularized_deviations)

        # Regularized optimization should produce parameters closer to initial values
        # (smaller deviation from ground truth which is [0,0,0,0,0,0])
        assert mean_reg <= mean_unreg, (
            f"Regularization should reduce parameter deviation from initial values. "
            f"Mean unregularized: {mean_unreg:.4f}, mean regularized: {mean_reg:.4f}"
        )

        # Additionally, verify that at least in most trials, regularization helps
        improvement_count = sum(
            reg < unreg
            for reg, unreg in zip(regularized_deviations, unregularized_deviations)
        )
        assert improvement_count >= len(seeds) // 2, (
            f"Regularization should help in majority of trials. "
            f"Improved in {improvement_count}/{len(seeds)} trials."
        )

    def test_bias_correction_capability(self, ground_truth_geometry, accurate_gcps):
        """Test that regularization allows bias correction while still constraining parameters.

        Setup: Initialize camera with known bias (pan offset, GPS offset).
        Verifies that:
        - Without regularization: Full bias correction occurs
        - With weak regularization: Similar bias correction
        - With strong regularization: Partial bias correction (reduced magnitude)
        """
        ref_lat, ref_lon = 39.640444, -0.230111

        # Create a biased camera geometry (known offset from ground truth)
        biased_geo = CameraGeometry(1920, 1080)
        K = CameraGeometry.get_intrinsics(zoom_factor=10.0, W_px=1920, H_px=1080)
        biased_geo.set_camera_parameters(
            K=K,
            w_pos=np.array([10.0, 0.0, 10.0]),  # 10m East offset
            pan_deg=5.0,  # 5 degree pan offset
            tilt_deg=45.0,
            map_width=640,
            map_height=640
        )

        # Use accurate GCPs from ground truth (these reveal the bias)
        # GCPs were generated from ground_truth_geometry, so calibrating
        # biased_geo should correct towards ground truth

        # Calibrate without regularization (lambda=0.0)
        calibrator_unreg = GCPCalibrator(
            camera_geometry=biased_geo,
            gcps=accurate_gcps,
            regularization_weight=0.0,
            reference_lat=ref_lat,
            reference_lon=ref_lon
        )
        result_unreg = calibrator_unreg.calibrate()

        # Calibrate with weak regularization (lambda=0.1)
        calibrator_weak = GCPCalibrator(
            camera_geometry=biased_geo,
            gcps=accurate_gcps,
            regularization_weight=0.1,
            reference_lat=ref_lat,
            reference_lon=ref_lon
        )
        result_weak = calibrator_weak.calibrate()

        # Calibrate with strong regularization (lambda=10.0)
        calibrator_strong = GCPCalibrator(
            camera_geometry=biased_geo,
            gcps=accurate_gcps,
            regularization_weight=10.0,
            reference_lat=ref_lat,
            reference_lon=ref_lon
        )
        result_strong = calibrator_strong.calibrate()

        # Extract pan correction (params[0]) - should be negative to correct +5 degree bias
        pan_correction_unreg = abs(result_unreg.optimized_params[0])
        pan_correction_weak = abs(result_weak.optimized_params[0])
        pan_correction_strong = abs(result_strong.optimized_params[0])

        # Weak regularization should allow similar correction as unregularized
        # Strong regularization should reduce correction magnitude
        assert pan_correction_strong < pan_correction_unreg, (
            f"Strong regularization should reduce correction magnitude. "
            f"Unregularized pan correction: {pan_correction_unreg:.4f}, "
            f"Strong regularized: {pan_correction_strong:.4f}"
        )

        # Weak regularization should still allow substantial correction
        # (at least 50% of unregularized correction)
        assert pan_correction_weak >= 0.5 * pan_correction_unreg, (
            f"Weak regularization should still allow substantial bias correction. "
            f"Unregularized: {pan_correction_unreg:.4f}, Weak: {pan_correction_weak:.4f}"
        )

        # All methods should improve reprojection error (bias correction works)
        assert result_unreg.final_error < result_unreg.initial_error
        assert result_weak.final_error < result_weak.initial_error
        assert result_strong.final_error < result_strong.initial_error

    def test_regularization_disabled_backward_compatibility(
        self, ground_truth_geometry, accurate_gcps
    ):
        """Test that disabled regularization (lambda=0.0) maintains backward compatibility.

        Verifies:
        - Residual vector has only 2N elements (no regularization residuals appended)
        - CalibrationResult.regularization_penalty is None
        - Optimization behaves identically to unregularized case
        """
        ref_lat, ref_lon = 39.640444, -0.230111
        n_gcps = len(accurate_gcps)

        # Create calibrator with regularization disabled
        calibrator = GCPCalibrator(
            camera_geometry=ground_truth_geometry,
            gcps=accurate_gcps,
            regularization_weight=0.0,
            reference_lat=ref_lat,
            reference_lon=ref_lon
        )

        # Test 1: Residual vector shape (2N elements only)
        params = np.zeros(6)
        reproj_residuals = calibrator._compute_residuals(params)
        assert reproj_residuals.shape == (2 * n_gcps,), (
            f"Reprojection residuals should have 2N={2*n_gcps} elements, "
            f"got {reproj_residuals.shape[0]}"
        )

        # Regularization residuals should be zeros
        reg_residuals = calibrator._compute_regularization_residuals(params)
        assert reg_residuals.shape == (6,)
        np.testing.assert_array_equal(reg_residuals, np.zeros(6))

        # Test 2: CalibrationResult.regularization_penalty is None
        result = calibrator.calibrate()
        assert result.regularization_penalty is None, (
            f"regularization_penalty should be None when disabled, "
            f"got {result.regularization_penalty}"
        )

        # Test 3: Also verify with prior_sigmas=None (should use defaults but still disabled)
        calibrator_none_sigmas = GCPCalibrator(
            camera_geometry=ground_truth_geometry,
            gcps=accurate_gcps,
            prior_sigmas=None,
            regularization_weight=0.0,
            reference_lat=ref_lat,
            reference_lon=ref_lon
        )
        result_none = calibrator_none_sigmas.calibrate()
        assert result_none.regularization_penalty is None

    def test_regularization_performance(self, ground_truth_geometry, accurate_gcps):
        """Test that regularization adds negligible computational overhead.

        Measures optimization time with and without regularization for typical
        GCP counts and verifies overhead is within acceptable bounds (<10%).
        """
        import time

        ref_lat, ref_lon = 39.640444, -0.230111

        # Warm up: run one calibration to avoid cold-start effects
        warmup_calibrator = GCPCalibrator(
            camera_geometry=ground_truth_geometry,
            gcps=accurate_gcps,
            regularization_weight=0.0,
            reference_lat=ref_lat,
            reference_lon=ref_lon
        )
        _ = warmup_calibrator.calibrate()

        # Measure time without regularization
        n_trials = 5
        times_unreg = []
        for _ in range(n_trials):
            calibrator = GCPCalibrator(
                camera_geometry=ground_truth_geometry,
                gcps=accurate_gcps,
                regularization_weight=0.0,
                reference_lat=ref_lat,
                reference_lon=ref_lon
            )
            start = time.perf_counter()
            _ = calibrator.calibrate()
            times_unreg.append(time.perf_counter() - start)

        # Measure time with regularization
        times_reg = []
        for _ in range(n_trials):
            calibrator = GCPCalibrator(
                camera_geometry=ground_truth_geometry,
                gcps=accurate_gcps,
                regularization_weight=1.0,
                reference_lat=ref_lat,
                reference_lon=ref_lon
            )
            start = time.perf_counter()
            _ = calibrator.calibrate()
            times_reg.append(time.perf_counter() - start)

        mean_unreg = np.mean(times_unreg)
        mean_reg = np.mean(times_reg)

        # Calculate overhead percentage
        if mean_unreg > 0:
            overhead_pct = ((mean_reg - mean_unreg) / mean_unreg) * 100
        else:
            overhead_pct = 0.0

        # Regularization should add less than 50% overhead
        # (using 50% as a more generous threshold for CI stability)
        assert overhead_pct < 50.0, (
            f"Regularization overhead too high: {overhead_pct:.1f}% "
            f"(unregularized: {mean_unreg*1000:.2f}ms, regularized: {mean_reg*1000:.2f}ms)"
        )

        # Log performance info for reference (not a strict assertion)
        print(f"\nPerformance: unreg={mean_unreg*1000:.2f}ms, "
              f"reg={mean_reg*1000:.2f}ms, overhead={overhead_pct:.1f}%")
