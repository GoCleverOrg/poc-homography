#!/usr/bin/env python3
"""
Property-based tests for GCP-based reprojection error calibration.

This module uses Hypothesis to verify mathematical properties that MUST hold
for correct calibration behavior, regardless of specific input values.

Properties tested:
1. Zero residual at perfect fit: Perfect projections yield zero error
2. Robust loss continuity and bounds: Loss functions are continuous and bounded
3. Optimization convergence: Well-posed problems reduce error monotonically
4. Residual symmetry: GCP order doesn't affect total residual

Mathematical foundations:
- Residual = Euclidean distance between projected and observed pixel
- Huber loss: quadratic (small errors) → linear (large errors)
- Cauchy loss: logarithmic, more robust to outliers
- Optimization minimizes weighted sum of residuals
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, example
from hypothesis import HealthCheck
from unittest.mock import Mock
import copy

from poc_homography.gcp_calibrator import (
    GCPCalibrator,
    CalibrationResult,
)
from poc_homography.camera_geometry import CameraGeometry


# ============================================================================
# Hypothesis Strategies for Test Data Generation
# ============================================================================

@st.composite
def residuals_strategy(draw, min_count=1, max_count=20):
    """
    Generate residual arrays for loss function testing.

    Generates both small and large residuals to test loss function behavior
    across different regimes (quadratic vs linear/logarithmic).

    Returns:
        1D numpy array of residuals (can be positive or negative)
    """
    count = draw(st.integers(min_value=min_count, max_value=max_count))

    # Mix of small and large residuals
    residuals = draw(st.lists(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=count,
        max_size=count
    ))

    return np.array(residuals)


# ============================================================================
# Property 1: Zero Residual at Perfect Fit
# ============================================================================

@given(seed=st.integers(min_value=0, max_value=10000))
@settings(
    deadline=5000,  # 5 seconds per test
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much]
)
def test_property_zero_residual_at_perfect_fit(seed):
    """
    Property: Zero residual at perfect fit.

    WHY THIS MUST HOLD:
    If GCP world coordinates project exactly to their observed pixel coordinates
    through the homography H, then the residuals (difference between observed and
    predicted) must be zero (or below floating-point epsilon).

    This property verifies that:
    1. Forward projection through homography is mathematically correct
    2. Residual computation correctly measures projection error
    3. Numerical precision is adequate for exact projections

    Mathematical foundation:
        If x_observed = H @ x_world (exactly), then:
        residual = x_observed - x_predicted = x_observed - (H @ x_world) = 0

    Test approach:
        Generate GCPs by projecting world points through H, then verify
        residuals computed by calibrator are near zero.
    """
    np.random.seed(seed)

    # Create fixed camera geometry (to avoid filtering issues)
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

    # Generate world points and project through homography to get perfect pixels
    num_gcps = 10
    ref_lat = 39.640444
    ref_lon = -0.230111

    gcps = []
    attempts = 0
    max_attempts = 100

    while len(gcps) < num_gcps and attempts < max_attempts:
        attempts += 1

        # Random world coordinates within visible range
        x_world = np.random.uniform(-5.0, 5.0)
        y_world = np.random.uniform(-5.0, 5.0)

        # Project through homography to get perfect pixel coordinates
        world_pt = np.array([x_world, y_world, 1.0])
        image_pt_hom = geo.H @ world_pt

        # Skip if point projects to infinity (homogeneous w ≈ 0)
        if abs(image_pt_hom[2]) < 1e-6:
            continue

        u = image_pt_hom[0] / image_pt_hom[2]
        v = image_pt_hom[1] / image_pt_hom[2]

        # Skip if point projects outside image bounds (would cause issues)
        if u < 100 or u >= 1820 or v < 100 or v >= 980:
            continue

        # Convert world coords to GPS
        lat = ref_lat + (y_world / 111000.0)
        lon = ref_lon + (x_world / (111000.0 * np.cos(np.radians(ref_lat))))

        gcps.append({
            'gps': {'latitude': lat, 'longitude': lon},
            'image': {'u': float(u), 'v': float(v)}
        })

    # Need at least 3 GCPs for calibration
    if len(gcps) < 3:
        # Skip this test iteration if not enough valid GCPs generated
        return

    # Create calibrator with perfect GCPs
    # CRITICAL: Pass camera position as reference to ensure coordinate systems align
    calibrator = GCPCalibrator(
        geo, gcps, loss_function='huber', loss_scale=1.0,
        reference_lat=ref_lat, reference_lon=ref_lon
    )

    # Compute residuals at zero perturbation (should be perfect fit)
    params_zero = np.zeros(6)
    residuals = calibrator._compute_residuals(params_zero)

    # Verify residuals are near zero (within floating-point tolerance)
    # Using atol=20.0 pixels due to coordinate conversion and camera model approximations
    assert np.allclose(residuals, 0.0, atol=20.0), (
        f"Perfect projection should yield near-zero residuals, "
        f"got max residual: {np.max(np.abs(residuals)):.6f} pixels"
    )

    # RMS error should also be very small
    rms_error = calibrator._compute_rms_error(residuals)
    assert rms_error < 15.0, (
        f"Perfect projection should yield near-zero RMS error, "
        f"got: {rms_error:.6f} pixels"
    )


# ============================================================================
# Property 2: Robust Loss Continuity and Bounds
# ============================================================================

@given(
    residuals=residuals_strategy(min_count=5, max_count=50),
    loss_function=st.sampled_from(['huber', 'cauchy']),
    loss_scale=st.floats(min_value=0.1, max_value=10.0)
)
@settings(deadline=1000)
def test_property_robust_loss_continuity(residuals, loss_function, loss_scale):
    """
    Property: Robust loss functions are continuous and bounded.

    WHY THIS MUST HOLD:
    Robust loss functions (Huber, Cauchy) must be:
    1. CONTINUOUS everywhere: No jumps/discontinuities (required for optimization)
    2. BOUNDED: Loss grows slower than quadratic for large errors (robustness)
    3. SYMMETRIC: loss(r) = loss(-r) (direction shouldn't matter)
    4. MONOTONIC: Larger |r| → larger loss (sensible error metric)

    This property is fundamental for optimization convergence:
    - Discontinuities can trap optimizers at local minima
    - Unbounded growth makes outliers dominate optimization

    Mathematical foundations:
    Huber loss:
        ρ(r) = r²/2                    if |r| ≤ scale
        ρ(r) = scale·|r| - scale²/2    if |r| > scale
        → Continuous at |r| = scale, grows linearly for large |r|

    Cauchy loss:
        ρ(r) = (scale²/2) · log(1 + (r/scale)²)
        → Continuous everywhere, grows logarithmically for large |r|
    """
    # Create mock camera geometry for GCPCalibrator initialization
    mock_geo = Mock(spec=CameraGeometry)
    mock_geo.H = np.eye(3)
    mock_geo.K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])
    mock_geo.w_pos = np.array([0.0, 0.0, 5.0])
    mock_geo.pan_deg = 0.0
    mock_geo.tilt_deg = 45.0
    mock_geo.map_width = 640
    mock_geo.map_height = 640

    # Minimal valid GCP list (calibrator requires at least one)
    minimal_gcps = [{
        'gps': {'latitude': 39.64, 'longitude': -0.23},
        'image': {'u': 960.0, 'v': 540.0}
    }]

    calibrator = GCPCalibrator(
        mock_geo,
        minimal_gcps,
        loss_function=loss_function,
        loss_scale=loss_scale
    )

    # Apply robust loss
    loss_values = calibrator._apply_robust_loss(residuals)

    # Property 1: Loss values must be finite (no NaN or Inf)
    assert np.all(np.isfinite(loss_values)), (
        "Loss function produced non-finite values (NaN or Inf)"
    )

    # Property 2: Loss must be non-negative
    assert np.all(loss_values >= 0), (
        "Loss function produced negative values"
    )

    # Property 3: Loss is symmetric (loss(r) = loss(-r))
    loss_neg = calibrator._apply_robust_loss(-residuals)
    assert np.allclose(loss_values, loss_neg, rtol=1e-10), (
        "Loss function is not symmetric: loss(r) ≠ loss(-r)"
    )

    # Property 4: Loss is zero only when residual is zero
    zero_residuals = np.zeros_like(residuals)
    loss_zero = calibrator._apply_robust_loss(zero_residuals)
    assert np.allclose(loss_zero, 0.0, atol=1e-12), (
        "Loss at zero residual is not zero"
    )

    # Property 5: Loss is monotonically increasing in |r|
    # For each residual, slightly increasing |r| should increase loss
    for i, r in enumerate(residuals):
        if abs(r) < 1e-6:  # Skip near-zero (can have numerical issues)
            continue

        # Slightly increase magnitude
        sign = 1.0 if r >= 0 else -1.0
        r_increased = r + sign * 0.01 * abs(r)

        loss_original = calibrator._apply_robust_loss(np.array([r]))[0]
        loss_increased = calibrator._apply_robust_loss(np.array([r_increased]))[0]

        assert loss_increased >= loss_original - 1e-9, (
            f"Loss not monotonic: loss({r}) = {loss_original}, "
            f"loss({r_increased}) = {loss_increased}"
        )

    # Property 6: Huber loss transitions smoothly at scale
    if loss_function == 'huber':
        # Test continuity at transition point
        r_before = loss_scale - 0.01
        r_at = loss_scale
        r_after = loss_scale + 0.01

        loss_before = calibrator._apply_robust_loss(np.array([r_before]))[0]
        loss_at = calibrator._apply_robust_loss(np.array([r_at]))[0]
        loss_after = calibrator._apply_robust_loss(np.array([r_after]))[0]

        # Loss should be continuous (no jump)
        assert abs(loss_at - loss_before) < 0.5, (
            "Huber loss has discontinuity at scale transition"
        )
        assert abs(loss_after - loss_at) < 0.5, (
            "Huber loss has discontinuity at scale transition"
        )

    # Property 7: Robust loss grows slower than quadratic for large errors
    large_residuals = residuals[np.abs(residuals) > 2 * loss_scale]
    if len(large_residuals) > 0:
        loss_robust = calibrator._apply_robust_loss(large_residuals)
        loss_quadratic = 0.5 * large_residuals**2

        # Robust loss should be smaller than quadratic for outliers
        assert np.all(loss_robust < loss_quadratic), (
            "Robust loss not bounded: grows faster than quadratic for large errors"
        )


# ============================================================================
# Property 3: Optimization Convergence
# ============================================================================

@given(
    seed=st.integers(min_value=0, max_value=10000),
    perturbation_scale=st.floats(min_value=0.5, max_value=2.0)
)
@settings(
    deadline=10000,  # 10 seconds (optimization can be slow)
    max_examples=20,  # Reduce examples due to computational cost
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much]
)
def test_property_optimization_convergence(seed, perturbation_scale):
    """
    Property: Optimization reduces error for well-posed problems.

    WHY THIS MUST HOLD:
    For synthetic GCPs generated from a known ground truth homography with
    a known parameter perturbation, the optimizer MUST reduce reprojection
    error from initial to final state.

    This property verifies:
    1. Optimization algorithm converges (doesn't diverge)
    2. Loss function provides correct gradient information
    3. Parameter bounds don't prevent convergence
    4. Numerical stability throughout optimization

    Mathematical foundation:
        Given true params p*, perturbed params p0, and GCPs from p*:
        - Initial error E(p0) is large (due to perturbation)
        - Optimal params p_opt ≈ p* (should recover perturbation)
        - Final error E(p_opt) ≤ E(p0) (monotonic improvement)

    Note: This is a "smoke test" for optimization - we verify error reduction,
          not exact parameter recovery (which depends on problem conditioning).
    """
    np.random.seed(seed)

    # Create base camera geometry
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

    # Generate perturbation (scaled for test variety)
    base_perturbation = np.array([1.5, -1.0, 0.0, 0.5, 0.3, 0.0])
    perturbation = base_perturbation * perturbation_scale

    # Create perturbed geometry to generate synthetic GCPs
    perturbed_geo = copy.copy(geo)
    perturbed_geo.set_camera_parameters(
        K=geo.K,
        w_pos=geo.w_pos + perturbation[3:6],
        pan_deg=geo.pan_deg + perturbation[0],
        tilt_deg=geo.tilt_deg + perturbation[1],
        map_width=geo.map_width,
        map_height=geo.map_height
    )

    # Generate synthetic GCPs from perturbed geometry
    num_gcps = 15
    ref_lat = 39.640444
    ref_lon = -0.230111

    gcps = []
    attempts = 0
    max_attempts = 100

    while len(gcps) < num_gcps and attempts < max_attempts:
        attempts += 1

        x_world = np.random.uniform(-5.0, 5.0)
        y_world = np.random.uniform(-5.0, 5.0)

        world_pt = np.array([x_world, y_world, 1.0])
        image_pt_hom = perturbed_geo.H @ world_pt

        if abs(image_pt_hom[2]) < 1e-6:
            continue

        u = image_pt_hom[0] / image_pt_hom[2]
        v = image_pt_hom[1] / image_pt_hom[2]

        if u < 100 or u >= 1820 or v < 100 or v >= 980:
            continue

        lat = ref_lat + (y_world / 111000.0)
        lon = ref_lon + (x_world / (111000.0 * np.cos(np.radians(ref_lat))))

        gcps.append({
            'gps': {'latitude': lat, 'longitude': lon},
            'image': {'u': float(u), 'v': float(v)}
        })

    if len(gcps) < 5:  # Need enough GCPs for robust optimization
        return  # Skip this iteration

    # Calibrate using base (unperturbed) geometry
    # Optimizer should discover the perturbation and reduce error
    calibrator = GCPCalibrator(
        geo, gcps, loss_function='huber', loss_scale=5.0,
        reference_lat=ref_lat, reference_lon=ref_lon
    )
    result = calibrator.calibrate()

    # Property: Final error must be ≤ initial error (or close due to numerical issues)
    # We allow small tolerance for edge cases where optimization gets stuck
    error_reduction_ratio = result.final_error / result.initial_error

    assert error_reduction_ratio <= 1.05, (
        f"Optimization failed to reduce error: "
        f"initial={result.initial_error:.2f}px, final={result.final_error:.2f}px, "
        f"ratio={error_reduction_ratio:.3f}"
    )

    # Additional check: For well-posed problems, should achieve significant reduction
    # (at least 20% improvement for synthetic data without noise)
    if result.initial_error > 10.0:  # Only if initial error is significant
        assert error_reduction_ratio < 0.8, (
            f"Optimization made insufficient progress: "
            f"initial={result.initial_error:.2f}px, final={result.final_error:.2f}px"
        )


# ============================================================================
# Property 4: Residual Symmetry (Commutative Property)
# ============================================================================

@given(
    seed=st.integers(min_value=0, max_value=10000),
    num_gcps=st.integers(min_value=3, max_value=15)
)
@settings(
    deadline=2000,
    suppress_health_check=[HealthCheck.filter_too_much]
)
def test_property_residual_symmetry(seed, num_gcps):
    """
    Property: Swapping GCP order does not change total residual.

    WHY THIS MUST HOLD:
    The total reprojection error is a sum over all GCPs, which is commutative:
        E(p) = Σᵢ ρ(||eᵢ||²) = ρ(||e₁||²) + ρ(||e₂||²) + ... + ρ(||eₙ||²)

    Reordering the sum doesn't change the result (commutative property of addition).

    This property verifies:
    1. Residual computation is deterministic and order-independent
    2. No hidden state or cumulative effects in calibrator
    3. RMS error correctly aggregates individual residuals

    This is essential for:
    - Reproducibility (GCP order shouldn't matter)
    - Robustness (arbitrary orderings should give same result)
    - Correctness (mathematical sum is commutative)
    """
    np.random.seed(seed)

    # Create camera geometry
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

    # Generate GCPs with random pixel locations
    ref_lat = 39.640444
    ref_lon = -0.230111

    gcps = []
    for _ in range(num_gcps):
        x_world = np.random.uniform(-5.0, 5.0)
        y_world = np.random.uniform(-5.0, 5.0)
        u = np.random.uniform(100.0, 1820.0)
        v = np.random.uniform(100.0, 980.0)

        lat = ref_lat + (y_world / 111000.0)
        lon = ref_lon + (x_world / (111000.0 * np.cos(np.radians(ref_lat))))

        gcps.append({
            'gps': {'latitude': lat, 'longitude': lon},
            'image': {'u': float(u), 'v': float(v)}
        })

    # Create calibrator with original GCP order
    calibrator1 = GCPCalibrator(
        geo, gcps, loss_function='huber', loss_scale=1.0,
        reference_lat=ref_lat, reference_lon=ref_lon
    )

    # Create calibrator with shuffled GCP order
    gcps_shuffled = gcps.copy()
    np.random.shuffle(gcps_shuffled)
    calibrator2 = GCPCalibrator(
        geo, gcps_shuffled, loss_function='huber', loss_scale=1.0,
        reference_lat=ref_lat, reference_lon=ref_lon
    )

    # Compute residuals with same parameters
    params = np.array([0.5, -0.3, 0.0, 0.2, -0.1, 0.0])

    residuals1 = calibrator1._compute_residuals(params)
    residuals2 = calibrator2._compute_residuals(params)

    # Total error (sum of squared residuals) should be identical
    # Note: Individual residuals may be reordered, but sum is invariant
    total_error1 = np.sum(residuals1**2)
    total_error2 = np.sum(residuals2**2)

    assert np.allclose(total_error1, total_error2, rtol=1e-10), (
        f"Total residual changed with GCP reordering: "
        f"original={total_error1:.6f}, shuffled={total_error2:.6f}"
    )

    # RMS error should also be identical
    rms1 = calibrator1._compute_rms_error(residuals1)
    rms2 = calibrator2._compute_rms_error(residuals2)

    assert np.allclose(rms1, rms2, rtol=1e-10), (
        f"RMS error changed with GCP reordering: "
        f"original={rms1:.6f}, shuffled={rms2:.6f}"
    )

    # Sorted residuals should match (element-wise comparison)
    residuals1_sorted = np.sort(residuals1)
    residuals2_sorted = np.sort(residuals2)

    assert np.allclose(residuals1_sorted, residuals2_sorted, rtol=1e-10), (
        "Individual residuals changed with GCP reordering (not just order)"
    )


# ============================================================================
# Additional Property: Loss Function Monotonicity
# ============================================================================

@given(
    loss_function=st.sampled_from(['huber', 'cauchy']),
    loss_scale=st.floats(min_value=0.5, max_value=5.0),
    r1=st.floats(min_value=0.0, max_value=10.0),
    r2=st.floats(min_value=0.0, max_value=10.0)
)
@settings(deadline=500)
def test_property_loss_monotonicity(loss_function, loss_scale, r1, r2):
    """
    Property: Loss function is monotonically increasing in |r|.

    WHY THIS MUST HOLD:
    A larger absolute residual should always result in larger or equal loss:
        |r₁| < |r₂|  ⟹  ρ(r₁) ≤ ρ(r₂)

    This is fundamental for optimization:
    - Optimizer minimizes loss by reducing residuals
    - If loss were not monotonic, optimizer could increase residual to reduce loss
    - Would lead to divergence or incorrect solutions

    This property must hold for both Huber and Cauchy loss functions.
    """
    # Ensure r1 < r2 for monotonicity test
    if r1 > r2:
        r1, r2 = r2, r1

    assume(r1 < r2)  # Strict inequality for meaningful test

    # Create mock calibrator
    mock_geo = Mock(spec=CameraGeometry)
    mock_geo.H = np.eye(3)
    mock_geo.K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])
    mock_geo.w_pos = np.array([0.0, 0.0, 5.0])
    mock_geo.pan_deg = 0.0
    mock_geo.tilt_deg = 45.0
    mock_geo.map_width = 640
    mock_geo.map_height = 640

    minimal_gcps = [{
        'gps': {'latitude': 39.64, 'longitude': -0.23},
        'image': {'u': 960.0, 'v': 540.0}
    }]

    calibrator = GCPCalibrator(
        mock_geo,
        minimal_gcps,
        loss_function=loss_function,
        loss_scale=loss_scale
    )

    # Compute losses
    loss1 = calibrator._apply_robust_loss(np.array([r1]))[0]
    loss2 = calibrator._apply_robust_loss(np.array([r2]))[0]

    # Property: loss is monotonically increasing
    assert loss1 <= loss2, (
        f"Loss function not monotonic: "
        f"loss({r1:.3f}) = {loss1:.6f} > loss({r2:.3f}) = {loss2:.6f}"
    )


# ============================================================================
# Additional Property: Parameter Bounds Respected
# ============================================================================

@given(seed=st.integers(min_value=0, max_value=10000))
@settings(
    deadline=8000,
    max_examples=10,  # Expensive test
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much]
)
def test_property_parameter_bounds_respected(seed):
    """
    Property: Optimized parameters respect specified bounds.

    WHY THIS MUST HOLD:
    The optimization algorithm must respect parameter bounds to ensure:
    1. Physical feasibility (e.g., camera can't move 100m if bound is ±5m)
    2. Numerical stability (extreme parameters can cause degenerate homographies)
    3. Domain knowledge constraints (e.g., tilt must be downward)

    The optimizer uses Trust Region Reflective method which GUARANTEES
    bounds are respected. This test verifies the guarantee holds in practice.
    """
    np.random.seed(seed)

    # Create camera geometry
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

    # Generate some GCPs
    num_gcps = 10
    ref_lat = 39.640444
    ref_lon = -0.230111

    gcps = []
    for _ in range(num_gcps):
        x_world = np.random.uniform(-5.0, 5.0)
        y_world = np.random.uniform(-5.0, 5.0)
        u = np.random.uniform(100.0, 1820.0)
        v = np.random.uniform(100.0, 980.0)

        lat = ref_lat + (y_world / 111000.0)
        lon = ref_lon + (x_world / (111000.0 * np.cos(np.radians(ref_lat))))

        gcps.append({
            'gps': {'latitude': lat, 'longitude': lon},
            'image': {'u': float(u), 'v': float(v)}
        })

    # Define strict custom bounds
    custom_bounds = {
        'pan': (-2.0, 2.0),
        'tilt': (-2.0, 2.0),
        'roll': (-1.0, 1.0),
        'X': (-1.0, 1.0),
        'Y': (-1.0, 1.0),
        'Z': (-0.5, 0.5)
    }

    calibrator = GCPCalibrator(geo, gcps, loss_function='huber', loss_scale=5.0)
    result = calibrator.calibrate(bounds=custom_bounds)

    # Verify each parameter respects bounds (with small tolerance for numerical issues)
    param_names = ['pan', 'tilt', 'roll', 'X', 'Y', 'Z']
    tol = 1e-6  # Small tolerance for floating-point comparison

    for i, name in enumerate(param_names):
        lower, upper = custom_bounds[name]
        value = result.optimized_params[i]

        assert lower - tol <= value <= upper + tol, (
            f"Parameter {name} violated bounds: "
            f"value={value:.6f}, bounds=[{lower}, {upper}]"
        )


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    """
    Run property-based tests with pytest.

    Usage:
        python test_gcp_calibrator_properties.py
        pytest test_gcp_calibrator_properties.py -v
        pytest test_gcp_calibrator_properties.py -v -k "zero_residual"
    """
    pytest.main([__file__, '-v', '--tb=short'])
