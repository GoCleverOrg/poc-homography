#!/usr/bin/env python3
"""
Cross-module consistency property-based tests using Hypothesis.

These tests verify that overlapping functionality across multiple modules
produces identical results, preventing implementation drift and ensuring
the single-source-of-truth principle.

Properties verified:
1. Rotation matrix equivalence: CameraGeometry and IntrinsicExtrinsicHomography
   produce identical rotation matrices for the same pan/tilt inputs
2. GPS conversion function equivalence: coordinate_converter, gps_distance_calculator,
   and feature_match_homography all use the same canonical GPS conversion functions
3. GCP calibration residual consistency: Residuals computed by GCPCalibrator match
   manually computed projection errors using CameraGeometry
4. Homography consistency: CameraGeometry.H and IntrinsicExtrinsicHomography
   produce identical homography matrices for identical camera parameters

Mathematical Context:
    Shared formulas must have canonical implementations. When multiple modules
    implement the same mathematical transformation (e.g., rotation matrices,
    coordinate conversions), they MUST produce identical results to prevent
    subtle bugs from implementation variations.

Run with: pytest tests/test_cross_module_consistency_properties.py -v
"""

import math
import os
import sys

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import modules under test
# pytest imports
import pytest

# Hypothesis imports
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

import poc_homography.feature_match_homography

# Import modules that should use shared GPS functions
import poc_homography.gps_distance_calculator
from poc_homography.camera_geometry import CameraGeometry
from poc_homography.coordinate_converter import EARTH_RADIUS_M, gps_to_local_xy, local_xy_to_gps
from poc_homography.gcp_calibrator import GCPCalibrator
from poc_homography.intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography

# =============================================================================
# Hypothesis Strategies for Camera Parameters
# =============================================================================


@st.composite
def valid_pan_degrees(draw):
    """
    Generate valid pan angles in degrees.

    Pan can be any angle (wraps modulo 360), but we test a representative range.
    """
    return draw(st.floats(min_value=-360.0, max_value=360.0, allow_nan=False, allow_infinity=False))


@st.composite
def valid_tilt_degrees(draw):
    """
    Generate valid tilt angles in degrees.

    Tilt must be positive (camera pointing down) for ground plane projection.
    Valid range: (0, 90] degrees
    Avoid extremes near 0° (horizontal, infinite projection) and 90° (looking straight down).
    """
    return draw(st.floats(min_value=5.0, max_value=85.0, allow_nan=False, allow_infinity=False))


@st.composite
def valid_camera_height(draw):
    """
    Generate valid camera heights in meters.

    Typical PTZ camera heights: 2m to 30m
    """
    return draw(st.floats(min_value=2.0, max_value=30.0, allow_nan=False, allow_infinity=False))


@st.composite
def valid_zoom_factor(draw):
    """
    Generate valid zoom factors.

    Zoom range: [1.0, 25.0] for typical PTZ cameras
    """
    return draw(st.floats(min_value=1.0, max_value=25.0, allow_nan=False, allow_infinity=False))


@st.composite
def valid_gps_coordinate(draw):
    """
    Generate valid GPS latitude/longitude pairs.

    Excludes polar regions (> 85°) where equirectangular approximation breaks down.
    """
    latitude = draw(
        st.floats(min_value=-85.0, max_value=85.0, allow_nan=False, allow_infinity=False)
    )
    longitude = draw(
        st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
    )
    return latitude, longitude


@st.composite
def valid_local_xy_offset(draw):
    """
    Generate valid local XY offsets in meters.

    Range: ±100m (typical PTZ camera coverage area)
    """
    x = draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    y = draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    return x, y


@st.composite
def valid_camera_parameters(draw):
    """
    Generate complete valid camera parameter set.

    Returns:
        Dictionary with keys: pan_deg, tilt_deg, height_m, zoom_factor, position
    """
    pan_deg = draw(valid_pan_degrees())
    tilt_deg = draw(valid_tilt_degrees())
    height_m = draw(valid_camera_height())
    zoom_factor = draw(valid_zoom_factor())

    # Camera position at origin with generated height
    position = np.array([0.0, 0.0, height_m])

    return {
        "pan_deg": pan_deg,
        "tilt_deg": tilt_deg,
        "height_m": height_m,
        "zoom_factor": zoom_factor,
        "position": position,
    }


# =============================================================================
# Property 1: Rotation Matrix Equivalence
# =============================================================================


class TestRotationMatrixEquivalence:
    """
    Property: CameraGeometry and IntrinsicExtrinsicHomography must produce
    identical rotation matrices for the same pan/tilt inputs.

    WHY: Both modules implement the same camera rotation model (pan=yaw, tilt=pitch).
    If their rotation matrices differ, one implementation is wrong and will produce
    incorrect projections. This property ensures mathematical consistency.
    """

    @given(pan_deg=valid_pan_degrees(), tilt_deg=valid_tilt_degrees())
    @settings(max_examples=100)
    @example(pan_deg=0.0, tilt_deg=30.0)  # Common test case
    @example(pan_deg=66.7, tilt_deg=30.0)  # Valencia test data case
    @example(pan_deg=180.0, tilt_deg=45.0)  # Looking south
    def test_rotation_matrices_identical(self, pan_deg: float, tilt_deg: float):
        """
        Rotation matrices from both modules must be exactly identical.

        Mathematical requirement:
            R_camera = CameraGeometry._get_rotation_matrix(pan, tilt)
            R_intrinsic = IntrinsicExtrinsicHomography._get_rotation_matrix(pan, tilt)

            Must satisfy: ||R_camera - R_intrinsic||_∞ < ε
            where ε = 1e-10 (floating point tolerance)
        """
        # Create instances
        geo = CameraGeometry(w=1920, h=1080)
        ieh = IntrinsicExtrinsicHomography(width=1920, height=1080)

        # Set parameters
        geo.pan_deg = pan_deg
        geo.tilt_deg = tilt_deg

        # Compute rotation matrices
        R_geo = geo._get_rotation_matrix()
        R_ieh = ieh._get_rotation_matrix(pan_deg, tilt_deg)

        # Verify exact match (within floating point tolerance)
        max_diff = np.max(np.abs(R_geo - R_ieh))
        assert max_diff < 1e-10, (
            f"Rotation matrices differ at pan={pan_deg:.2f}°, tilt={tilt_deg:.2f}°. "
            f"Max element-wise difference: {max_diff:.2e}. "
            f"This violates the single-source-of-truth principle."
        )

    @given(pan_deg=valid_pan_degrees(), tilt_deg=valid_tilt_degrees())
    @settings(max_examples=100)
    def test_rotation_matrices_are_orthogonal(self, pan_deg: float, tilt_deg: float):
        """
        All rotation matrices must be proper orthogonal matrices.

        Mathematical requirement:
            1. det(R) = 1 (proper rotation, not reflection)
            2. R @ R^T = I (orthonormality)

        WHY: Non-orthogonal matrices indicate numerical errors or incorrect
        implementation. Orthogonality is a fundamental property of rotation matrices.
        """
        geo = CameraGeometry(w=1920, h=1080)
        ieh = IntrinsicExtrinsicHomography(width=1920, height=1080)

        geo.pan_deg = pan_deg
        geo.tilt_deg = tilt_deg

        R_geo = geo._get_rotation_matrix()
        R_ieh = ieh._get_rotation_matrix(pan_deg, tilt_deg)

        for name, R in [("CameraGeometry", R_geo), ("IntrinsicExtrinsicHomography", R_ieh)]:
            # Check determinant is 1 (proper rotation)
            det = np.linalg.det(R)
            assert abs(det - 1.0) < 1e-10, (
                f"{name}: det(R) = {det:.10f} at pan={pan_deg:.2f}°, tilt={tilt_deg:.2f}°. "
                f"Expected det(R) = 1.0 for proper rotation matrix."
            )

            # Check R @ R^T = I (orthonormality)
            RRT = R @ R.T
            identity_diff = np.max(np.abs(RRT - np.eye(3)))
            assert identity_diff < 1e-10, (
                f"{name}: R @ R^T differs from identity by {identity_diff:.2e} "
                f"at pan={pan_deg:.2f}°, tilt={tilt_deg:.2f}°. "
                f"Rotation matrix is not orthogonal."
            )


# =============================================================================
# Property 2: GPS Conversion Function Equivalence
# =============================================================================


class TestGPSConversionEquivalence:
    """
    Property: All modules must use the same canonical GPS conversion functions.

    WHY: GPS coordinate conversion is a shared mathematical operation. If different
    modules use different implementations (or different constants like Earth radius),
    they will compute different local XY coordinates from the same GPS input,
    causing systematic errors across the system.

    This property verifies the single-source-of-truth principle for coordinate conversion.
    """

    def test_all_modules_import_shared_functions(self):
        """
        All modules must import GPS functions from coordinate_converter.

        WHY: This ensures all modules use the same implementation, preventing
        implementation drift where different modules compute different results
        from the same GPS input.
        """
        # Verify gps_distance_calculator imports shared functions
        assert hasattr(poc_homography.gps_distance_calculator, "gps_to_local_xy"), (
            "gps_distance_calculator must import gps_to_local_xy from coordinate_converter"
        )
        assert hasattr(poc_homography.gps_distance_calculator, "local_xy_to_gps"), (
            "gps_distance_calculator must import local_xy_to_gps from coordinate_converter"
        )
        assert hasattr(poc_homography.gps_distance_calculator, "EARTH_RADIUS_M"), (
            "gps_distance_calculator must import EARTH_RADIUS_M from coordinate_converter"
        )

        # Verify feature_match_homography imports shared functions
        assert hasattr(poc_homography.feature_match_homography, "gps_to_local_xy"), (
            "feature_match_homography must import gps_to_local_xy from coordinate_converter"
        )
        assert hasattr(poc_homography.feature_match_homography, "local_xy_to_gps"), (
            "feature_match_homography must import local_xy_to_gps from coordinate_converter"
        )

    def test_all_modules_use_same_earth_radius(self):
        """
        All modules must use the same Earth radius constant.

        WHY: Different Earth radius values produce different conversion results.
        Issue #30 documented a 0.076% systematic error from using 111111 m/degree
        instead of the correct 6371000m radius. This test prevents regression.
        """

        # Verify gps_distance_calculator uses shared constant
        assert poc_homography.gps_distance_calculator.EARTH_RADIUS_M == EARTH_RADIUS_M, (
            f"gps_distance_calculator uses EARTH_RADIUS_M={poc_homography.gps_distance_calculator.EARTH_RADIUS_M}, "
            f"but coordinate_converter uses {EARTH_RADIUS_M}. These must be identical."
        )

        # Verify correct value (not the old 111111 approximation)
        assert EARTH_RADIUS_M == 6371000.0, (
            f"EARTH_RADIUS_M must be 6371000.0 (WGS84 mean radius), got {EARTH_RADIUS_M}. "
            f"This prevents the 0.076% systematic error from issue #30."
        )

    @given(ref_gps=valid_gps_coordinate(), target_gps=valid_gps_coordinate())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    @example(
        ref_gps=(39.640472, -0.230194), target_gps=(39.640583, -0.230111)
    )  # Valencia test data
    def test_modules_produce_identical_gps_to_local_results(
        self, ref_gps: tuple[float, float], target_gps: tuple[float, float]
    ):
        """
        GPS-to-local conversion must produce identical results across all modules.

        Mathematical requirement:
            For same inputs (ref_lat, ref_lon, target_lat, target_lon):

            coordinate_converter.gps_to_local_xy() ==
            gps_distance_calculator.gps_to_local_xy() ==
            feature_match_homography.gps_to_local_xy()

        WHY: If modules produce different local XY coordinates from the same GPS
        input, they will have misaligned world coordinate systems, causing projection
        errors and calibration failures.
        """
        ref_lat, ref_lon = ref_gps
        target_lat, target_lon = target_gps

        # Skip if reference and target are too far apart (> 1000km)
        # Equirectangular approximation breaks down at large distances
        lat_diff_deg = abs(target_lat - ref_lat)
        lon_diff_deg = abs(target_lon - ref_lon)
        assume(lat_diff_deg < 10.0 and lon_diff_deg < 10.0)

        # Compute using coordinate_converter (canonical implementation)
        x1, y1 = gps_to_local_xy(ref_lat, ref_lon, target_lat, target_lon)

        # Compute using gps_distance_calculator (should be identical)
        x2, y2 = poc_homography.gps_distance_calculator.gps_to_local_xy(
            ref_lat, ref_lon, target_lat, target_lon
        )

        # Compute using feature_match_homography (should be identical)
        x3, y3 = poc_homography.feature_match_homography.gps_to_local_xy(
            ref_lat, ref_lon, target_lat, target_lon
        )

        # All results must be exactly identical
        assert x1 == x2, (
            f"coordinate_converter and gps_distance_calculator produce different X values: "
            f"{x1:.10f} vs {x2:.10f} for ref=({ref_lat}, {ref_lon}), target=({target_lat}, {target_lon})"
        )
        assert x1 == x3, (
            f"coordinate_converter and feature_match_homography produce different X values: "
            f"{x1:.10f} vs {x3:.10f}"
        )
        assert y1 == y2, (
            f"coordinate_converter and gps_distance_calculator produce different Y values: "
            f"{y1:.10f} vs {y2:.10f}"
        )
        assert y1 == y3, (
            f"coordinate_converter and feature_match_homography produce different Y values: "
            f"{y1:.10f} vs {y3:.10f}"
        )

    @given(ref_gps=valid_gps_coordinate(), offset=valid_local_xy_offset())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    @example(ref_gps=(39.640472, -0.230194), offset=(50.0, 100.0))  # Valencia test data
    def test_modules_produce_identical_local_to_gps_results(
        self, ref_gps: tuple[float, float], offset: tuple[float, float]
    ):
        """
        Local-to-GPS conversion must produce identical results across all modules.

        Mathematical requirement:
            For same inputs (ref_lat, ref_lon, x_meters, y_meters):

            coordinate_converter.local_xy_to_gps() ==
            gps_distance_calculator.local_xy_to_gps() ==
            feature_match_homography.local_xy_to_gps()
        """
        ref_lat, ref_lon = ref_gps
        x_meters, y_meters = offset

        # Compute using coordinate_converter (canonical implementation)
        lat1, lon1 = local_xy_to_gps(ref_lat, ref_lon, x_meters, y_meters)

        # Compute using gps_distance_calculator (should be identical)
        lat2, lon2 = poc_homography.gps_distance_calculator.local_xy_to_gps(
            ref_lat, ref_lon, x_meters, y_meters
        )

        # Compute using feature_match_homography (should be identical)
        lat3, lon3 = poc_homography.feature_match_homography.local_xy_to_gps(
            ref_lat, ref_lon, x_meters, y_meters
        )

        # All results must be exactly identical
        assert lat1 == lat2, (
            f"coordinate_converter and gps_distance_calculator produce different latitude: "
            f"{lat1:.10f} vs {lat2:.10f} for ref=({ref_lat}, {ref_lon}), offset=({x_meters}, {y_meters})"
        )
        assert lat1 == lat3, (
            f"coordinate_converter and feature_match_homography produce different latitude: "
            f"{lat1:.10f} vs {lat3:.10f}"
        )
        assert lon1 == lon2, (
            f"coordinate_converter and gps_distance_calculator produce different longitude: "
            f"{lon1:.10f} vs {lon2:.10f}"
        )
        assert lon1 == lon3, (
            f"coordinate_converter and feature_match_homography produce different longitude: "
            f"{lon1:.10f} vs {lon3:.10f}"
        )


# =============================================================================
# Property 3: Homography Matrix Consistency
# =============================================================================


class TestHomographyConsistency:
    """
    Property: CameraGeometry.H and IntrinsicExtrinsicHomography must produce
    identical homography matrices for identical camera parameters.

    WHY: Both modules implement ground plane homography using the same mathematical
    model (H = K @ [r1, r2, t]). If they produce different homography matrices,
    they will project world points to different image pixels, causing inconsistent
    behavior across the system.
    """

    @given(params=valid_camera_parameters())
    @settings(max_examples=100)
    @example(
        params={
            "pan_deg": 66.7,
            "tilt_deg": 30.0,
            "height_m": 3.4,
            "zoom_factor": 1.0,
            "position": np.array([0.0, 0.0, 3.4]),
        }
    )  # Valencia test data
    def test_homography_matrices_identical(self, params: dict):
        """
        Homography matrices must be identical for same camera parameters.

        Mathematical requirement:
            For same inputs (K, position, pan, tilt):

            CameraGeometry.H == IntrinsicExtrinsicHomography._calculate_ground_homography()

            where matrices are normalized to H[2,2] = 1
        """
        pan_deg = params["pan_deg"]
        tilt_deg = params["tilt_deg"]
        height_m = params["height_m"]
        zoom_factor = params["zoom_factor"]
        position = params["position"]

        width, height = 1920, 1080

        # Create instances
        geo = CameraGeometry(w=width, h=height)
        ieh = IntrinsicExtrinsicHomography(width=width, height=height)

        # Compute intrinsic matrix (both modules should use same method)
        K = CameraGeometry.get_intrinsics(zoom_factor, width, height, sensor_width_mm=7.18)

        # Set parameters for CameraGeometry
        geo.set_camera_parameters(K, position, pan_deg, tilt_deg, map_width=640, map_height=640)
        H_geo = geo.H

        # Compute homography for IntrinsicExtrinsicHomography
        H_ieh = ieh._calculate_ground_homography(K, position, pan_deg, tilt_deg)

        # Both should already be normalized to H[2,2] = 1, but verify
        assert abs(H_geo[2, 2] - 1.0) < 1e-10, "CameraGeometry.H not normalized"
        assert abs(H_ieh[2, 2] - 1.0) < 1e-10, "IntrinsicExtrinsicHomography.H not normalized"

        # Verify matrices are identical
        max_diff = np.max(np.abs(H_geo - H_ieh))
        assert max_diff < 1e-6, (
            f"Homography matrices differ at pan={pan_deg:.2f}°, tilt={tilt_deg:.2f}°, "
            f"height={height_m:.2f}m, zoom={zoom_factor:.2f}. "
            f"Max element-wise difference: {max_diff:.2e}. "
            f"\nH_geo:\n{H_geo}\n\nH_ieh:\n{H_ieh}"
        )

    @given(params=valid_camera_parameters(), world_point=valid_local_xy_offset())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_projection_consistency(self, params: dict, world_point: tuple[float, float]):
        """
        World point projection must be identical using both homography matrices.

        Mathematical requirement:
            For same world point [X, Y, 1]:

            pixel_geo = H_geo @ [X, Y, 1]
            pixel_ieh = H_ieh @ [X, Y, 1]

            ||pixel_geo - pixel_ieh|| < ε (after normalization)

        WHY: If homographies project the same world point to different image pixels,
        the modules are using inconsistent camera models.
        """
        pan_deg = params["pan_deg"]
        tilt_deg = params["tilt_deg"]
        position = params["position"]
        zoom_factor = params["zoom_factor"]

        x_world, y_world = world_point

        width, height = 1920, 1080

        # Create instances and compute homographies
        geo = CameraGeometry(w=width, h=height)
        ieh = IntrinsicExtrinsicHomography(width=width, height=height)

        K = CameraGeometry.get_intrinsics(zoom_factor, width, height, sensor_width_mm=7.18)

        geo.set_camera_parameters(K, position, pan_deg, tilt_deg, map_width=640, map_height=640)
        H_geo = geo.H

        H_ieh = ieh._calculate_ground_homography(K, position, pan_deg, tilt_deg)

        # Project world point using both homographies
        pt = np.array([[x_world], [y_world], [1.0]])

        p_geo = H_geo @ pt
        p_ieh = H_ieh @ pt

        # Normalize to pixel coordinates
        # Skip if point is at/near horizon (w ≈ 0)
        assume(abs(p_geo[2, 0]) > 1e-6 and abs(p_ieh[2, 0]) > 1e-6)

        u_geo = p_geo[0, 0] / p_geo[2, 0]
        v_geo = p_geo[1, 0] / p_geo[2, 0]
        u_ieh = p_ieh[0, 0] / p_ieh[2, 0]
        v_ieh = p_ieh[1, 0] / p_ieh[2, 0]

        # Verify projections match
        pixel_diff = math.sqrt((u_geo - u_ieh) ** 2 + (v_geo - v_ieh) ** 2)
        assert pixel_diff < 0.01, (
            f"Projection differs by {pixel_diff:.4f} pixels for world point ({x_world:.2f}, {y_world:.2f}m) "
            f"at pan={pan_deg:.2f}°, tilt={tilt_deg:.2f}°. "
            f"CameraGeometry: ({u_geo:.2f}, {v_geo:.2f}), "
            f"IntrinsicExtrinsicHomography: ({u_ieh:.2f}, {v_ieh:.2f})"
        )


# =============================================================================
# Property 4: GCP Calibration Residual Consistency
# =============================================================================


class TestGCPResidualConsistency:
    """
    Property: Residuals computed by GCPCalibrator must match manually computed
    projection errors using CameraGeometry.

    WHY: GCPCalibrator optimizes camera parameters by minimizing reprojection error.
    If its residual computation differs from the actual projection behavior of
    CameraGeometry, the optimization will converge to incorrect parameters.

    This property ensures the calibrator and the geometry engine are using the
    same mathematical model.
    """

    @given(params=valid_camera_parameters())
    @settings(
        max_examples=50, suppress_health_check=[HealthCheck.filter_too_much]
    )  # Fewer examples due to computational cost
    @example(
        params={
            "pan_deg": 0.0,
            "tilt_deg": 30.0,
            "height_m": 5.0,
            "zoom_factor": 10.0,
            "position": np.array([0.0, 0.0, 5.0]),
        }
    )
    def test_calibrator_residuals_match_manual_projection_errors(self, params: dict):
        """
        GCPCalibrator residuals must match manually computed projection errors.

        Mathematical requirement:
            For each GCP with world coords [X, Y] and observed pixel [u_obs, v_obs]:

            residual_calibrator = calibrator._compute_residuals()

            [u_pred, v_pred, 1] = H @ [X, Y, 1]  (normalize by w)
            residual_manual = [u_obs - u_pred, v_obs - v_pred]

            ||residual_calibrator - residual_manual|| < ε

        Note: Tolerance is 0.01 pixels to account for GPS round-trip numerical precision.
        The calibrator converts GPS → local XY → back to GPS internally, introducing
        small numerical errors. This is acceptable as long as both computations use
        the same coordinate conversion pipeline.
        """
        pan_deg = params["pan_deg"]
        tilt_deg = params["tilt_deg"]
        position = params["position"]
        zoom_factor = params["zoom_factor"]

        width, height = 1920, 1080

        # Create camera geometry
        geo = CameraGeometry(w=width, h=height)
        K = CameraGeometry.get_intrinsics(zoom_factor, width, height, sensor_width_mm=7.18)
        geo.set_camera_parameters(K, position, pan_deg, tilt_deg, map_width=640, map_height=640)

        # Create synthetic GCPs at known world locations
        # Use camera GPS position (0, 0) as reference for GPS conversion
        ref_lat, ref_lon = 39.640472, -0.230194  # Valencia test location

        # Generate 5 GCPs at different world positions
        world_positions = [
            (10.0, 20.0),  # North-East
            (-10.0, 20.0),  # North-West
            (10.0, -20.0),  # South-East
            (-10.0, -20.0),  # South-West
            (0.0, 15.0),  # North
        ]

        gcps = []
        for x_world, y_world in world_positions:
            # Convert world position to GPS
            lat, lon = local_xy_to_gps(ref_lat, ref_lon, x_world, y_world)

            # Project to image using CameraGeometry
            pt = np.array([[x_world], [y_world], [1.0]])
            img_pt = geo.H @ pt

            # Skip if point projects near horizon
            if abs(img_pt[2, 0]) < 1e-6:
                continue

            u_obs = img_pt[0, 0] / img_pt[2, 0]
            v_obs = img_pt[1, 0] / img_pt[2, 0]

            # Skip if point is outside image bounds (would be rejected by real GCP selection)
            if not (0 <= u_obs < width and 0 <= v_obs < height):
                continue

            gcps.append(
                {
                    "gps": {"latitude": lat, "longitude": lon},
                    "image": {"u": u_obs, "v": v_obs},
                    "world": (x_world, y_world),  # Store for manual computation
                }
            )

        # Need at least 3 GCPs for calibration
        assume(len(gcps) >= 3)

        # Create calibrator with zero parameter perturbation
        # This makes initial parameters = current parameters, so residuals should be near zero
        calibrator = GCPCalibrator(
            camera_geometry=geo,
            gcps=gcps,
            loss_function="huber",
            loss_scale=1.0,
            reference_lat=ref_lat,
            reference_lon=ref_lon,
        )

        # Compute residuals using calibrator (at zero perturbation)
        params_zero = np.zeros(6)  # [Δpan, Δtilt, Δroll, ΔX, ΔY, ΔZ] = [0, 0, 0, 0, 0, 0]
        residuals_calibrator = calibrator._compute_residuals(params_zero)

        # Manually compute projection errors using CameraGeometry
        # Use the calibrator's internal world coordinates (which went through GPS round-trip)
        # to ensure we're comparing apples to apples
        residuals_manual = []
        for i, gcp in enumerate(gcps):
            u_obs = gcp["image"]["u"]
            v_obs = gcp["image"]["v"]

            # Use calibrator's world coordinates (which include GPS round-trip)
            x_world, y_world = calibrator._world_coords[i]

            # Project using CameraGeometry homography
            pt = np.array([[x_world], [y_world], [1.0]])
            img_pt = geo.H @ pt

            if abs(img_pt[2, 0]) < 1e-10:
                # Point at horizon - calibrator uses INFINITY_RESIDUAL
                residuals_manual.extend(
                    [calibrator.INFINITY_RESIDUAL, calibrator.INFINITY_RESIDUAL]
                )
                continue

            u_pred = img_pt[0, 0] / img_pt[2, 0]
            v_pred = img_pt[1, 0] / img_pt[2, 0]

            # Residual = observed - predicted
            residuals_manual.append(u_obs - u_pred)
            residuals_manual.append(v_obs - v_pred)

        residuals_manual = np.array(residuals_manual)

        # Verify residuals match
        assert len(residuals_calibrator) == len(residuals_manual), (
            f"Residual array length mismatch: calibrator={len(residuals_calibrator)}, "
            f"manual={len(residuals_manual)}"
        )

        max_diff = np.max(np.abs(residuals_calibrator - residuals_manual))
        # Use 0.01 pixel tolerance to account for GPS round-trip numerical precision
        assert max_diff < 0.01, (
            f"GCPCalibrator residuals differ from manual projection error computation. "
            f"Max difference: {max_diff:.2e} pixels. "
            f"\nCalibrator residuals: {residuals_calibrator}"
            f"\nManual residuals: {residuals_manual}"
            f"\nThis indicates GCPCalibrator is using a different projection model than CameraGeometry."
        )

        # Additionally verify that residuals are small (since zero perturbation)
        rms_error = np.sqrt(np.mean(residuals_calibrator**2))
        # Use 0.01 pixel tolerance for RMS as well
        assert rms_error < 0.01, (
            f"RMS projection error should be near zero with zero perturbation, got {rms_error:.2e} pixels. "
            f"This indicates numerical issues in the projection pipeline."
        )


# =============================================================================
# Additional Consistency Properties
# =============================================================================


class TestIntrinsicMatrixConsistency:
    """
    Property: Intrinsic matrix computation must be consistent across modules.

    WHY: If CameraGeometry.get_intrinsics() and IntrinsicExtrinsicHomography.get_intrinsics()
    produce different camera matrices for the same inputs, they will have different
    focal lengths and field-of-view, causing projection inconsistencies.
    """

    @given(zoom_factor=valid_zoom_factor())
    @settings(max_examples=100)
    @example(zoom_factor=1.0)  # Wide angle
    @example(zoom_factor=10.0)  # Medium zoom
    @example(zoom_factor=25.0)  # Maximum zoom
    def test_intrinsic_matrix_identical(self, zoom_factor: float):
        """
        Both modules must produce identical intrinsic matrices for same zoom.

        Mathematical requirement:
            K_geo = CameraGeometry.get_intrinsics(zoom, w, h, sensor_w)
            K_ieh = IntrinsicExtrinsicHomography.get_intrinsics(zoom, w, h, sensor_w)

            Must satisfy: ||K_geo - K_ieh||_∞ < ε
        """
        width, height = 1920, 1080
        sensor_width_mm = 7.18

        # Compute using CameraGeometry static method
        K_geo = CameraGeometry.get_intrinsics(zoom_factor, width, height, sensor_width_mm)

        # Compute using IntrinsicExtrinsicHomography instance method
        ieh = IntrinsicExtrinsicHomography(width, height, sensor_width_mm=sensor_width_mm)
        K_ieh = ieh.get_intrinsics(zoom_factor, width, height, sensor_width_mm)

        # Verify exact match
        max_diff = np.max(np.abs(K_geo - K_ieh))
        assert max_diff < 1e-10, (
            f"Intrinsic matrices differ at zoom={zoom_factor:.2f}. "
            f"Max element-wise difference: {max_diff:.2e}. "
            f"\nK_geo:\n{K_geo}\n\nK_ieh:\n{K_ieh}"
        )

        # Verify focal length is correct
        f_px_expected = 5.9 * zoom_factor * (width / sensor_width_mm)
        f_px_geo = K_geo[0, 0]
        f_px_ieh = K_ieh[0, 0]

        assert abs(f_px_geo - f_px_expected) < 1e-6, (
            f"CameraGeometry focal length {f_px_geo:.2f} doesn't match expected {f_px_expected:.2f}"
        )
        assert abs(f_px_ieh - f_px_expected) < 1e-6, (
            f"IntrinsicExtrinsicHomography focal length {f_px_ieh:.2f} doesn't match expected {f_px_expected:.2f}"
        )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
