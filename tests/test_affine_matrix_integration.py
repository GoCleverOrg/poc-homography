#!/usr/bin/env python3
"""
Integration test suite for affine matrix A backward compatibility and homography composition.

Tests Issue #122: Verify A matrix works correctly in the full homography pipeline and
maintains backward compatibility with code that doesn't use geotiff_params.

These integration tests verify:
1. Backward compatibility: Code without geotiff_params continues working (A = I)
2. Homography composition: H_total = H @ A produces correct projections
3. Equivalence: A matrix matches previous inline T_pixel_to_localXY computation
4. Numerical precision: Large UTM coordinates maintain precision
5. Reset capability: A matrix can be reset to identity
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.camera_geometry import CameraGeometry


class TestAffineMatrixIntegration:
    """Integration tests for A matrix in the full homography pipeline."""

    def test_backward_compatibility_default_a_matrix(self):
        """
        Test backward compatibility with default A matrix.

        Verifies that:
        - A remains identity when set_geotiff_params is not called
        - H_total = geo.H @ geo.A equals geo.H when A is identity
        - Code that doesn't use geotiff_params continues working unchanged
        """
        # Setup CameraGeometry without calling set_geotiff_params
        geo = CameraGeometry(1920, 1080)

        # Set camera parameters (realistic PTZ setup)
        K = CameraGeometry.get_intrinsics(zoom_factor=2.0)
        w_pos = np.array([0.0, 0.0, 5.0])  # 5m height
        pan_deg = 0.0
        tilt_deg = 45.0

        geo.set_camera_parameters(K, w_pos, pan_deg, tilt_deg, 640, 640)

        # Verify A is identity (backward compatibility)
        expected_A = np.eye(3)
        np.testing.assert_array_equal(
            geo.A,
            expected_A,
            err_msg="A matrix should default to identity when geotiff_params not set",
        )

        # Verify H_total = H @ A equals H when A is identity
        H_total = geo.H @ geo.A
        np.testing.assert_array_almost_equal(
            H_total, geo.H, decimal=10, err_msg="H_total should equal H when A is identity"
        )

        # Verify this is mathematically equivalent (H @ I = H)
        assert np.allclose(H_total, geo.H), "Matrix multiplication H @ I should preserve H"

    def test_homography_composition_correct(self):
        """
        Test homography composition produces correct results.

        Verifies that:
        - H_total = geo.H @ geo.A produces valid projections
        - Reference image point projects correctly to camera frame
        - Composition maintains mathematical consistency
        """
        # Setup CameraGeometry with realistic geotiff parameters
        geo = CameraGeometry(1920, 1080)

        # Realistic GeoTIFF parameters (0.5 m/pixel, UTM Zone 32N coordinates)
        geotiff_params = {
            "pixel_size_x": 0.5,  # 0.5 meters per pixel
            "pixel_size_y": 0.5,
            "origin_easting": 500000.0,  # Reference image origin in UTM
            "origin_northing": 4000000.0,
        }

        # Camera position in UTM (10m East, 20m North from origin)
        camera_utm_position = (500010.0, 4000020.0)

        # Set geotiff params to compute A matrix
        geo.set_geotiff_params(geotiff_params, camera_utm_position)

        # Set camera parameters
        K = CameraGeometry.get_intrinsics(zoom_factor=2.0)
        w_pos = np.array([0.0, 0.0, 5.0])  # 5m height (local coordinates)
        pan_deg = 0.0
        tilt_deg = 45.0

        geo.set_camera_parameters(K, w_pos, pan_deg, tilt_deg, 640, 640)

        # Verify A is not identity (was actually set)
        assert not np.allclose(geo.A, np.eye(3)), (
            "A matrix should differ from identity when geotiff_params are set"
        )

        # Compute H_total = H @ A (composition of homographies)
        H_total = geo.H @ geo.A

        # Verify H_total is valid (invertible, reasonable condition number)
        det_H_total = np.linalg.det(H_total)
        assert abs(det_H_total) > 1e-10, "H_total should be invertible (non-zero determinant)"

        cond_H_total = np.linalg.cond(H_total)
        assert cond_H_total < 1e10, f"H_total condition number {cond_H_total} should be reasonable"

        # Test projection: reference image point -> world -> camera
        # Pick a reference image point (100 pixels, 150 pixels from origin)
        ref_pixel = np.array([100.0, 150.0, 1.0])

        # Project through A: reference pixel -> world coordinates
        world_point = geo.A @ ref_pixel
        world_x = world_point[0] / world_point[2]
        world_y = world_point[1] / world_point[2]

        # Expected world coordinates:
        # x = pixel_size_x * 100 + t_x = 0.5 * 100 + (-10) = 50 - 10 = 40 meters
        # y = pixel_size_y * 150 + t_y = 0.5 * 150 + (-20) = 75 - 20 = 55 meters
        expected_world_x = 40.0
        expected_world_y = 55.0

        np.testing.assert_almost_equal(
            world_x,
            expected_world_x,
            decimal=6,
            err_msg="World X coordinate should match expected transformation",
        )
        np.testing.assert_almost_equal(
            world_y,
            expected_world_y,
            decimal=6,
            err_msg="World Y coordinate should match expected transformation",
        )

        # Project through H_total directly: reference pixel -> camera
        camera_point_total = H_total @ ref_pixel

        # Project through H after A: (reference -> world -> camera)
        camera_point_composed = geo.H @ (geo.A @ ref_pixel)

        # Verify composition is consistent
        np.testing.assert_array_almost_equal(
            camera_point_total,
            camera_point_composed,
            decimal=10,
            err_msg="H_total projection should equal composed projection H @ (A @ pixel)",
        )

    def test_equivalence_with_inline_computation(self):
        """
        Test equivalence with inline T_pixel_to_localXY computation.

        This is the critical regression test that verifies the new A matrix approach
        produces identical results to the previous inline computation method.

        Verifies:
        - A matrix computed via set_geotiff_params
        - Equivalent T_pixel_to_localXY computed inline (old method)
        - np.allclose(geo.A, T_pixel_to_localXY) confirms equivalence
        """
        geo = CameraGeometry(1920, 1080)

        # Realistic GeoTIFF parameters
        pixel_size_x = 0.5
        pixel_size_y = 0.5
        origin_easting = 500000.0
        origin_northing = 4000000.0
        camera_easting = 500010.0
        camera_northing = 4000020.0

        geotiff_params = {
            "pixel_size_x": pixel_size_x,
            "pixel_size_y": pixel_size_y,
            "origin_easting": origin_easting,
            "origin_northing": origin_northing,
        }
        camera_utm_position = (camera_easting, camera_northing)

        # Compute A matrix using new method
        geo.set_geotiff_params(geotiff_params, camera_utm_position)
        A_new = geo.A.copy()

        # Compute equivalent T_pixel_to_localXY inline (old method)
        # This is how it would have been computed before the A matrix abstraction:
        # t_x = origin_easting - camera_easting
        # t_y = origin_northing - camera_northing
        t_x = origin_easting - camera_easting
        t_y = origin_northing - camera_northing

        T_pixel_to_localXY = np.array(
            [[pixel_size_x, 0.0, t_x], [0.0, pixel_size_y, t_y], [0.0, 0.0, 1.0]]
        )

        # Critical regression test: verify exact equivalence
        np.testing.assert_allclose(
            A_new,
            T_pixel_to_localXY,
            rtol=1e-15,
            atol=1e-15,
            err_msg="A matrix should be identical to inline T_pixel_to_localXY computation",
        )

        # Verify element-by-element for clarity
        assert A_new[0, 0] == pixel_size_x, "A[0,0] should equal pixel_size_x"
        assert A_new[1, 1] == pixel_size_y, "A[1,1] should equal pixel_size_y"
        assert A_new[0, 2] == t_x, "A[0,2] should equal t_x"
        assert A_new[1, 2] == t_y, "A[1,2] should equal t_y"
        assert A_new[2, 2] == 1.0, "A[2,2] should equal 1.0"

        # Verify test transformation produces identical results
        test_pixel = np.array([100.0, 200.0, 1.0])

        result_new = A_new @ test_pixel
        result_old = T_pixel_to_localXY @ test_pixel

        np.testing.assert_array_equal(
            result_new,
            result_old,
            err_msg="Transformation results should be identical (new A vs old inline)",
        )

    def test_numerical_precision_large_utm(self):
        """
        Test A matrix numerical precision with large UTM coordinates.

        Verifies:
        - Large UTM coordinates (typical: 500000 easting, 4000000 northing)
        - No significant precision loss in A matrix computation
        - Homography condition number remains reasonable
        """
        geo = CameraGeometry(1920, 1080)

        # Realistic large UTM coordinates (Zone 32N, Central Europe)
        # Typical easting: ~500,000 meters (500 km from central meridian)
        # Typical northing: ~4,000,000 meters (4000 km from equator)
        geotiff_params = {
            "pixel_size_x": 0.25,  # Fine resolution: 25cm per pixel
            "pixel_size_y": 0.25,
            "origin_easting": 523456.789,  # Realistic large coordinates
            "origin_northing": 4987654.321,
        }

        # Camera nearby but with different large coordinates
        camera_utm_position = (523500.123, 4987700.456)

        geo.set_geotiff_params(geotiff_params, camera_utm_position)

        # Verify A matrix is computed correctly despite large values
        # Expected translation components:
        t_x = 523456.789 - 523500.123  # = -43.334
        t_y = 4987654.321 - 4987700.456  # = -46.135

        expected_A = np.array([[0.25, 0.0, t_x], [0.0, 0.25, t_y], [0.0, 0.0, 1.0]])

        np.testing.assert_allclose(
            geo.A,
            expected_A,
            rtol=1e-12,
            atol=1e-12,
            err_msg="A matrix should maintain precision with large UTM coordinates",
        )

        # Verify no catastrophic cancellation in translation components
        assert abs(geo.A[0, 2] - t_x) < 1e-10, "Translation t_x should be computed precisely"
        assert abs(geo.A[1, 2] - t_y) < 1e-10, "Translation t_y should be computed precisely"

        # Set camera parameters and verify homography is well-conditioned
        K = CameraGeometry.get_intrinsics(zoom_factor=2.0)
        w_pos = np.array([0.0, 0.0, 5.0])

        geo.set_camera_parameters(K, w_pos, 0.0, 45.0, 640, 640)

        # Compute H_total and verify condition number
        H_total = geo.H @ geo.A
        cond_H_total = np.linalg.cond(H_total)

        # Condition number should be reasonable (not catastrophically large)
        assert cond_H_total < 1e8, (
            f"H_total condition number {cond_H_total} should be reasonable "
            f"(large UTM coordinates should not cause ill-conditioning)"
        )

        # Verify projection precision: small pixel offset should produce
        # expected world coordinate change
        pixel1 = np.array([100.0, 100.0, 1.0])
        pixel2 = np.array([104.0, 100.0, 1.0])  # 4 pixels to the right

        world1 = geo.A @ pixel1
        world2 = geo.A @ pixel2

        world1_x = world1[0] / world1[2]
        world2_x = world2[0] / world2[2]

        # Expected difference: 4 pixels * 0.25 m/pixel = 1.0 meter
        expected_diff = 1.0
        actual_diff = world2_x - world1_x

        np.testing.assert_almost_equal(
            actual_diff,
            expected_diff,
            decimal=10,
            err_msg="Small pixel offsets should produce precise world coordinate changes",
        )

    def test_a_matrix_reset_to_identity(self):
        """
        Test A matrix reset capability.

        Verifies:
        - Set A matrix with geotiff_params
        - Call set_geotiff_params(None, None)
        - Verify A resets to identity
        """
        geo = CameraGeometry(1920, 1080)

        # Initially A is identity
        assert np.allclose(geo.A, np.eye(3)), "A should be identity on initialization"

        # Set geotiff params to compute non-identity A
        geotiff_params = {
            "pixel_size_x": 0.5,
            "pixel_size_y": 0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500010.0, 4000020.0)

        geo.set_geotiff_params(geotiff_params, camera_utm_position)

        # Verify A is not identity
        assert not np.allclose(geo.A, np.eye(3)), (
            "A should be non-identity after setting geotiff params"
        )

        # Reset A to identity by passing None
        geo.set_geotiff_params(None, None)

        # Verify A is reset to identity
        expected = np.eye(3)
        np.testing.assert_array_equal(
            geo.A,
            expected,
            err_msg="A should reset to identity when set_geotiff_params(None, None) is called",
        )

        # Test partial reset: geotiff_params is None
        geo.set_geotiff_params(geotiff_params, camera_utm_position)
        assert not np.allclose(geo.A, np.eye(3))

        geo.set_geotiff_params(None, camera_utm_position)
        np.testing.assert_array_equal(
            geo.A, np.eye(3), err_msg="A should reset to identity when geotiff_params is None"
        )

        # Test partial reset: camera_utm_position is None
        geo.set_geotiff_params(geotiff_params, camera_utm_position)
        assert not np.allclose(geo.A, np.eye(3))

        geo.set_geotiff_params(geotiff_params, None)
        np.testing.assert_array_equal(
            geo.A, np.eye(3), err_msg="A should reset to identity when camera_utm_position is None"
        )


class TestAffineMatrixCompositionEdgeCases:
    """Test edge cases and boundary conditions for homography composition."""

    def test_composition_with_identity_a(self):
        """Test that H @ I = H (composition with identity A)."""
        geo = CameraGeometry(1920, 1080)

        K = CameraGeometry.get_intrinsics(zoom_factor=1.0)
        w_pos = np.array([0.0, 0.0, 5.0])

        geo.set_camera_parameters(K, w_pos, 0.0, 45.0, 640, 640)

        # A is identity by default
        H_total = geo.H @ geo.A

        # Should equal H exactly (within floating point precision)
        np.testing.assert_allclose(
            H_total, geo.H, rtol=1e-15, atol=1e-15, err_msg="H @ I should equal H exactly"
        )

    def test_composition_associativity(self):
        """Test that homography composition is associative: H @ (A @ p) = (H @ A) @ p."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": 0.5,
            "pixel_size_y": 0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500010.0, 4000020.0)

        geo.set_geotiff_params(geotiff_params, camera_utm_position)

        K = CameraGeometry.get_intrinsics(zoom_factor=2.0)
        w_pos = np.array([0.0, 0.0, 5.0])
        geo.set_camera_parameters(K, w_pos, 0.0, 45.0, 640, 640)

        # Test point
        p = np.array([100.0, 150.0, 1.0])

        # Compute both ways
        result1 = geo.H @ (geo.A @ p)  # H @ (A @ p)
        result2 = (geo.H @ geo.A) @ p  # (H @ A) @ p

        # Should be identical (matrix multiplication is associative)
        np.testing.assert_allclose(
            result1,
            result2,
            rtol=1e-15,
            atol=1e-15,
            err_msg="Matrix multiplication should be associative: H @ (A @ p) = (H @ A) @ p",
        )

    def test_composition_with_zero_translation(self):
        """Test A matrix when camera is exactly at reference origin (zero translation)."""
        geo = CameraGeometry(1920, 1080)

        # Camera at exactly the reference origin
        geotiff_params = {
            "pixel_size_x": 0.5,
            "pixel_size_y": 0.5,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500000.0, 4000000.0)  # Same as origin

        geo.set_geotiff_params(geotiff_params, camera_utm_position)

        # Expected A with zero translation
        expected_A = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])

        np.testing.assert_array_almost_equal(
            geo.A,
            expected_A,
            decimal=10,
            err_msg="A should have zero translation when camera is at origin",
        )

    def test_composition_preserves_homogeneous_scaling(self):
        """Test that homogeneous coordinate scaling is preserved through composition."""
        geo = CameraGeometry(1920, 1080)

        geotiff_params = {
            "pixel_size_x": 1.0,
            "pixel_size_y": 1.0,
            "origin_easting": 500000.0,
            "origin_northing": 4000000.0,
        }
        camera_utm_position = (500050.0, 4000100.0)

        geo.set_geotiff_params(geotiff_params, camera_utm_position)

        K = CameraGeometry.get_intrinsics(zoom_factor=1.0)
        w_pos = np.array([0.0, 0.0, 5.0])
        geo.set_camera_parameters(K, w_pos, 0.0, 45.0, 640, 640)

        # Test point in homogeneous coordinates
        p = np.array([100.0, 150.0, 1.0])

        # Scale homogeneous coordinate (should produce equivalent point)
        p_scaled = p * 2.0  # [200, 300, 2]

        # Project both
        result1 = geo.H @ (geo.A @ p)
        result2 = geo.H @ (geo.A @ p_scaled)

        # Normalize to compare
        result1_norm = result1 / result1[2]
        result2_norm = result2 / result2[2]

        # Should be equivalent after normalization
        np.testing.assert_allclose(
            result1_norm,
            result2_norm,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Homogeneous scaling should be preserved through composition",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
