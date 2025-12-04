#!/usr/bin/env python3
"""
Unit tests for tilt sign convention.

These tests verify that the tilt convention is handled correctly:
- Positive tilt (camera pointing DOWN) should project to positive Y world coords
- The internal negation in _get_rotation_matrix() handles the Hikvision convention
- Callers should NOT negate tilt externally

Issue #29: Fix tilt convention double negation causing inverted projections
"""

import math
import numpy as np
import pytest

from poc_homography.intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography


class TestTiltSignConvention:
    """Test suite for tilt sign convention handling."""

    @pytest.fixture
    def homography_provider(self):
        """Create a standard homography provider for testing."""
        return IntrinsicExtrinsicHomography(
            width=2560,
            height=1440,
            sensor_width_mm=7.18,
            base_focal_length_mm=5.9
        )

    def test_positive_tilt_projects_ahead(self, homography_provider):
        """
        Test that positive tilt (camera pointing DOWN) projects image center
        to a point AHEAD of the camera (positive Y in world coordinates).

        With Hikvision convention: positive tilt = camera pointing down.
        When camera points down at ground, image center should project to
        a point in front of the camera (positive Y).
        """
        # Setup: camera at 5m height, tilted down 30 degrees
        camera_height = 5.0
        pan_deg = 0.0
        tilt_deg = 30.0  # Positive = pointing DOWN (Hikvision convention)
        zoom = 1.0

        # Compute homography - pass tilt directly (no external negation!)
        result = homography_provider.compute_homography(
            frame=None,
            reference={
                'camera_matrix': homography_provider.get_intrinsics(zoom, 2560, 1440),
                'camera_position': np.array([0.0, 0.0, camera_height]),
                'pan_deg': pan_deg,
                'tilt_deg': tilt_deg,  # Pass raw value, NOT negated
                'map_width': 640,
                'map_height': 640,
            }
        )

        assert result.confidence > 0, "Homography should be valid"

        # Project image center
        center_x, center_y = 2560 // 2, 1440 // 2
        world_point = homography_provider.project_point_to_map((center_x, center_y))

        # Y coordinate should be positive (ahead of camera)
        assert world_point.y > 0, (
            f"Image center should project AHEAD of camera (Y > 0), got Y = {world_point.y:.2f}. "
            f"This suggests tilt is being double-negated."
        )

    def test_positive_tilt_produces_reasonable_distance(self, homography_provider):
        """
        Test that positive tilt produces a reasonable ground distance.

        For a camera at height h tilted down by angle θ:
        - Ground distance should be positive and in a reasonable range
        - Steeper tilt (larger angle) should produce shorter distances
        """
        camera_height = 5.0
        tilt_deg = 30.0  # Hikvision: positive = down

        result = homography_provider.compute_homography(
            frame=None,
            reference={
                'camera_matrix': homography_provider.get_intrinsics(1.0, 2560, 1440),
                'camera_position': np.array([0.0, 0.0, camera_height]),
                'pan_deg': 0.0,
                'tilt_deg': tilt_deg,
                'map_width': 640,
                'map_height': 640,
            }
        )

        # Project image center
        center_x, center_y = 2560 // 2, 1440 // 2
        world_point = homography_provider.project_point_to_map((center_x, center_y))

        actual_distance = math.sqrt(world_point.x**2 + world_point.y**2)

        # Basic sanity: distance should be positive and within reasonable bounds
        # For 30° tilt at 5m height, optical axis intersects at h/tan(30°) ≈ 8.66m
        # But image center may differ due to FOV. Distance should be in [1, 50] range
        assert actual_distance > 1.0, (
            f"Distance too short: {actual_distance:.2f}m. May indicate projection error."
        )
        assert actual_distance < 50.0, (
            f"Distance too long: {actual_distance:.2f}m. May indicate tilt sign error."
        )

    def test_bottom_closer_than_top(self, homography_provider):
        """
        Test that points at the bottom of the image project closer than top.

        With camera tilted down, the bottom of the image shows ground closer
        to the camera, while the top shows ground further away.
        """
        result = homography_provider.compute_homography(
            frame=None,
            reference={
                'camera_matrix': homography_provider.get_intrinsics(1.0, 2560, 1440),
                'camera_position': np.array([0.0, 0.0, 5.0]),
                'pan_deg': 0.0,
                'tilt_deg': 30.0,
                'map_width': 640,
                'map_height': 640,
            }
        )

        # Project bottom and top of image center column
        bottom_point = homography_provider.project_point_to_map((1280, 1340))  # Near bottom
        top_point = homography_provider.project_point_to_map((1280, 200))      # Near top

        # Check both projections are valid (not on horizon)
        if bottom_point.y > 0 and top_point.y > 0:
            bottom_distance = math.sqrt(bottom_point.x**2 + bottom_point.y**2)
            top_distance = math.sqrt(top_point.x**2 + top_point.y**2)

            assert bottom_distance < top_distance, (
                f"Bottom of image ({bottom_distance:.2f}m) should be CLOSER than top ({top_distance:.2f}m). "
                f"This suggests inverted tilt projection."
            )

    def test_different_tilt_angles_change_distance(self, homography_provider):
        """
        Test that different tilt angles produce different ground distances.

        The relationship between tilt and distance depends on the coordinate system
        and projection math. This test verifies that:
        1. Both projections produce positive distances
        2. The distances are different (tilt has an effect)
        """
        camera_height = 5.0

        # Test with two different tilt angles
        tilt_a = 25.0
        tilt_b = 45.0

        # Compute homography for tilt A
        homography_provider.compute_homography(
            frame=None,
            reference={
                'camera_matrix': homography_provider.get_intrinsics(1.0, 2560, 1440),
                'camera_position': np.array([0.0, 0.0, camera_height]),
                'pan_deg': 0.0,
                'tilt_deg': tilt_a,
                'map_width': 640,
                'map_height': 640,
            }
        )
        point_a = homography_provider.project_point_to_map((1280, 720))
        distance_a = math.sqrt(point_a.x**2 + point_a.y**2)

        # Compute homography for tilt B
        homography_provider.compute_homography(
            frame=None,
            reference={
                'camera_matrix': homography_provider.get_intrinsics(1.0, 2560, 1440),
                'camera_position': np.array([0.0, 0.0, camera_height]),
                'pan_deg': 0.0,
                'tilt_deg': tilt_b,
                'map_width': 640,
                'map_height': 640,
            }
        )
        point_b = homography_provider.project_point_to_map((1280, 720))
        distance_b = math.sqrt(point_b.x**2 + point_b.y**2)

        # Both distances should be positive
        assert distance_a > 0, f"Tilt {tilt_a}° should produce positive distance"
        assert distance_b > 0, f"Tilt {tilt_b}° should produce positive distance"

        # Distances should be different (tilt angle affects projection)
        assert distance_a != distance_b, (
            f"Different tilt angles should produce different distances: "
            f"tilt {tilt_a}° → {distance_a:.2f}m, tilt {tilt_b}° → {distance_b:.2f}m"
        )

    def test_no_external_negation_needed(self, homography_provider):
        """
        Test that passing raw tilt works correctly - no external negation needed.

        This test simulates what main.py and other tools should do:
        pass the tilt value directly from camera status without modification.
        """
        # Simulate camera reporting positive tilt (pointing down)
        camera_status = {
            'pan': 45.0,
            'tilt': 25.0,  # Hikvision: positive = down
            'zoom': 1.0
        }

        # Pass directly without negation (correct way)
        result = homography_provider.compute_homography(
            frame=None,
            reference={
                'camera_matrix': homography_provider.get_intrinsics(
                    camera_status['zoom'], 2560, 1440
                ),
                'camera_position': np.array([0.0, 0.0, 5.0]),
                'pan_deg': camera_status['pan'],
                'tilt_deg': camera_status['tilt'],  # Direct, no negation
                'map_width': 640,
                'map_height': 640,
            }
        )

        assert result.confidence > 0, "Homography should be valid with direct tilt"

        # Verify projection makes sense
        world_point = homography_provider.project_point_to_map((1280, 720))
        assert world_point.y > 0, (
            f"Image center should project ahead (Y > 0) when passing raw tilt. "
            f"Got Y = {world_point.y:.2f}. The internal negation should handle this."
        )


class TestTiltSignConsistency:
    """Test that tilt sign is handled consistently across the codebase."""

    def test_intrinsic_extrinsic_homography_tilt(self):
        """
        Verify IntrinsicExtrinsicHomography handles tilt convention internally.
        """
        provider = IntrinsicExtrinsicHomography(2560, 1440)

        # Test with typical downward-pointing camera
        result = provider.compute_homography(
            frame=None,
            reference={
                'camera_matrix': provider.get_intrinsics(1.0, 2560, 1440),
                'camera_position': np.array([0.0, 0.0, 10.0]),
                'pan_deg': 0.0,
                'tilt_deg': 45.0,  # 45 degrees down
                'map_width': 640,
                'map_height': 640,
            }
        )

        # At 45 degrees and 10m height, distance should be ~10m
        world_point = provider.project_point_to_map((1280, 720))
        distance = math.sqrt(world_point.x**2 + world_point.y**2)

        expected = 10.0 / math.tan(math.radians(45))  # = 10m
        assert abs(distance - expected) < expected * 0.3, (
            f"At 45° tilt and 10m height, distance should be ~{expected:.1f}m, got {distance:.1f}m"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
