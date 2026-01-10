#!/usr/bin/env python3
"""
Unit tests for GCP (Ground Control Point) validation functions.

Tests verify validation logic for map pixel coordinates, image pixel coordinates,
duplicate detection, and overall GCP configuration.
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.gcp_validation import (
    MAX_GCP_COUNT,
    _is_valid_finite_number,
    _validate_image_dimension,
    detect_duplicate_gcps,
    validate_gcp_image_coordinates,
    validate_gcp_map_coordinates,
    validate_ground_control_points,
)
from poc_homography.homography_config import HomographyConfig


class TestValidateGCPMapCoordinates(unittest.TestCase):
    """Test map pixel coordinate validation for ground control points."""

    def test_valid_map_coordinates_pass_validation(self):
        """Test that valid map coordinates pass validation."""
        gcp = {
            "map_id": "map_valte",
            "map_pixel_x": 1234.5,
            "map_pixel_y": 567.8,
        }
        # Should not raise any exception
        validate_gcp_map_coordinates(gcp, 0)

    def test_valid_map_coordinates_with_dimensions_pass(self):
        """Test that valid map coordinates pass with map dimensions provided."""
        gcp = {
            "map_id": "map_valte",
            "map_pixel_x": 1234.5,
            "map_pixel_y": 567.8,
        }
        # Should not raise any exception
        validate_gcp_map_coordinates(gcp, 0, map_width=2000, map_height=1500)

    def test_missing_map_id_raises_value_error(self):
        """Test that missing map_id field raises ValueError."""
        gcp = {"map_pixel_x": 1234.5, "map_pixel_y": 567.8}
        with self.assertRaisesRegex(ValueError, "missing required 'map_id'"):
            validate_gcp_map_coordinates(gcp, 0)

    def test_empty_map_id_raises_value_error(self):
        """Test that empty map_id raises ValueError."""
        gcp = {"map_id": "", "map_pixel_x": 1234.5, "map_pixel_y": 567.8}
        with self.assertRaisesRegex(ValueError, "must be a non-empty string"):
            validate_gcp_map_coordinates(gcp, 0)

    def test_whitespace_only_map_id_raises_value_error(self):
        """Test that whitespace-only map_id raises ValueError."""
        gcp = {"map_id": "   ", "map_pixel_x": 1234.5, "map_pixel_y": 567.8}
        with self.assertRaisesRegex(ValueError, "must be a non-empty string"):
            validate_gcp_map_coordinates(gcp, 0)

    def test_non_string_map_id_raises_value_error(self):
        """Test that non-string map_id raises ValueError."""
        gcp = {"map_id": 123, "map_pixel_x": 1234.5, "map_pixel_y": 567.8}
        with self.assertRaisesRegex(ValueError, "must be a non-empty string"):
            validate_gcp_map_coordinates(gcp, 0)

    def test_missing_map_pixel_x_raises_value_error(self):
        """Test that missing map_pixel_x field raises ValueError."""
        gcp = {"map_id": "map_valte", "map_pixel_y": 567.8}
        with self.assertRaisesRegex(ValueError, "missing required 'map_pixel_x'"):
            validate_gcp_map_coordinates(gcp, 0)

    def test_missing_map_pixel_y_raises_value_error(self):
        """Test that missing map_pixel_y field raises ValueError."""
        gcp = {"map_id": "map_valte", "map_pixel_x": 1234.5}
        with self.assertRaisesRegex(ValueError, "missing required 'map_pixel_y'"):
            validate_gcp_map_coordinates(gcp, 0)

    def test_non_numeric_map_pixel_x_raises_value_error(self):
        """Test that non-numeric map_pixel_x raises ValueError."""
        gcp = {
            "map_id": "map_valte",
            "map_pixel_x": "1234.5",  # String instead of number
            "map_pixel_y": 567.8,
        }
        with self.assertRaisesRegex(ValueError, "map_pixel_x must be a number"):
            validate_gcp_map_coordinates(gcp, 0)

    def test_non_numeric_map_pixel_y_raises_value_error(self):
        """Test that non-numeric map_pixel_y raises ValueError."""
        gcp = {
            "map_id": "map_valte",
            "map_pixel_x": 1234.5,
            "map_pixel_y": "567.8",  # String instead of number
        }
        with self.assertRaisesRegex(ValueError, "map_pixel_y must be a number"):
            validate_gcp_map_coordinates(gcp, 0)

    def test_map_pixel_x_outside_map_width_raises_value_error(self):
        """Test that map_pixel_x outside map width raises ValueError."""
        gcp = {
            "map_id": "map_valte",
            "map_pixel_x": 2000.0,  # At or beyond right edge
            "map_pixel_y": 567.8,
        }
        with self.assertRaisesRegex(ValueError, "map_pixel_x 2000.0 outside map width"):
            validate_gcp_map_coordinates(gcp, 0, map_width=2000, map_height=1500)

    def test_map_pixel_x_negative_raises_value_error(self):
        """Test that negative map_pixel_x raises ValueError."""
        gcp = {"map_id": "map_valte", "map_pixel_x": -1.0, "map_pixel_y": 567.8}
        with self.assertRaisesRegex(ValueError, "map_pixel_x -1.0 outside map width"):
            validate_gcp_map_coordinates(gcp, 0, map_width=2000, map_height=1500)

    def test_map_pixel_y_outside_map_height_raises_value_error(self):
        """Test that map_pixel_y outside map height raises ValueError."""
        gcp = {
            "map_id": "map_valte",
            "map_pixel_x": 1234.5,
            "map_pixel_y": 1500.0,  # At or beyond bottom edge
        }
        with self.assertRaisesRegex(ValueError, "map_pixel_y 1500.0 outside map height"):
            validate_gcp_map_coordinates(gcp, 0, map_width=2000, map_height=1500)

    def test_map_pixel_y_negative_raises_value_error(self):
        """Test that negative map_pixel_y raises ValueError."""
        gcp = {"map_id": "map_valte", "map_pixel_x": 1234.5, "map_pixel_y": -1.0}
        with self.assertRaisesRegex(ValueError, "map_pixel_y -1.0 outside map height"):
            validate_gcp_map_coordinates(gcp, 0, map_width=2000, map_height=1500)

    def test_validation_passes_when_dimensions_not_provided(self):
        """Test that validation passes when dimensions not provided (skips bounds check)."""
        gcp = {
            "map_id": "map_valte",
            "map_pixel_x": 5000.0,  # Would be out of bounds if we had dimensions
            "map_pixel_y": 3000.0,
        }
        # Should not raise - bounds check is skipped
        validate_gcp_map_coordinates(gcp, 0)

    def test_validation_passes_at_boundaries(self):
        """Test that coordinates at valid boundaries pass."""
        # At origin (top-left corner)
        gcp_origin = {"map_id": "map_valte", "map_pixel_x": 0.0, "map_pixel_y": 0.0}
        # Just before max (bottom-right corner)
        gcp_max = {"map_id": "map_valte", "map_pixel_x": 1999.9, "map_pixel_y": 1499.9}
        # Should not raise
        validate_gcp_map_coordinates(gcp_origin, 0, map_width=2000, map_height=1500)
        validate_gcp_map_coordinates(gcp_max, 1, map_width=2000, map_height=1500)

    def test_only_width_provided_validates_x(self):
        """Test that map_pixel_x is validated when only width is provided."""
        gcp = {
            "map_id": "map_valte",
            "map_pixel_x": 3000.0,  # Out of bounds for width 2000
            "map_pixel_y": 500.0,
        }
        with self.assertRaisesRegex(ValueError, "map_pixel_x 3000.0 outside map width"):
            validate_gcp_map_coordinates(gcp, 0, map_width=2000, map_height=None)

    def test_only_height_provided_validates_y(self):
        """Test that map_pixel_y is validated when only height is provided."""
        gcp = {
            "map_id": "map_valte",
            "map_pixel_x": 500.0,
            "map_pixel_y": 2000.0,  # Out of bounds for height 1000
        }
        with self.assertRaisesRegex(ValueError, "map_pixel_y 2000.0 outside map height"):
            validate_gcp_map_coordinates(gcp, 0, map_width=None, map_height=1000)

    def test_error_message_includes_description(self):
        """Test that error messages include GCP description when available."""
        gcp = {
            "map_id": "map_valte",
            "map_pixel_x": -1.0,
            "map_pixel_y": 567.8,
            "metadata": {"description": "Building corner NW"},
        }
        with self.assertRaisesRegex(ValueError, "Building corner NW"):
            validate_gcp_map_coordinates(gcp, 0, map_width=2000, map_height=1500)


class TestValidateGCPImageCoordinates(unittest.TestCase):
    """Test image pixel coordinate validation for ground control points."""

    def test_valid_image_coordinates_pass(self):
        """Test that valid image coordinates pass validation."""
        gcp = {"image_u": 1250.5, "image_v": 680.0}
        # Should not raise any exception
        validate_gcp_image_coordinates(gcp, 0)

    def test_valid_image_coordinates_with_dimensions_pass(self):
        """Test that valid image coordinates pass with image dimensions provided."""
        gcp = {"image_u": 1250.5, "image_v": 680.0}
        # Should not raise any exception
        validate_gcp_image_coordinates(gcp, 0, image_width=2560, image_height=1440)

    def test_missing_image_u_field_raises_value_error(self):
        """Test that missing image_u field raises ValueError."""
        gcp = {"image_v": 680.0}
        with self.assertRaisesRegex(ValueError, "missing required 'image_u'"):
            validate_gcp_image_coordinates(gcp, 0)

    def test_missing_image_v_field_raises_value_error(self):
        """Test that missing image_v field raises ValueError."""
        gcp = {"image_u": 1250.5}
        with self.assertRaisesRegex(ValueError, "missing required 'image_v'"):
            validate_gcp_image_coordinates(gcp, 0)

    def test_image_u_outside_image_width_raises_value_error(self):
        """Test that image_u outside image width raises ValueError when dimensions provided."""
        gcp = {
            "image_u": 2560.0,  # At or beyond right edge
            "image_v": 680.0,
        }
        with self.assertRaisesRegex(ValueError, "image_u 2560.0 outside image width"):
            validate_gcp_image_coordinates(gcp, 0, image_width=2560, image_height=1440)

    def test_image_u_negative_raises_value_error(self):
        """Test that negative image_u coordinate raises ValueError when dimensions provided."""
        gcp = {"image_u": -1.0, "image_v": 680.0}
        with self.assertRaisesRegex(ValueError, "image_u -1.0 outside image width"):
            validate_gcp_image_coordinates(gcp, 0, image_width=2560, image_height=1440)

    def test_image_v_outside_image_height_raises_value_error(self):
        """Test that image_v outside image height raises ValueError when dimensions provided."""
        gcp = {
            "image_u": 1250.5,
            "image_v": 1440.0,  # At or beyond bottom edge
        }
        with self.assertRaisesRegex(ValueError, "image_v 1440.0 outside image height"):
            validate_gcp_image_coordinates(gcp, 0, image_width=2560, image_height=1440)

    def test_image_v_negative_raises_value_error(self):
        """Test that negative image_v coordinate raises ValueError when dimensions provided."""
        gcp = {"image_u": 1250.5, "image_v": -1.0}
        with self.assertRaisesRegex(ValueError, "image_v -1.0 outside image height"):
            validate_gcp_image_coordinates(gcp, 0, image_width=2560, image_height=1440)

    def test_validation_passes_when_dimensions_not_provided(self):
        """Test that validation passes when dimensions not provided (skips bounds check)."""
        gcp = {
            "image_u": 5000.0,  # Would be out of bounds if we had dimensions
            "image_v": 3000.0,
        }
        # Should not raise - bounds check is skipped
        validate_gcp_image_coordinates(gcp, 0)

    def test_validation_passes_at_boundaries(self):
        """Test that coordinates at valid boundaries pass."""
        # At origin (top-left corner)
        gcp_origin = {"image_u": 0.0, "image_v": 0.0}
        # Just before max (bottom-right corner)
        gcp_max = {"image_u": 2559.9, "image_v": 1439.9}
        # Should not raise
        validate_gcp_image_coordinates(gcp_origin, 0, image_width=2560, image_height=1440)
        validate_gcp_image_coordinates(gcp_max, 1, image_width=2560, image_height=1440)

    def test_non_numeric_image_u_raises_value_error(self):
        """Test that non-numeric image_u coordinate raises ValueError."""
        gcp = {
            "image_u": "1250.5",  # String instead of number
            "image_v": 680.0,
        }
        with self.assertRaisesRegex(ValueError, "image_u must be a number"):
            validate_gcp_image_coordinates(gcp, 0)

    def test_non_numeric_image_v_raises_value_error(self):
        """Test that non-numeric image_v coordinate raises ValueError."""
        gcp = {
            "image_u": 1250.5,
            "image_v": "680.0",  # String instead of number
        }
        with self.assertRaisesRegex(ValueError, "image_v must be a number"):
            validate_gcp_image_coordinates(gcp, 0)

    def test_only_width_provided_validates_u(self):
        """Test that image_u is validated when only width is provided."""
        gcp = {
            "image_u": 3000.0,  # Out of bounds for width 2000
            "image_v": 500.0,
        }
        with self.assertRaisesRegex(ValueError, "image_u 3000.0 outside image width"):
            validate_gcp_image_coordinates(gcp, 0, image_width=2000, image_height=None)

    def test_only_height_provided_validates_v(self):
        """Test that image_v is validated when only height is provided."""
        gcp = {
            "image_u": 500.0,
            "image_v": 2000.0,  # Out of bounds for height 1000
        }
        with self.assertRaisesRegex(ValueError, "image_v 2000.0 outside image height"):
            validate_gcp_image_coordinates(gcp, 0, image_width=None, image_height=1000)

    def test_only_width_provided_allows_any_v(self):
        """Test that image_v is not validated when only width is provided."""
        gcp = {
            "image_u": 500.0,
            "image_v": 999999.0,  # Would be out of bounds if height were checked
        }
        # Should not raise - v is not validated when height not provided
        validate_gcp_image_coordinates(gcp, 0, image_width=1000, image_height=None)

    def test_only_height_provided_allows_any_u(self):
        """Test that image_u is not validated when only height is provided."""
        gcp = {
            "image_u": 999999.0,  # Would be out of bounds if width were checked
            "image_v": 500.0,
        }
        # Should not raise - u is not validated when width not provided
        validate_gcp_image_coordinates(gcp, 0, image_width=None, image_height=1000)


class TestDetectDuplicateGCPs(unittest.TestCase):
    """Test duplicate GCP detection."""

    def test_no_duplicates_passes(self):
        """Test that list with no duplicates passes validation."""
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            },
            {
                "map_id": "map_valte",
                "map_pixel_x": 2456.2,
                "map_pixel_y": 695.5,
                "image_u": 2456.2,
                "image_v": 695.5,
            },
            {
                "map_id": "map_valte",
                "map_pixel_x": 1180.0,
                "map_pixel_y": 1820.3,
                "image_u": 1180.0,
                "image_v": 1820.3,
            },
        ]
        # Should not raise any exception
        detect_duplicate_gcps(gcps)

    def test_duplicate_map_and_image_pixels_raises_value_error(self):
        """Test that duplicate map AND image pixel coordinates raises ValueError."""
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
                "metadata": {"description": "Point A"},
            },
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,  # Same map coords
                "map_pixel_y": 567.8,
                "image_u": 1250.5,  # Same image coords
                "image_v": 680.0,
                "metadata": {"description": "Point B"},
            },
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate GCP detected"):
            detect_duplicate_gcps(gcps)

    def test_same_map_different_image_does_not_raise(self):
        """Test that same map coords but different image coords does NOT raise."""
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            },
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,  # Same map coords
                "map_pixel_y": 567.8,
                "image_u": 2456.2,  # Different image coords
                "image_v": 695.5,
            },
        ]
        # Should not raise - different image coordinates
        detect_duplicate_gcps(gcps)

    def test_same_image_different_map_does_not_raise(self):
        """Test that same image coords but different map coords does NOT raise."""
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            },
            {
                "map_id": "map_valte",
                "map_pixel_x": 2456.2,  # Different map coords
                "map_pixel_y": 695.5,
                "image_u": 1250.5,  # Same image coords
                "image_v": 680.0,
            },
        ]
        # Should not raise - different map coordinates
        detect_duplicate_gcps(gcps)

    def test_different_map_ids_not_duplicate(self):
        """Test that same coordinates on different maps are not duplicates."""
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            },
            {
                "map_id": "map_other",  # Different map_id
                "map_pixel_x": 1234.5,  # Same map pixel coords
                "map_pixel_y": 567.8,
                "image_u": 1250.5,  # Same image coords
                "image_v": 680.0,
            },
        ]
        # Should not raise - different map IDs
        detect_duplicate_gcps(gcps)

    def test_near_duplicate_within_epsilon_raises(self):
        """Test that near-duplicates within epsilon threshold raise ValueError."""
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            },
            {
                "map_id": "map_valte",
                # Map pixels within default epsilon (0.5 pixels)
                "map_pixel_x": 1234.6,
                "map_pixel_y": 567.9,
                # Image pixels within default epsilon (0.5 pixels)
                "image_u": 1250.6,
                "image_v": 680.1,
            },
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate GCP detected"):
            detect_duplicate_gcps(gcps)

    def test_near_duplicate_outside_epsilon_passes(self):
        """Test that near-duplicates outside epsilon threshold pass."""
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            },
            {
                "map_id": "map_valte",
                # Map pixels outside default epsilon (0.5 pixels)
                "map_pixel_x": 1235.5,
                "map_pixel_y": 568.8,
                # Image pixels outside default epsilon (0.5 pixels)
                "image_u": 1251.5,
                "image_v": 681.0,
            },
        ]
        # Should not raise - outside epsilon thresholds
        detect_duplicate_gcps(gcps)

    def test_custom_epsilon_values(self):
        """Test duplicate detection with custom epsilon values."""
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            },
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.6,
                "map_pixel_y": 567.9,
                "image_u": 1250.6,
                "image_v": 680.1,
            },
        ]
        # With larger epsilon, these should be considered duplicates
        with self.assertRaisesRegex(ValueError, "Duplicate GCP detected"):
            detect_duplicate_gcps(gcps, map_pixel_epsilon=1.0, image_pixel_epsilon=1.0)

    def test_single_gcp_passes(self):
        """Test that a single GCP passes (no duplicates possible)."""
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            }
        ]
        # Should not raise
        detect_duplicate_gcps(gcps)

    def test_empty_list_passes(self):
        """Test that empty list passes (no duplicates possible)."""
        gcps = []
        # Should not raise
        detect_duplicate_gcps(gcps)


class TestValidateGroundControlPoints(unittest.TestCase):
    """Test overall ground control points validation."""

    def test_valid_list_format_passes(self):
        """Test that valid list format passes validation."""
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            },
            {
                "map_id": "map_valte",
                "map_pixel_x": 2456.2,
                "map_pixel_y": 695.5,
                "image_u": 2456.2,
                "image_v": 695.5,
            },
        ]
        result = validate_ground_control_points(gcps)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result, list)

    def test_valid_dict_format_passes(self):
        """Test that valid dict format passes (multiple sets)."""
        gcps = {
            "set_1": [
                {
                    "map_id": "map_valte",
                    "map_pixel_x": 1234.5,
                    "map_pixel_y": 567.8,
                    "image_u": 1250.5,
                    "image_v": 680.0,
                },
                {
                    "map_id": "map_valte",
                    "map_pixel_x": 2456.2,
                    "map_pixel_y": 695.5,
                    "image_u": 2456.2,
                    "image_v": 695.5,
                },
            ]
        }
        result = validate_ground_control_points(gcps)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result, list)

    def test_empty_dict_logs_warning(self):
        """Test that empty dict logs warning and returns empty list."""
        gcps = {}
        result = validate_ground_control_points(gcps)
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, list)

    def test_fewer_than_min_gcp_count_logs_warning(self):
        """Test that fewer than min_gcp_count logs warning but passes."""
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            },
            {
                "map_id": "map_valte",
                "map_pixel_x": 2456.2,
                "map_pixel_y": 695.5,
                "image_u": 2456.2,
                "image_v": 695.5,
            },
        ]
        # Should not raise, but logs warning (default min is 6)
        result = validate_ground_control_points(gcps, min_gcp_count=6)
        self.assertEqual(len(result), 2)

    def test_invalid_format_raises_value_error(self):
        """Test that invalid format raises ValueError."""
        gcps = "not a list or dict"
        with self.assertRaisesRegex(ValueError, "must be a list or dict"):
            validate_ground_control_points(gcps)

    def test_dict_with_non_list_value_raises_value_error(self):
        """Test that dict with non-list value raises ValueError."""
        gcps = {"set_1": "not a list"}
        with self.assertRaisesRegex(ValueError, "values must be lists"):
            validate_ground_control_points(gcps)

    def test_nested_validation_errors_are_raised_map_coords(self):
        """Test that nested map coordinate validation errors are raised."""
        gcps = [
            {
                "map_id": "",  # Invalid map_id
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            }
        ]
        with self.assertRaisesRegex(ValueError, "must be a non-empty string"):
            validate_ground_control_points(gcps)

    def test_nested_validation_errors_are_raised_image_coords(self):
        """Test that nested image coordinate validation errors are raised."""
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 3000.0,  # Out of bounds
                "image_v": 680.0,
            }
        ]
        with self.assertRaisesRegex(ValueError, "image_u 3000.0 outside image width"):
            validate_ground_control_points(gcps, image_width=2560, image_height=1440)

    def test_nested_validation_errors_are_raised_duplicates(self):
        """Test that duplicate detection errors are raised."""
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            },
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            },
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate GCP detected"):
            validate_ground_control_points(gcps)

    def test_non_dict_gcp_item_raises_value_error(self):
        """Test that non-dict GCP item raises ValueError."""
        gcps = [
            "not a dict",
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            },
        ]
        with self.assertRaisesRegex(ValueError, "must be a dictionary"):
            validate_ground_control_points(gcps)

    def test_image_dimensions_used_for_bounds_checking(self):
        """Test that image dimensions are used for pixel bounds checking."""
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            }
        ]
        # Should pass with correct dimensions
        result = validate_ground_control_points(gcps, image_width=2560, image_height=1440)
        self.assertEqual(len(result), 1)

        # Should fail with too-small dimensions
        with self.assertRaisesRegex(ValueError, "image_u 1250.5 outside image width"):
            validate_ground_control_points(gcps, image_width=1000, image_height=1440)

    def test_map_dimensions_used_for_bounds_checking(self):
        """Test that map dimensions are used for map pixel bounds checking."""
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": 1234.5,
                "map_pixel_y": 567.8,
                "image_u": 1250.5,
                "image_v": 680.0,
            }
        ]
        # Should pass with correct dimensions
        result = validate_ground_control_points(gcps, map_width=2000, map_height=1500)
        self.assertEqual(len(result), 1)

        # Should fail with too-small dimensions
        with self.assertRaisesRegex(ValueError, "map_pixel_x 1234.5 outside map width"):
            validate_ground_control_points(gcps, map_width=1000, map_height=1500)

    def test_multiple_gcp_sets_uses_first_set(self):
        """Test that when multiple GCP sets provided, first set is used."""
        gcps = {
            "set_1": [
                {
                    "map_id": "map_valte",
                    "map_pixel_x": 1234.5,
                    "map_pixel_y": 567.8,
                    "image_u": 1250.5,
                    "image_v": 680.0,
                }
            ],
            "set_2": [
                {
                    "map_id": "map_valte",
                    "map_pixel_x": 2456.2,
                    "map_pixel_y": 695.5,
                    "image_u": 2456.2,
                    "image_v": 695.5,
                }
            ],
        }
        result = validate_ground_control_points(gcps)
        # Should return first set
        self.assertEqual(len(result), 1)


class TestHomographyConfigIntegration(unittest.TestCase):
    """Integration tests with HomographyConfig.from_yaml."""

    def test_loading_config_with_gcps_succeeds(self):
        """Test loading the actual homography_config.yaml with GCPs succeeds."""
        # Get the path to the config file
        test_dir = Path(__file__).parent
        config_path = test_dir.parent / "homography_config.yaml"

        # Skip test if config file doesn't exist
        if not config_path.exists():
            self.skipTest(f"Config file not found at {config_path}")

        # Should load without errors
        config = HomographyConfig.from_yaml(str(config_path))

        # Verify it's a valid config object
        self.assertIsInstance(config, HomographyConfig)

    def test_gcps_are_accessible_after_loading(self):
        """Test that GCPs are accessible after loading config."""
        # Get the path to the config file
        test_dir = Path(__file__).parent
        config_path = test_dir.parent / "homography_config.yaml"

        # Skip test if config file doesn't exist
        if not config_path.exists():
            self.skipTest(f"Config file not found at {config_path}")

        # Load config
        config = HomographyConfig.from_yaml(str(config_path))

        # Get feature_match config
        feature_match_config = config.get_approach_config(
            config.approach if hasattr(config, "approach") else None
        )

        # If feature_match approach is not the primary, try getting it explicitly
        if "ground_control_points" not in feature_match_config:
            from homography_interface import HomographyApproach

            feature_match_config = config.get_approach_config(HomographyApproach.FEATURE_MATCH)

        # Verify GCPs exist and are accessible
        if "ground_control_points" in feature_match_config:
            gcps = feature_match_config["ground_control_points"]
            self.assertIsInstance(gcps, list)
            self.assertGreater(len(gcps), 0, "Config should have at least one GCP")

            # Verify first GCP has expected structure
            first_gcp = gcps[0]
            self.assertIn("map_id", first_gcp)
            self.assertIn("map_pixel_x", first_gcp)
            self.assertIn("map_pixel_y", first_gcp)
            self.assertIn("image_u", first_gcp)
            self.assertIn("image_v", first_gcp)

    def test_invalid_gcp_in_config_raises_error(self):
        """Test that loading config with invalid GCP raises appropriate error."""
        # Create a temporary config with invalid GCP
        test_config = {
            "homography": {
                "approach": "feature_match",
                "feature_match": {
                    "detector": "sift",
                    "min_matches": 4,
                    "ransac_threshold": 5.0,
                    "ground_control_points": [
                        {
                            "map_id": "",  # Invalid - empty string
                            "map_pixel_x": 1234.5,
                            "map_pixel_y": 567.8,
                            "image_u": 100.0,
                            "image_v": 200.0,
                        }
                    ],
                },
            }
        }

        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(test_config, f)
            temp_path = f.name

        try:
            with self.assertRaisesRegex(ValueError, "Ground control points validation failed"):
                HomographyConfig.from_yaml(temp_path)
        finally:
            os.unlink(temp_path)


class TestIsValidFiniteNumber(unittest.TestCase):
    """Test the _is_valid_finite_number helper function."""

    def test_valid_integers(self):
        """Test that integers are valid."""
        self.assertTrue(_is_valid_finite_number(0))
        self.assertTrue(_is_valid_finite_number(42))
        self.assertTrue(_is_valid_finite_number(-100))

    def test_valid_floats(self):
        """Test that finite floats are valid."""
        self.assertTrue(_is_valid_finite_number(0.0))
        self.assertTrue(_is_valid_finite_number(3.14159))
        self.assertTrue(_is_valid_finite_number(-273.15))

    def test_nan_is_invalid(self):
        """Test that NaN is rejected."""
        self.assertFalse(_is_valid_finite_number(float("nan")))

    def test_infinity_is_invalid(self):
        """Test that infinity is rejected."""
        self.assertFalse(_is_valid_finite_number(float("inf")))
        self.assertFalse(_is_valid_finite_number(float("-inf")))

    def test_strings_are_invalid(self):
        """Test that strings are rejected."""
        self.assertFalse(_is_valid_finite_number("42"))
        self.assertFalse(_is_valid_finite_number("3.14"))

    def test_none_is_invalid(self):
        """Test that None is rejected."""
        self.assertFalse(_is_valid_finite_number(None))

    def test_complex_numbers_are_invalid(self):
        """Test that complex numbers are rejected."""
        self.assertFalse(_is_valid_finite_number(complex(1, 2)))


class TestNaNInfinityValidation(unittest.TestCase):
    """Test that NaN and Infinity values are properly rejected."""

    def test_nan_map_pixel_x_raises_value_error(self):
        """Test that NaN map_pixel_x raises ValueError."""
        gcp = {"map_id": "map_valte", "map_pixel_x": float("nan"), "map_pixel_y": 567.8}
        with self.assertRaisesRegex(ValueError, "must be a finite number"):
            validate_gcp_map_coordinates(gcp, 0)

    def test_infinity_map_pixel_x_raises_value_error(self):
        """Test that infinite map_pixel_x raises ValueError."""
        gcp = {"map_id": "map_valte", "map_pixel_x": float("inf"), "map_pixel_y": 567.8}
        with self.assertRaisesRegex(ValueError, "must be a finite number"):
            validate_gcp_map_coordinates(gcp, 0)

    def test_nan_map_pixel_y_raises_value_error(self):
        """Test that NaN map_pixel_y raises ValueError."""
        gcp = {"map_id": "map_valte", "map_pixel_x": 1234.5, "map_pixel_y": float("nan")}
        with self.assertRaisesRegex(ValueError, "must be a finite number"):
            validate_gcp_map_coordinates(gcp, 0)

    def test_nan_image_u_raises_value_error(self):
        """Test that NaN image_u raises ValueError."""
        gcp = {"image_u": float("nan"), "image_v": 100.0}
        with self.assertRaisesRegex(ValueError, "must be a finite number"):
            validate_gcp_image_coordinates(gcp, 0)

    def test_infinity_image_u_raises_value_error(self):
        """Test that infinite image_u raises ValueError."""
        gcp = {"image_u": float("inf"), "image_v": 100.0}
        with self.assertRaisesRegex(ValueError, "must be a finite number"):
            validate_gcp_image_coordinates(gcp, 0)

    def test_nan_image_v_raises_value_error(self):
        """Test that NaN image_v raises ValueError."""
        gcp = {"image_u": 100.0, "image_v": float("nan")}
        with self.assertRaisesRegex(ValueError, "must be a finite number"):
            validate_gcp_image_coordinates(gcp, 0)


class TestImageDimensionValidation(unittest.TestCase):
    """Test image dimension parameter validation."""

    def test_valid_dimensions(self):
        """Test that valid dimensions are accepted."""
        self.assertEqual(_validate_image_dimension(1920, "image_width"), 1920)
        self.assertEqual(_validate_image_dimension(1080, "image_height"), 1080)

    def test_none_dimension_returns_none(self):
        """Test that None dimension returns None."""
        self.assertIsNone(_validate_image_dimension(None, "image_width"))

    def test_zero_dimension_raises_value_error(self):
        """Test that zero dimension raises ValueError."""
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _validate_image_dimension(0, "image_width")

    def test_negative_dimension_raises_value_error(self):
        """Test that negative dimension raises ValueError."""
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _validate_image_dimension(-100, "image_height")

    def test_nan_dimension_raises_value_error(self):
        """Test that NaN dimension raises ValueError."""
        with self.assertRaisesRegex(ValueError, "must be a finite positive integer"):
            _validate_image_dimension(float("nan"), "image_width")

    def test_infinity_dimension_raises_value_error(self):
        """Test that infinite dimension raises ValueError."""
        with self.assertRaisesRegex(ValueError, "must be a finite positive integer"):
            _validate_image_dimension(float("inf"), "image_height")

    def test_string_dimension_raises_value_error(self):
        """Test that string dimension raises ValueError."""
        with self.assertRaisesRegex(ValueError, "must be a positive integer"):
            _validate_image_dimension("1920", "image_width")

    def test_excessively_large_dimension_raises_value_error(self):
        """Test that excessively large dimension raises ValueError."""
        with self.assertRaisesRegex(ValueError, "exceeds maximum allowed"):
            _validate_image_dimension(100001, "image_width")


class TestMaxGCPCount(unittest.TestCase):
    """Test maximum GCP count limit."""

    def test_max_gcp_count_constant_exists(self):
        """Test that MAX_GCP_COUNT constant is defined."""
        self.assertEqual(MAX_GCP_COUNT, 1000)

    def test_exceeding_max_gcp_count_raises_value_error(self):
        """Test that exceeding MAX_GCP_COUNT raises ValueError."""
        # Create a list with too many GCPs
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": float(i),
                "map_pixel_y": float(i),
                "image_u": float(i),
                "image_v": float(i),
            }
            for i in range(MAX_GCP_COUNT + 1)
        ]
        with self.assertRaisesRegex(ValueError, "Too many GCPs provided"):
            validate_ground_control_points(gcps)

    def test_max_gcp_count_exactly_is_allowed(self):
        """Test that exactly MAX_GCP_COUNT GCPs is allowed."""
        # Create a list with exactly max GCPs
        gcps = [
            {
                "map_id": "map_valte",
                "map_pixel_x": float(i * 10),
                "map_pixel_y": float(i * 10),
                "image_u": float(i * 10),
                "image_v": float(i * 10),
            }
            for i in range(MAX_GCP_COUNT)
        ]
        # Should not raise
        result = validate_ground_control_points(gcps, min_gcp_count=0)
        self.assertEqual(len(result), MAX_GCP_COUNT)


class TestDescriptionSanitization(unittest.TestCase):
    """Test that description field is properly sanitized in error messages."""

    def test_long_description_is_truncated(self):
        """Test that very long descriptions are truncated."""
        long_desc = "A" * 500  # 500 characters
        gcp = {
            "map_id": "map_valte",
            "map_pixel_x": -1.0,  # Invalid to trigger error
            "map_pixel_y": 567.8,
            "metadata": {"description": long_desc},
        }
        try:
            validate_gcp_map_coordinates(gcp, 0, map_width=2000, map_height=1500)
            self.fail("Expected ValueError")
        except ValueError as e:
            error_msg = str(e)
            # Description should be truncated with "..."
            self.assertIn("...", error_msg)
            # Error message shouldn't contain the full 500 character description
            self.assertLess(len(error_msg), 400)

    def test_control_characters_are_removed(self):
        """Test that control characters are removed from description."""
        bad_desc = "Point\x00with\nnewline\rand\ttab"
        gcp = {
            "map_id": "map_valte",
            "map_pixel_x": -1.0,  # Invalid to trigger error
            "map_pixel_y": 567.8,
            "metadata": {"description": bad_desc},
        }
        try:
            validate_gcp_map_coordinates(gcp, 0, map_width=2000, map_height=1500)
            self.fail("Expected ValueError")
        except ValueError as e:
            error_msg = str(e)
            # Null byte should be removed
            self.assertNotIn("\x00", error_msg)


class TestNumpyTypeSupport(unittest.TestCase):
    """Test that numpy types are properly handled (if numpy is available)."""

    def setUp(self):
        """Check if numpy is available."""
        try:
            import numpy as np

            self.np = np
            self.numpy_available = True
        except ImportError:
            self.numpy_available = False

    def test_numpy_float64_is_valid(self):
        """Test that numpy.float64 is accepted."""
        if not self.numpy_available:
            self.skipTest("numpy not available")

        gcp = {
            "map_id": "map_valte",
            "map_pixel_x": self.np.float64(1234.5),
            "map_pixel_y": self.np.float64(567.8),
            "image_u": self.np.float64(1250.5),
            "image_v": self.np.float64(680.0),
        }
        # Should not raise
        validate_gcp_map_coordinates(gcp, 0)
        validate_gcp_image_coordinates(gcp, 0)

    def test_numpy_int64_is_valid(self):
        """Test that numpy.int64 is accepted."""
        if not self.numpy_available:
            self.skipTest("numpy not available")

        gcp = {
            "map_id": "map_valte",
            "map_pixel_x": self.np.int64(1234),
            "map_pixel_y": self.np.int64(567),
            "image_u": self.np.int64(1250),
            "image_v": self.np.int64(680),
        }
        # Should not raise
        validate_gcp_map_coordinates(gcp, 0)
        validate_gcp_image_coordinates(gcp, 0)

    def test_numpy_nan_is_rejected(self):
        """Test that numpy.nan is rejected."""
        if not self.numpy_available:
            self.skipTest("numpy not available")

        gcp = {"map_id": "map_valte", "map_pixel_x": self.np.nan, "map_pixel_y": 567.8}
        with self.assertRaisesRegex(ValueError, "must be a finite number"):
            validate_gcp_map_coordinates(gcp, 0)

    def test_numpy_infinity_is_rejected(self):
        """Test that numpy.inf is rejected."""
        if not self.numpy_available:
            self.skipTest("numpy not available")

        gcp = {"map_id": "map_valte", "map_pixel_x": self.np.inf, "map_pixel_y": 567.8}
        with self.assertRaisesRegex(ValueError, "must be a finite number"):
            validate_gcp_map_coordinates(gcp, 0)


class TestConfigGCPPixelBoundsCI(unittest.TestCase):
    """CI test to verify GCPs in config/homography_config.yaml are within image bounds.

    This test ensures that all GCP pixel coordinates in the production config file
    are valid with respect to the image dimensions specified in camera_capture_context.
    This prevents issues where invalid GCPs cause incorrect homography computation.
    """

    def test_config_gcps_within_image_bounds(self):
        """Test that all GCPs in homography_config.yaml have valid pixel coordinates.

        This test:
        1. Loads the production config file
        2. Extracts image dimensions from camera_capture_context
        3. Validates all GCPs have image_u < image_width and image_v < image_height

        If this test fails, the config contains GCPs that would cause RANSAC to fit
        invalid points, resulting in incorrect homography matrices.
        """
        # Get the path to the config file
        test_dir = Path(__file__).parent
        config_path = test_dir.parent / "config" / "homography_config.yaml"

        # Skip test if config file doesn't exist
        if not config_path.exists():
            self.skipTest(f"Config file not found at {config_path}")

        # Load config
        config = HomographyConfig.from_yaml(str(config_path))

        # Get feature_match config
        feature_match_config = config.approach_specific_config.get("feature_match", {})

        # Get camera capture context for image dimensions
        camera_context = feature_match_config.get("camera_capture_context", {})
        image_width = camera_context.get("image_width")
        image_height = camera_context.get("image_height")

        self.assertIsNotNone(image_width, "Config missing image_width in camera_capture_context")
        self.assertIsNotNone(image_height, "Config missing image_height in camera_capture_context")

        # Get GCPs - require exactly 6 as specified in Issue #31
        gcps = feature_match_config.get("ground_control_points", [])
        self.assertEqual(len(gcps), 6, f"Config should have exactly 6 GCPs, found {len(gcps)}")

        # Validate each GCP's pixel coordinates are within bounds
        errors = []
        for i, gcp in enumerate(gcps):
            image_u = gcp["image_u"]
            image_v = gcp["image_v"]
            desc = gcp.get("metadata", {}).get("description", f"GCP {i + 1}")

            if image_u < 0 or image_u >= image_width:
                errors.append(f"{desc}: image_u={image_u} outside [0, {image_width})")
            if image_v < 0 or image_v >= image_height:
                errors.append(f"{desc}: image_v={image_v} outside [0, {image_height})")

        if errors:
            self.fail(
                f"GCPs with invalid pixel coordinates found in config:\n"
                f"  Image dimensions: {image_width} x {image_height}\n"
                f"  Invalid GCPs:\n    " + "\n    ".join(errors)
            )


def main():
    """Run all tests."""
    # Run tests with unittest's test runner
    unittest.main(argv=[""], verbosity=2, exit=False)


if __name__ == "__main__":
    main()
