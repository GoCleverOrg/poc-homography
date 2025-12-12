#!/usr/bin/env python3
"""
Unit and integration tests for outlier removal in capture_gcps_web.py.

Tests verify that:
1. get_outliers() returns GCP indices based on RANSAC inlier mask (not error threshold)
2. remove_outliers() removes exactly the GCPs shown as outliers in the UI
3. The outlier count displayed matches the actual removal count

Issue #62: Fix inconsistent outlier removal (RANSAC vs threshold mismatch)
"""

import unittest
import sys
import os
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))


class TestGetOutliersRANSACBased(unittest.TestCase):
    """Test that get_outliers() uses RANSAC inlier mask instead of error threshold."""

    def setUp(self):
        """Create a mock session with controllable inlier mask."""
        # We need to import after path setup
        from capture_gcps_web import GCPCaptureWebSession

        # Create a minimal frame for the session
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Create session - this will fail if coordinate converter not available
        # but we'll mock the relevant attributes
        with patch.object(GCPCaptureWebSession, '__init__', lambda self, **kwargs: None):
            self.session = GCPCaptureWebSession()
            self.session.gcps = []
            self.session.inlier_mask = None
            self.session.last_reproj_errors = []
            self.session.current_homography = None
            self.session.frame_width = 1920
            self.session.frame_height = 1080
            self.session.reference_lat = None
            self.session.reference_lon = None
            self.session.coord_converter = None

    def test_get_outliers_returns_ransac_outlier_indices(self):
        """Verify get_outliers returns indices where inlier_mask is False."""
        # Setup: predetermined RANSAC mask with known outliers at indices 2, 4, 5
        self.session.inlier_mask = [1, 1, 0, 1, 0, 0, 1]  # 0 = outlier
        self.session.gcps = [{'id': i} for i in range(7)]

        # Mock update_homography to avoid actual computation
        self.session.update_homography = Mock(return_value={})

        # Act
        outliers = self.session.get_outliers()

        # Assert
        self.assertEqual(outliers, [2, 4, 5])

    def test_get_outliers_ignores_error_magnitude(self):
        """Verify outliers are based on RANSAC mask, not error values."""
        # Setup: GCP at index 0 has high error (20px) but is RANSAC inlier
        # GCP at index 1 has low error (3px) but is RANSAC outlier
        self.session.inlier_mask = [1, 0, 1, 1]  # Only index 1 is outlier
        self.session.last_reproj_errors = [20.0, 3.0, 5.0, 2.0]  # Index 0 has highest error
        self.session.gcps = [{'id': i} for i in range(4)]

        self.session.update_homography = Mock(return_value={})

        # Act
        outliers = self.session.get_outliers()

        # Assert: Only index 1 should be an outlier (RANSAC says so)
        # Index 0 should NOT be an outlier despite 20px error
        self.assertEqual(outliers, [1])
        self.assertNotIn(0, outliers)

    def test_get_outliers_returns_empty_when_all_inliers(self):
        """Verify empty list when RANSAC marks all GCPs as inliers."""
        self.session.inlier_mask = [1, 1, 1, 1, 1]
        self.session.gcps = [{'id': i} for i in range(5)]
        self.session.update_homography = Mock(return_value={})

        outliers = self.session.get_outliers()

        self.assertEqual(outliers, [])

    def test_get_outliers_returns_all_when_all_outliers(self):
        """Verify all indices returned when RANSAC marks all GCPs as outliers."""
        self.session.inlier_mask = [0, 0, 0, 0]
        self.session.gcps = [{'id': i} for i in range(4)]
        self.session.update_homography = Mock(return_value={})

        outliers = self.session.get_outliers()

        self.assertEqual(outliers, [0, 1, 2, 3])

    def test_get_outliers_empty_when_no_inlier_mask(self):
        """Verify empty list when inlier_mask is None (no homography computed)."""
        self.session.inlier_mask = None
        self.session.gcps = [{'id': i} for i in range(5)]
        # Mock update_homography to keep inlier_mask as None (failed computation)
        self.session.update_homography = Mock(return_value={})

        outliers = self.session.get_outliers()

        self.assertEqual(outliers, [])

    def test_get_outliers_recalculates_when_mask_stale(self):
        """Verify homography is recalculated if mask length doesn't match GCPs."""
        # Setup: stale mask with wrong length
        self.session.inlier_mask = [1, 1, 0]  # 3 elements
        self.session.gcps = [{'id': i} for i in range(5)]  # 5 GCPs

        # Mock update_homography to set correct mask on call
        def update_mock():
            self.session.inlier_mask = [1, 0, 1, 0, 1]  # 5 elements
            return {}

        self.session.update_homography = Mock(side_effect=update_mock)

        outliers = self.session.get_outliers()

        # Should have recalculated and returned correct outliers
        self.session.update_homography.assert_called()
        self.assertEqual(outliers, [1, 3])


class TestRemoveOutliersConsistency(unittest.TestCase):
    """Test that remove_outliers removes exactly what get_outliers returns."""

    def setUp(self):
        """Create a mock session for testing removal."""
        from capture_gcps_web import GCPCaptureWebSession

        with patch.object(GCPCaptureWebSession, '__init__', lambda self, **kwargs: None):
            self.session = GCPCaptureWebSession()
            self.session.gcps = []
            self.session.inlier_mask = None
            self.session.last_reproj_errors = []
            self.session.current_homography = None
            self.session.frame_width = 1920
            self.session.frame_height = 1080
            self.session.reference_lat = None
            self.session.reference_lon = None
            self.session.coord_converter = None

    def test_remove_outliers_removes_exactly_ransac_outliers(self):
        """Verify removal count equals RANSAC outlier count, not threshold-based."""
        # Setup: 7 GCPs with 3 RANSAC outliers
        self.session.gcps = [
            {'metadata': {'description': f'GCP {i}'}} for i in range(7)
        ]
        self.session.inlier_mask = [1, 1, 0, 1, 0, 0, 1]  # Outliers at 2, 4, 5

        # Mock update_homography to maintain current state after removal
        def update_mock():
            # After removal, all remaining should be inliers
            self.session.inlier_mask = [1] * len(self.session.gcps)
            return {}

        self.session.update_homography = Mock(side_effect=update_mock)

        # Act
        result = self.session.remove_outliers()

        # Assert
        self.assertEqual(result['removed_count'], 3)
        self.assertEqual(result['removed_indices'], [2, 4, 5])
        self.assertEqual(result['remaining_gcps'], 4)
        self.assertEqual(len(self.session.gcps), 4)

    def test_remove_outliers_preserves_inliers(self):
        """Verify inliers are preserved after outlier removal."""
        # Setup: 5 GCPs, outliers at indices 1 and 3
        self.session.gcps = [
            {'metadata': {'description': f'GCP {i}'}} for i in range(5)
        ]
        self.session.inlier_mask = [1, 0, 1, 0, 1]  # Outliers at 1, 3

        self.session.update_homography = Mock(return_value={})

        # Act
        result = self.session.remove_outliers()

        # Assert: Should have 3 remaining GCPs (indices 0, 2, 4)
        self.assertEqual(result['removed_count'], 2)
        self.assertEqual(len(self.session.gcps), 3)

        # Verify the correct GCPs were kept
        remaining_descriptions = [
            g['metadata']['description'] for g in self.session.gcps
        ]
        self.assertIn('GCP 0', remaining_descriptions)
        self.assertIn('GCP 2', remaining_descriptions)
        self.assertIn('GCP 4', remaining_descriptions)
        self.assertNotIn('GCP 1', remaining_descriptions)
        self.assertNotIn('GCP 3', remaining_descriptions)

    def test_remove_outliers_returns_zero_when_no_outliers(self):
        """Verify zero removal when all GCPs are inliers."""
        self.session.gcps = [
            {'metadata': {'description': f'GCP {i}'}} for i in range(5)
        ]
        self.session.inlier_mask = [1, 1, 1, 1, 1]  # All inliers

        self.session.update_homography = Mock(return_value={})

        result = self.session.remove_outliers()

        self.assertEqual(result['removed_count'], 0)
        self.assertEqual(result['removed_indices'], [])
        self.assertEqual(len(self.session.gcps), 5)


class TestOutlierUIConsistency(unittest.TestCase):
    """Test that UI display matches removal behavior."""

    def setUp(self):
        """Create session for UI consistency tests."""
        from capture_gcps_web import GCPCaptureWebSession

        with patch.object(GCPCaptureWebSession, '__init__', lambda self, **kwargs: None):
            self.session = GCPCaptureWebSession()
            self.session.gcps = []
            self.session.inlier_mask = None
            self.session.last_reproj_errors = []
            self.session.current_homography = Mock()  # Non-None = homography computed
            self.session.frame_width = 1920
            self.session.frame_height = 1080
            self.session.reference_lat = 39.0
            self.session.reference_lon = -0.2
            self.session.reference_utm_easting = 728000
            self.session.reference_utm_northing = 4390000
            self.session.coord_converter = Mock()
            self.session.utm_crs = "EPSG:25830"

    def test_ui_outlier_count_matches_removal_count(self):
        """Integration test: UI outlier count equals actual removal count."""
        # This test simulates the full workflow:
        # 1. update_homography() computes RANSAC mask and returns UI info
        # 2. UI displays outlier count from homography['outliers']
        # 3. remove_outliers() removes GCPs
        # 4. Verify counts match

        # Setup: 10 GCPs with controlled errors and RANSAC mask
        self.session.gcps = [
            {
                'pixel': {'u': 100 + i * 100, 'v': 100 + i * 50},
                'gps': {'latitude': 39.0 + i * 0.001, 'longitude': -0.2 + i * 0.001},
                'utm': {'easting': 728000 + i * 100, 'northing': 4390000 + i * 100},
                'metadata': {'description': f'GCP {i+1}'}
            }
            for i in range(10)
        ]

        # Simulate RANSAC marking 4 GCPs as outliers (indices 2, 5, 7, 9)
        self.session.inlier_mask = [1, 1, 0, 1, 1, 0, 1, 0, 1, 0]
        self.session.last_reproj_errors = [2.0, 3.0, 8.0, 1.5, 4.0, 12.0, 3.0, 7.0, 2.5, 9.0]

        # Build UI info similar to update_homography() output
        gcp_errors = []
        for i, (gcp, error, is_inlier) in enumerate(
            zip(self.session.gcps, self.session.last_reproj_errors, self.session.inlier_mask)
        ):
            gcp_errors.append({
                'index': i,
                'description': gcp['metadata']['description'],
                'error_px': float(error),
                'is_inlier': bool(is_inlier),
                'status': 'good' if error < 5.0 else 'warning' if error < 10.0 else 'bad'
            })

        # UI outliers = GCPs where is_inlier is False
        ui_outliers = [e for e in gcp_errors if not e['is_inlier']]
        ui_outlier_count = len(ui_outliers)

        # Mock update_homography to set new mask after removal
        def update_mock():
            self.session.inlier_mask = [1] * len(self.session.gcps)
            return {}

        self.session.update_homography = Mock(side_effect=update_mock)

        # Act: Remove outliers
        result = self.session.remove_outliers()

        # Assert: UI count matches removal count
        self.assertEqual(
            result['removed_count'],
            ui_outlier_count,
            f"UI showed {ui_outlier_count} outliers but removed {result['removed_count']}"
        )
        self.assertEqual(ui_outlier_count, 4)  # Sanity check
        self.assertEqual(len(self.session.gcps), 6)  # 10 - 4 = 6


class TestEdgeCases(unittest.TestCase):
    """Test edge cases for outlier handling."""

    def setUp(self):
        """Create session for edge case tests."""
        from capture_gcps_web import GCPCaptureWebSession

        with patch.object(GCPCaptureWebSession, '__init__', lambda self, **kwargs: None):
            self.session = GCPCaptureWebSession()
            self.session.gcps = []
            self.session.inlier_mask = None
            self.session.last_reproj_errors = []
            self.session.current_homography = None
            self.session.frame_width = 1920
            self.session.frame_height = 1080
            self.session.reference_lat = None
            self.session.reference_lon = None
            self.session.coord_converter = None

    def test_get_outliers_with_fewer_than_four_gcps(self):
        """Verify empty list when fewer than 4 GCPs (homography cannot be computed)."""
        self.session.gcps = [{'id': i} for i in range(3)]
        self.session.inlier_mask = None  # Not computed

        # Mock update_homography to keep inlier_mask as None
        self.session.update_homography = Mock(return_value={})

        outliers = self.session.get_outliers()

        self.assertEqual(outliers, [])

    def test_remove_outliers_with_empty_gcp_list(self):
        """Verify graceful handling when GCP list is empty."""
        self.session.gcps = []
        self.session.inlier_mask = None

        self.session.update_homography = Mock(return_value={})

        result = self.session.remove_outliers()

        self.assertEqual(result['removed_count'], 0)
        self.assertEqual(result['remaining_gcps'], 0)

    def test_inlier_mask_with_boolean_values(self):
        """Verify handling of boolean values in inlier_mask (True/False vs 1/0)."""
        self.session.gcps = [{'id': i} for i in range(5)]
        self.session.inlier_mask = [True, False, True, False, True]

        self.session.update_homography = Mock(return_value={})

        outliers = self.session.get_outliers()

        self.assertEqual(outliers, [1, 3])


if __name__ == '__main__':
    unittest.main()
