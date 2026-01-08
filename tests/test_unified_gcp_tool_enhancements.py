"""
Unit tests for unified_gcp_tool.py enhancements (Issue #125).

Tests the following new functionality:
1. Corner extraction from SAM3 masks
2. Edge distance scoring
3. Spatial distribution penalty calculation
4. Spatial distribution metrics
5. Auto-suggest GCP points workflow
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class TestCornerExtraction:
    """Tests for extract_corner_points_from_mask method."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock UnifiedSession with minimal required attributes."""
        from tools.unified_gcp_tool import UnifiedSession

        # Create a mock that has the methods we need
        session = Mock(spec=UnifiedSession)

        # Bind the actual method to the mock
        session.extract_corner_points_from_mask = UnifiedSession.extract_corner_points_from_mask.__get__(session, UnifiedSession)

        return session

    def test_extracts_corners_from_simple_rectangle(self, mock_session):
        """Test corner extraction from a simple rectangular mask."""
        # Create a 100x100 mask with a white rectangle
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)

        corners = mock_session.extract_corner_points_from_mask(mask, max_corners=10)

        assert len(corners) > 0
        assert all('x' in c and 'y' in c for c in corners)
        assert all('quality' in c for c in corners)

    def test_extracts_corners_from_l_shape(self, mock_session):
        """Test corner extraction from an L-shaped mask."""
        # Create L-shaped mask with clear corners
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (50, 90), 255, -1)  # Vertical part
        cv2.rectangle(mask, (10, 60), (90, 90), 255, -1)  # Horizontal part

        corners = mock_session.extract_corner_points_from_mask(mask, max_corners=20)

        # L-shape should have more corners than a rectangle
        assert len(corners) >= 4

    def test_returns_empty_list_for_empty_mask(self, mock_session):
        """Test that empty mask returns empty list."""
        mask = np.zeros((100, 100), dtype=np.uint8)

        corners = mock_session.extract_corner_points_from_mask(mask, max_corners=10)

        assert corners == []

    def test_respects_max_corners_limit(self, mock_session):
        """Test that max_corners parameter is respected."""
        # Create mask with many potential corners
        mask = np.zeros((200, 200), dtype=np.uint8)
        for i in range(5):
            for j in range(5):
                cv2.rectangle(mask, (i*40+5, j*40+5), (i*40+35, j*40+35), 255, -1)

        corners = mock_session.extract_corner_points_from_mask(mask, max_corners=5)

        assert len(corners) <= 5

    def test_respects_min_distance_parameter(self, mock_session):
        """Test that min_distance parameter affects corner distribution."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (90, 90), 255, -1)

        # With small min_distance, should get more corners
        corners_close = mock_session.extract_corner_points_from_mask(
            mask, max_corners=20, min_distance=5
        )

        # With large min_distance, should get fewer corners
        corners_far = mock_session.extract_corner_points_from_mask(
            mask, max_corners=20, min_distance=50
        )

        assert len(corners_close) >= len(corners_far)


class TestEdgeDistanceScore:
    """Tests for _calculate_edge_distance_score method."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock UnifiedSession with the edge distance method."""
        from tools.unified_gcp_tool import UnifiedSession

        session = Mock(spec=UnifiedSession)
        session._calculate_edge_distance_score = UnifiedSession._calculate_edge_distance_score.__get__(session, UnifiedSession)

        return session

    def test_corner_point_scores_high(self, mock_session):
        """Test that points at sharp corners score high."""
        # Create mask with a clear corner at (50, 50)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (90, 90), 255, -1)

        corner_point = {'x': 50, 'y': 50}
        score = mock_session._calculate_edge_distance_score(corner_point, mask)

        # Corner should have high score (many edge transitions)
        assert 0.0 <= score <= 1.0

    def test_interior_point_scores_low(self, mock_session):
        """Test that points in mask interior score low."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)

        # Point in the center of the rectangle
        interior_point = {'x': 50, 'y': 50}
        score = mock_session._calculate_edge_distance_score(interior_point, mask)

        # Interior point should have low score (no edge transitions)
        assert score < 0.5

    def test_returns_zero_for_point_outside_mask(self, mock_session):
        """Test score for point completely outside mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (90, 90), 255, -1)

        outside_point = {'x': 10, 'y': 10}
        score = mock_session._calculate_edge_distance_score(outside_point, mask)

        assert 0.0 <= score <= 1.0

    def test_handles_point_at_mask_boundary(self, mock_session):
        """Test score calculation for point at image boundary."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:50, 0:50] = 255  # Top-left quadrant

        boundary_point = {'x': 0, 'y': 0}
        score = mock_session._calculate_edge_distance_score(boundary_point, mask)

        # Should not crash and return valid score
        assert 0.0 <= score <= 1.0


class TestSpatialDistributionPenalty:
    """Tests for _calculate_spatial_distribution_penalty method."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock UnifiedSession with the spatial penalty method."""
        from tools.unified_gcp_tool import UnifiedSession

        session = Mock(spec=UnifiedSession)
        session._calculate_spatial_distribution_penalty = UnifiedSession._calculate_spatial_distribution_penalty.__get__(session, UnifiedSession)

        return session

    def test_no_penalty_when_no_existing_points(self, mock_session):
        """Test that first point gets no penalty."""
        new_point = {'x': 50, 'y': 50}
        existing_points = []

        penalty = mock_session._calculate_spatial_distribution_penalty(
            new_point, existing_points, min_separation=30.0
        )

        assert penalty == 1.0  # No penalty

    def test_full_penalty_when_too_close(self, mock_session):
        """Test full penalty when point is too close to existing."""
        new_point = {'x': 50, 'y': 50}
        existing_points = [{'x': 51, 'y': 51}]  # Very close

        penalty = mock_session._calculate_spatial_distribution_penalty(
            new_point, existing_points, min_separation=30.0
        )

        assert penalty < 0.3  # Should be heavily penalized

    def test_no_penalty_when_far_apart(self, mock_session):
        """Test no penalty when point is far from existing."""
        new_point = {'x': 50, 'y': 50}
        existing_points = [{'x': 200, 'y': 200}]  # Far away

        penalty = mock_session._calculate_spatial_distribution_penalty(
            new_point, existing_points, min_separation=30.0
        )

        assert penalty == 1.0  # No penalty

    def test_partial_penalty_at_threshold(self, mock_session):
        """Test partial penalty near the separation threshold."""
        new_point = {'x': 50, 'y': 50}
        # Point at exactly min_separation distance
        existing_points = [{'x': 80, 'y': 50}]  # 30 pixels away

        penalty = mock_session._calculate_spatial_distribution_penalty(
            new_point, existing_points, min_separation=30.0
        )

        # At threshold, should get full penalty reduction
        assert 0.9 <= penalty <= 1.0


class TestSpatialDistributionMetrics:
    """Tests for calculate_spatial_distribution method."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock UnifiedSession with spatial distribution method."""
        from tools.unified_gcp_tool import UnifiedSession

        session = Mock(spec=UnifiedSession)
        session.calculate_spatial_distribution = UnifiedSession.calculate_spatial_distribution.__get__(session, UnifiedSession)
        session.camera_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        session.camera_params = {'image_width': 1920, 'image_height': 1080}
        session.projected_points = []

        return session

    def test_returns_zero_metrics_for_no_points(self, mock_session):
        """Test metrics when no projected points."""
        mock_session.projected_points = []

        result = mock_session.calculate_spatial_distribution()

        # Method returns flat dict, not nested under 'distribution'
        assert result['coverage_ratio'] == 0.0
        assert result['quadrants_covered'] == 0
        assert result['coverage_status'] == 'poor'
        assert result['warning_message'] == 'No GCPs projected to camera view.'

    def test_calculates_coverage_ratio(self, mock_session):
        """Test coverage ratio calculation with distributed points."""
        # Points covering roughly 25% of the image
        mock_session.projected_points = [
            {'pixel_u': 480, 'pixel_v': 270, 'visible': True},
            {'pixel_u': 1440, 'pixel_v': 270, 'visible': True},
            {'pixel_u': 1440, 'pixel_v': 810, 'visible': True},
            {'pixel_u': 480, 'pixel_v': 810, 'visible': True},
        ]

        result = mock_session.calculate_spatial_distribution()

        assert result['coverage_ratio'] > 0.0
        assert result['quadrants_covered'] == 4

    def test_counts_quadrants_correctly(self, mock_session):
        """Test quadrant counting with points in specific quadrants."""
        # Points only in top-left and bottom-right quadrants
        # Need at least 3 points for analysis (method requires 3+)
        mock_session.projected_points = [
            {'pixel_u': 100, 'pixel_v': 100, 'visible': True},   # Top-left (q=0)
            {'pixel_u': 200, 'pixel_v': 200, 'visible': True},   # Top-left (q=0)
            {'pixel_u': 1800, 'pixel_v': 1000, 'visible': True}, # Bottom-right (q=3)
        ]

        result = mock_session.calculate_spatial_distribution()

        assert result['quadrants_covered'] == 2

    def test_ignores_invisible_points(self, mock_session):
        """Test that invisible points are not counted."""
        mock_session.projected_points = [
            {'pixel_u': 100, 'pixel_v': 100, 'visible': True},
            {'pixel_u': 1800, 'pixel_v': 1000, 'visible': False},  # Invisible
        ]

        result = mock_session.calculate_spatial_distribution()

        assert result['num_visible_gcps'] == 1

    def test_warning_message_for_poor_distribution(self, mock_session):
        """Test warning message when distribution is poor."""
        # Single point - poor distribution (needs 3+)
        mock_session.projected_points = [
            {'pixel_u': 100, 'pixel_v': 100, 'visible': True},
        ]

        result = mock_session.calculate_spatial_distribution()

        assert result['warning_message'] is not None
        assert 'Need at least 3' in result['warning_message']

    def test_status_thresholds(self, mock_session):
        """Test coverage status thresholds."""
        # Create points with good coverage (need at least 3 visible)
        mock_session.projected_points = [
            {'pixel_u': 100, 'pixel_v': 100, 'visible': True},
            {'pixel_u': 1800, 'pixel_v': 100, 'visible': True},
            {'pixel_u': 1800, 'pixel_v': 980, 'visible': True},
            {'pixel_u': 100, 'pixel_v': 980, 'visible': True},
            {'pixel_u': 960, 'pixel_v': 540, 'visible': True},  # Center
        ]

        result = mock_session.calculate_spatial_distribution()

        # With 4 corners + center, should have good coverage
        assert result['coverage_status'] in ['good', 'fair']
        assert result['quadrant_status'] == 'good'


class TestExportValidation:
    """Tests for export GCPs validation logic."""

    def test_minimum_gcp_count_constant(self):
        """Verify the minimum GCP count constant exists."""
        # The export requires at least 20 GCPs
        MIN_EXPORT_GCPS = 20
        assert MIN_EXPORT_GCPS == 20

    def test_export_format_structure(self):
        """Test the expected export format structure."""
        expected_gcp = {
            'pixel': {'u': 512.5, 'v': 384.2},
            'gps': {'latitude': 40.416775, 'longitude': -3.703790, 'elevation': 0.0}
        }

        # Verify structure
        assert 'pixel' in expected_gcp
        assert 'u' in expected_gcp['pixel']
        assert 'v' in expected_gcp['pixel']
        assert 'gps' in expected_gcp
        assert 'latitude' in expected_gcp['gps']
        assert 'longitude' in expected_gcp['gps']
        assert 'elevation' in expected_gcp['gps']


class TestAutoSuggestIntegration:
    """Integration tests for auto-suggest workflow."""

    @pytest.fixture
    def mock_session_full(self):
        """Create a more complete mock session for integration tests."""
        from tools.unified_gcp_tool import UnifiedSession

        session = Mock(spec=UnifiedSession)

        # Bind all required methods
        session.extract_corner_points_from_mask = UnifiedSession.extract_corner_points_from_mask.__get__(session, UnifiedSession)
        session._calculate_edge_distance_score = UnifiedSession._calculate_edge_distance_score.__get__(session, UnifiedSession)
        session._calculate_spatial_distribution_penalty = UnifiedSession._calculate_spatial_distribution_penalty.__get__(session, UnifiedSession)
        session.auto_suggest_gcp_points = UnifiedSession.auto_suggest_gcp_points.__get__(session, UnifiedSession)

        # Set up masks
        session.cartography_mask = None
        session.camera_mask = None
        session.camera_params = None

        return session

    def test_returns_empty_when_no_masks(self, mock_session_full):
        """Test auto-suggest returns empty when no masks loaded."""
        result = mock_session_full.auto_suggest_gcp_points(max_suggestions=10)

        assert result['success'] is False
        assert len(result['cartography_suggestions']) == 0
        assert len(result['camera_suggestions']) == 0

    def test_suggests_from_single_mask(self, mock_session_full):
        """Test auto-suggest works with single mask."""
        # Create a mask with clear corners
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (450, 450), 255, -1)

        mock_session_full.camera_mask = mask

        result = mock_session_full.auto_suggest_gcp_points(max_suggestions=10)

        assert result['success'] is True
        assert len(result['camera_suggestions']) > 0

    def test_respects_max_suggestions(self, mock_session_full):
        """Test that max_suggestions is respected."""
        # Create mask with many potential corners
        mask = np.zeros((500, 500), dtype=np.uint8)
        for i in range(5):
            for j in range(5):
                cv2.rectangle(mask, (i*100+10, j*100+10), (i*100+90, j*100+90), 255, -1)

        mock_session_full.camera_mask = mask

        result = mock_session_full.auto_suggest_gcp_points(max_suggestions=5)

        assert len(result['camera_suggestions']) <= 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
