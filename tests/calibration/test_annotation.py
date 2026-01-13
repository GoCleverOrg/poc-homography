"""
Unit tests for Annotation serialization (to_dict/from_dict).

Tests cover:
- to_dict includes pose_id field
- from_dict correctly parses pose_id
- Backward compatibility (missing pose_id defaults to empty string)
- Round-trip serialization preserves pose_id
"""

from __future__ import annotations

import pytest

from poc_homography.domain.entities.annotation import Annotation
from poc_homography.domain.vo.pixel_point import PixelPoint

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_annotation() -> Annotation:
    """Create a sample annotation for testing."""
    return Annotation(
        gcp_id="GCP_001",
        camera_pose="pose_123",
        pixel=PixelPoint(x=100.5, y=200.5),
    )


@pytest.fixture
def annotation_without_pose_id() -> Annotation:
    """Create an annotation with empty pose_id for testing."""
    return Annotation(
        gcp_id="GCP_002",
        camera_pose="",
        pixel=PixelPoint(x=300.0, y=400.0),
    )


# =============================================================================
# Annotation Serialization Tests
# =============================================================================


class TestAnnotationSerialization:
    """Test Annotation to_dict and from_dict methods."""

    def test_annotation_to_dict_includes_pose_id(self, sample_annotation: Annotation):
        """Test that to_dict() includes pose_id."""
        result = sample_annotation.to_dict()

        assert "pose_id" in result
        assert result["pose_id"] == "pose_123"
        assert result["gcp_id"] == "GCP_001"
        assert result["pixel"]["x"] == 100.5
        assert result["pixel"]["y"] == 200.5

    def test_annotation_from_dict_with_pose_id(self):
        """Test that from_dict() correctly parses pose_id."""
        data = {
            "gcp_id": "GCP_TEST",
            "pose_id": "my_pose_id",
            "pixel": {"x": 150.0, "y": 250.0},
        }

        annotation = Annotation.from_dict(data)

        assert annotation.gcp_id == "GCP_TEST"
        assert annotation.pose_id == "my_pose_id"
        assert annotation.pixel.x == 150.0
        assert annotation.pixel.y == 250.0

    def test_annotation_from_dict_backward_compat(self):
        """Test that from_dict() without pose_id defaults to empty string."""
        # Simulate old data format without pose_id
        data = {
            "gcp_id": "GCP_OLD",
            "pixel": {"x": 50.0, "y": 75.0},
        }

        annotation = Annotation.from_dict(data)

        assert annotation.gcp_id == "GCP_OLD"
        assert annotation.pose_id == ""
        assert annotation.pixel.x == 50.0
        assert annotation.pixel.y == 75.0

    def test_annotation_round_trip_preserves_pose_id(self, sample_annotation: Annotation):
        """Test that round-trip (to_dict -> from_dict) preserves pose_id."""
        # Convert to dict and back
        data = sample_annotation.to_dict()
        restored = Annotation.from_dict(data)

        # All fields should be preserved
        assert restored.gcp_id == sample_annotation.gcp_id
        assert restored.pose_id == sample_annotation.camera_pose
        assert restored.pixel.x == sample_annotation.pixel.x
        assert restored.pixel.y == sample_annotation.pixel.y

        # Verify pose_id specifically
        assert restored.pose_id == "pose_123"

    def test_annotation_round_trip_empty_pose_id(self, annotation_without_pose_id: Annotation):
        """Test that round-trip preserves empty pose_id."""
        data = annotation_without_pose_id.to_dict()
        restored = Annotation.from_dict(data)

        assert restored.pose_id == ""
        assert restored.gcp_id == annotation_without_pose_id.gcp_id
