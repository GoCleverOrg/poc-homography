#!/usr/bin/env python3
"""
Unit tests for GCPRegistry YAML serialization.

Tests cover:
- YAML serialization/deserialization round-trip
- Format detection by file extension
- Error handling for invalid content
- Edge cases (empty registry, unicode, special characters)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from poc_homography.map_points import GCPRegistry, MapPoint

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_registry() -> GCPRegistry:
    """Create a sample registry for testing."""
    points = {
        "P1": MapPoint(pixel_x=100.5, pixel_y=200.5),
        "P2": MapPoint(pixel_x=300.0, pixel_y=400.0),
        "P3": MapPoint(pixel_x=-50.25, pixel_y=150.75),
    }
    return GCPRegistry(map_id="test_map", points=points)


@pytest.fixture
def empty_registry() -> GCPRegistry:
    """Create an empty registry for testing."""
    return GCPRegistry(map_id="empty_map", points={})


@pytest.fixture
def unicode_registry() -> GCPRegistry:
    """Create a registry with unicode characters for testing."""
    points = {
        "点1": MapPoint(pixel_x=100.0, pixel_y=200.0),
        "Pöint_2": MapPoint(pixel_x=300.0, pixel_y=400.0),
        "точка_3": MapPoint(pixel_x=500.0, pixel_y=600.0),
    }
    return GCPRegistry(map_id="地图_карта", points=points)


# =============================================================================
# YAML Serialization Tests
# =============================================================================


class TestYAMLSerialization:
    """Test YAML serialization and deserialization."""

    def test_to_yaml_returns_valid_yaml(self, sample_registry: GCPRegistry):
        """Test that to_yaml() produces valid YAML."""
        yaml_str = sample_registry.to_yaml()
        assert isinstance(yaml_str, str)
        assert len(yaml_str) > 0

        # Should be parseable as YAML
        data = yaml.safe_load(yaml_str)
        assert data is not None
        assert "map_id" in data
        assert "points" in data

    def test_from_yaml_parses_valid_yaml(self, sample_registry: GCPRegistry):
        """Test that from_yaml() correctly parses valid YAML."""
        yaml_str = sample_registry.to_yaml()
        restored = GCPRegistry.from_yaml(yaml_str)

        assert restored.map_id == sample_registry.map_id
        assert len(restored.points) == len(sample_registry.points)

    def test_yaml_round_trip_preserves_data(self, sample_registry: GCPRegistry):
        """Test YAML serialization round-trip preserves all data."""
        yaml_str = sample_registry.to_yaml()
        restored = GCPRegistry.from_yaml(yaml_str)

        assert restored.map_id == sample_registry.map_id
        assert set(restored.points.keys()) == set(sample_registry.points.keys())

        for point_id, original_point in sample_registry.points.items():
            restored_point = restored.points[point_id]
            assert restored_point.pixel_x == original_point.pixel_x
            assert restored_point.pixel_y == original_point.pixel_y

    def test_from_yaml_empty_content_raises_error(self):
        """Test that from_yaml() raises ValueError for empty content."""
        with pytest.raises(ValueError, match="empty"):
            GCPRegistry.from_yaml("")

    def test_from_yaml_whitespace_only_raises_error(self):
        """Test that from_yaml() raises ValueError for whitespace-only content."""
        with pytest.raises(ValueError, match="empty"):
            GCPRegistry.from_yaml("   \n\n   ")

    def test_from_yaml_invalid_yaml_raises_error(self):
        """Test that from_yaml() raises error for invalid YAML syntax."""
        invalid_yaml = "map_id: test\npoints: [invalid: yaml: here"
        with pytest.raises(yaml.YAMLError):
            GCPRegistry.from_yaml(invalid_yaml)

    def test_from_yaml_missing_map_id_raises_error(self):
        """Test that from_yaml() raises KeyError for missing map_id."""
        yaml_str = "points: []"
        with pytest.raises(KeyError):
            GCPRegistry.from_yaml(yaml_str)

    def test_yaml_empty_registry(self, empty_registry: GCPRegistry):
        """Test YAML serialization of empty registry."""
        yaml_str = empty_registry.to_yaml()
        restored = GCPRegistry.from_yaml(yaml_str)

        assert restored.map_id == empty_registry.map_id
        assert len(restored.points) == 0

    def test_yaml_unicode_characters(self, unicode_registry: GCPRegistry):
        """Test YAML handles unicode characters correctly."""
        yaml_str = unicode_registry.to_yaml()
        restored = GCPRegistry.from_yaml(yaml_str)

        assert restored.map_id == unicode_registry.map_id
        assert set(restored.points.keys()) == set(unicode_registry.points.keys())


# =============================================================================
# File I/O Tests
# =============================================================================


class TestFileIO:
    """Test file save/load."""

    def test_save_yaml_extension(self, sample_registry: GCPRegistry):
        """Test that .yaml extension saves as YAML format."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            sample_registry.save(temp_path)
            content = temp_path.read_text(encoding="utf-8")

            # Should be valid YAML
            data = yaml.safe_load(content)
            assert data["map_id"] == sample_registry.map_id

            # YAML format check: should NOT start with JSON brace
            assert not content.strip().startswith("{")
        finally:
            temp_path.unlink()

    def test_save_yml_extension(self, sample_registry: GCPRegistry):
        """Test that .yml extension saves as YAML format."""
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            sample_registry.save(temp_path)
            content = temp_path.read_text(encoding="utf-8")

            # Should be valid YAML
            data = yaml.safe_load(content)
            assert data["map_id"] == sample_registry.map_id
            assert not content.strip().startswith("{")
        finally:
            temp_path.unlink()

    def test_load_yaml_extension(self, sample_registry: GCPRegistry):
        """Test that .yaml extension loads using YAML parser."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            sample_registry.save(temp_path)
            restored = GCPRegistry.load(temp_path)

            assert restored.map_id == sample_registry.map_id
            assert len(restored.points) == len(sample_registry.points)
        finally:
            temp_path.unlink()
