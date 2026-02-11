#!/usr/bin/env python3
"""
Unit tests for GCPRegistry serialization (JSON and YAML).

Tests cover:
- JSON serialization/deserialization round-trip
- YAML serialization/deserialization round-trip
- Format detection by file extension
- Cross-format conversion (JSON <-> YAML)
- Error handling for invalid content
- Edge cases (empty registry, unicode, special characters)
"""

from __future__ import annotations

import json
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
# JSON Serialization Tests
# =============================================================================


class TestJSONSerialization:
    """Test JSON serialization and deserialization."""

    def test_to_json_returns_valid_json(self, sample_registry: GCPRegistry):
        """Test that to_json() produces valid JSON."""
        json_str = sample_registry.to_json()
        assert isinstance(json_str, str)

        # Should be parseable as JSON
        data = json.loads(json_str)
        assert "map_id" in data
        assert "points" in data

    def test_json_round_trip_preserves_data(self, sample_registry: GCPRegistry):
        """Test JSON serialization round-trip preserves all data."""
        json_str = sample_registry.to_json()
        restored = GCPRegistry.from_json(json_str)

        assert restored.map_id == sample_registry.map_id
        assert set(restored.points.keys()) == set(sample_registry.points.keys())

        for point_id, original_point in sample_registry.points.items():
            restored_point = restored.points[point_id]
            assert restored_point.pixel_x == original_point.pixel_x
            assert restored_point.pixel_y == original_point.pixel_y

    def test_from_json_invalid_json_raises_error(self):
        """Test that from_json() raises error for invalid JSON."""
        with pytest.raises(json.JSONDecodeError):
            GCPRegistry.from_json("{invalid json")


# =============================================================================
# File I/O and Format Detection Tests
# =============================================================================


class TestFileIOFormatDetection:
    """Test file save/load with format detection by extension."""

    def test_save_yaml_extension(self, sample_registry: GCPRegistry):
        """Test that .yaml extension saves as YAML format."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            sample_registry.save(temp_path)
            content = temp_path.read_text(encoding="utf-8")

            # Should be valid YAML (not JSON)
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

    def test_save_json_extension(self, sample_registry: GCPRegistry):
        """Test that .json extension saves as JSON format."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            sample_registry.save(temp_path)
            content = temp_path.read_text(encoding="utf-8")

            # Should be valid JSON
            data = json.loads(content)
            assert data["map_id"] == sample_registry.map_id

            # JSON format check: should start with brace
            assert content.strip().startswith("{")
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

    def test_load_json_extension(self, sample_registry: GCPRegistry):
        """Test that .json extension loads using JSON parser."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            sample_registry.save(temp_path)
            restored = GCPRegistry.load(temp_path)

            assert restored.map_id == sample_registry.map_id
            assert len(restored.points) == len(sample_registry.points)
        finally:
            temp_path.unlink()

    def test_unknown_extension_defaults_to_json(self, sample_registry: GCPRegistry):
        """Test that unknown extension defaults to JSON format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = Path(f.name)

        try:
            sample_registry.save(temp_path)
            content = temp_path.read_text(encoding="utf-8")

            # Should be valid JSON
            data = json.loads(content)
            assert data["map_id"] == sample_registry.map_id
        finally:
            temp_path.unlink()


# =============================================================================
# Cross-Format Conversion Tests
# =============================================================================


class TestCrossFormatConversion:
    """Test conversion between JSON and YAML formats."""

    def test_yaml_to_json_preserves_data(self, sample_registry: GCPRegistry):
        """Test converting from YAML to JSON preserves data."""
        # Save as YAML
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            yaml_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = Path(f.name)

        try:
            sample_registry.save(yaml_path)

            # Load YAML and save as JSON
            loaded = GCPRegistry.load(yaml_path)
            loaded.save(json_path)

            # Load JSON and verify
            final = GCPRegistry.load(json_path)
            assert final.map_id == sample_registry.map_id
            assert set(final.points.keys()) == set(sample_registry.points.keys())
        finally:
            yaml_path.unlink()
            json_path.unlink()

    def test_json_to_yaml_preserves_data(self, sample_registry: GCPRegistry):
        """Test converting from JSON to YAML preserves data."""
        # Save as JSON
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            yaml_path = Path(f.name)

        try:
            sample_registry.save(json_path)

            # Load JSON and save as YAML
            loaded = GCPRegistry.load(json_path)
            loaded.save(yaml_path)

            # Load YAML and verify
            final = GCPRegistry.load(yaml_path)
            assert final.map_id == sample_registry.map_id
            assert set(final.points.keys()) == set(sample_registry.points.keys())
        finally:
            json_path.unlink()
            yaml_path.unlink()

    def test_multiple_format_conversions(self, sample_registry: GCPRegistry):
        """Test multiple format conversions preserve data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # YAML -> JSON -> YAML -> JSON
            path1 = tmp_path / "step1.yaml"
            path2 = tmp_path / "step2.json"
            path3 = tmp_path / "step3.yaml"
            path4 = tmp_path / "step4.json"

            sample_registry.save(path1)
            GCPRegistry.load(path1).save(path2)
            GCPRegistry.load(path2).save(path3)
            GCPRegistry.load(path3).save(path4)

            final = GCPRegistry.load(path4)

            assert final.map_id == sample_registry.map_id
            assert len(final.points) == len(sample_registry.points)

            for point_id, original in sample_registry.points.items():
                restored = final.points[point_id]
                assert restored.pixel_x == original.pixel_x
                assert restored.pixel_y == original.pixel_y
