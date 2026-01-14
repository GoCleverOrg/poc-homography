"""
Unit tests for GroundControlPointCollection serialization (YAML).

Tests cover:
- YAML serialization/deserialization round-trip
- Format detection by file extension
- Error handling for invalid content
- Edge cases (empty registry, unicode, special characters)

Architecture note:
- Tests use FakeFileSystem to avoid real filesystem side-effects
- This follows seams-based testing pattern for DDD
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

from poc_homography.domain.vo.pixel_point import PixelPoint
from poc_homography.map_points import GroundControlPointCollection, MapPoint

# =============================================================================
# Fake FileSystem for Testing (No Side Effects)
# =============================================================================


class FakeFileSystem:
    """In-memory filesystem for testing without side effects."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def read_text(self, path: str | Path) -> str:
        """Read text from in-memory storage."""
        key = str(path)
        if key not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        return self.files[key]

    def write_text(self, path: str | Path, content: str) -> None:
        """Write text to in-memory storage."""
        self.files[str(path)] = content


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def fake_fs() -> FakeFileSystem:
    """Create a fresh in-memory filesystem for each test."""
    return FakeFileSystem()


@pytest.fixture
def sample_registry() -> GroundControlPointCollection:
    """Create a sample registry for testing."""
    points = {
        "P1": MapPoint(map_id="test_map", pixel_point=PixelPoint(_x=100.5, _y=200.5)),
        "P2": MapPoint(map_id="test_map", pixel_point=PixelPoint(_x=300.0, _y=400.0)),
        "P3": MapPoint(map_id="test_map", pixel_point=PixelPoint(_x=-50.25, _y=150.75)),
    }
    return GroundControlPointCollection(map_id="test_map", points=points)


@pytest.fixture
def empty_registry() -> GroundControlPointCollection:
    """Create an empty registry for testing."""
    return GroundControlPointCollection(map_id="empty_map", points={})


@pytest.fixture
def unicode_registry() -> GroundControlPointCollection:
    """Create a registry with unicode characters for testing."""
    map_id = "地图_карта"
    points = {
        "点1": MapPoint(map_id=map_id, pixel_point=PixelPoint(_x=100.0, _y=200.0)),
        "Pöint_2": MapPoint(map_id=map_id, pixel_point=PixelPoint(_x=300.0, _y=400.0)),
        "точка_3": MapPoint(map_id=map_id, pixel_point=PixelPoint(_x=500.0, _y=600.0)),
    }
    return GroundControlPointCollection(map_id=map_id, points=points)


# =============================================================================
# YAML Serialization Tests
# =============================================================================


class TestYAMLSerialization:
    """Test YAML serialization and deserialization."""

    def test_to_yaml_returns_valid_yaml(self, sample_registry: GroundControlPointCollection):
        """Test that to_yaml() produces valid YAML."""
        yaml_str = sample_registry.to_yaml()
        assert isinstance(yaml_str, str)
        assert len(yaml_str) > 0

        # Should be parseable as YAML
        data = yaml.safe_load(yaml_str)
        assert data is not None
        assert "map_id" in data
        assert "points" in data

    def test_from_yaml_parses_valid_yaml(self, sample_registry: GroundControlPointCollection):
        """Test that from_yaml() correctly parses valid YAML."""
        yaml_str = sample_registry.to_yaml()
        restored = GroundControlPointCollection.from_yaml(yaml_str)

        assert restored.map_id == sample_registry.map_id
        assert len(restored.points) == len(sample_registry.points)

    def test_yaml_round_trip_preserves_data(self, sample_registry: GroundControlPointCollection):
        """Test YAML serialization round-trip preserves all data."""
        yaml_str = sample_registry.to_yaml()
        restored = GroundControlPointCollection.from_yaml(yaml_str)

        assert restored.map_id == sample_registry.map_id
        assert set(restored.points.keys()) == set(sample_registry.points.keys())

        for point_id, original_point in sample_registry.points.items():
            restored_point = restored.points[point_id]
            assert float(restored_point.pixel_point.x) == float(original_point.pixel_point.x)
            assert float(restored_point.pixel_point.y) == float(original_point.pixel_point.y)

    def test_from_yaml_empty_content_raises_error(self):
        """Test that from_yaml() raises ValueError for empty content."""
        with pytest.raises(ValueError, match="empty"):
            GroundControlPointCollection.from_yaml("")

    def test_from_yaml_whitespace_only_raises_error(self):
        """Test that from_yaml() raises ValueError for whitespace-only content."""
        with pytest.raises(ValueError, match="empty"):
            GroundControlPointCollection.from_yaml("   \n\n   ")

    def test_from_yaml_invalid_yaml_raises_error(self):
        """Test that from_yaml() raises error for invalid YAML syntax."""
        invalid_yaml = "map_id: test\npoints: [invalid: yaml: here"
        with pytest.raises(yaml.YAMLError):
            GroundControlPointCollection.from_yaml(invalid_yaml)

    def test_from_yaml_missing_map_id_raises_error(self):
        """Test that from_yaml() raises KeyError for missing map_id."""
        yaml_str = "points: []"
        with pytest.raises(KeyError):
            GroundControlPointCollection.from_yaml(yaml_str)

    def test_yaml_empty_registry(self, empty_registry: GroundControlPointCollection):
        """Test YAML serialization of empty registry."""
        yaml_str = empty_registry.to_yaml()
        restored = GroundControlPointCollection.from_yaml(yaml_str)

        assert restored.map_id == empty_registry.map_id
        assert len(restored.points) == 0

    def test_yaml_unicode_characters(self, unicode_registry: GroundControlPointCollection):
        """Test YAML handles unicode characters correctly."""
        yaml_str = unicode_registry.to_yaml()
        restored = GroundControlPointCollection.from_yaml(yaml_str)

        assert restored.map_id == unicode_registry.map_id
        assert set(restored.points.keys()) == set(unicode_registry.points.keys())


# =============================================================================
# File I/O and Format Detection Tests (Using FakeFileSystem - No Side Effects)
# =============================================================================


class TestFileIOFormatDetection:
    """Test file save/load with format detection by extension.

    All tests use FakeFileSystem to avoid real filesystem side effects.
    """

    def test_save_yaml_extension(
        self, sample_registry: GroundControlPointCollection, fake_fs: FakeFileSystem
    ):
        """Test that .yaml extension saves as YAML format."""
        sample_registry.save("/fake/path/test.yaml", fs=fake_fs)
        content = fake_fs.files["/fake/path/test.yaml"]

        # Should be valid YAML (not JSON)
        data = yaml.safe_load(content)
        assert data["map_id"] == sample_registry.map_id

        # YAML format check: should NOT start with JSON brace
        assert not content.strip().startswith("{")

    def test_save_yml_extension(
        self, sample_registry: GroundControlPointCollection, fake_fs: FakeFileSystem
    ):
        """Test that .yml extension saves as YAML format."""
        sample_registry.save("/fake/path/test.yml", fs=fake_fs)
        content = fake_fs.files["/fake/path/test.yml"]

        # Should be valid YAML
        data = yaml.safe_load(content)
        assert data["map_id"] == sample_registry.map_id
        assert not content.strip().startswith("{")

    def test_save_unsupported_extension_raises_error(
        self, sample_registry: GroundControlPointCollection, fake_fs: FakeFileSystem
    ):
        """Test that unsupported extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            sample_registry.save("/fake/path/test.json", fs=fake_fs)

    def test_load_yaml_extension(
        self, sample_registry: GroundControlPointCollection, fake_fs: FakeFileSystem
    ):
        """Test that .yaml extension loads using YAML parser."""
        sample_registry.save("/fake/path/test.yaml", fs=fake_fs)
        restored = GroundControlPointCollection.load("/fake/path/test.yaml", fs=fake_fs)

        assert restored.map_id == sample_registry.map_id
        assert len(restored.points) == len(sample_registry.points)

    def test_load_unsupported_extension_raises_error(
        self, sample_registry: GroundControlPointCollection, fake_fs: FakeFileSystem
    ):
        """Test that loading unsupported extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            GroundControlPointCollection.load("/fake/path/test.json", fs=fake_fs)


# =============================================================================
# YAML File Round-Trip Tests (Using FakeFileSystem - No Side Effects)
# =============================================================================


class TestYAMLFileRoundTrip:
    """Test YAML file save/load round-trips.

    All tests use FakeFileSystem to avoid real filesystem side effects.
    """

    def test_yaml_file_round_trip_preserves_data(
        self, sample_registry: GroundControlPointCollection, fake_fs: FakeFileSystem
    ):
        """Test saving and loading YAML file preserves data."""
        sample_registry.save("/fake/path/test.yaml", fs=fake_fs)
        restored = GroundControlPointCollection.load("/fake/path/test.yaml", fs=fake_fs)

        assert restored.map_id == sample_registry.map_id
        assert len(restored.points) == len(sample_registry.points)

        for point_id, original in sample_registry.points.items():
            loaded = restored.points[point_id]
            assert float(loaded.pixel_point.x) == float(original.pixel_point.x)
            assert float(loaded.pixel_point.y) == float(original.pixel_point.y)

    def test_yml_file_round_trip_preserves_data(
        self, sample_registry: GroundControlPointCollection, fake_fs: FakeFileSystem
    ):
        """Test saving and loading .yml file preserves data."""
        sample_registry.save("/fake/path/test.yml", fs=fake_fs)
        restored = GroundControlPointCollection.load("/fake/path/test.yml", fs=fake_fs)

        assert restored.map_id == sample_registry.map_id
        assert len(restored.points) == len(sample_registry.points)

    def test_multiple_yaml_round_trips(
        self, sample_registry: GroundControlPointCollection, fake_fs: FakeFileSystem
    ):
        """Test multiple YAML round-trips preserve data."""
        # YAML -> YAML -> YAML
        sample_registry.save("/fake/step1.yaml", fs=fake_fs)
        GroundControlPointCollection.load("/fake/step1.yaml", fs=fake_fs).save(
            "/fake/step2.yml", fs=fake_fs
        )
        GroundControlPointCollection.load("/fake/step2.yml", fs=fake_fs).save(
            "/fake/step3.yaml", fs=fake_fs
        )

        final = GroundControlPointCollection.load("/fake/step3.yaml", fs=fake_fs)

        assert final.map_id == sample_registry.map_id
        assert len(final.points) == len(sample_registry.points)

        for point_id, original in sample_registry.points.items():
            restored = final.points[point_id]
            assert float(restored.pixel_point.x) == float(original.pixel_point.x)
            assert float(restored.pixel_point.y) == float(original.pixel_point.y)


# =============================================================================
# Iteration Protocol Tests
# =============================================================================


class TestIterationProtocol:
    """Test __iter__ and __len__ protocol methods."""

    def test_len_returns_number_of_points(self, sample_registry: GroundControlPointCollection):
        """Test that len(registry) returns correct count."""
        assert len(sample_registry) == 3

    def test_iter_returns_gcp_id_mappoint_tuples(
        self, sample_registry: GroundControlPointCollection
    ):
        """Test that iterating yields (gcp_id, MapPoint) tuples."""
        items = list(sample_registry)

        # Should have 3 items
        assert len(items) == 3

        # Each item should be a (str, MapPoint) tuple
        for gcp_id, map_point in items:
            assert isinstance(gcp_id, str)
            assert isinstance(map_point, MapPoint)

        # Check that the expected GCP IDs are present
        gcp_ids = {gcp_id for gcp_id, _ in items}
        assert gcp_ids == {"P1", "P2", "P3"}

        # Verify correct point is associated with each ID
        items_dict = dict(items)
        assert float(items_dict["P1"].pixel_point.x) == 100.5
        assert float(items_dict["P1"].pixel_point.y) == 200.5
        assert float(items_dict["P2"].pixel_point.x) == 300.0
        assert float(items_dict["P2"].pixel_point.y) == 400.0

    def test_iter_empty_registry(self, empty_registry: GroundControlPointCollection):
        """Test that iteration over empty registry yields nothing."""
        items = list(empty_registry)
        assert items == []

        # Also verify with for loop
        count = 0
        for _ in empty_registry:
            count += 1
        assert count == 0

    def test_len_empty_registry(self, empty_registry: GroundControlPointCollection):
        """Test that len(empty_registry) returns 0."""
        assert len(empty_registry) == 0

    def test_list_conversion(self, sample_registry: GroundControlPointCollection):
        """Test that list(registry) produces expected list of tuples."""
        result = list(sample_registry)

        # Verify it's a list
        assert isinstance(result, list)

        # Verify length matches
        assert len(result) == len(sample_registry)

        # Verify content matches points dictionary items
        expected = list(sample_registry.points.items())
        assert result == expected
