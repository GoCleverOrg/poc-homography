"""Unit tests for poc_homography.kml.geo_point_registry module."""

import xml.etree.ElementTree as ET

import pytest

from poc_homography.kml import GeoConfig, GeoPointRegistry, KmlPoint, PixelPoint


class TestGeoPointRegistry:
    """Tests for GeoPointRegistry frozen dataclass."""

    @pytest.fixture
    def geo_config(self) -> GeoConfig:
        """Standard geo config for Valencia area (EPSG:25830)."""
        return GeoConfig(
            crs="EPSG:25830",
            geotransform=(725140.0, 0.05, 0.0, 4373490.0, 0.0, -0.05),
        )

    def test_frozen(self, geo_config: GeoConfig) -> None:
        """Test GeoPointRegistry is immutable."""
        registry = GeoPointRegistry(geo_config=geo_config, points={})

        with pytest.raises(AttributeError):
            registry.geo_config = geo_config  # type: ignore[misc]

    def test_empty_registry(self, geo_config: GeoConfig) -> None:
        """Test empty registry creation."""
        registry = GeoPointRegistry(geo_config=geo_config, points={})

        assert registry.geo_config == geo_config
        assert len(registry.points) == 0


class TestGeoPointRegistryFromPixelPoints:
    """Tests for GeoPointRegistry.from_pixel_points factory method."""

    @pytest.fixture
    def geo_config(self) -> GeoConfig:
        """Standard geo config for Valencia area (EPSG:25830)."""
        return GeoConfig(
            crs="EPSG:25830",
            geotransform=(725140.0, 0.05, 0.0, 4373490.0, 0.0, -0.05),
        )

    @pytest.mark.parametrize(
        "pixel_points,category",
        [
            ({"P1": PixelPoint(x=100.0, y=200.0)}, "zebra"),
            (
                {
                    "P1": PixelPoint(x=0.0, y=0.0),
                    "P2": PixelPoint(x=1000.0, y=1000.0),
                },
                "arrow",
            ),
            ({}, "other"),
        ],
        ids=["single-point", "multiple-points", "empty"],
    )
    def test_from_pixel_points_creates_registry(
        self,
        geo_config: GeoConfig,
        pixel_points: dict[str, PixelPoint],
        category: str,
    ) -> None:
        """Test from_pixel_points creates registry with correct structure."""
        registry = GeoPointRegistry.from_pixel_points(geo_config, pixel_points, category=category)

        assert registry.geo_config == geo_config
        assert len(registry.points) == len(pixel_points)

        for name, pixel_point in pixel_points.items():
            assert name in registry.points
            stored_pixel, stored_kml = registry.points[name]
            assert stored_pixel == pixel_point
            assert stored_kml.category == category
            assert stored_kml.name == name

    def test_from_pixel_points_converts_to_latlon(self, geo_config: GeoConfig) -> None:
        """Test pixel points are converted to valid lat/lon coordinates."""
        pixel_points = {"P1": PixelPoint(x=500.0, y=500.0)}

        registry = GeoPointRegistry.from_pixel_points(geo_config, pixel_points)

        _, kml_point = registry.points["P1"]

        # Lat/lon should be in valid ranges
        assert -90 <= kml_point.lat <= 90
        assert -180 <= kml_point.lon <= 180

        # For Valencia area, expect roughly these ranges
        assert 39.0 < kml_point.lat < 40.0
        assert -1.0 < kml_point.lon < 0.0


class TestGeoPointRegistryFromKmlPoints:
    """Tests for GeoPointRegistry.from_kml_points factory method."""

    @pytest.fixture
    def geo_config(self) -> GeoConfig:
        """Standard geo config for Valencia area (EPSG:25830)."""
        return GeoConfig(
            crs="EPSG:25830",
            geotransform=(725140.0, 0.05, 0.0, 4373490.0, 0.0, -0.05),
        )

    @pytest.mark.parametrize(
        "kml_points",
        [
            {"P1": KmlPoint(name="P1", category="zebra", lat=39.5, lon=-0.4)},
            {
                "P1": KmlPoint(name="P1", category="zebra", lat=39.5, lon=-0.4),
                "P2": KmlPoint(name="P2", category="arrow", lat=39.6, lon=-0.3),
            },
            {},
        ],
        ids=["single-point", "multiple-points", "empty"],
    )
    def test_from_kml_points_creates_registry(
        self, geo_config: GeoConfig, kml_points: dict[str, KmlPoint]
    ) -> None:
        """Test from_kml_points creates registry with correct structure."""
        registry = GeoPointRegistry.from_kml_points(geo_config, kml_points)

        assert registry.geo_config == geo_config
        assert len(registry.points) == len(kml_points)

        for name, kml_point in kml_points.items():
            assert name in registry.points
            _, stored_kml = registry.points[name]
            assert stored_kml == kml_point

    def test_from_kml_points_converts_to_pixel(self, geo_config: GeoConfig) -> None:
        """Test KML points are converted to pixel coordinates."""
        kml_points = {"P1": KmlPoint(name="P1", category="zebra", lat=39.5, lon=-0.4)}

        registry = GeoPointRegistry.from_kml_points(geo_config, kml_points)

        pixel_point, _ = registry.points["P1"]

        # Pixel coordinates should be reasonable (not NaN or inf)
        assert pixel_point.x == pixel_point.x  # Not NaN
        assert pixel_point.y == pixel_point.y  # Not NaN


class TestGeoPointRegistryRoundTrip:
    """Tests for coordinate conversion round-trip consistency."""

    @pytest.fixture
    def geo_config(self) -> GeoConfig:
        """Standard geo config for Valencia area (EPSG:25830)."""
        return GeoConfig(
            crs="EPSG:25830",
            geotransform=(725140.0, 0.05, 0.0, 4373490.0, 0.0, -0.05),
        )

    @pytest.mark.parametrize(
        "original_pixel",
        [
            PixelPoint(x=100.0, y=200.0),
            PixelPoint(x=500.0, y=500.0),
            PixelPoint(x=0.0, y=0.0),
            PixelPoint(x=1920.0, y=1080.0),
        ],
        ids=["small", "medium", "origin", "hd"],
    )
    def test_pixel_to_kml_to_pixel_round_trip(
        self, geo_config: GeoConfig, original_pixel: PixelPoint
    ) -> None:
        """Test pixel -> latlon -> pixel round trip preserves coordinates."""
        # Pixel to KML
        registry1 = GeoPointRegistry.from_pixel_points(geo_config, {"P1": original_pixel})
        _, kml_point = registry1.points["P1"]

        # KML back to pixel
        registry2 = GeoPointRegistry.from_kml_points(geo_config, {"P1": kml_point})
        recovered_pixel, _ = registry2.points["P1"]

        # Should match within small tolerance
        assert recovered_pixel.x == pytest.approx(original_pixel.x, abs=0.01)
        assert recovered_pixel.y == pytest.approx(original_pixel.y, abs=0.01)


class TestGeoPointRegistryRenderKml:
    """Tests for GeoPointRegistry.render_kml method."""

    KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}

    @pytest.fixture
    def geo_config(self) -> GeoConfig:
        """Standard geo config for Valencia area (EPSG:25830)."""
        return GeoConfig(
            crs="EPSG:25830",
            geotransform=(725140.0, 0.05, 0.0, 4373490.0, 0.0, -0.05),
        )

    def test_render_kml_empty_registry(self, geo_config: GeoConfig) -> None:
        """Test render_kml with empty registry produces valid KML."""
        registry = GeoPointRegistry(geo_config=geo_config, points={})
        kml_str = registry.render_kml()

        root = ET.fromstring(kml_str)
        assert root.tag.endswith("kml")

    def test_render_kml_structure(self, geo_config: GeoConfig) -> None:
        """Test render_kml produces valid KML structure."""
        pixel_points = {"P1": PixelPoint(x=100.0, y=200.0)}
        registry = GeoPointRegistry.from_pixel_points(geo_config, pixel_points, category="zebra")

        kml_str = registry.render_kml()

        root = ET.fromstring(kml_str)
        assert root.tag.endswith("kml")

        doc = root.find("kml:Document", self.KML_NS)
        assert doc is not None

        styles = doc.findall("kml:Style", self.KML_NS)
        assert len(styles) >= 1

        placemarks = root.findall(".//kml:Placemark", self.KML_NS)
        assert len(placemarks) == 1

    @pytest.mark.parametrize(
        "num_points",
        [1, 3, 10],
        ids=["single", "few", "many"],
    )
    def test_render_kml_correct_placemark_count(
        self, geo_config: GeoConfig, num_points: int
    ) -> None:
        """Test render_kml produces correct number of placemarks."""
        pixel_points = {
            f"P{i}": PixelPoint(x=float(i * 100), y=float(i * 100)) for i in range(num_points)
        }
        registry = GeoPointRegistry.from_pixel_points(geo_config, pixel_points, category="zebra")

        kml_str = registry.render_kml()

        root = ET.fromstring(kml_str)
        placemarks = root.findall(".//kml:Placemark", self.KML_NS)

        assert len(placemarks) == num_points

    def test_render_kml_contains_point_names(self, geo_config: GeoConfig) -> None:
        """Test render_kml includes point names in output."""
        pixel_points = {
            "AlphaPoint": PixelPoint(x=100.0, y=100.0),
            "BetaPoint": PixelPoint(x=200.0, y=200.0),
        }
        registry = GeoPointRegistry.from_pixel_points(geo_config, pixel_points, category="arrow")

        kml_str = registry.render_kml()

        assert "AlphaPoint" in kml_str
        assert "BetaPoint" in kml_str

    def test_render_kml_has_all_styles(self, geo_config: GeoConfig) -> None:
        """Test render_kml includes all required styles."""
        registry = GeoPointRegistry(geo_config=geo_config, points={})
        kml_str = registry.render_kml()

        assert 'id="zebra"' in kml_str
        assert 'id="arrow"' in kml_str
        assert 'id="parking"' in kml_str
        assert 'id="other"' in kml_str
