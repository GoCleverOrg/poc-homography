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


# Data extracted from Cartografia_valencia.kml - each tuple represents one placemark
# Format: (name, pixel_x, pixel_y, utm_easting, utm_northing, lat, lon, category)
VALENCIA_PLACEMARKS = [
    ("Z1", 848.59, 649.7, 737702.3385, 4391497.995, 39.64025855, -0.22997144, "zebra"),
    ("Z25", 692.4, 680.73, 737678.91, 4391493.3405, 39.64022316, -0.23024584, "zebra"),
    ("Z48", 818.06, 734.42, 737697.759, 4391485.287, 39.64014545, -0.23002932, "zebra"),
    ("A1", 733.5, 649.2, 737685.075, 4391498.07, 39.64026402, -0.23017237, "arrow"),
    ("A4", 971.9, 815.29, 737720.835, 4391473.1565, 39.64002987, -0.22976506, "arrow"),
    ("A7", 835.34, 468.0, 737700.351, 4391525.25, 39.64050439, -0.22998478, "arrow"),
    ("P1", 822.59, 728.21, 737698.4385, 4391486.2185, 39.64015364, -0.23002107, "parking"),
    ("P10", 1068.54, 560.45, 737735.331, 4391511.3825, 39.64036987, -0.22958258, "parking"),
    ("P18", 1173.74, 373.88, 737751.111, 4391539.368, 39.64061734, -0.22938883, "parking"),
    ("X1", 1179.27, 359.62, 737751.9405, 4391541.507, 39.64063636, -0.22937841, "other"),
    ("X12", 1057.8, 309.29, 737733.72, 4391549.0565, 39.64070937, -0.22958779, "other"),
    ("X23", 739.71, 561.95, 737686.0065, 4391511.1575, 39.64038154, -0.23015683, "other"),
]


class TestGeoPointRegistryValenciaData:
    """Data-driven tests using real placemarks from Cartografia_valencia.kml.

    Each test validates coordinate conversions against known ground-truth data
    extracted from the Valencia cartography KML file.

    Geotransform derived from KML data:
        origin_x=737575.05, pixel_width=0.15, origin_y=4391595.45, pixel_height=-0.15
    """

    @pytest.fixture
    def valencia_geo_config(self) -> GeoConfig:
        """GeoConfig for Cartografia_valencia.tif (EPSG:25830)."""
        return GeoConfig(
            crs="EPSG:25830",
            geotransform=(737575.05, 0.15, 0.0, 4391595.45, 0.0, -0.15),
        )

    @pytest.mark.parametrize(
        "name,pixel_x,pixel_y,utm_easting,utm_northing,lat,lon,category",
        VALENCIA_PLACEMARKS,
        ids=[p[0] for p in VALENCIA_PLACEMARKS],
    )
    def test_latlon_to_pixel_conversion(
        self,
        valencia_geo_config: GeoConfig,
        name: str,
        pixel_x: float,
        pixel_y: float,
        utm_easting: float,
        utm_northing: float,
        lat: float,
        lon: float,
        category: str,
    ) -> None:
        """Test from_kml_points converts lat/lon to correct pixel coordinates."""
        kml_point = KmlPoint(name=name, category=category, lat=lat, lon=lon)
        registry = GeoPointRegistry.from_kml_points(valencia_geo_config, {name: kml_point})

        recovered_pixel, _ = registry.points[name]

        assert recovered_pixel.x == pytest.approx(pixel_x, abs=0.5)
        assert recovered_pixel.y == pytest.approx(pixel_y, abs=0.5)

    @pytest.mark.parametrize(
        "name,pixel_x,pixel_y,utm_easting,utm_northing,lat,lon,category",
        VALENCIA_PLACEMARKS,
        ids=[p[0] for p in VALENCIA_PLACEMARKS],
    )
    def test_pixel_to_latlon_conversion(
        self,
        valencia_geo_config: GeoConfig,
        name: str,
        pixel_x: float,
        pixel_y: float,
        utm_easting: float,
        utm_northing: float,
        lat: float,
        lon: float,
        category: str,
    ) -> None:
        """Test from_pixel_points converts pixel to correct lat/lon coordinates."""
        pixel_point = PixelPoint(x=pixel_x, y=pixel_y)
        registry = GeoPointRegistry.from_pixel_points(
            valencia_geo_config, {name: pixel_point}, category=category
        )

        _, recovered_kml = registry.points[name]

        # Lat/lon tolerance ~0.00001 degrees ≈ ~1m at this latitude
        assert recovered_kml.lat == pytest.approx(lat, abs=1e-5)
        assert recovered_kml.lon == pytest.approx(lon, abs=1e-5)

    @pytest.mark.parametrize(
        "name,pixel_x,pixel_y,utm_easting,utm_northing,lat,lon,category",
        VALENCIA_PLACEMARKS,
        ids=[p[0] for p in VALENCIA_PLACEMARKS],
    )
    def test_pixel_to_utm_conversion(
        self,
        valencia_geo_config: GeoConfig,
        name: str,
        pixel_x: float,
        pixel_y: float,
        utm_easting: float,
        utm_northing: float,
        lat: float,
        lon: float,
        category: str,
    ) -> None:
        """Test pixel to UTM conversion matches expected easting/northing."""
        from poc_homography.geotiff_utils import apply_geotransform

        easting, northing = apply_geotransform(
            pixel_x, pixel_y, list(valencia_geo_config.geotransform)
        )

        # UTM tolerance 0.01m (1cm)
        assert easting == pytest.approx(utm_easting, abs=0.01)
        assert northing == pytest.approx(utm_northing, abs=0.01)
