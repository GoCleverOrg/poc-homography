"""Unit tests for poc_homography.kml.geo_config module."""

import pytest

from poc_homography.kml import GeoConfig, Geotransform


class TestGeoConfig:
    """Tests for GeoConfig frozen dataclass."""

    @pytest.mark.parametrize(
        "crs,geotransform",
        [
            ("EPSG:25830", (725140.0, 0.05, 0.0, 4373490.0, 0.0, -0.05)),
            ("EPSG:4326", (0.0, 0.001, 0.0, 0.0, 0.0, -0.001)),
            ("EPSG:32630", (500000.0, 1.0, 0.0, 4500000.0, 0.0, -1.0)),
        ],
        ids=["utm-spain", "wgs84", "utm-30n"],
    )
    def test_creation(self, crs: str, geotransform: Geotransform) -> None:
        """Test GeoConfig can be created with various CRS and geotransforms."""
        config = GeoConfig(crs=crs, geotransform=geotransform)

        assert config.crs == crs
        assert config.geotransform == geotransform

    def test_frozen(self) -> None:
        """Test GeoConfig is immutable."""
        config = GeoConfig(crs="EPSG:25830", geotransform=(0, 1, 0, 0, 0, -1))

        with pytest.raises(AttributeError):
            config.crs = "EPSG:4326"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Test GeoConfig equality comparison."""
        gt: Geotransform = (725140.0, 0.05, 0.0, 4373490.0, 0.0, -0.05)
        config1 = GeoConfig(crs="EPSG:25830", geotransform=gt)
        config2 = GeoConfig(crs="EPSG:25830", geotransform=gt)
        config3 = GeoConfig(crs="EPSG:4326", geotransform=gt)

        assert config1 == config2
        assert config1 != config3
