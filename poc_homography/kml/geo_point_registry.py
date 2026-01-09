"""Georeferenced point registry with KML support."""

from dataclasses import dataclass, field

from jinja2 import Environment, PackageLoader
from pyproj import Transformer

from poc_homography.geotiff_utils import apply_geotransform
from poc_homography.kml.geo_config import GeoConfig
from poc_homography.kml.kml_point import KmlPoint
from poc_homography.kml.pixel_point import PixelPoint

# Set up Jinja2 template environment
_template_env = Environment(
    loader=PackageLoader("poc_homography.kml", "templates"),
    autoescape=False,  # KML is XML, we handle escaping manually
    trim_blocks=True,
    lstrip_blocks=True,
)


def _pixel_to_latlon(
    px: float, py: float, geo_config: GeoConfig, transformer: Transformer
) -> tuple[float, float]:
    """Convert pixel coordinates to lat/lon (WGS84)."""
    easting, northing = apply_geotransform(px, py, list(geo_config.geotransform))
    lon, lat = transformer.transform(easting, northing)
    return lat, lon


def _latlon_to_pixel(
    lat: float, lon: float, geo_config: GeoConfig, reverse_transformer: Transformer
) -> tuple[float, float]:
    """Convert lat/lon to pixel coordinates using inverse affine transform."""
    easting, northing = reverse_transformer.transform(lon, lat)
    gt = geo_config.geotransform

    det = gt[1] * gt[5] - gt[2] * gt[4]
    if abs(det) < 1e-10:
        raise ValueError("Geotransform matrix is singular (cannot invert)")

    de = easting - gt[0]
    dn = northing - gt[3]

    px = (gt[5] * de - gt[2] * dn) / det
    py = (-gt[4] * de + gt[1] * dn) / det

    return px, py


@dataclass(frozen=True)
class GeoPointRegistry:
    """Immutable georeferenced point registry with KML support.

    This class stores a mapping from point names to (PixelPoint, KmlPoint) tuples,
    allowing efficient lookup by name and clear separation between pixel and
    geographic coordinate spaces.

    Use factory methods `from_kml_points` or `from_pixel_points` to create instances.

    Attributes:
        geo_config: Georeferencing configuration with CRS and geotransform.
        points: Mapping from point name to (PixelPoint, KmlPoint) tuple.

    Example:
        >>> geo_config = GeoConfig(
        ...     crs="EPSG:25830",
        ...     geotransform=(725140.0, 0.05, 0.0, 4373490.0, 0.0, -0.05),
        ... )
        >>> kml_points = {"P1": KmlPoint(name="P1", category="zebra", lat=39.5, lon=-0.4)}
        >>> registry = GeoPointRegistry.from_kml_points(geo_config, kml_points)
    """

    geo_config: GeoConfig
    points: dict[str, tuple[PixelPoint, KmlPoint]] = field(default_factory=dict, hash=False)

    @classmethod
    def from_kml_points(
        cls, geo_config: GeoConfig, kml_points: dict[str, KmlPoint]
    ) -> "GeoPointRegistry":
        """Create registry from KML geographic points.

        Converts lat/lon coordinates to pixel coordinates using the configured
        geotransform.

        Args:
            geo_config: Georeferencing configuration with CRS and geotransform.
            kml_points: Dict mapping point names to KmlPoint objects.

        Returns:
            New GeoPointRegistry instance.
        """
        reverse_transformer = Transformer.from_crs("EPSG:4326", geo_config.crs, always_xy=True)
        points: dict[str, tuple[PixelPoint, KmlPoint]] = {}

        for name, kml_point in kml_points.items():
            px, py = _latlon_to_pixel(kml_point.lat, kml_point.lon, geo_config, reverse_transformer)
            pixel_point = PixelPoint(x=px, y=py)
            points[name] = (pixel_point, kml_point)

        return cls(geo_config=geo_config, points=points)

    @classmethod
    def from_pixel_points(
        cls,
        geo_config: GeoConfig,
        pixel_points: dict[str, PixelPoint],
        category: str = "other",
    ) -> "GeoPointRegistry":
        """Create registry from pixel coordinate points.

        Converts pixel coordinates to lat/lon using the configured geotransform.
        All points are assigned the same category.

        Args:
            geo_config: Georeferencing configuration with CRS and geotransform.
            pixel_points: Dict mapping point names to PixelPoint objects.
            category: Category for all points (e.g., "zebra", "arrow", "parking").

        Returns:
            New GeoPointRegistry instance.
        """
        transformer = Transformer.from_crs(geo_config.crs, "EPSG:4326", always_xy=True)
        points: dict[str, tuple[PixelPoint, KmlPoint]] = {}

        for name, pixel_point in pixel_points.items():
            lat, lon = _pixel_to_latlon(pixel_point.x, pixel_point.y, geo_config, transformer)
            kml_point = KmlPoint(name=name, category=category, lat=lat, lon=lon)
            points[name] = (pixel_point, kml_point)

        return cls(geo_config=geo_config, points=points)

    def render_kml(self) -> str:
        """Render points to KML format.

        Uses Jinja2 template for KML generation.

        Returns:
            KML content as a string.
        """
        kml_points = [kml_point for _, kml_point in self.points.values()]
        template = _template_env.get_template("points.kml.j2")
        return template.render(points=kml_points)
