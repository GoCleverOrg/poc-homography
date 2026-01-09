"""Point extractor for georeferenced images with KML import/export."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import cached_property

from jinja2 import Environment, PackageLoader
from pyproj import Transformer

from poc_homography.geotiff_utils import apply_geotransform

# Set up Jinja2 template environment
_template_env = Environment(
    loader=PackageLoader("poc_homography.kml", "templates"),
    autoescape=False,  # KML is XML, we handle escaping manually
    trim_blocks=True,
    lstrip_blocks=True,
)

# Type alias for the 6-parameter GDAL geotransform
Geotransform = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class PixelPoint:
    """Pixel coordinates in an image.

    Attributes:
        x: Pixel x coordinate (column).
        y: Pixel y coordinate (row).
    """

    x: float
    y: float


@dataclass(frozen=True)
class KmlPoint:
    """Geographic point for KML export.

    Attributes:
        category: Point category (e.g., "zebra", "arrow", "parking", "other").
        lat: Latitude in degrees (WGS84).
        lon: Longitude in degrees (WGS84).
        style: Normalized style identifier for KML rendering (computed).
    """

    category: str
    lat: float
    lon: float
    style: str = field(init=False)

    def __post_init__(self) -> None:
        """Compute style from category."""
        style = self.category.lower().replace(" ", "_")
        if style not in ["zebra", "arrow", "parking"]:
            style = "other"
        object.__setattr__(self, "style", style)


class Kml:
    """Parser for KML files containing geographic points.

    Parses KML text and extracts points as a cached property.

    Args:
        kml_text: KML file content as string.

    Example:
        >>> kml = Kml(kml_content)
        >>> for name, point in kml.points.items():
        ...     print(f"{name}: {point.lat}, {point.lon}")
    """

    def __init__(self, kml_text: str):
        self._kml_text = kml_text

    @cached_property
    def points(self) -> dict[str, KmlPoint]:
        """Parse KML and return geographic points.

        Returns:
            Dict mapping point names to KmlPoint objects.
        """
        # Remove namespace for easier parsing
        text = re.sub(r'\sxmlns="[^"]+"', "", self._kml_text, count=1)

        root = ET.fromstring(text)
        points: dict[str, KmlPoint] = {}
        unnamed_count = 0

        for placemark in root.iter("Placemark"):
            name_elem = placemark.find("name")
            if name_elem is not None and name_elem.text:
                name = name_elem.text
            else:
                unnamed_count += 1
                name = f"Point_{unnamed_count}"

            # Try to extract category from styleUrl or description
            style_elem = placemark.find("styleUrl")
            desc_elem = placemark.find("description")

            category = "other"
            if style_elem is not None and style_elem.text:
                style = style_elem.text.replace("#", "")
                if style in ["zebra", "arrow", "parking"]:
                    category = style

            # Also check description for category
            if desc_elem is not None and desc_elem.text:
                desc_lower = desc_elem.text.lower()
                if "category: zebra" in desc_lower:
                    category = "zebra"
                elif "category: arrow" in desc_lower:
                    category = "arrow"
                elif "category: parking" in desc_lower:
                    category = "parking"

            # Get coordinates
            coords_elem = placemark.find(".//coordinates")
            if coords_elem is not None and coords_elem.text:
                coords_text = coords_elem.text.strip()
                parts = coords_text.split(",")
                if len(parts) >= 2:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    points[name] = KmlPoint(category=category, lat=lat, lon=lon)

        return points


@dataclass(frozen=True)
class GeoConfig:
    """Configuration for georeferenced coordinate transformations.

    Attributes:
        crs: Coordinate reference system identifier (e.g., "EPSG:25830").
        geotransform: 6-parameter GDAL affine geotransform as
            (origin_x, pixel_width, row_rotation, origin_y, col_rotation, pixel_height).
            For north-up images, row_rotation and col_rotation are typically 0.
    """

    crs: str
    geotransform: Geotransform


class PointExtractor:
    """Extract and manage georeferenced points from images with KML support.

    This class handles coordinate transformations between pixel coordinates,
    UTM coordinates, and lat/lon (WGS84) using GDAL-style 6-parameter affine
    geotransforms.

    Points are stored as a mapping from name to (PixelPoint, KmlPoint) tuples,
    allowing efficient lookup by name and clear separation between pixel and
    geographic coordinate spaces.

    Args:
        geo_config: Georeferencing configuration with CRS and geotransform.

    Example:
        >>> geo_config = GeoConfig(
        ...     crs="EPSG:25830",
        ...     geotransform=(725140.0, 0.05, 0.0, 4373490.0, 0.0, -0.05),
        ... )
        >>> extractor = PointExtractor(geo_config)
        >>> extractor.add_point(100, 200, "P1", "zebra")
        >>> kml_content = extractor.render_kml()
    """

    def __init__(self, geo_config: GeoConfig):
        self.geo_config = geo_config
        self.points: dict[str, tuple[PixelPoint, KmlPoint]] = {}
        self.transformer = Transformer.from_crs(geo_config.crs, "EPSG:4326", always_xy=True)
        self.reverse_transformer = Transformer.from_crs("EPSG:4326", geo_config.crs, always_xy=True)

    def pixel_to_utm(self, px: float, py: float) -> tuple[float, float]:
        """Convert pixel coordinates to UTM using 6-parameter affine geotransform.

        Uses the GDAL GeoTransform formula:
            easting = GT[0] + px*GT[1] + py*GT[2]
            northing = GT[3] + px*GT[4] + py*GT[5]

        Args:
            px: Pixel x coordinate (column).
            py: Pixel y coordinate (row).

        Returns:
            Tuple of (easting, northing) in UTM coordinates.
        """
        easting, northing = apply_geotransform(px, py, list(self.geo_config.geotransform))
        return easting, northing

    def pixel_to_latlon(self, px: float, py: float) -> tuple[float, float]:
        """Convert pixel coordinates to lat/lon (WGS84).

        Args:
            px: Pixel x coordinate (column).
            py: Pixel y coordinate (row).

        Returns:
            Tuple of (latitude, longitude) in degrees.
        """
        easting, northing = self.pixel_to_utm(px, py)
        lon, lat = self.transformer.transform(easting, northing)
        return lat, lon

    def latlon_to_utm(self, lat: float, lon: float) -> tuple[float, float]:
        """Convert lat/lon to UTM coordinates.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.

        Returns:
            Tuple of (easting, northing) in UTM coordinates.
        """
        easting, northing = self.reverse_transformer.transform(lon, lat)
        return easting, northing

    def latlon_to_pixel(self, lat: float, lon: float) -> tuple[float, float]:
        """Convert lat/lon to pixel coordinates using inverse affine transform.

        For north-up images (GT[2]=0, GT[4]=0), this simplifies to:
            px = (easting - GT[0]) / GT[1]
            py = (northing - GT[3]) / GT[5]

        For rotated images, we solve the full 2x2 system.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.

        Returns:
            Tuple of (px, py) pixel coordinates.

        Raises:
            ValueError: If the geotransform matrix is singular (cannot be inverted).
        """
        easting, northing = self.latlon_to_utm(lat, lon)
        gt = self.geo_config.geotransform

        # Inverse affine transform
        # For general case (with rotation):
        # [easting - GT[0]]   [GT[1]  GT[2]]   [px]
        # [northing - GT[3]] = [GT[4]  GT[5]] * [py]
        #
        # Solve: [px, py] = inv([[GT[1], GT[2]], [GT[4], GT[5]]]) @ [easting-GT[0], northing-GT[3]]

        det = gt[1] * gt[5] - gt[2] * gt[4]
        if abs(det) < 1e-10:
            raise ValueError("Geotransform matrix is singular (cannot invert)")

        de = easting - gt[0]
        dn = northing - gt[3]

        px = (gt[5] * de - gt[2] * dn) / det
        py = (-gt[4] * de + gt[1] * dn) / det

        return px, py

    def import_kml(self, kml_points: dict[str, KmlPoint]) -> dict[str, tuple[PixelPoint, KmlPoint]]:
        """Convert KML geographic points to pixel coordinates.

        Converts lat/lon coordinates to pixel coordinates using the configured
        geotransform.

        Args:
            kml_points: Dict mapping point names to KmlPoint objects.

        Returns:
            Dict mapping point names to (PixelPoint, KmlPoint) tuples.
        """
        result: dict[str, tuple[PixelPoint, KmlPoint]] = {}

        for name, kml_point in kml_points.items():
            px, py = self.latlon_to_pixel(kml_point.lat, kml_point.lon)
            pixel_point = PixelPoint(x=px, y=py)
            result[name] = (pixel_point, kml_point)

        return result

    def add_point(
        self, px: float, py: float, name: str, category: str
    ) -> tuple[PixelPoint, KmlPoint]:
        """Add a reference point.

        Args:
            px: Pixel x coordinate.
            py: Pixel y coordinate.
            name: Point name/label (used as key in points dict).
            category: Point category (e.g., "zebra", "arrow", "parking", "other").

        Returns:
            Tuple of (PixelPoint, KmlPoint) representing the added point.
        """
        lat, lon = self.pixel_to_latlon(px, py)
        pixel_point = PixelPoint(x=px, y=py)
        kml_point = KmlPoint(category=category, lat=lat, lon=lon)
        self.points[name] = (pixel_point, kml_point)
        return pixel_point, kml_point

    def render_kml(self) -> str:
        """Render points to KML format.

        Uses Jinja2 template for KML generation.

        Returns:
            KML content as a string.
        """
        # Prepare points for template
        template_points = []
        for name, (_, kml_point) in self.points.items():
            template_points.append(
                {
                    "name": name,
                    "category": kml_point.category,
                    "lat": kml_point.lat,
                    "lon": kml_point.lon,
                    "style": kml_point.style,
                }
            )

        # Render template
        template = _template_env.get_template("points.kml.j2")
        return template.render(points=template_points)
