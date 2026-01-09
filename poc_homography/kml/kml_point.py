"""Geographic point for KML export."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cache

from jinja2 import Environment, PackageLoader

DEFAULT_POINT_NAME = "unnamed"
_KNOWN_STYLES = frozenset(["zebra", "arrow", "parking"])

_template_env = Environment(
    loader=PackageLoader("poc_homography.kml", "templates"),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)
_placemark_template = _template_env.get_template("placemark.kml.j2")


@dataclass(frozen=True)
class KmlPoint:
    """Geographic point for KML export.

    Attributes:
        name: Point name/label.
        category: Point category (e.g., "zebra", "arrow", "parking", "other").
        lat: Latitude in degrees (WGS84).
        lon: Longitude in degrees (WGS84).
    """

    name: str
    category: str
    lat: float
    lon: float

    @property
    @cache  # noqa: B019 - safe on frozen dataclass
    def style(self) -> str:
        """Normalized style identifier for KML rendering (cached)."""
        style = self.category.lower().replace(" ", "_")
        return style if style in _KNOWN_STYLES else "other"

    @cache  # noqa: B019 - safe on frozen dataclass
    def to_kml(self) -> str:
        """Render this point as a KML Placemark element.

        Returns:
            KML Placemark XML string (cached).
        """
        return _placemark_template.render(point=self)

    @classmethod
    def from_placemark(cls, placemark: ET.Element) -> "KmlPoint | None":
        """Create KmlPoint from a KML Placemark element.

        Args:
            placemark: XML Element representing a KML Placemark.

        Returns:
            KmlPoint if coordinates found, None otherwise.
        """
        # Extract name
        name_elem = placemark.find("name")
        name = name_elem.text if name_elem is not None and name_elem.text else DEFAULT_POINT_NAME

        # Extract category from styleUrl or description
        category = cls._parse_category(placemark)

        # Get coordinates
        coords_elem = placemark.find(".//coordinates")
        if coords_elem is None or not coords_elem.text:
            return None

        parts = coords_elem.text.strip().split(",")
        if len(parts) < 2:
            return None

        lon = float(parts[0])
        lat = float(parts[1])
        return cls(name=name, category=category, lat=lat, lon=lon)

    @staticmethod
    def _parse_category(placemark: ET.Element) -> str:
        """Extract category from Placemark's styleUrl or description."""
        # Try styleUrl first
        style_elem = placemark.find("styleUrl")
        if style_elem is not None and style_elem.text:
            style = style_elem.text.replace("#", "")
            if style in ["zebra", "arrow", "parking"]:
                return style

        # Fall back to description
        desc_elem = placemark.find("description")
        if desc_elem is not None and desc_elem.text:
            desc_lower = desc_elem.text.lower()
            if "category: zebra" in desc_lower:
                return "zebra"
            if "category: arrow" in desc_lower:
                return "arrow"
            if "category: parking" in desc_lower:
                return "parking"

        return "other"
