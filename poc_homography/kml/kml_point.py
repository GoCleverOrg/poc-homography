"""Geographic point for KML export."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


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

    @classmethod
    def from_placemark(cls, placemark: ET.Element) -> "KmlPoint | None":
        """Create KmlPoint from a KML Placemark element.

        Args:
            placemark: XML Element representing a KML Placemark.

        Returns:
            KmlPoint if coordinates found, None otherwise.
        """
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
        return cls(category=category, lat=lat, lon=lon)

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
