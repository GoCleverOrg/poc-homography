"""KML file parser."""

import re
import xml.etree.ElementTree as ET
from functools import cached_property

from poc_homography.kml.kml_point import KmlPoint


class Kml:
    """Parser for KML files containing geographic points.

    Args:
        kml_text: KML file content as string.

    Example:
        >>> kml = Kml(kml_content)
        >>> for name, point in kml.points.items():
        ...     print(f"{name}: {point.lat}, {point.lon}")
    """

    def __init__(self, kml_text: str):
        text = re.sub(r'\sxmlns="[^"]+"', "", kml_text, count=1)
        self._root = ET.fromstring(text)

    @cached_property
    def points(self) -> dict[str, KmlPoint]:
        """Parse KML and return geographic points.

        Returns:
            Dict mapping point names to KmlPoint objects.
        """
        points: dict[str, KmlPoint] = {}
        unnamed_count = 0

        for placemark in self._root.iter("Placemark"):
            point = KmlPoint.from_placemark(placemark)
            if point is None:
                continue

            name_elem = placemark.find("name")
            if name_elem is not None and name_elem.text:
                name = name_elem.text
            else:
                unnamed_count += 1
                name = f"Point_{unnamed_count}"

            points[name] = point

        return points
