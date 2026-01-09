"""Point extractor for georeferenced images with KML import/export."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from pyproj import Transformer

from poc_homography.geotiff_utils import apply_geotransform


class PointExtractor:
    """Extract and manage georeferenced points from images with KML support.

    This class handles coordinate transformations between pixel coordinates,
    UTM coordinates, and lat/lon (WGS84) using GDAL-style 6-parameter affine
    geotransforms.

    Args:
        image_path: Path to the source image file.
        config: Configuration dict containing:
            - crs: Coordinate reference system (e.g., "EPSG:25830")
            - geotransform: 6-parameter GDAL geotransform array
              [origin_x, pixel_width, row_rotation, origin_y, col_rotation, pixel_height]

    Example:
        >>> config = {
        ...     "crs": "EPSG:25830",
        ...     "geotransform": [725140.0, 0.05, 0, 4373490.0, 0, -0.05]
        ... }
        >>> extractor = PointExtractor("image.tif", config)
        >>> extractor.add_point(100, 200, "P1", "zebra")
        >>> extractor.export_kml("output.kml")
    """

    def __init__(self, image_path: str, config: dict):
        self.image_path = Path(image_path)
        self.config = config
        self.points: list[dict] = []
        self.transformer = Transformer.from_crs(config["crs"], "EPSG:4326", always_xy=True)
        self.reverse_transformer = Transformer.from_crs("EPSG:4326", config["crs"], always_xy=True)

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
        easting, northing = apply_geotransform(px, py, self.config["geotransform"])
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
        gt = self.config["geotransform"]

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

    def parse_kml(self, kml_text: str) -> list[dict]:
        """Parse KML file and extract points with pixel coordinates.

        Args:
            kml_text: KML file content as string.

        Returns:
            List of point dictionaries with keys:
                - name: Point name
                - category: Point category (zebra, arrow, parking, other)
                - px, py: Pixel coordinates
                - lat, lon: Geographic coordinates
        """
        # Remove namespace for easier parsing
        kml_text = re.sub(r'\sxmlns="[^"]+"', "", kml_text, count=1)

        root = ET.fromstring(kml_text)
        points = []

        for placemark in root.iter("Placemark"):
            name_elem = placemark.find("name")
            name = name_elem.text if name_elem is not None else f"Point_{len(points) + 1}"

            # Try to extract category from styleUrl or description
            style_elem = placemark.find("styleUrl")
            desc_elem = placemark.find("description")

            category = "other"
            if style_elem is not None:
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
                    px, py = self.latlon_to_pixel(lat, lon)
                    points.append(
                        {
                            "name": name,
                            "category": category,
                            "px": px,
                            "py": py,
                            "lat": lat,
                            "lon": lon,
                        }
                    )

        return points

    def add_point(self, px: float, py: float, name: str, category: str) -> dict:
        """Add a reference point.

        Args:
            px: Pixel x coordinate.
            py: Pixel y coordinate.
            name: Point name/label.
            category: Point category (e.g., "zebra", "arrow", "parking", "other").

        Returns:
            Dictionary containing the added point's data.
        """
        lat, lon = self.pixel_to_latlon(px, py)
        easting, northing = self.pixel_to_utm(px, py)
        self.points.append(
            {
                "name": name,
                "category": category,
                "pixel_x": px,
                "pixel_y": py,
                "easting": easting,
                "northing": northing,
                "lat": lat,
                "lon": lon,
            }
        )
        return self.points[-1]

    def export_kml(self, output_path: str) -> str:
        """Export points to KML file with both GPS and UTM coordinates.

        Args:
            output_path: Path for the output KML file.

        Returns:
            The output path.
        """
        kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>Reference Points</name>
    <description>Extracted from {self.image_path.name}</description>

    <Schema name="GCPData" id="GCPData">
        <SimpleField type="string" name="category"/>
        <SimpleField type="float" name="pixel_x"/>
        <SimpleField type="float" name="pixel_y"/>
        <SimpleField type="float" name="utm_easting"/>
        <SimpleField type="float" name="utm_northing"/>
        <SimpleField type="string" name="utm_crs"/>
    </Schema>

    <Style id="zebra">
        <IconStyle><color>ff0000ff</color><scale>0.8</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon>
        </IconStyle>
    </Style>
    <Style id="arrow">
        <IconStyle><color>ff00ff00</color><scale>0.8</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/paddle/grn-circle.png</href></Icon>
        </IconStyle>
    </Style>
    <Style id="parking">
        <IconStyle><color>ffff0000</color><scale>0.8</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/paddle/blu-circle.png</href></Icon>
        </IconStyle>
    </Style>
    <Style id="other">
        <IconStyle><color>ff00ffff</color><scale>0.8</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png</href></Icon>
        </IconStyle>
    </Style>
"""

        for pt in self.points:
            style = pt["category"].lower().replace(" ", "_")
            if style not in ["zebra", "arrow", "parking"]:
                style = "other"

            kml_content += """
    <Placemark>
        <name>{name}</name>
        <description>Category: {category}
Pixel: ({pixel_x:.1f}, {pixel_y:.1f})
UTM: E {easting:.2f}, N {northing:.2f}
CRS: {crs}</description>
        <styleUrl>#{style}</styleUrl>
        <ExtendedData>
            <SchemaData schemaUrl="#GCPData">
                <SimpleData name="category">{category}</SimpleData>
                <SimpleData name="pixel_x">{pixel_x:.2f}</SimpleData>
                <SimpleData name="pixel_y">{pixel_y:.2f}</SimpleData>
                <SimpleData name="utm_easting">{easting:.4f}</SimpleData>
                <SimpleData name="utm_northing">{northing:.4f}</SimpleData>
                <SimpleData name="utm_crs">{crs}</SimpleData>
            </SchemaData>
        </ExtendedData>
        <Point>
            <coordinates>{lon:.8f},{lat:.8f},0</coordinates>
        </Point>
    </Placemark>
""".format(
                name=pt["name"],
                category=pt["category"],
                pixel_x=pt["pixel_x"],
                pixel_y=pt["pixel_y"],
                easting=pt["easting"],
                northing=pt["northing"],
                crs=self.config["crs"],
                style=style,
                lon=pt["lon"],
                lat=pt["lat"],
            )

        kml_content += """
</Document>
</kml>"""

        with open(output_path, "w") as f:
            f.write(kml_content)

        return output_path
