"""Unit tests for poc_homography.kml.kml_point module."""

import xml.etree.ElementTree as ET

import pytest

from poc_homography.kml import KmlPoint


class TestKmlPoint:
    """Tests for KmlPoint frozen dataclass."""

    @pytest.mark.parametrize(
        "name,category,lat,lon",
        [
            ("P1", "zebra", 39.640600, -0.230200),
            ("Point_2", "arrow", 40.416775, -3.703790),
            ("Parking Spot", "parking", 41.385064, 2.173404),
            ("Other Point", "unknown", 0.0, 0.0),
        ],
        ids=["zebra-valencia", "arrow-madrid", "parking-barcelona", "other-equator"],
    )
    def test_creation(self, name: str, category: str, lat: float, lon: float) -> None:
        """Test KmlPoint can be created with various data."""
        point = KmlPoint(name=name, category=category, lat=lat, lon=lon)

        assert point.name == name
        assert point.category == category
        assert point.lat == lat
        assert point.lon == lon

    def test_frozen(self) -> None:
        """Test KmlPoint is immutable."""
        point = KmlPoint(name="P1", category="zebra", lat=39.5, lon=-0.4)

        with pytest.raises(AttributeError):
            point.lat = 40.0  # type: ignore[misc]


class TestKmlPointStyle:
    """Tests for KmlPoint.style property."""

    @pytest.mark.parametrize(
        "category,expected_style",
        [
            ("zebra", "zebra"),
            ("ZEBRA", "zebra"),
            ("Zebra", "zebra"),
            ("arrow", "arrow"),
            ("ARROW", "arrow"),
            ("parking", "parking"),
            ("PARKING", "parking"),
            ("Parking Space", "other"),
            ("unknown", "other"),
            ("", "other"),
            ("something else", "other"),
        ],
        ids=[
            "zebra-lower",
            "zebra-upper",
            "zebra-title",
            "arrow-lower",
            "arrow-upper",
            "parking-lower",
            "parking-upper",
            "parking-with-space",
            "unknown",
            "empty",
            "random",
        ],
    )
    def test_style_property(self, category: str, expected_style: str) -> None:
        """Test style property normalizes category to known styles."""
        point = KmlPoint(name="Test", category=category, lat=0.0, lon=0.0)

        assert point.style == expected_style

    def test_style_is_cached(self) -> None:
        """Test style property is cached (same object returned)."""
        point = KmlPoint(name="Test", category="zebra", lat=0.0, lon=0.0)

        style1 = point.style
        style2 = point.style

        assert style1 == style2
        assert style1 == "zebra"


class TestKmlPointToKml:
    """Tests for KmlPoint.to_kml method."""

    @pytest.mark.parametrize(
        "name,category,lat,lon",
        [
            ("P1", "zebra", 39.640600, -0.230200),
            ("Test Point", "arrow", 40.0, -3.5),
            ("Simple", "other", 0.0, 0.0),
        ],
        ids=["zebra-point", "arrow-point", "other-at-origin"],
    )
    def test_to_kml_structure(self, name: str, category: str, lat: float, lon: float) -> None:
        """Test to_kml returns valid KML placemark XML."""
        point = KmlPoint(name=name, category=category, lat=lat, lon=lon)
        kml_str = point.to_kml()

        elem = ET.fromstring(kml_str)

        assert elem.tag == "Placemark"

        name_elem = elem.find("name")
        assert name_elem is not None
        assert name_elem.text == name

        desc_elem = elem.find("description")
        assert desc_elem is not None
        assert f"Category: {category}" in desc_elem.text

        style_elem = elem.find("styleUrl")
        assert style_elem is not None
        assert style_elem.text == f"#{point.style}"

        point_elem = elem.find("Point")
        assert point_elem is not None

        coords_elem = point_elem.find("coordinates")
        assert coords_elem is not None

    @pytest.mark.parametrize(
        "lat,lon,expected_substr",
        [
            (39.64060000, -0.23020000, "-0.23020000,39.64060000,0"),
            (0.0, 0.0, "0.00000000,0.00000000,0"),
            (45.12345678, 12.87654321, "12.87654321,45.12345678,0"),
        ],
        ids=["valencia", "origin", "italy"],
    )
    def test_to_kml_coordinate_format(self, lat: float, lon: float, expected_substr: str) -> None:
        """Test to_kml uses correct coordinate format (lon,lat,0)."""
        point = KmlPoint(name="Test", category="zebra", lat=lat, lon=lon)
        kml_str = point.to_kml()

        assert expected_substr in kml_str

    def test_to_kml_is_cached(self) -> None:
        """Test to_kml result is cached."""
        point = KmlPoint(name="Test", category="zebra", lat=39.5, lon=-0.4)

        kml1 = point.to_kml()
        kml2 = point.to_kml()

        assert kml1 == kml2


class TestKmlPointFromPlacemark:
    """Tests for KmlPoint.from_placemark class method."""

    @pytest.mark.parametrize(
        "xml_str,expected_name,expected_category,expected_lat,expected_lon",
        [
            (
                """<Placemark>
                    <name>P1</name>
                    <Point><coordinates>-0.230200,39.640600,0</coordinates></Point>
                </Placemark>""",
                "P1",
                "other",
                39.640600,
                -0.230200,
            ),
            (
                """<Placemark>
                    <name>Zebra Point</name>
                    <styleUrl>#zebra</styleUrl>
                    <Point><coordinates>-3.703790,40.416775,0</coordinates></Point>
                </Placemark>""",
                "Zebra Point",
                "zebra",
                40.416775,
                -3.703790,
            ),
            (
                """<Placemark>
                    <name>Arrow</name>
                    <styleUrl>#arrow</styleUrl>
                    <Point><coordinates>2.173404,41.385064,0</coordinates></Point>
                </Placemark>""",
                "Arrow",
                "arrow",
                41.385064,
                2.173404,
            ),
            (
                """<Placemark>
                    <name>Parking</name>
                    <styleUrl>#parking</styleUrl>
                    <Point><coordinates>0.0,0.0,0</coordinates></Point>
                </Placemark>""",
                "Parking",
                "parking",
                0.0,
                0.0,
            ),
            (
                """<Placemark>
                    <name>Desc Point</name>
                    <description>Category: zebra</description>
                    <Point><coordinates>1.0,2.0,0</coordinates></Point>
                </Placemark>""",
                "Desc Point",
                "zebra",
                2.0,
                1.0,
            ),
            (
                """<Placemark>
                    <Point><coordinates>5.0,10.0,0</coordinates></Point>
                </Placemark>""",
                "unnamed",
                "other",
                10.0,
                5.0,
            ),
        ],
        ids=[
            "basic",
            "style-zebra",
            "style-arrow",
            "style-parking",
            "desc-category",
            "no-name",
        ],
    )
    def test_from_placemark_valid(
        self,
        xml_str: str,
        expected_name: str,
        expected_category: str,
        expected_lat: float,
        expected_lon: float,
    ) -> None:
        """Test from_placemark parses valid placemarks correctly."""
        placemark = ET.fromstring(xml_str)
        point = KmlPoint.from_placemark(placemark)

        assert point is not None
        assert point.name == expected_name
        assert point.category == expected_category
        assert point.lat == pytest.approx(expected_lat)
        assert point.lon == pytest.approx(expected_lon)

    @pytest.mark.parametrize(
        "xml_str",
        [
            "<Placemark><name>NoCoords</name></Placemark>",
            "<Placemark><Point><coordinates></coordinates></Point></Placemark>",
            "<Placemark><Point><coordinates>1.0</coordinates></Point></Placemark>",
        ],
        ids=["no-coords", "empty-coords", "single-value"],
    )
    def test_from_placemark_returns_none(self, xml_str: str) -> None:
        """Test from_placemark returns None for invalid placemarks."""
        placemark = ET.fromstring(xml_str)
        result = KmlPoint.from_placemark(placemark)

        assert result is None


class TestKmlPointParseCategory:
    """Tests for KmlPoint._parse_category static method."""

    @pytest.mark.parametrize(
        "xml_str,expected_category",
        [
            ("<Placemark><styleUrl>#zebra</styleUrl></Placemark>", "zebra"),
            ("<Placemark><styleUrl>#arrow</styleUrl></Placemark>", "arrow"),
            ("<Placemark><styleUrl>#parking</styleUrl></Placemark>", "parking"),
            ("<Placemark><styleUrl>#custom</styleUrl></Placemark>", "other"),
            (
                "<Placemark><description>Category: zebra</description></Placemark>",
                "zebra",
            ),
            (
                "<Placemark><description>CATEGORY: ARROW</description></Placemark>",
                "arrow",
            ),
            (
                "<Placemark><description>Some text. Category: parking. More text.</description></Placemark>",
                "parking",
            ),
            ("<Placemark><name>Test</name></Placemark>", "other"),
            (
                "<Placemark><styleUrl>#zebra</styleUrl><description>Category: arrow</description></Placemark>",
                "zebra",
            ),
        ],
        ids=[
            "style-zebra",
            "style-arrow",
            "style-parking",
            "style-unknown",
            "desc-zebra",
            "desc-arrow-upper",
            "desc-parking-embedded",
            "no-style-no-desc",
            "style-precedence",
        ],
    )
    def test_parse_category(self, xml_str: str, expected_category: str) -> None:
        """Test _parse_category extracts category correctly."""
        placemark = ET.fromstring(xml_str)
        category = KmlPoint._parse_category(placemark)

        assert category == expected_category
