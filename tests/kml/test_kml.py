"""Unit tests for poc_homography.kml.kml module (Kml parser)."""

import pytest

from poc_homography.kml import Kml


class TestKml:
    """Tests for Kml parser class."""

    @pytest.mark.parametrize(
        "kml_text,expected_count",
        [
            (
                """<?xml version="1.0"?>
                <kml xmlns="http://www.opengis.net/kml/2.2">
                <Document>
                    <Placemark>
                        <name>P1</name>
                        <Point><coordinates>-0.23,39.64,0</coordinates></Point>
                    </Placemark>
                </Document>
                </kml>""",
                1,
            ),
            (
                """<?xml version="1.0"?>
                <kml xmlns="http://www.opengis.net/kml/2.2">
                <Document>
                    <Placemark>
                        <name>P1</name>
                        <Point><coordinates>-0.23,39.64,0</coordinates></Point>
                    </Placemark>
                    <Placemark>
                        <name>P2</name>
                        <Point><coordinates>-0.24,39.65,0</coordinates></Point>
                    </Placemark>
                    <Placemark>
                        <name>P3</name>
                        <Point><coordinates>-0.25,39.66,0</coordinates></Point>
                    </Placemark>
                </Document>
                </kml>""",
                3,
            ),
            (
                """<?xml version="1.0"?>
                <kml xmlns="http://www.opengis.net/kml/2.2">
                <Document></Document>
                </kml>""",
                0,
            ),
        ],
        ids=["single-point", "multiple-points", "empty"],
    )
    def test_points_count(self, kml_text: str, expected_count: int) -> None:
        """Test Kml parser extracts correct number of points."""
        kml = Kml(kml_text)

        assert len(kml.points) == expected_count

    def test_points_dict_keys_are_names(self) -> None:
        """Test points dict uses point names as keys."""
        kml_text = """<?xml version="1.0"?>
        <kml xmlns="http://www.opengis.net/kml/2.2">
        <Document>
            <Placemark>
                <name>Alpha</name>
                <Point><coordinates>1.0,2.0,0</coordinates></Point>
            </Placemark>
            <Placemark>
                <name>Beta</name>
                <Point><coordinates>3.0,4.0,0</coordinates></Point>
            </Placemark>
        </Document>
        </kml>"""

        kml = Kml(kml_text)

        assert "Alpha" in kml.points
        assert "Beta" in kml.points
        assert kml.points["Alpha"].lat == pytest.approx(2.0)
        assert kml.points["Beta"].lat == pytest.approx(4.0)

    def test_unnamed_points_get_generated_names(self) -> None:
        """Test unnamed placemarks get auto-generated names."""
        kml_text = """<?xml version="1.0"?>
        <kml xmlns="http://www.opengis.net/kml/2.2">
        <Document>
            <Placemark>
                <Point><coordinates>1.0,2.0,0</coordinates></Point>
            </Placemark>
            <Placemark>
                <Point><coordinates>3.0,4.0,0</coordinates></Point>
            </Placemark>
        </Document>
        </kml>"""

        kml = Kml(kml_text)

        assert "Point_1" in kml.points
        assert "Point_2" in kml.points

    def test_points_is_cached_property(self) -> None:
        """Test points property is cached (same dict returned)."""
        kml_text = """<?xml version="1.0"?>
        <kml xmlns="http://www.opengis.net/kml/2.2">
        <Document>
            <Placemark>
                <name>P1</name>
                <Point><coordinates>1.0,2.0,0</coordinates></Point>
            </Placemark>
        </Document>
        </kml>"""

        kml = Kml(kml_text)

        points1 = kml.points
        points2 = kml.points

        assert points1 is points2

    def test_skips_placemarks_without_coordinates(self) -> None:
        """Test placemarks without coordinates are skipped."""
        kml_text = """<?xml version="1.0"?>
        <kml xmlns="http://www.opengis.net/kml/2.2">
        <Document>
            <Placemark>
                <name>ValidPoint</name>
                <Point><coordinates>1.0,2.0,0</coordinates></Point>
            </Placemark>
            <Placemark>
                <name>InvalidPoint</name>
            </Placemark>
        </Document>
        </kml>"""

        kml = Kml(kml_text)

        assert len(kml.points) == 1
        assert "ValidPoint" in kml.points
        assert "InvalidPoint" not in kml.points

    def test_handles_nested_folders(self) -> None:
        """Test parser finds placemarks in nested folders."""
        kml_text = """<?xml version="1.0"?>
        <kml xmlns="http://www.opengis.net/kml/2.2">
        <Document>
            <Folder>
                <name>Folder1</name>
                <Placemark>
                    <name>P1</name>
                    <Point><coordinates>1.0,2.0,0</coordinates></Point>
                </Placemark>
                <Folder>
                    <name>Nested</name>
                    <Placemark>
                        <name>P2</name>
                        <Point><coordinates>3.0,4.0,0</coordinates></Point>
                    </Placemark>
                </Folder>
            </Folder>
        </Document>
        </kml>"""

        kml = Kml(kml_text)

        assert len(kml.points) == 2
        assert "P1" in kml.points
        assert "P2" in kml.points
