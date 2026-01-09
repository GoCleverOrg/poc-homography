"""Unit tests for poc_homography.kml.pixel_point module."""

import pytest

from poc_homography.kml import PixelPoint


class TestPixelPoint:
    """Tests for PixelPoint frozen dataclass."""

    @pytest.mark.parametrize(
        "x,y",
        [
            (0.0, 0.0),
            (100.5, 200.75),
            (1920.0, 1080.0),
            (-10.0, -20.0),
            (0.123456789, 0.987654321),
        ],
        ids=["origin", "fractional", "hd-resolution", "negative", "high-precision"],
    )
    def test_creation(self, x: float, y: float) -> None:
        """Test PixelPoint can be created with various coordinates."""
        point = PixelPoint(x=x, y=y)

        assert point.x == x
        assert point.y == y

    def test_frozen(self) -> None:
        """Test PixelPoint is immutable."""
        point = PixelPoint(x=100.0, y=200.0)

        with pytest.raises(AttributeError):
            point.x = 50.0  # type: ignore[misc]

    @pytest.mark.parametrize(
        "p1,p2,expected_equal",
        [
            ((100.0, 200.0), (100.0, 200.0), True),
            ((100.0, 200.0), (100.0, 201.0), False),
            ((0.0, 0.0), (0.0, 0.0), True),
        ],
        ids=["same", "different-y", "both-origin"],
    )
    def test_equality(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        expected_equal: bool,
    ) -> None:
        """Test PixelPoint equality comparison."""
        point1 = PixelPoint(x=p1[0], y=p1[1])
        point2 = PixelPoint(x=p2[0], y=p2[1])

        assert (point1 == point2) == expected_equal
