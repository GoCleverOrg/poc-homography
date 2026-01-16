"""Test suite for ImageDimensions value object.

Tests cover:
- Creation via factory method with validation
- Computed properties (area, aspect_ratio, center, center_x, center_y)
- contains() method for bounds checking
- Serialization (to_tuple, to_dict, from_dict)
- Immutability
"""

import pytest

from poc_homography.domain.vo.image_dimensions import ImageDimensions
from poc_homography.domain.vo.pixel_point import PixelPoint
from poc_homography.types import Pixels


class TestImageDimensionsCreation:
    """Test ImageDimensions creation via factory method."""

    def test_create_valid_dimensions(self):
        """Test creating valid ImageDimensions."""
        dims = ImageDimensions.create(width=1920, height=1080)

        assert dims.width == 1920
        assert dims.height == 1080

    def test_create_with_pixels_type(self):
        """Test creating with Pixels type directly."""
        dims = ImageDimensions(width=Pixels(1920), height=Pixels(1080))

        assert dims.width == 1920
        assert dims.height == 1080

    def test_create_rejects_zero_width(self):
        """Test that zero width is rejected."""
        with pytest.raises(ValueError, match="width must be positive"):
            ImageDimensions.create(width=0, height=1080)

    def test_create_rejects_negative_width(self):
        """Test that negative width is rejected."""
        with pytest.raises(ValueError, match="width must be positive"):
            ImageDimensions.create(width=-1, height=1080)

    def test_create_rejects_zero_height(self):
        """Test that zero height is rejected."""
        with pytest.raises(ValueError, match="height must be positive"):
            ImageDimensions.create(width=1920, height=0)

    def test_create_rejects_negative_height(self):
        """Test that negative height is rejected."""
        with pytest.raises(ValueError, match="height must be positive"):
            ImageDimensions.create(width=1920, height=-1)


class TestImageDimensionsComputedProperties:
    """Test computed properties."""

    def test_area(self):
        """Test area computation."""
        dims = ImageDimensions.create(width=1920, height=1080)
        assert dims.area == 1920 * 1080

    def test_area_square(self):
        """Test area for square image."""
        dims = ImageDimensions.create(width=100, height=100)
        assert dims.area == 10000

    def test_aspect_ratio_16_9(self):
        """Test aspect ratio for 16:9."""
        dims = ImageDimensions.create(width=1920, height=1080)
        assert dims.aspect_ratio == pytest.approx(16.0 / 9.0)

    def test_aspect_ratio_4_3(self):
        """Test aspect ratio for 4:3."""
        dims = ImageDimensions.create(width=1024, height=768)
        assert dims.aspect_ratio == pytest.approx(4.0 / 3.0)

    def test_aspect_ratio_square(self):
        """Test aspect ratio for square image."""
        dims = ImageDimensions.create(width=100, height=100)
        assert dims.aspect_ratio == 1.0

    def test_center_x(self):
        """Test center_x computation."""
        dims = ImageDimensions.create(width=1920, height=1080)
        assert dims.center_x == pytest.approx(960.0)

    def test_center_y(self):
        """Test center_y computation."""
        dims = ImageDimensions.create(width=1920, height=1080)
        assert dims.center_y == pytest.approx(540.0)

    def test_center_odd_dimensions(self):
        """Test center for odd dimensions."""
        dims = ImageDimensions.create(width=101, height=51)
        assert dims.center_x == pytest.approx(50.5)
        assert dims.center_y == pytest.approx(25.5)

    def test_center_returns_pixel_point(self):
        """Test that center returns a PixelPoint."""
        dims = ImageDimensions.create(width=1920, height=1080)
        center = dims.center

        assert isinstance(center, PixelPoint)
        assert center.x == pytest.approx(960.0)
        assert center.y == pytest.approx(540.0)


class TestImageDimensionsContains:
    """Test contains() method."""

    def test_contains_center_point(self):
        """Test that center point is contained."""
        dims = ImageDimensions.create(width=100, height=100)
        point = PixelPoint.create(50.0, 50.0)
        assert dims.contains(point) is True

    def test_contains_origin(self):
        """Test that origin is contained."""
        dims = ImageDimensions.create(width=100, height=100)
        point = PixelPoint.create(0.0, 0.0)
        assert dims.contains(point) is True

    def test_not_contains_at_width_boundary(self):
        """Test that point at width boundary is NOT contained."""
        dims = ImageDimensions.create(width=100, height=100)
        point = PixelPoint.create(100.0, 50.0)
        assert dims.contains(point) is False

    def test_not_contains_at_height_boundary(self):
        """Test that point at height boundary is NOT contained."""
        dims = ImageDimensions.create(width=100, height=100)
        point = PixelPoint.create(50.0, 100.0)
        assert dims.contains(point) is False

    def test_not_contains_negative_x(self):
        """Test that point with negative x is NOT contained."""
        dims = ImageDimensions.create(width=100, height=100)
        point = PixelPoint.create(-1.0, 50.0)
        assert dims.contains(point) is False

    def test_not_contains_negative_y(self):
        """Test that point with negative y is NOT contained."""
        dims = ImageDimensions.create(width=100, height=100)
        point = PixelPoint.create(50.0, -1.0)
        assert dims.contains(point) is False

    def test_contains_just_inside_boundary(self):
        """Test that point just inside boundary is contained."""
        dims = ImageDimensions.create(width=100, height=100)
        point = PixelPoint.create(99.9, 99.9)
        assert dims.contains(point) is True


class TestImageDimensionsSerialization:
    """Test serialization methods."""

    def test_to_tuple(self):
        """Test to_tuple conversion."""
        dims = ImageDimensions.create(width=1920, height=1080)
        result = dims.to_tuple()

        assert result == (1920, 1080)
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_to_dict(self):
        """Test to_dict conversion."""
        dims = ImageDimensions.create(width=1920, height=1080)
        result = dims.to_dict()

        assert result == {"width": 1920, "height": 1080}
        assert isinstance(result["width"], int)
        assert isinstance(result["height"], int)

    def test_from_dict(self):
        """Test from_dict creation."""
        data = {"width": 1920, "height": 1080}
        dims = ImageDimensions.from_dict(data)

        assert dims.width == 1920
        assert dims.height == 1080

    def test_round_trip_serialization(self):
        """Test that to_dict and from_dict are inverse operations."""
        original = ImageDimensions.create(width=2560, height=1440)
        data = original.to_dict()
        restored = ImageDimensions.from_dict(data)

        assert restored == original


class TestImageDimensionsFrozen:
    """Test that ImageDimensions is immutable."""

    def test_cannot_modify_width(self):
        """Test that width cannot be modified."""
        dims = ImageDimensions.create(width=1920, height=1080)

        with pytest.raises(AttributeError):
            dims.width = Pixels(1000)  # type: ignore[misc]

    def test_cannot_modify_height(self):
        """Test that height cannot be modified."""
        dims = ImageDimensions.create(width=1920, height=1080)

        with pytest.raises(AttributeError):
            dims.height = Pixels(1000)  # type: ignore[misc]


class TestImageDimensionsEquality:
    """Test equality comparisons."""

    def test_equal_dimensions(self):
        """Test that equal dimensions are equal."""
        dims1 = ImageDimensions.create(width=1920, height=1080)
        dims2 = ImageDimensions.create(width=1920, height=1080)

        assert dims1 == dims2

    def test_different_width_not_equal(self):
        """Test that different widths are not equal."""
        dims1 = ImageDimensions.create(width=1920, height=1080)
        dims2 = ImageDimensions.create(width=1280, height=1080)

        assert dims1 != dims2

    def test_different_height_not_equal(self):
        """Test that different heights are not equal."""
        dims1 = ImageDimensions.create(width=1920, height=1080)
        dims2 = ImageDimensions.create(width=1920, height=720)

        assert dims1 != dims2

    def test_hashable(self):
        """Test that ImageDimensions can be used in sets and as dict keys."""
        dims1 = ImageDimensions.create(width=1920, height=1080)
        dims2 = ImageDimensions.create(width=1920, height=1080)
        dims3 = ImageDimensions.create(width=1280, height=720)

        # Can be used in set
        dims_set = {dims1, dims2, dims3}
        assert len(dims_set) == 2  # dims1 and dims2 are equal

        # Can be used as dict key
        dims_dict = {dims1: "Full HD", dims3: "HD"}
        assert dims_dict[dims2] == "Full HD"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
