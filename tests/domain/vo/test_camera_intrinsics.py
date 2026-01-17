"""Test suite for CameraIntrinsics value object.

Tests cover:
- Creation via factory method with validation
- Private constructor enforcement
- Computed properties (focal_length_px, cx, cy)
- Intrinsic matrix K access via to_K()
- Immutability
"""

import numpy as np
import pytest

from poc_homography.domain.vo.camera_intrinsics import CameraIntrinsics
from poc_homography.domain.vo.image_dimensions import ImageDimensions
from poc_homography.domain.vo.matrix3x3 import Matrix3x3
from poc_homography.types import Millimeters, Pixels

# Common test parameters (based on Hikvision DS-2DF8425IX-AELW specs)
SENSOR_WIDTH = Millimeters(6.78)
BASE_FOCAL_LENGTH = Millimeters(5.9)
IMAGE_WIDTH = Pixels(1920)
IMAGE_HEIGHT = Pixels(1080)
FOCAL_LENGTH_1X = Millimeters(5.9)  # At 1x zoom
FOCAL_LENGTH_10X = Millimeters(59.0)  # At 10x zoom


class TestCameraIntrinsicsCreation:
    """Test CameraIntrinsics creation via factory method."""

    def test_create_valid_intrinsics(self):
        """Test creating valid CameraIntrinsics."""
        intrinsics = CameraIntrinsics.create(
            sensor_width=SENSOR_WIDTH,
            base_focal_length=BASE_FOCAL_LENGTH,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            focal_length=FOCAL_LENGTH_1X,
        )

        assert intrinsics.sensor_width == SENSOR_WIDTH
        assert intrinsics.base_focal_length == BASE_FOCAL_LENGTH
        assert intrinsics.image_width == IMAGE_WIDTH
        assert intrinsics.image_height == IMAGE_HEIGHT
        assert intrinsics.focal_length == FOCAL_LENGTH_1X

    def test_create_rejects_non_positive_sensor_width(self):
        """Test that non-positive sensor_width is rejected."""
        with pytest.raises(ValueError, match="sensor_width must be positive"):
            CameraIntrinsics.create(
                sensor_width=Millimeters(0),
                base_focal_length=BASE_FOCAL_LENGTH,
                image_width=IMAGE_WIDTH,
                image_height=IMAGE_HEIGHT,
                focal_length=FOCAL_LENGTH_1X,
            )

        with pytest.raises(ValueError, match="sensor_width must be positive"):
            CameraIntrinsics.create(
                sensor_width=Millimeters(-1.0),
                base_focal_length=BASE_FOCAL_LENGTH,
                image_width=IMAGE_WIDTH,
                image_height=IMAGE_HEIGHT,
                focal_length=FOCAL_LENGTH_1X,
            )

    def test_create_rejects_non_positive_base_focal_length(self):
        """Test that non-positive base_focal_length is rejected."""
        with pytest.raises(ValueError, match="base_focal_length must be positive"):
            CameraIntrinsics.create(
                sensor_width=SENSOR_WIDTH,
                base_focal_length=Millimeters(0),
                image_width=IMAGE_WIDTH,
                image_height=IMAGE_HEIGHT,
                focal_length=FOCAL_LENGTH_1X,
            )

    def test_create_rejects_non_positive_image_width(self):
        """Test that non-positive image_width is rejected."""
        with pytest.raises(ValueError, match="image_width must be positive"):
            CameraIntrinsics.create(
                sensor_width=SENSOR_WIDTH,
                base_focal_length=BASE_FOCAL_LENGTH,
                image_width=Pixels(0),
                image_height=IMAGE_HEIGHT,
                focal_length=FOCAL_LENGTH_1X,
            )

    def test_create_rejects_non_positive_image_height(self):
        """Test that non-positive image_height is rejected."""
        with pytest.raises(ValueError, match="image_height must be positive"):
            CameraIntrinsics.create(
                sensor_width=SENSOR_WIDTH,
                base_focal_length=BASE_FOCAL_LENGTH,
                image_width=IMAGE_WIDTH,
                image_height=Pixels(0),
                focal_length=FOCAL_LENGTH_1X,
            )

    def test_create_rejects_non_positive_focal_length(self):
        """Test that non-positive focal_length is rejected."""
        with pytest.raises(ValueError, match="focal_length must be positive"):
            CameraIntrinsics.create(
                sensor_width=SENSOR_WIDTH,
                base_focal_length=BASE_FOCAL_LENGTH,
                image_width=IMAGE_WIDTH,
                image_height=IMAGE_HEIGHT,
                focal_length=Millimeters(0),
            )

    def test_direct_constructor_raises_error(self):
        """Test that direct constructor access is blocked."""
        K = Matrix3x3.create(np.eye(3))
        dims = ImageDimensions.create(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)

        with pytest.raises(TypeError, match="cannot be instantiated directly"):
            CameraIntrinsics(
                sensor_width=SENSOR_WIDTH,
                base_focal_length=BASE_FOCAL_LENGTH,
                dimensions=dims,
                focal_length=FOCAL_LENGTH_1X,
                _K=K,
            )


class TestCameraIntrinsicsComputedProperties:
    """Test computed properties."""

    def test_focal_length_px(self):
        """Test focal_length_px computation."""
        intrinsics = CameraIntrinsics.create(
            sensor_width=SENSOR_WIDTH,
            base_focal_length=BASE_FOCAL_LENGTH,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            focal_length=FOCAL_LENGTH_1X,
        )

        # focal_length_px = focal_length_mm * (image_width / sensor_width)
        # = 5.9 * (1920 / 6.78) ≈ 1671.68
        expected_f_px = FOCAL_LENGTH_1X * (IMAGE_WIDTH / SENSOR_WIDTH)
        assert intrinsics.focal_length_px == pytest.approx(expected_f_px, rel=1e-6)

    def test_focal_length_px_at_zoom(self):
        """Test focal_length_px scales with zoom."""
        intrinsics_1x = CameraIntrinsics.create(
            sensor_width=SENSOR_WIDTH,
            base_focal_length=BASE_FOCAL_LENGTH,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            focal_length=FOCAL_LENGTH_1X,
        )

        intrinsics_10x = CameraIntrinsics.create(
            sensor_width=SENSOR_WIDTH,
            base_focal_length=BASE_FOCAL_LENGTH,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            focal_length=FOCAL_LENGTH_10X,
        )

        # Focal length in pixels should scale with zoom
        assert intrinsics_10x.focal_length_px == pytest.approx(
            intrinsics_1x.focal_length_px * 10.0, rel=1e-6
        )

    def test_cx_is_image_center_x(self):
        """Test cx is at image center."""
        intrinsics = CameraIntrinsics.create(
            sensor_width=SENSOR_WIDTH,
            base_focal_length=BASE_FOCAL_LENGTH,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            focal_length=FOCAL_LENGTH_1X,
        )

        assert intrinsics.cx == pytest.approx(IMAGE_WIDTH / 2.0)

    def test_cy_is_image_center_y(self):
        """Test cy is at image center."""
        intrinsics = CameraIntrinsics.create(
            sensor_width=SENSOR_WIDTH,
            base_focal_length=BASE_FOCAL_LENGTH,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            focal_length=FOCAL_LENGTH_1X,
        )

        assert intrinsics.cy == pytest.approx(IMAGE_HEIGHT / 2.0)


class TestCameraIntrinsicsMatrix:
    """Test intrinsic matrix K access."""

    def test_to_K_returns_matrix3x3(self):
        """Test that to_K returns a Matrix3x3 instance."""
        intrinsics = CameraIntrinsics.create(
            sensor_width=SENSOR_WIDTH,
            base_focal_length=BASE_FOCAL_LENGTH,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            focal_length=FOCAL_LENGTH_1X,
        )

        K = intrinsics.to_K()

        assert isinstance(K, Matrix3x3)

    def test_to_K_matrix_structure(self):
        """Test that K has correct structure."""
        intrinsics = CameraIntrinsics.create(
            sensor_width=SENSOR_WIDTH,
            base_focal_length=BASE_FOCAL_LENGTH,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            focal_length=FOCAL_LENGTH_1X,
        )

        K = intrinsics.to_K()._to_array()

        # K should be:
        # [fx  0  cx]
        # [0  fy  cy]
        # [0   0   1]
        f = intrinsics.focal_length_px
        cx = intrinsics.cx
        cy = intrinsics.cy

        expected_K = np.array(
            [
                [f, 0.0, cx],
                [0.0, f, cy],
                [0.0, 0.0, 1.0],
            ]
        )

        np.testing.assert_array_almost_equal(K, expected_K)

    def test_to_K_returns_same_instance(self):
        """Test that to_K returns the same Matrix3x3 instance."""
        intrinsics = CameraIntrinsics.create(
            sensor_width=SENSOR_WIDTH,
            base_focal_length=BASE_FOCAL_LENGTH,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            focal_length=FOCAL_LENGTH_1X,
        )

        K1 = intrinsics.to_K()
        K2 = intrinsics.to_K()

        assert K1 is K2  # Same instance, not recomputed

    def test_K_matrix_is_immutable(self):
        """Test that K matrix array is immutable."""
        intrinsics = CameraIntrinsics.create(
            sensor_width=SENSOR_WIDTH,
            base_focal_length=BASE_FOCAL_LENGTH,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            focal_length=FOCAL_LENGTH_1X,
        )

        K = intrinsics.to_K()._to_array()

        with pytest.raises(ValueError):
            K[0, 0] = 999.0


class TestCameraIntrinsicsFrozen:
    """Test that CameraIntrinsics is immutable."""

    def test_cannot_modify_attributes(self):
        """Test that attributes cannot be modified after creation."""
        intrinsics = CameraIntrinsics.create(
            sensor_width=SENSOR_WIDTH,
            base_focal_length=BASE_FOCAL_LENGTH,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            focal_length=FOCAL_LENGTH_1X,
        )

        with pytest.raises(AttributeError):
            intrinsics.focal_length = Millimeters(10.0)  # type: ignore[misc]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
