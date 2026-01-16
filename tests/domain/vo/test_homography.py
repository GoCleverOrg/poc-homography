"""Test suite for Homography value object.

Tests cover:
- Creation and validation (via create() factory)
- Private constructor enforcement
- Matrix access (to_matrix() returns Matrix3x3)
- Inverse property (returns Homography)
- Projection method
- Serialization
- Hashability and immutability
"""

import numpy as np
import pytest

from poc_homography.domain.vo.homography import Homography
from poc_homography.domain.vo.matrix3x3 import Matrix3x3
from poc_homography.domain.vo.pixel_point import PixelPoint


def make_simple_homography() -> np.ndarray:
    """Create a simple, well-conditioned homography matrix."""
    return np.array(
        [
            [100.0, 0.0, 960.0],
            [0.0, -100.0, 540.0],
            [0.0, 0.0, 1.0],
        ]
    )


def make_perspective_homography() -> np.ndarray:
    """Create a perspective homography matrix with non-trivial third row."""
    return np.array(
        [
            [85.3, 12.7, 960.0],
            [-8.4, 92.1, 540.0],
            [-0.001, -0.002, 1.0],
        ]
    )


class TestHomographyCreation:
    """Test Homography creation via factory method."""

    def test_create_valid_homography(self):
        """Test creating a valid Homography."""
        H = make_simple_homography()
        homography = Homography.create(H)

        np.testing.assert_array_almost_equal(homography.to_matrix().to_array(), H)

    def test_create_computes_condition_number(self):
        """Test that condition number is computed correctly."""
        H = make_simple_homography()
        expected_cond = np.linalg.cond(H)

        homography = Homography.create(H)

        assert homography.condition_number == pytest.approx(expected_cond, rel=1e-10)

    def test_create_computes_determinant(self):
        """Test that determinant is computed correctly."""
        H = make_simple_homography()
        expected_det = np.linalg.det(H)

        homography = Homography.create(H)

        assert homography.determinant == pytest.approx(expected_det, rel=1e-10)

    def test_create_rejects_wrong_shape(self):
        """Test that non-3x3 matrices are rejected."""
        with pytest.raises(ValueError, match="3x3"):
            Homography.create(np.eye(4))

    def test_create_rejects_nan_values(self):
        """Test that matrices with NaN are rejected."""
        H = make_simple_homography()
        H[1, 1] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            Homography.create(H)

    def test_create_rejects_inf_values(self):
        """Test that matrices with infinity are rejected."""
        H = make_simple_homography()
        H[0, 0] = np.inf

        with pytest.raises(ValueError, match="Infinity"):
            Homography.create(H)

    def test_create_rejects_singular_matrix(self):
        """Test that singular matrices are rejected."""
        H_singular = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [0.0, 0.0, 1.0],
            ]
        )

        with pytest.raises(ValueError, match="singular"):
            Homography.create(H_singular)

    def test_create_rejects_ill_conditioned_matrix(self):
        """Test that ill-conditioned matrices are rejected."""
        H = np.array(
            [
                [1e12, 0.0, 0.0],
                [0.0, 1e-3, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        with pytest.raises(ValueError, match="ill-conditioned"):
            Homography.create(H)

    def test_direct_constructor_raises_error(self):
        """Test that direct constructor access is blocked."""
        m = Matrix3x3.create(make_simple_homography())

        with pytest.raises(TypeError, match="cannot be instantiated directly"):
            Homography(_matrix=m, _inverse_matrix=m)


class TestHomographyMatrixAccess:
    """Test matrix access via to_matrix()."""

    def test_to_matrix_returns_matrix3x3(self):
        """Test that to_matrix returns a Matrix3x3 instance."""
        homography = Homography.create(make_simple_homography())

        result = homography.to_matrix()

        assert isinstance(result, Matrix3x3)

    def test_to_matrix_values_correct(self):
        """Test that to_matrix returns correct values."""
        H = make_simple_homography()
        homography = Homography.create(H)

        result = homography.to_matrix()

        np.testing.assert_array_almost_equal(result.to_array(), H)

    def test_condition_number_accessible(self):
        """Test that condition number is accessible."""
        homography = Homography.create(make_simple_homography())

        assert homography.condition_number > 0

    def test_determinant_accessible(self):
        """Test that determinant is accessible."""
        homography = Homography.create(make_simple_homography())

        assert homography.determinant != 0


class TestHomographyInverse:
    """Test Homography inverse property."""

    def test_inverse_returns_homography(self):
        """Test that inverse returns a Homography instance."""
        homography = Homography.create(make_simple_homography())

        inv = homography.inverse

        assert isinstance(inv, Homography)

    def test_inverse_matrix_is_correct(self):
        """Test that inverse contains the correct matrix."""
        H = make_simple_homography()
        homography = Homography.create(H)

        inv = homography.inverse
        expected_inv = np.linalg.inv(H)

        np.testing.assert_array_almost_equal(inv.to_matrix().to_array(), expected_inv)

    def test_inverse_determinant_is_reciprocal(self):
        """Test that inverse determinant is 1/original."""
        homography = Homography.create(make_simple_homography())

        inv = homography.inverse

        assert inv.determinant == pytest.approx(1.0 / homography.determinant, rel=1e-10)

    def test_inverse_condition_number_preserved(self):
        """Test that inverse has same condition number."""
        homography = Homography.create(make_simple_homography())

        inv = homography.inverse

        assert inv.condition_number == pytest.approx(homography.condition_number, rel=1e-10)

    def test_double_inverse_returns_original_matrix(self):
        """Test that inverse of inverse equals original matrix."""
        H = make_perspective_homography()
        homography = Homography.create(H)

        double_inv = homography.inverse.inverse

        np.testing.assert_array_almost_equal(
            double_inv.to_matrix().to_array(), homography.to_matrix().to_array()
        )


class TestHomographyProjection:
    """Test Homography projection method."""

    def test_project_origin(self):
        """Test projecting origin through simple homography."""
        homography = Homography.create(make_simple_homography())

        # (0, 0) -> (960, 540)
        result = homography.project(PixelPoint(_x=0.0, _y=0.0))

        assert float(result.x) == pytest.approx(960.0, abs=0.001)
        assert float(result.y) == pytest.approx(540.0, abs=0.001)

    def test_project_with_scale(self):
        """Test that scaling works correctly."""
        homography = Homography.create(make_simple_homography())

        # (1, 1) -> (100*1 + 960, -100*1 + 540) = (1060, 440)
        result = homography.project(PixelPoint(_x=1.0, _y=1.0))

        assert float(result.x) == pytest.approx(1060.0, abs=0.001)
        assert float(result.y) == pytest.approx(440.0, abs=0.001)

    def test_project_roundtrip(self):
        """Test that project followed by inverse.project returns original."""
        homography = Homography.create(make_perspective_homography())

        test_points = [
            PixelPoint(_x=100.0, _y=200.0),
            PixelPoint(_x=500.0, _y=300.0),
            PixelPoint(_x=960.0, _y=540.0),
        ]

        for point in test_points:
            projected = homography.project(point)
            back = homography.inverse.project(projected)

            assert float(back.x) == pytest.approx(float(point.x), abs=0.01)
            assert float(back.y) == pytest.approx(float(point.y), abs=0.01)

    def test_project_inverse_roundtrip(self):
        """Test that inverse.project followed by project returns original."""
        homography = Homography.create(make_perspective_homography())

        test_points = [
            PixelPoint(_x=800.0, _y=400.0),
            PixelPoint(_x=960.0, _y=540.0),
            PixelPoint(_x=1200.0, _y=700.0),
        ]

        for point in test_points:
            projected = homography.inverse.project(point)
            back = homography.project(projected)

            assert float(back.x) == pytest.approx(float(point.x), abs=0.01)
            assert float(back.y) == pytest.approx(float(point.y), abs=0.01)

    def test_project_infinity_raises(self):
        """Test that projecting to infinity raises ValueError."""
        H = np.array(
            [
                [100.0, 0.0, 960.0],
                [0.0, 100.0, 540.0],
                [0.0, 0.01, 1.0],
            ]
        )
        homography = Homography.create(H)

        # w = 0.01*y + 1 = 0 when y = -100
        point_at_infinity = PixelPoint(_x=0.0, _y=-100.0)

        with pytest.raises(ValueError, match="infinity"):
            homography.project(point_at_infinity)


class TestHomographySerialization:
    """Test Homography serialization methods."""

    def test_to_dict_contains_required_fields(self):
        """Test that to_dict returns required fields."""
        homography = Homography.create(make_simple_homography())
        data = homography.to_dict()

        assert "matrix" in data
        assert "condition_number" in data
        assert "determinant" in data

    def test_to_dict_matrix_format(self):
        """Test that matrix is serialized as nested lists."""
        homography = Homography.create(make_simple_homography())
        data = homography.to_dict()

        assert isinstance(data["matrix"], list)
        assert len(data["matrix"]) == 3
        assert len(data["matrix"][0]) == 3

    def test_from_dict_roundtrip(self):
        """Test that to_dict -> from_dict preserves data."""
        original = Homography.create(make_perspective_homography())
        data = original.to_dict()
        restored = Homography.from_dict(data)

        np.testing.assert_array_almost_equal(
            restored.to_matrix().to_array(), original.to_matrix().to_array()
        )
        assert restored.condition_number == pytest.approx(original.condition_number, rel=1e-10)
        assert restored.determinant == pytest.approx(original.determinant, rel=1e-10)


class TestHomographyHashability:
    """Test Homography can be used as dict key and in sets."""

    def test_hash_consistency(self):
        """Test that same homography always produces same hash."""
        H = make_simple_homography()
        h1 = Homography.create(H)
        h2 = Homography.create(H.copy())

        assert hash(h1) == hash(h2)

    def test_can_be_dict_key(self):
        """Test that Homography can be used as dictionary key."""
        homography = Homography.create(make_simple_homography())
        d = {homography: "test_value"}

        assert d[homography] == "test_value"

    def test_can_be_in_set(self):
        """Test that Homography can be added to set."""
        h1 = Homography.create(make_simple_homography())
        h2 = Homography.create(make_simple_homography())

        s = {h1, h2}
        assert len(s) == 1


class TestHomographyFrozen:
    """Test that Homography is immutable."""

    def test_cannot_modify_attributes(self):
        """Test that attributes cannot be modified after creation."""
        homography = Homography.create(make_simple_homography())
        another_matrix = Matrix3x3.create(make_perspective_homography())

        with pytest.raises(AttributeError):
            homography._matrix = another_matrix  # type: ignore[misc]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
