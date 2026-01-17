"""Test suite for Matrix3x3 value object.

Tests cover:
- Creation and validation
- Array access and immutability
- Mathematical operations (determinant, condition number, inverse)
- Matrix multiplication
- Serialization
- Hashability and private constructor enforcement
"""

import numpy as np
import pytest

from poc_homography.domain.vo.matrix3x3 import Matrix3x3


def make_identity() -> np.ndarray:
    """Create 3x3 identity matrix."""
    return np.eye(3)


def make_simple_matrix() -> np.ndarray:
    """Create a simple well-conditioned 3x3 matrix."""
    return np.array(
        [
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
            [1.0, 0.0, 2.0],
        ]
    )


def make_rotation_matrix(angle_rad: float) -> np.ndarray:
    """Create a 2D rotation matrix embedded in 3x3."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


class TestMatrix3x3Creation:
    """Test Matrix3x3 creation and validation."""

    def test_create_valid_matrix(self):
        """Test creating a valid Matrix3x3."""
        arr = make_simple_matrix()
        m = Matrix3x3.create(arr)

        np.testing.assert_array_almost_equal(m._to_array(), arr)

    def test_create_from_list(self):
        """Test creating from nested list."""
        data = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        m = Matrix3x3.create(np.array(data))

        np.testing.assert_array_almost_equal(m._to_array(), np.eye(3))

    def test_create_rejects_wrong_shape_2x3(self):
        """Test that 2x3 matrices are rejected."""
        with pytest.raises(ValueError, match="3x3"):
            Matrix3x3.create(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_create_rejects_wrong_shape_4x4(self):
        """Test that 4x4 matrices are rejected."""
        with pytest.raises(ValueError, match="3x3"):
            Matrix3x3.create(np.eye(4))

    def test_create_rejects_wrong_shape_1d(self):
        """Test that 1D arrays are rejected."""
        with pytest.raises(ValueError, match="3x3"):
            Matrix3x3.create(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    def test_create_rejects_nan(self):
        """Test that matrices with NaN are rejected."""
        arr = make_simple_matrix()
        arr[1, 1] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            Matrix3x3.create(arr)

    def test_create_rejects_positive_infinity(self):
        """Test that matrices with +inf are rejected."""
        arr = make_simple_matrix()
        arr[0, 0] = np.inf

        with pytest.raises(ValueError, match="Infinity"):
            Matrix3x3.create(arr)

    def test_create_rejects_negative_infinity(self):
        """Test that matrices with -inf are rejected."""
        arr = make_simple_matrix()
        arr[2, 2] = -np.inf

        with pytest.raises(ValueError, match="Infinity"):
            Matrix3x3.create(arr)

    def test_direct_constructor_raises_error(self):
        """Test that direct constructor access is blocked."""
        with pytest.raises(TypeError, match="cannot be instantiated directly"):
            Matrix3x3(_data=b"test")


class TestMatrix3x3ArrayAccess:
    """Test array access and immutability."""

    def test_to_array_returns_correct_values(self):
        """Test that to_array returns the original values."""
        arr = make_simple_matrix()
        m = Matrix3x3.create(arr)

        result = m._to_array()

        np.testing.assert_array_equal(result, arr)

    def test_to_array_is_immutable(self):
        """Test that returned array cannot be modified."""
        m = Matrix3x3.create(make_simple_matrix())
        arr = m._to_array()

        with pytest.raises(ValueError):
            arr[0, 0] = 999.0

    def test_to_array_returns_fresh_view(self):
        """Test that multiple calls return consistent data."""
        m = Matrix3x3.create(make_simple_matrix())

        arr1 = m._to_array()
        arr2 = m._to_array()

        np.testing.assert_array_equal(arr1, arr2)


class TestMatrix3x3MathOperations:
    """Test mathematical operations."""

    def test_determinant_identity(self):
        """Test determinant of identity is 1."""
        m = Matrix3x3.create(make_identity())

        assert m.determinant == pytest.approx(1.0)

    def test_determinant_scaled_identity(self):
        """Test determinant of scaled identity."""
        arr = 2.0 * np.eye(3)
        m = Matrix3x3.create(arr)

        # det(2I) = 2^3 = 8
        assert m.determinant == pytest.approx(8.0)

    def test_determinant_simple_matrix(self):
        """Test determinant of simple matrix."""
        m = Matrix3x3.create(make_simple_matrix())
        expected = np.linalg.det(make_simple_matrix())

        assert m.determinant == pytest.approx(expected)

    def test_condition_number_identity(self):
        """Test condition number of identity is 1."""
        m = Matrix3x3.create(make_identity())

        assert m.condition_number == pytest.approx(1.0)

    def test_condition_number_rotation(self):
        """Test condition number of rotation matrix is 1."""
        m = Matrix3x3.create(make_rotation_matrix(np.pi / 4))

        # Rotation matrices are orthogonal, condition number = 1
        assert m.condition_number == pytest.approx(1.0, rel=1e-10)

    def test_condition_number_positive(self):
        """Test condition number is always positive."""
        m = Matrix3x3.create(make_simple_matrix())

        assert m.condition_number > 0


class TestMatrix3x3Inverse:
    """Test inverse computation."""

    def test_inverse_identity(self):
        """Test inverse of identity is identity."""
        m = Matrix3x3.create(make_identity())
        m_inv = m.inverse()

        np.testing.assert_array_almost_equal(m_inv._to_array(), np.eye(3))

    def test_inverse_simple_matrix(self):
        """Test inverse of simple matrix."""
        arr = make_simple_matrix()
        m = Matrix3x3.create(arr)
        m_inv = m.inverse()

        expected_inv = np.linalg.inv(arr)
        np.testing.assert_array_almost_equal(m_inv._to_array(), expected_inv)

    def test_inverse_returns_matrix3x3(self):
        """Test that inverse returns a Matrix3x3 instance."""
        m = Matrix3x3.create(make_simple_matrix())
        m_inv = m.inverse()

        assert isinstance(m_inv, Matrix3x3)

    def test_inverse_product_is_identity(self):
        """Test that M @ M^-1 = I."""
        m = Matrix3x3.create(make_simple_matrix())
        m_inv = m.inverse()

        product = m._to_array() @ m_inv._to_array()

        np.testing.assert_array_almost_equal(product, np.eye(3))

    def test_double_inverse_returns_original(self):
        """Test that (M^-1)^-1 = M."""
        arr = make_simple_matrix()
        m = Matrix3x3.create(arr)
        m_double_inv = m.inverse().inverse()

        np.testing.assert_array_almost_equal(m_double_inv._to_array(), arr)

    def test_inverse_singular_matrix_raises(self):
        """Test that inverse of singular matrix raises error."""
        singular = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],  # Linear dependent row
                [0.0, 0.0, 1.0],
            ]
        )
        m = Matrix3x3.create(singular)

        with pytest.raises(ValueError, match="singular"):
            m.inverse()


class TestMatrix3x3RotationFactories:
    """Test rotation matrix factory methods."""

    def test_rotation_x_zero(self):
        """Test rotation_x(0) returns identity."""
        m = Matrix3x3.rotation_x(0.0)

        np.testing.assert_array_almost_equal(m._to_array(), np.eye(3))

    def test_rotation_x_90_degrees(self):
        """Test rotation_x(90 deg) rotates Y to Z."""
        m = Matrix3x3.rotation_x(np.pi / 2)
        v = np.array([0.0, 1.0, 0.0])

        result = m @ v

        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 1.0])

    def test_rotation_y_zero(self):
        """Test rotation_y(0) returns identity."""
        m = Matrix3x3.rotation_y(0.0)

        np.testing.assert_array_almost_equal(m._to_array(), np.eye(3))

    def test_rotation_y_90_degrees(self):
        """Test rotation_y(90 deg) rotates Z to X."""
        m = Matrix3x3.rotation_y(np.pi / 2)
        v = np.array([0.0, 0.0, 1.0])

        result = m @ v

        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0])

    def test_rotation_z_zero(self):
        """Test rotation_z(0) returns identity."""
        m = Matrix3x3.rotation_z(0.0)

        np.testing.assert_array_almost_equal(m._to_array(), np.eye(3))

    def test_rotation_z_90_degrees(self):
        """Test rotation_z(90 deg) rotates X to Y."""
        m = Matrix3x3.rotation_z(np.pi / 2)
        v = np.array([1.0, 0.0, 0.0])

        result = m @ v

        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 0.0])

    def test_rotation_matrices_are_orthogonal(self):
        """Test that rotation matrices have determinant 1."""
        for angle in [0.0, np.pi / 6, np.pi / 4, np.pi / 2, np.pi]:
            mx = Matrix3x3.rotation_x(angle)
            my = Matrix3x3.rotation_y(angle)
            mz = Matrix3x3.rotation_z(angle)

            assert mx.determinant == pytest.approx(1.0)
            assert my.determinant == pytest.approx(1.0)
            assert mz.determinant == pytest.approx(1.0)

    def test_rotation_matrices_condition_number(self):
        """Test that rotation matrices have condition number 1."""
        m = Matrix3x3.rotation_x(np.pi / 3)

        assert m.condition_number == pytest.approx(1.0)


class TestMatrix3x3Multiplication:
    """Test matrix multiplication."""

    def test_matmul_with_vector(self):
        """Test multiplication with 3D vector."""
        m = Matrix3x3.create(make_simple_matrix())
        v = np.array([1.0, 2.0, 3.0])

        result = m @ v
        expected = make_simple_matrix() @ v

        np.testing.assert_array_almost_equal(result, expected)

    def test_matmul_with_identity(self):
        """Test multiplication with identity preserves vector."""
        m = Matrix3x3.create(make_identity())
        v = np.array([1.0, 2.0, 3.0])

        result = m @ v

        np.testing.assert_array_almost_equal(result, v)

    def test_matmul_with_ndarray(self):
        """Test multiplication with numpy array."""
        m = Matrix3x3.create(make_simple_matrix())
        other = make_rotation_matrix(np.pi / 6)

        result = m @ other
        expected = make_simple_matrix() @ other

        np.testing.assert_array_almost_equal(result, expected)

    def test_matmul_with_matrix3x3(self):
        """Test multiplication of Matrix3x3 @ Matrix3x3 returns Matrix3x3."""
        m1 = Matrix3x3.create(make_simple_matrix())
        m2 = Matrix3x3.create(make_rotation_matrix(np.pi / 6))

        result = m1 @ m2

        assert isinstance(result, Matrix3x3)
        expected = make_simple_matrix() @ make_rotation_matrix(np.pi / 6)
        np.testing.assert_array_almost_equal(result._to_array(), expected)

    def test_matmul_rotation_composition(self):
        """Test composing rotations via @."""
        rx = Matrix3x3.rotation_x(np.pi / 4)
        ry = Matrix3x3.rotation_y(np.pi / 6)

        result = rx @ ry

        assert isinstance(result, Matrix3x3)
        expected = rx._to_array() @ ry._to_array()
        np.testing.assert_array_almost_equal(result._to_array(), expected)


class TestMatrix3x3Serialization:
    """Test serialization methods."""

    def test_to_list_format(self):
        """Test that to_list returns nested list."""
        m = Matrix3x3.create(make_simple_matrix())
        result = m.to_list()

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(len(row) == 3 for row in result)

    def test_to_list_values(self):
        """Test that to_list returns correct values."""
        arr = make_simple_matrix()
        m = Matrix3x3.create(arr)
        result = m.to_list()

        for i in range(3):
            for j in range(3):
                assert result[i][j] == pytest.approx(arr[i, j])

    def test_from_list_roundtrip(self):
        """Test to_list -> from_list roundtrip."""
        original = Matrix3x3.create(make_simple_matrix())
        data = original.to_list()
        restored = Matrix3x3.from_list(data)

        np.testing.assert_array_almost_equal(restored._to_array(), original._to_array())


class TestMatrix3x3Hashability:
    """Test hashability and equality."""

    def test_hash_consistency(self):
        """Test that same matrix produces same hash."""
        arr = make_simple_matrix()
        m1 = Matrix3x3.create(arr)
        m2 = Matrix3x3.create(arr.copy())

        assert hash(m1) == hash(m2)

    def test_can_be_dict_key(self):
        """Test that Matrix3x3 can be used as dict key."""
        m = Matrix3x3.create(make_simple_matrix())
        d = {m: "test_value"}

        assert d[m] == "test_value"

    def test_can_be_in_set(self):
        """Test that Matrix3x3 can be added to set."""
        m1 = Matrix3x3.create(make_simple_matrix())
        m2 = Matrix3x3.create(make_simple_matrix())

        s = {m1, m2}
        assert len(s) == 1  # Same matrix, should deduplicate

    def test_different_matrices_different_hash(self):
        """Test that different matrices have different hashes."""
        m1 = Matrix3x3.create(make_simple_matrix())
        m2 = Matrix3x3.create(make_identity())

        assert hash(m1) != hash(m2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
