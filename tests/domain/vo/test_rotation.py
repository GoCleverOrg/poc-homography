"""Test suite for Rotation value object.

Tests cover:
- Creation via from_euler() factory
- Matrix access
- Rotation composition
- Hashability
"""

import numpy as np
import pytest

from poc_homography.domain.vo.matrix3x3 import Matrix3x3
from poc_homography.domain.vo.rotation import Rotation
from poc_homography.types import Degrees


class TestRotationCreation:
    """Test Rotation creation and validation."""

    def test_from_euler_zero_angles(self):
        """Test from_euler(0, 0, 0) creates identity rotation."""
        r = Rotation.from_euler(Degrees(0), Degrees(0), Degrees(0))

        np.testing.assert_array_almost_equal(r.to_matrix()._to_array(), np.eye(3))

    def test_from_euler_rejects_nan_yaw(self):
        """Test that NaN yaw is rejected."""
        with pytest.raises(ValueError, match="yaw must be finite"):
            Rotation.from_euler(Degrees(float("nan")), Degrees(0), Degrees(0))

    def test_from_euler_rejects_nan_pitch(self):
        """Test that NaN pitch is rejected."""
        with pytest.raises(ValueError, match="pitch must be finite"):
            Rotation.from_euler(Degrees(0), Degrees(float("nan")), Degrees(0))

    def test_from_euler_rejects_nan_roll(self):
        """Test that NaN roll is rejected."""
        with pytest.raises(ValueError, match="roll must be finite"):
            Rotation.from_euler(Degrees(0), Degrees(0), Degrees(float("nan")))

    def test_from_euler_rejects_inf(self):
        """Test that infinite angles are rejected."""
        with pytest.raises(ValueError, match="yaw must be finite"):
            Rotation.from_euler(Degrees(float("inf")), Degrees(0), Degrees(0))

class TestRotationMatrixAccess:
    """Test matrix access methods."""

    def test_to_matrix_returns_matrix3x3(self):
        """Test that to_matrix returns a Matrix3x3."""
        r = Rotation.from_euler(Degrees(45), Degrees(30), Degrees(15))

        result = r.to_matrix()

        assert isinstance(result, Matrix3x3)

    def test_to_matrix_determinant_is_one(self):
        """Test rotation matrix has determinant 1."""
        r = Rotation.from_euler(Degrees(45), Degrees(30), Degrees(15))

        assert r.to_matrix().determinant == pytest.approx(1.0)

    def test_to_matrix_condition_number_is_one(self):
        """Test rotation matrix has condition number 1."""
        r = Rotation.from_euler(Degrees(45), Degrees(30), Degrees(15))

        assert r.to_matrix().condition_number == pytest.approx(1.0)


class TestRotationComposition:
    """Test rotation composition."""

    def test_compose_identity(self):
        """Test composing with identity returns same rotation."""
        r = Rotation.from_euler(Degrees(45), Degrees(30), Degrees(15))
        identity = Rotation.from_euler(Degrees(0), Degrees(0), Degrees(0))

        result = r.compose(identity)

        np.testing.assert_array_almost_equal(
            result.to_matrix()._to_array(),
            r.to_matrix()._to_array(),
        )

    def test_compose_two_yaw_rotations(self):
        """Test composing two yaw rotations is additive."""
        r1 = Rotation.from_euler(Degrees(30), Degrees(0), Degrees(0))
        r2 = Rotation.from_euler(Degrees(45), Degrees(0), Degrees(0))

        result = r1.compose(r2)
        expected = Rotation.from_euler(Degrees(75), Degrees(0), Degrees(0))

        np.testing.assert_array_almost_equal(
            result.to_matrix()._to_array(),
            expected.to_matrix()._to_array(),
        )

    def test_compose_returns_rotation(self):
        """Test that compose returns a Rotation."""
        r1 = Rotation.from_euler(Degrees(45), Degrees(0), Degrees(0))
        r2 = Rotation.from_euler(Degrees(0), Degrees(30), Degrees(0))

        result = r1.compose(r2)

        assert isinstance(result, Rotation)

    def test_compose_order_matters(self):
        """Test that rotation composition is non-commutative."""
        r1 = Rotation.from_euler(Degrees(90), Degrees(0), Degrees(0))
        r2 = Rotation.from_euler(Degrees(0), Degrees(90), Degrees(0))

        result1 = r1.compose(r2)
        result2 = r2.compose(r1)

        # Rotations don't commute in general
        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(
                result1.to_matrix()._to_array(),
                result2.to_matrix()._to_array(),
            )


class TestRotationHashability:
    """Test hashability and equality."""

    def test_hash_consistency(self):
        """Test that same rotation produces same hash."""
        r1 = Rotation.from_euler(Degrees(45), Degrees(30), Degrees(15))
        r2 = Rotation.from_euler(Degrees(45), Degrees(30), Degrees(15))

        assert hash(r1) == hash(r2)

    def test_can_be_dict_key(self):
        """Test that Rotation can be used as dict key."""
        r = Rotation.from_euler(Degrees(45), Degrees(30), Degrees(15))
        d = {r: "test_value"}

        assert d[r] == "test_value"

    def test_can_be_in_set(self):
        """Test that Rotation can be added to set."""
        r1 = Rotation.from_euler(Degrees(45), Degrees(30), Degrees(15))
        r2 = Rotation.from_euler(Degrees(45), Degrees(30), Degrees(15))

        s = {r1, r2}
        assert len(s) == 1

    def test_equality(self):
        """Test rotation equality."""
        r1 = Rotation.from_euler(Degrees(45), Degrees(30), Degrees(15))
        r2 = Rotation.from_euler(Degrees(45), Degrees(30), Degrees(15))

        assert r1 == r2

    def test_inequality(self):
        """Test rotation inequality."""
        r1 = Rotation.from_euler(Degrees(45), Degrees(30), Degrees(15))
        r2 = Rotation.from_euler(Degrees(90), Degrees(30), Degrees(15))

        assert r1 != r2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
