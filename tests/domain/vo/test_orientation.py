"""Test suite for Orientation value object.

Tests cover:
- Creation via create() factory
- Creation via from_rotation() factory
- Serialization (to_dict/from_dict)
- Addition operator (__add__)
- Rotation composition (compose)
- Access to underlying rotation
- Validation
- Private constructor enforcement
"""

import pytest

from poc_homography.domain.vo.orientation import Orientation
from poc_homography.domain.vo.rotation import Rotation
from poc_homography.types import Degrees


class TestOrientationCreation:
    """Test Orientation creation and validation."""

    def test_create_valid_orientation(self):
        """Test creating a valid Orientation."""
        o = Orientation.create(yaw=Degrees(45), pitch=Degrees(30), roll=Degrees(15))

        assert o.yaw == 45.0
        assert o.pitch == 30.0
        assert o.roll == 15.0

    def test_create_default_roll(self):
        """Test that roll defaults to 0."""
        o = Orientation.create(yaw=Degrees(45), pitch=Degrees(30))

        assert o.roll == 0.0

    def test_create_rejects_nan_yaw(self):
        """Test that NaN yaw is rejected."""
        with pytest.raises(ValueError, match="yaw must be finite"):
            Orientation.create(yaw=Degrees(float("nan")), pitch=Degrees(0))

    def test_create_rejects_nan_pitch(self):
        """Test that NaN pitch is rejected."""
        with pytest.raises(ValueError, match="pitch must be finite"):
            Orientation.create(yaw=Degrees(0), pitch=Degrees(float("nan")))

    def test_create_rejects_nan_roll(self):
        """Test that NaN roll is rejected."""
        with pytest.raises(ValueError, match="roll must be finite"):
            Orientation.create(yaw=Degrees(0), pitch=Degrees(0), roll=Degrees(float("nan")))

    def test_create_rejects_inf(self):
        """Test that infinite angles are rejected."""
        with pytest.raises(ValueError, match="yaw must be finite"):
            Orientation.create(yaw=Degrees(float("inf")), pitch=Degrees(0))

    def test_create_rejects_pitch_over_90(self):
        """Test that pitch > 90 is rejected."""
        with pytest.raises(ValueError, match="pitch must be in range"):
            Orientation.create(yaw=Degrees(0), pitch=Degrees(91))

    def test_create_rejects_pitch_under_minus_90(self):
        """Test that pitch < -90 is rejected."""
        with pytest.raises(ValueError, match="pitch must be in range"):
            Orientation.create(yaw=Degrees(0), pitch=Degrees(-91))

    def test_create_accepts_pitch_at_boundaries(self):
        """Test that pitch at +-90 is accepted."""
        o1 = Orientation.create(yaw=Degrees(0), pitch=Degrees(90))
        o2 = Orientation.create(yaw=Degrees(0), pitch=Degrees(-90))

        assert o1.pitch == 90.0
        assert o2.pitch == -90.0

    def test_direct_constructor_raises_error(self):
        """Test that direct constructor access is blocked."""
        r = Rotation.from_euler(Degrees(0), Degrees(0), Degrees(0))
        with pytest.raises(TypeError, match="cannot be instantiated directly"):
            Orientation(yaw=Degrees(0), pitch=Degrees(0), roll=Degrees(0), _rotation=r)


class TestOrientationFromRotation:
    """Test from_rotation() factory."""

    def test_from_rotation_identity(self):
        """Test from_rotation with identity rotation."""
        r = Rotation.from_euler(Degrees(0), Degrees(0), Degrees(0))
        o = Orientation.from_rotation(r)

        assert o.yaw == pytest.approx(0.0, abs=1e-10)
        assert o.pitch == pytest.approx(0.0, abs=1e-10)
        assert o.roll == pytest.approx(0.0, abs=1e-10)

    def test_from_rotation_yaw_only(self):
        """Test from_rotation extracts yaw correctly."""
        r = Rotation.from_euler(Degrees(45), Degrees(0), Degrees(0))
        o = Orientation.from_rotation(r)

        assert o.yaw == pytest.approx(45.0, abs=1e-10)
        assert o.pitch == pytest.approx(0.0, abs=1e-10)
        assert o.roll == pytest.approx(0.0, abs=1e-10)

    def test_from_rotation_pitch_only(self):
        """Test from_rotation extracts pitch correctly."""
        r = Rotation.from_euler(Degrees(0), Degrees(30), Degrees(0))
        o = Orientation.from_rotation(r)

        assert o.yaw == pytest.approx(0.0, abs=1e-10)
        assert o.pitch == pytest.approx(30.0, abs=1e-10)
        assert o.roll == pytest.approx(0.0, abs=1e-10)

    def test_from_rotation_roundtrip(self):
        """Test create -> to_rotation -> from_rotation roundtrip."""
        original = Orientation.create(yaw=Degrees(45), pitch=Degrees(30), roll=Degrees(15))
        rotation = original.to_rotation()
        restored = Orientation.from_rotation(rotation)

        assert restored.yaw == pytest.approx(original.yaw, abs=1e-10)
        assert restored.pitch == pytest.approx(original.pitch, abs=1e-10)
        assert restored.roll == pytest.approx(original.roll, abs=1e-10)


class TestOrientationAddition:
    """Test __add__ operator."""

    def test_add_zero(self):
        """Test adding zero orientation."""
        o1 = Orientation.create(yaw=Degrees(45), pitch=Degrees(30), roll=Degrees(15))
        zero = Orientation.create(yaw=Degrees(0), pitch=Degrees(0), roll=Degrees(0))

        result = o1 + zero

        assert result.yaw == 45.0
        assert result.pitch == 30.0
        assert result.roll == 15.0

    def test_add_angles(self):
        """Test angle addition."""
        o1 = Orientation.create(yaw=Degrees(10), pitch=Degrees(20), roll=Degrees(5))
        o2 = Orientation.create(yaw=Degrees(30), pitch=Degrees(10), roll=Degrees(10))

        result = o1 + o2

        assert result.yaw == 40.0
        assert result.pitch == 30.0
        assert result.roll == 15.0

    def test_add_returns_orientation(self):
        """Test that add returns an Orientation."""
        o1 = Orientation.create(yaw=Degrees(10), pitch=Degrees(20))
        o2 = Orientation.create(yaw=Degrees(30), pitch=Degrees(10))

        result = o1 + o2

        assert isinstance(result, Orientation)

    def test_add_validates_result(self):
        """Test that addition validates the result."""
        o1 = Orientation.create(yaw=Degrees(0), pitch=Degrees(80))
        o2 = Orientation.create(yaw=Degrees(0), pitch=Degrees(20))

        with pytest.raises(ValueError, match="pitch must be in range"):
            _ = o1 + o2


class TestOrientationComposition:
    """Test compose() method."""

    def test_compose_identity(self):
        """Test composing with identity."""
        o1 = Orientation.create(yaw=Degrees(45), pitch=Degrees(30), roll=Degrees(15))
        identity = Orientation.create(yaw=Degrees(0), pitch=Degrees(0), roll=Degrees(0))

        result = o1.compose(identity)

        assert result.yaw == pytest.approx(o1.yaw, abs=1e-10)
        assert result.pitch == pytest.approx(o1.pitch, abs=1e-10)
        assert result.roll == pytest.approx(o1.roll, abs=1e-10)

    def test_compose_returns_orientation(self):
        """Test that compose returns an Orientation."""
        o1 = Orientation.create(yaw=Degrees(45), pitch=Degrees(30))
        o2 = Orientation.create(yaw=Degrees(10), pitch=Degrees(5))

        result = o1.compose(o2)

        assert isinstance(result, Orientation)

    def test_compose_differs_from_add_for_large_angles(self):
        """Test that compose differs from add for large rotations."""
        o1 = Orientation.create(yaw=Degrees(45), pitch=Degrees(30))
        o2 = Orientation.create(yaw=Degrees(30), pitch=Degrees(20))

        add_result = o1 + o2
        compose_result = o1.compose(o2)

        # For large angles, results should differ
        # (they're the same for very small angles)
        assert (
            add_result.yaw != pytest.approx(compose_result.yaw, abs=0.1)
            or add_result.pitch != pytest.approx(compose_result.pitch, abs=0.1)
            or add_result.roll != pytest.approx(compose_result.roll, abs=0.1)
        )


class TestOrientationRotationAccess:
    """Test to_rotation() method."""

    def test_to_rotation_returns_rotation(self):
        """Test that to_rotation returns a Rotation."""
        o = Orientation.create(yaw=Degrees(45), pitch=Degrees(30), roll=Degrees(15))

        result = o.to_rotation()

        assert isinstance(result, Rotation)

    def test_to_rotation_matrix_is_orthogonal(self):
        """Test rotation matrix has determinant 1."""
        o = Orientation.create(yaw=Degrees(45), pitch=Degrees(30), roll=Degrees(15))

        m = o.to_rotation().to_matrix()

        assert m.determinant == pytest.approx(1.0)


class TestOrientationSerialization:
    """Test serialization methods."""

    def test_to_dict_format(self):
        """Test that to_dict returns correct keys."""
        o = Orientation.create(yaw=Degrees(45), pitch=Degrees(30), roll=Degrees(15))

        d = o.to_dict()

        assert "yaw" in d
        assert "pitch" in d
        assert "roll" in d

    def test_to_dict_values(self):
        """Test that to_dict returns correct values."""
        o = Orientation.create(yaw=Degrees(45), pitch=Degrees(30), roll=Degrees(15))

        d = o.to_dict()

        assert d["yaw"] == 45.0
        assert d["pitch"] == 30.0
        assert d["roll"] == 15.0

    def test_from_dict_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = Orientation.create(yaw=Degrees(45), pitch=Degrees(30), roll=Degrees(15))
        d = original.to_dict()
        restored = Orientation.from_dict(d)

        assert restored.yaw == original.yaw
        assert restored.pitch == original.pitch
        assert restored.roll == original.roll

    def test_from_dict_default_roll(self):
        """Test that from_dict handles missing roll."""
        d = {"yaw": 45.0, "pitch": 30.0}
        o = Orientation.from_dict(d)

        assert o.roll == 0.0


class TestOrientationEquality:
    """Test equality and hashing."""

    def test_equality(self):
        """Test orientation equality."""
        o1 = Orientation.create(yaw=Degrees(45), pitch=Degrees(30), roll=Degrees(15))
        o2 = Orientation.create(yaw=Degrees(45), pitch=Degrees(30), roll=Degrees(15))

        assert o1 == o2

    def test_inequality(self):
        """Test orientation inequality."""
        o1 = Orientation.create(yaw=Degrees(45), pitch=Degrees(30))
        o2 = Orientation.create(yaw=Degrees(90), pitch=Degrees(30))

        assert o1 != o2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
