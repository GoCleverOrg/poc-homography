"""Test suite for PTZState value object.

Tests cover:
- Basic creation
- to_orientation() conversion with different tilt conventions
- Serialization (to_dict/from_dict)
"""

import pytest

from poc_homography.domain.enums import TiltConvention
from poc_homography.domain.vo.orientation import Orientation
from poc_homography.domain.vo.ptz_state import PTZState
from poc_homography.types import Degrees, Unitless


class TestPTZStateCreation:
    """Test PTZState creation."""

    def test_create_ptz_state(self):
        """Test creating a PTZState."""
        ptz = PTZState(pan_raw=Degrees(45), tilt_deg=Degrees(30), zoom=Unitless(2.0))

        assert ptz.pan_raw == 45.0
        assert ptz.tilt_deg == 30.0
        assert ptz.zoom == 2.0


class TestPTZStateToOrientation:
    """Test to_orientation() conversion."""

    def test_to_orientation_positive_up_convention(self):
        """Test conversion with POSITIVE_UP convention."""
        ptz = PTZState(pan_raw=Degrees(45), tilt_deg=Degrees(30), zoom=Unitless(1.0))

        result = ptz.to_orientation(TiltConvention.POSITIVE_UP)

        assert isinstance(result, Orientation)
        assert result.yaw == 45.0
        assert result.pitch == 30.0  # sign = 1
        assert result.roll == 0.0

    def test_to_orientation_positive_down_convention(self):
        """Test conversion with POSITIVE_DOWN convention (Hikvision)."""
        ptz = PTZState(pan_raw=Degrees(45), tilt_deg=Degrees(30), zoom=Unitless(1.0))

        result = ptz.to_orientation(TiltConvention.POSITIVE_DOWN)

        assert isinstance(result, Orientation)
        assert result.yaw == 45.0
        assert result.pitch == -30.0  # sign = -1
        assert result.roll == 0.0

    def test_to_orientation_negative_tilt_positive_up(self):
        """Test negative tilt with POSITIVE_UP convention."""
        ptz = PTZState(pan_raw=Degrees(0), tilt_deg=Degrees(-20), zoom=Unitless(1.0))

        result = ptz.to_orientation(TiltConvention.POSITIVE_UP)

        assert result.pitch == -20.0

    def test_to_orientation_negative_tilt_positive_down(self):
        """Test negative tilt with POSITIVE_DOWN convention."""
        ptz = PTZState(pan_raw=Degrees(0), tilt_deg=Degrees(-20), zoom=Unitless(1.0))

        result = ptz.to_orientation(TiltConvention.POSITIVE_DOWN)

        assert result.pitch == 20.0  # -20 * -1 = 20

    def test_to_orientation_zero_tilt(self):
        """Test zero tilt gives zero pitch regardless of convention."""
        ptz = PTZState(pan_raw=Degrees(90), tilt_deg=Degrees(0), zoom=Unitless(1.0))

        result_up = ptz.to_orientation(TiltConvention.POSITIVE_UP)
        result_down = ptz.to_orientation(TiltConvention.POSITIVE_DOWN)

        assert result_up.pitch == 0.0
        assert result_down.pitch == 0.0

    def test_to_orientation_zoom_ignored(self):
        """Test that zoom is not part of orientation."""
        ptz1 = PTZState(pan_raw=Degrees(45), tilt_deg=Degrees(30), zoom=Unitless(1.0))
        ptz2 = PTZState(pan_raw=Degrees(45), tilt_deg=Degrees(30), zoom=Unitless(10.0))

        result1 = ptz1.to_orientation(TiltConvention.POSITIVE_UP)
        result2 = ptz2.to_orientation(TiltConvention.POSITIVE_UP)

        assert result1.yaw == result2.yaw
        assert result1.pitch == result2.pitch
        assert result1.roll == result2.roll


class TestPTZStateSerialization:
    """Test serialization methods."""

    def test_to_dict_format(self):
        """Test that to_dict returns correct keys."""
        ptz = PTZState(pan_raw=Degrees(45), tilt_deg=Degrees(30), zoom=Unitless(2.0))

        d = ptz.to_dict()

        assert "pan_raw" in d
        assert "tilt_deg" in d
        assert "zoom" in d

    def test_to_dict_values(self):
        """Test that to_dict returns correct values."""
        ptz = PTZState(pan_raw=Degrees(45), tilt_deg=Degrees(30), zoom=Unitless(2.0))

        d = ptz.to_dict()

        assert d["pan_raw"] == 45.0
        assert d["tilt_deg"] == 30.0
        assert d["zoom"] == 2.0

    def test_from_dict_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = PTZState(pan_raw=Degrees(45), tilt_deg=Degrees(30), zoom=Unitless(2.0))
        d = original.to_dict()
        restored = PTZState.from_dict(d)

        assert restored.pan_raw == original.pan_raw
        assert restored.tilt_deg == original.tilt_deg
        assert restored.zoom == original.zoom


class TestPTZStateEquality:
    """Test equality."""

    def test_equality(self):
        """Test PTZState equality."""
        ptz1 = PTZState(pan_raw=Degrees(45), tilt_deg=Degrees(30), zoom=Unitless(2.0))
        ptz2 = PTZState(pan_raw=Degrees(45), tilt_deg=Degrees(30), zoom=Unitless(2.0))

        assert ptz1 == ptz2

    def test_inequality(self):
        """Test PTZState inequality."""
        ptz1 = PTZState(pan_raw=Degrees(45), tilt_deg=Degrees(30), zoom=Unitless(2.0))
        ptz2 = PTZState(pan_raw=Degrees(90), tilt_deg=Degrees(30), zoom=Unitless(2.0))

        assert ptz1 != ptz2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
