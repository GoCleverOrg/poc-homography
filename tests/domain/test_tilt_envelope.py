"""Tests for the TiltEnvelope VO: defaults, round-trip, schema, interpolation."""

from __future__ import annotations

import json

import pytest
import yaml

from poc_homography.domain.vo.tilt_envelope import (
    TILT_ENVELOPE_SCHEMA_VERSION,
    TiltEnvelope,
)


def _populated() -> TiltEnvelope:
    """A non-default envelope with several calibrated azimuths."""
    return TiltEnvelope(
        bounds={0.0: -13.0, 90.0: -10.0, 180.0: -16.0, 270.0: -12.0},
        tilt_offset_deg=-31.0,
        vfov_deg=36.0,
        zoom=1.0,
    )


class TestTiltEnvelopeDefaults:
    def test_constructs_with_no_args(self) -> None:
        env = TiltEnvelope()
        assert env.schema_version == "1"
        assert env.bounds == {}
        assert env.interpolation == "linear"

    def test_independent_default_bounds(self) -> None:
        assert TiltEnvelope().bounds is not TiltEnvelope().bounds


class TestTiltEnvelopeFrozen:
    def test_mutate_raises(self) -> None:
        env = TiltEnvelope()
        with pytest.raises(AttributeError):
            env.zoom = 5.0  # type: ignore[misc]


class TestTiltEnvelopeRoundTrip:
    def test_from_dict_to_dict_equals_original(self) -> None:
        env = _populated()
        assert TiltEnvelope.from_dict(env.to_dict()) == env

    def test_to_dict_idempotent(self) -> None:
        env = _populated()
        restored = TiltEnvelope.from_dict(env.to_dict())
        assert restored.to_dict() == env.to_dict()

    def test_to_dict_is_json_serialisable_with_string_keys(self) -> None:
        decoded = json.loads(json.dumps(_populated().to_dict()))
        assert all(isinstance(k, str) for k in decoded["bounds"])

    def test_yaml_sidecar_round_trip(self, tmp_path) -> None:
        env = _populated()
        sidecar = tmp_path / "envelope.yaml"
        sidecar.write_text(yaml.safe_dump(env.to_dict()))
        loaded = yaml.safe_load(sidecar.read_text())
        assert TiltEnvelope.from_dict(loaded) == env


class TestTiltEnvelopeSchemaVersion:
    def test_to_dict_schema_version_is_one(self) -> None:
        assert _populated().to_dict()["schema_version"] == "1"
        assert TILT_ENVELOPE_SCHEMA_VERSION == "1"

    def test_unknown_version_raises(self) -> None:
        data = _populated().to_dict()
        data["schema_version"] = "0.9"
        with pytest.raises(ValueError, match="schema_version"):
            TiltEnvelope.from_dict(data)


class TestTiltEnvelopeInterpolation:
    def test_exact_azimuth_returns_exact_bound(self) -> None:
        env = _populated()
        assert env.upper_bound(90.0) == -10.0
        assert env.upper_bound(180.0) == -16.0

    def test_pan_reduced_modulo_360(self) -> None:
        env = _populated()
        assert env.upper_bound(360.0) == env.upper_bound(0.0)
        assert env.upper_bound(450.0) == env.upper_bound(90.0)

    def test_linear_midpoint_between_two_azimuths(self) -> None:
        env = _populated()
        # Halfway between 90° (-10) and 180° (-16) → -13.
        assert env.upper_bound(135.0) == pytest.approx(-13.0)

    def test_wrap_around_seam_interpolates(self) -> None:
        env = _populated()
        # Halfway between 270° (-12) and 0°/360° (-13) → -12.5.
        assert env.upper_bound(315.0) == pytest.approx(-12.5)

    def test_single_azimuth_is_constant(self) -> None:
        env = TiltEnvelope(bounds={42.0: -9.0})
        assert env.upper_bound(0.0) == -9.0
        assert env.upper_bound(200.0) == -9.0

    def test_empty_envelope_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            TiltEnvelope().upper_bound(0.0)
