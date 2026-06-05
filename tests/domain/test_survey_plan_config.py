"""Tests for SurveyPlanConfig frozen VO: defaults, round-trip, schema, yaml sidecar."""

from __future__ import annotations

import json

import pytest
import yaml

from poc_homography.domain.vo.survey_plan_config import (
    PLAN_CONFIG_SCHEMA_VERSION,
    SurveyPlanConfig,
)


def _populated() -> SurveyPlanConfig:
    """A non-default, fully-populated config exercising every field shape."""
    return SurveyPlanConfig(
        enabled_phases=frozenset({1, 2, 4, 8, 9}),
        phase_pan_range={2: (-170.0, 170.0), 5: (-90.0, 90.0)},
        phase_tilt_range={5: (-50.0, 60.0)},
        phase_zoom_range={3: (1.0, 25.0), 6: (5.0, 12.0)},
        grid_overlap_pct={4: 75.0, 5: 85.0},
        burst_frame_count={2: 5, 3: 4, 7: 6},
        jitter_burst_duration_s=45.0,
        jitter_burst_fps=10.0,
        jitter_zoom_levels=[2.0, 8.0, 20.0],
        jitter_pose_count=3,
        zoom_levels=[1.0, 10.0, 25.0],
        repeat_count={2: 4, 7: 5},
        holdout_fraction=0.2,
    )


class TestSurveyPlanConfigDefaults:
    def test_constructs_with_no_args(self) -> None:
        cfg = SurveyPlanConfig()
        assert cfg.schema_version == "1"

    def test_enabled_phases_all_nine(self) -> None:
        assert SurveyPlanConfig().enabled_phases == frozenset(range(1, 10))

    def test_grid_overlap_defaults(self) -> None:
        assert SurveyPlanConfig().grid_overlap_pct == {4: 80.0, 5: 80.0}

    def test_holdout_fraction_default(self) -> None:
        assert SurveyPlanConfig().holdout_fraction == 0.15

    def test_jitter_defaults(self) -> None:
        cfg = SurveyPlanConfig()
        assert cfg.jitter_burst_duration_s == 30.0
        assert cfg.jitter_burst_fps == 0.0
        assert cfg.jitter_zoom_levels == [1.0, 5.0, 12.0, 25.0]
        assert cfg.jitter_pose_count == 1

    def test_zoom_and_repeat_defaults(self) -> None:
        cfg = SurveyPlanConfig()
        assert cfg.zoom_levels == [1.0, 5.0, 12.0, 25.0]
        assert cfg.repeat_count == {2: 3, 7: 3}

    def test_range_dicts_default_empty(self) -> None:
        cfg = SurveyPlanConfig()
        assert cfg.phase_pan_range == {}
        assert cfg.phase_tilt_range == {}
        assert cfg.phase_zoom_range == {}
        assert cfg.burst_frame_count == {}

    def test_independent_default_instances(self) -> None:
        a = SurveyPlanConfig()
        b = SurveyPlanConfig()
        assert a.grid_overlap_pct is not b.grid_overlap_pct


class TestSurveyPlanConfigFrozen:
    def test_mutate_holdout_raises(self) -> None:
        cfg = SurveyPlanConfig()
        with pytest.raises(AttributeError):
            cfg.holdout_fraction = 0.5  # type: ignore[misc]

    def test_mutate_enabled_phases_raises(self) -> None:
        cfg = SurveyPlanConfig()
        with pytest.raises(AttributeError):
            cfg.enabled_phases = frozenset({1})  # type: ignore[misc]


class TestSurveyPlanConfigRoundTrip:
    def test_from_dict_to_dict_equals_original(self) -> None:
        cfg = _populated()
        assert SurveyPlanConfig.from_dict(cfg.to_dict()) == cfg

    def test_to_dict_idempotent(self) -> None:
        cfg = _populated()
        restored = SurveyPlanConfig.from_dict(cfg.to_dict())
        assert restored.to_dict() == cfg.to_dict()

    def test_default_round_trip(self) -> None:
        cfg = SurveyPlanConfig()
        assert SurveyPlanConfig.from_dict(cfg.to_dict()) == cfg

    def test_to_dict_is_json_serialisable(self) -> None:
        cfg = _populated()
        encoded = json.dumps(cfg.to_dict())
        decoded = json.loads(encoded)
        _assert_all_dict_keys_are_strings(decoded)

    def test_partial_dict_loads_with_defaults(self) -> None:
        cfg = SurveyPlanConfig.from_dict({"schema_version": "1", "holdout_fraction": 0.3})
        assert cfg.holdout_fraction == 0.3
        assert cfg.enabled_phases == frozenset(range(1, 10))
        assert cfg.grid_overlap_pct == {4: 80.0, 5: 80.0}


class TestSurveyPlanConfigSchemaVersion:
    def test_to_dict_schema_version_is_one(self) -> None:
        assert SurveyPlanConfig().to_dict()["schema_version"] == "1"
        assert PLAN_CONFIG_SCHEMA_VERSION == "1"

    def test_unknown_version_raises(self) -> None:
        data = SurveyPlanConfig().to_dict()
        data["schema_version"] = "0.9"
        with pytest.raises(ValueError, match="schema_version"):
            SurveyPlanConfig.from_dict(data)


class TestSurveyPlanConfigYamlReload:
    def test_yaml_sidecar_round_trip(self, tmp_path) -> None:
        cfg = _populated()
        sidecar = tmp_path / "plan_config.yaml"
        sidecar.write_text(yaml.safe_dump(cfg.to_dict()))
        loaded = yaml.safe_load(sidecar.read_text())
        assert SurveyPlanConfig.from_dict(loaded) == cfg


def _assert_all_dict_keys_are_strings(obj: object) -> None:
    """Recursively assert no dict in the decoded payload has non-str keys."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert isinstance(key, str)
            _assert_all_dict_keys_are_strings(value)
    elif isinstance(obj, list):
        for item in obj:
            _assert_all_dict_keys_are_strings(item)
