"""Tests for the checked-in calibration-sweep plan config (#343).

Covers the Definition of Done:
  * the ``calibration-sweep.yaml`` round-trips through ``SurveyPlanConfig`` and
    selects the ground-spanning phase subset;
  * the config drives ``fov_grid`` (via ``SurveyPlan.from_plan_config`` →
    ``_partition_main_grid``) over the configured visible-ground pan/tilt extent
    at the chosen zoom levels;
  * the generated grid spans that extent and excludes above-envelope sky tiles;
  * no new ``SurveyPhase`` member is introduced.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

import poc_homography.survey as survey_pkg
from poc_homography.domain.enums.camera_spec import CameraSpec
from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig
from poc_homography.domain.vo.tilt_envelope import TiltEnvelope
from poc_homography.survey.phases.runner import SurveyPlan
from poc_homography.survey.planner.generators import fov_grid

CONFIG_PATH = Path(survey_pkg.__file__).parent / "configs" / "calibration-sweep.yaml"

# The ground-spanning calibration subset: camera inventory, zoom
# characterization (feeds intrinsic self-calibration #339), and the main survey.
_EXPECTED_PHASES = frozenset({1, 3, 5})

_SPEC = CameraSpec.HIKVISION_DS_2DF8425IX


def _load_config() -> SurveyPlanConfig:
    """Load the checked-in calibration-sweep config from its YAML sidecar."""
    return SurveyPlanConfig.from_dict(yaml.safe_load(CONFIG_PATH.read_text()))


class TestCalibrationSweepYaml:
    def test_yaml_file_exists(self) -> None:
        assert CONFIG_PATH.is_file()

    def test_loads_and_round_trips(self) -> None:
        cfg = _load_config()
        assert SurveyPlanConfig.from_dict(cfg.to_dict()) == cfg

    def test_selects_ground_spanning_phase_subset(self) -> None:
        cfg = _load_config()
        assert cfg.enabled_phases == _EXPECTED_PHASES

    def test_main_survey_spans_full_azimuth_ground(self) -> None:
        cfg = _load_config()
        # Phase 5 (main survey) configures a non-trivial pan/tilt ground extent.
        pan_lo, pan_hi = cfg.phase_pan_range[5]
        tilt_lo, tilt_hi = cfg.phase_tilt_range[5]
        assert pan_lo < pan_hi
        assert tilt_lo < tilt_hi
        assert cfg.zoom_levels  # at least one chosen zoom level


class TestFromPlanConfigBridgesGroundSpan:
    def test_main_pan_tilt_zoom_bridged_from_config(self) -> None:
        cfg = _load_config()
        plan = SurveyPlan.from_plan_config(cfg)
        assert plan.main_pan_range == cfg.phase_pan_range[5]
        assert plan.main_tilt_range == cfg.phase_tilt_range[5]
        assert plan.main_zoom_levels == tuple(cfg.zoom_levels)

    def test_tilt_envelope_forwarded(self) -> None:
        env = _constant_envelope(0.0)
        plan = SurveyPlan.from_plan_config(replace(_load_config(), tilt_envelope=env))
        assert plan.tilt_envelope == env

    def test_default_config_keeps_runner_defaults(self) -> None:
        # A config that does not configure phase 5 leaves the main-survey knobs
        # at their defaults (no behaviour change for existing callers).
        defaults = SurveyPlan()
        plan = SurveyPlan.from_plan_config(SurveyPlanConfig())
        assert plan.main_pan_range == defaults.main_pan_range
        assert plan.main_tilt_range == defaults.main_tilt_range
        assert plan.main_zoom_levels == defaults.main_zoom_levels
        assert plan.tilt_envelope is None


class TestConfigDrivesFovGrid:
    def test_grid_spans_configured_ground_extent(self) -> None:
        cfg = _single_zoom(_load_config())
        plan = SurveyPlan.from_plan_config(cfg)
        grid = _grid_from_plan(plan)

        pans = [float(p.pan) for p in grid]
        tilts = [float(p.tilt) for p in grid]
        pan_lo, pan_hi = plan.main_pan_range
        tilt_lo, tilt_hi = plan.main_tilt_range
        # Both extent endpoints are covered (fov_grid clamps to the bounds).
        assert min(pans) == pan_lo
        assert max(pans) == pan_hi
        assert min(tilts) == tilt_lo
        assert max(tilts) == tilt_hi

    def test_envelope_excludes_sky_tiles(self) -> None:
        cfg = _single_zoom(_load_config())
        tilt_lo, tilt_hi = cfg.phase_tilt_range[5]
        # A constant bound strictly inside the tilt extent: rows above it (tilt
        # numerically below it, positive = down) are sky and must be skipped.
        bound = (tilt_lo + tilt_hi) / 2.0
        env = _constant_envelope(bound)

        plan_no_env = SurveyPlan.from_plan_config(cfg)
        plan_env = SurveyPlan.from_plan_config(replace(cfg, tilt_envelope=env))
        grid_no_env = _grid_from_plan(plan_no_env)
        grid_env = _grid_from_plan(plan_env)

        # The envelope removes a non-empty, strict subset of tiles...
        assert 0 < len(grid_env) < len(grid_no_env)
        # ...every surviving tile is on or below the per-azimuth bound (ground)...
        assert all(float(p.tilt) >= env.upper_bound(float(p.pan)) - 1e-9 for p in grid_env)
        # ...and every removed tile was a sky tile above the bound.
        removed = {_key(p) for p in grid_no_env} - {_key(p) for p in grid_env}
        assert removed
        assert all(tilt < env.upper_bound(pan) for pan, tilt, _ in removed)


class TestNoNewPhaseMember:
    def test_phase_catalog_still_nine(self) -> None:
        # #343 forbids adding a new SurveyPhase member; the catalog stays at 9.
        assert len(SurveyPhase) == 9


def _single_zoom(cfg: SurveyPlanConfig) -> SurveyPlanConfig:
    """Restrict the config to the widest single zoom for a small, fast grid."""
    return replace(cfg, zoom_levels=[min(cfg.zoom_levels)])


def _grid_from_plan(plan: SurveyPlan) -> list:
    """Generate the main-survey FOV grid exactly as the runner does."""
    return fov_grid(
        _SPEC,
        plan.main_pan_range,
        plan.main_tilt_range,
        plan.main_zoom_levels,
        plan.main_overlap_fraction,
        tilt_envelope=plan.tilt_envelope,
    )


def _key(pose) -> tuple[float, float, float]:
    """A hashable (pan, tilt, zoom) identity for a pose."""
    return (float(pose.pan), float(pose.tilt), float(pose.zoom))


def _constant_envelope(bound: float) -> TiltEnvelope:
    """A tilt envelope with a constant max-up useful tilt at every azimuth."""
    return TiltEnvelope(
        bounds={0.0: bound, 90.0: bound, 180.0: bound, 270.0: bound},
        tilt_offset_deg=-30.0,
        vfov_deg=40.0,
        zoom=1.0,
    )
