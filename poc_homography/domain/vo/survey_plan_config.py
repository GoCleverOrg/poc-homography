"""Survey plan configuration value object (reproducible, camera-free sidecar)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from poc_homography.domain.vo.tilt_envelope import TiltEnvelope

PLAN_CONFIG_SCHEMA_VERSION = "1"
"""Schema version for :class:`SurveyPlanConfig`.

Deliberately the literal ``"1"`` and distinct from the survey-entities
``SURVEY_SCHEMA_VERSION`` (``"1.0"``); the plan-config sidecar evolves
independently of the run/frame dataset schema.
"""


@dataclass(frozen=True)
class SurveyPlanConfig:
    """Reproducible configuration for a nine-phase survey plan.

    Captures every knob the survey planner needs so a run can be replayed
    deterministically from a ``plan_config.yaml`` sidecar without touching a
    live camera. Phase numbers (1..9) map to the C4 nine-phase survey catalog:

        1. camera inventory
        2. PTZ characterization
        3. zoom characterization
        4. dense nadir
        5. main survey
        6. cross zoom
        7. repeatability
        8. static jitter
        9. validation

    Phases absent from the ``phase_*_range`` dicts fall back to the live camera
    capabilities (#256 ``CameraCapabilities``) at plan time.

    Attributes:
        schema_version: Sidecar schema version (always ``"1"``).
        enabled_phases: Subset of ``{1..9}`` to execute; defaults to all nine.
        phase_pan_range: Per-phase ``(lo, hi)`` pan bounds in degrees.
        phase_tilt_range: Per-phase ``(lo, hi)`` tilt bounds in degrees.
        phase_zoom_range: Per-phase ``(lo, hi)`` zoom-factor bounds.
        grid_overlap_pct: Per-phase tile overlap percentage (phases 4 and 5).
        burst_frame_count: Per-phase snapshot-burst frame count (phases 2, 3, 7).
        jitter_burst_duration_s: Phase 8 jitter burst duration in seconds.
        jitter_burst_fps: Phase 8 jitter burst frame rate; ``0.0`` is a sentinel
            meaning "use the native FPS target".
        jitter_zoom_levels: Phase 8 zoom factors (wide/mid/high/max).
        jitter_pose_count: Phase 8 representative poses per camera.
        zoom_levels: Zoom factors swept by phases 3, 6 and 8.
        repeat_count: Per-phase repetition count (phases 2 and 7).
        holdout_fraction: Phase 9 validation holdout fraction.
        tilt_envelope: Optional per-azimuth max-up-useful-tilt envelope from the
            Phase-0 horizon calibration. ``None`` (the default) reproduces the
            pre-horizon behaviour exactly; when present the C3 pose generators
            skip sky tiles above the per-azimuth bound.
    """

    schema_version: str = PLAN_CONFIG_SCHEMA_VERSION
    enabled_phases: frozenset[int] = field(default=frozenset(range(1, 10)))
    phase_pan_range: dict[int, tuple[float, float]] = field(default_factory=dict)
    phase_tilt_range: dict[int, tuple[float, float]] = field(default_factory=dict)
    phase_zoom_range: dict[int, tuple[float, float]] = field(default_factory=dict)
    grid_overlap_pct: dict[int, float] = field(default_factory=lambda: {4: 80.0, 5: 80.0})
    burst_frame_count: dict[int, int] = field(default_factory=dict)
    jitter_burst_duration_s: float = 30.0
    jitter_burst_fps: float = 0.0
    jitter_zoom_levels: list[float] = field(default_factory=lambda: [1.0, 5.0, 12.0, 25.0])
    jitter_pose_count: int = 1
    zoom_levels: list[float] = field(default_factory=lambda: [1.0, 5.0, 12.0, 25.0])
    repeat_count: dict[int, int] = field(default_factory=lambda: {2: 3, 7: 3})
    holdout_fraction: float = 0.15
    tilt_envelope: TiltEnvelope | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dictionary.

        ``frozenset`` becomes a sorted list, every ``int``-keyed dict has its
        keys stringified, and range tuples become 2-element lists so the result
        survives ``json.dumps`` / ``yaml.safe_dump`` round-trips.
        """
        return {
            "schema_version": self.schema_version,
            "enabled_phases": sorted(self.enabled_phases),
            "phase_pan_range": _ranges_to_dict(self.phase_pan_range),
            "phase_tilt_range": _ranges_to_dict(self.phase_tilt_range),
            "phase_zoom_range": _ranges_to_dict(self.phase_zoom_range),
            "grid_overlap_pct": {str(k): v for k, v in self.grid_overlap_pct.items()},
            "burst_frame_count": {str(k): v for k, v in self.burst_frame_count.items()},
            "jitter_burst_duration_s": self.jitter_burst_duration_s,
            "jitter_burst_fps": self.jitter_burst_fps,
            "jitter_zoom_levels": list(self.jitter_zoom_levels),
            "jitter_pose_count": self.jitter_pose_count,
            "zoom_levels": list(self.zoom_levels),
            "repeat_count": {str(k): v for k, v in self.repeat_count.items()},
            "holdout_fraction": self.holdout_fraction,
            "tilt_envelope": (
                self.tilt_envelope.to_dict() if self.tilt_envelope is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurveyPlanConfig:
        """Create a :class:`SurveyPlanConfig` from a (partial) dictionary.

        Tolerates missing keys by falling back to field defaults, but a full
        ``to_dict()`` payload round-trips to an equal object.

        Raises:
            ValueError: If ``schema_version`` is not the supported version.
        """
        version = data.get("schema_version", PLAN_CONFIG_SCHEMA_VERSION)
        if version != PLAN_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version!r}; expected {PLAN_CONFIG_SCHEMA_VERSION!r}"
            )
        defaults = cls()
        enabled = data.get("enabled_phases")
        return cls(
            schema_version=PLAN_CONFIG_SCHEMA_VERSION,
            enabled_phases=(
                frozenset(int(p) for p in enabled)
                if enabled is not None
                else defaults.enabled_phases
            ),
            phase_pan_range=_ranges_from_dict(
                data.get("phase_pan_range"), defaults.phase_pan_range
            ),
            phase_tilt_range=_ranges_from_dict(
                data.get("phase_tilt_range"), defaults.phase_tilt_range
            ),
            phase_zoom_range=_ranges_from_dict(
                data.get("phase_zoom_range"), defaults.phase_zoom_range
            ),
            grid_overlap_pct=_int_keyed(
                data.get("grid_overlap_pct"), float, defaults.grid_overlap_pct
            ),
            burst_frame_count=_int_keyed(
                data.get("burst_frame_count"), int, defaults.burst_frame_count
            ),
            jitter_burst_duration_s=float(
                data.get("jitter_burst_duration_s", defaults.jitter_burst_duration_s)
            ),
            jitter_burst_fps=float(data.get("jitter_burst_fps", defaults.jitter_burst_fps)),
            jitter_zoom_levels=_float_list(
                data.get("jitter_zoom_levels"), defaults.jitter_zoom_levels
            ),
            jitter_pose_count=int(data.get("jitter_pose_count", defaults.jitter_pose_count)),
            zoom_levels=_float_list(data.get("zoom_levels"), defaults.zoom_levels),
            repeat_count=_int_keyed(data.get("repeat_count"), int, defaults.repeat_count),
            holdout_fraction=float(data.get("holdout_fraction", defaults.holdout_fraction)),
            tilt_envelope=_tilt_envelope_from_dict(data.get("tilt_envelope")),
        )


def _ranges_to_dict(
    ranges: dict[int, tuple[float, float]],
) -> dict[str, list[float]]:
    """Serialise an int-keyed range dict to string keys and 2-element lists."""
    return {str(k): [v[0], v[1]] for k, v in ranges.items()}


def _ranges_from_dict(
    raw: dict[Any, Any] | None,
    default: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """Rebuild an int-keyed range dict from a serialised payload."""
    if raw is None:
        return dict(default)
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


def _int_keyed(
    raw: dict[Any, Any] | None,
    value_cast: Any,
    default: dict[int, Any],
) -> dict[int, Any]:
    """Rebuild an int-keyed scalar dict, coercing values with ``value_cast``."""
    if raw is None:
        return dict(default)
    return {int(k): value_cast(v) for k, v in raw.items()}


def _tilt_envelope_from_dict(raw: dict[Any, Any] | None) -> TiltEnvelope | None:
    """Rebuild the optional tilt envelope; ``None`` preserves legacy behaviour."""
    if raw is None:
        return None
    return TiltEnvelope.from_dict(raw)


def _float_list(raw: list[Any] | None, default: list[float]) -> list[float]:
    """Rebuild a float list from a serialised payload."""
    if raw is None:
        return list(default)
    return [float(v) for v in raw]
