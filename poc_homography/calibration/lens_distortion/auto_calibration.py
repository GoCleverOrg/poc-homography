"""Automatic lens-distortion calibration orchestrator.

Drives a PTZ camera (or an offline frame source) through a set of zoom levels,
detects painted floor lines, gates each view through the visibility precheck,
solves the distortion per zoom, and assembles a per-camera
:class:`LensCalibrationTable` that is then folded into the shareable per-model
:class:`ModelCalibrationTable`.

Design goals:

* **Pure-ish core.** The orchestrator depends only on a small injected
  :class:`CalibrationDevice` protocol, so it is fully testable against a fake
  that returns synthetic frames -- no hardware, no DB.
* **Never leave a production camera moved.** The current PTZ state is saved
  first and restored in a ``finally`` block, even if a zoom raises.
* **Maximize runtime accuracy (decision D).** After a coarse zoom sweep, an
  adaptive pass inserts extra zooms wherever neighbouring calibrated
  coefficients disagree with their midpoint interpolation by more than a
  tolerance -- denser sampling naturally lands at the wide end where distortion
  changes fastest.
* **Precision grows with runs.** ``runs > 1`` revisits the calibrated zooms and
  folds every run into the model table, so its per-bin std shrinks.

The thin :class:`DeviceFrameCapturer` adapts the real
:class:`~poc_homography.domain.protocols.camera_device.CameraDevice` (Hikvision
ISAPI) onto :class:`CalibrationDevice`.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np
import yaml

from poc_homography.calibration.lens_distortion.model_aggregation import (
    AggregationConfig,
    fold_camera_table,
)
from poc_homography.calibration.lens_distortion.models import PTZPosition
from poc_homography.calibration.lens_distortion.painted_line_detection import (
    PaintedLineDetector,
)
from poc_homography.calibration.lens_distortion.scene_self_calibration import (
    ScenePerZoomConfig,
    SkippedZoom,
    calibrate_zoom_from_lines,
)
from poc_homography.calibration.lens_distortion.visibility import (
    NoCalibratableViewError,
    VisibilityCriteria,
    assess_camera_lines,
    require_calibratable_view,
)
from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
from poc_homography.domain.enums.camera_spec import CameraSpec
from poc_homography.types import Degrees

if TYPE_CHECKING:
    from collections.abc import Sequence

    from poc_homography.calibration.lens_distortion.models import CameraLine
    from poc_homography.domain.entities.model_calibration_table import (
        ModelCalibrationTable,
    )
    from poc_homography.domain.protocols.camera_device import CameraDevice
    from poc_homography.domain.vo.zoom_calibration_entry import ZoomCalibrationEntry

logger = logging.getLogger(__name__)

# A saved PTZ state: (pan_deg, tilt_deg, zoom_factor).
PTZTriple = tuple[float, float, float]


class CalibrationDevice(Protocol):
    """Minimal device surface the orchestrator needs.

    A deliberately small protocol so the core is testable against a fake. The
    real adapter (:class:`DeviceFrameCapturer`) maps these onto the wider
    :class:`~poc_homography.domain.protocols.camera_device.CameraDevice`.
    """

    def get_ptz(self) -> PTZTriple:
        """Read the current ``(pan_deg, tilt_deg, zoom_factor)``."""
        ...

    def set_pan_tilt(self, pan: float, tilt: float) -> None:
        """Aim at ``(pan, tilt)`` (zoom unchanged) and wait for stabilization."""
        ...

    def set_zoom(self, zoom: float) -> None:
        """Move to ``zoom`` (pan/tilt unchanged) and wait for stabilization."""
        ...

    def capture(self) -> np.ndarray:
        """Capture one frame as a BGR ``np.ndarray``."""
        ...

    def restore_ptz(self, state: PTZTriple) -> None:
        """Restore a previously saved ``(pan, tilt, zoom)`` state."""
        ...


@dataclass(frozen=True)
class AutoCalibrationConfig:
    """Tunable knobs for the automatic calibration orchestrator.

    Attributes:
        zoom_min: Lowest zoom of the coarse sweep.
        zoom_max: Highest zoom of the coarse sweep (clamped to the model max).
        coarse_steps: Number of evenly spaced coarse zoom levels.
        zoom_tol: Adaptive tolerance on ``|k1 - midpoint interpolation|``; a new
            zoom is inserted between a calibrated pair whose midpoint k1 deviates
            by more than this.
        max_adaptive_levels: Cap on adaptive refinement passes.
        frames_per_zoom: Frames captured (and detected) at each zoom.
        runs: How many times to repeat the whole capture/solve, folding each run
            into the model table (precision grows with runs).
        survey_enabled: When True, before the zoom sweep the orchestrator scans a
            small grid of tilt-down (and optional pan) poses at wide zoom and
            aims at the pose whose view scores best for floor paint. This is the
            fix for "calibrating whatever the camera happened to point at".
        survey_tilt_degrees: Candidate tilt angles (positive = down) to probe.
        survey_pan_offsets: Pan offsets (degrees) from the current pan to probe;
            ``(0.0,)`` keeps the current pan and surveys tilt only.
        survey_zoom: Zoom factor used while surveying (wide = most context).
    """

    zoom_min: float = 1.0
    zoom_max: float = 25.0
    coarse_steps: int = 5
    zoom_tol: float = 0.01
    max_adaptive_levels: int = 3
    frames_per_zoom: int = 1
    runs: int = 1
    survey_enabled: bool = True
    survey_tilt_degrees: tuple[float, ...] = (20.0, 30.0, 40.0, 50.0)
    survey_pan_offsets: tuple[float, ...] = (0.0,)
    survey_zoom: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.zoom_min <= 0:
            raise ValueError(f"zoom_min must be positive, got {self.zoom_min}")
        if self.zoom_max < self.zoom_min:
            raise ValueError(f"zoom_max ({self.zoom_max}) must be >= zoom_min ({self.zoom_min})")
        if self.coarse_steps < 1:
            raise ValueError(f"coarse_steps must be >= 1, got {self.coarse_steps}")
        if self.runs < 1:
            raise ValueError(f"runs must be >= 1, got {self.runs}")
        if self.survey_enabled and not self.survey_tilt_degrees:
            raise ValueError("survey_enabled requires at least one survey_tilt_degrees value")


@dataclass(frozen=True)
class AutoCalibrationResult:
    """Outcome of an automatic calibration run.

    Attributes:
        camera_table: The assembled per-camera lens calibration table.
        model_table: The per-model table after folding in every run.
        skipped_zooms: Zoom levels that could not be calibrated, with reasons.
        restored: Whether the camera PTZ was restored to its saved state.
    """

    camera_table: LensCalibrationTable
    model_table: ModelCalibrationTable
    skipped_zooms: tuple[SkippedZoom, ...]
    restored: bool = False


class DeviceFrameCapturer:
    """Adapt a real :class:`CameraDevice` onto :class:`CalibrationDevice`.

    Maps:

    * ``get_ptz``  -> ``device.get_ptz_status()`` -> ``(pan_raw, tilt_deg, zoom)``
    * ``set_zoom`` -> ``device.move_absolute(zoom=z)`` + ``wait_for_stabilization``
    * ``capture``  -> ``device.capture_snapshot()`` (JPEG) decoded via OpenCV
    * ``restore_ptz`` -> ``device.move_absolute(pan, tilt, zoom)`` + stabilize
    """

    def __init__(self, device: CameraDevice, *, stabilize_timeout_s: float = 5.0) -> None:
        """Wrap ``device``.

        Args:
            device: The concrete PTZ camera device (e.g. Hikvision ISAPI client).
            stabilize_timeout_s: Seconds to wait for PTZ moves to settle.
        """
        self._device = device
        self._timeout = stabilize_timeout_s

    def get_ptz(self) -> PTZTriple:
        """Read ``(pan, tilt, zoom)`` from the device hardware."""
        state = self._device.get_ptz_status()
        return (float(state.pan_raw), float(state.tilt_deg), float(state.zoom))

    def set_pan_tilt(self, pan: float, tilt: float) -> None:
        """Aim at ``(pan, tilt)`` (zoom unchanged) and wait for stabilization."""
        self._device.move_absolute(pan=pan, tilt=tilt)
        self._device.wait_for_stabilization(timeout_s=self._timeout)

    def set_zoom(self, zoom: float) -> None:
        """Move to ``zoom`` (pan/tilt unchanged) and wait for stabilization."""
        self._device.move_absolute(zoom=zoom)
        self._device.wait_for_stabilization(timeout_s=self._timeout)

    def capture(self) -> np.ndarray:
        """Capture a JPEG snapshot and decode it to a BGR image array."""
        data = self._device.capture_snapshot()
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode camera snapshot")
        return image

    def restore_ptz(self, state: PTZTriple) -> None:
        """Restore the saved ``(pan, tilt, zoom)`` and wait for stabilization."""
        pan, tilt, zoom = state
        self._device.move_absolute(pan=pan, tilt=tilt, zoom=zoom)
        self._device.wait_for_stabilization(timeout_s=self._timeout)


@dataclass
class _ZoomSolveContext:
    """Shared inputs threaded through the per-zoom solve helpers."""

    device: CalibrationDevice
    camera_spec: CameraSpec
    criteria: VisibilityCriteria
    scene_config: ScenePerZoomConfig
    frames_per_zoom: int
    detector: PaintedLineDetector
    skipped: list[SkippedZoom] = field(default_factory=list)


def _detect_lines_at_zoom(
    ctx: _ZoomSolveContext, zoom: float, *, image_tag: str
) -> list[CameraLine]:
    """Capture frame(s) at ``zoom`` and detect painted lines as camera lines."""
    ptz = PTZPosition(pan_deg=Degrees(0.0), tilt_deg=Degrees(0.0), zoom_factor=zoom)
    lines: list[CameraLine] = []
    for frame_idx in range(ctx.frames_per_zoom):
        frame = ctx.device.capture()
        image_path = f"{image_tag}_zoom{zoom:g}_img{frame_idx}"
        for line_idx, painted in enumerate(ctx.detector.detect(frame)):
            lines.append(
                painted.to_camera_line(
                    line_id=f"{image_path}_line{line_idx}",
                    image_path=image_path,
                    ptz_position=ptz,
                )
            )
    return lines


def _solve_zoom(
    ctx: _ZoomSolveContext, zoom: float, *, image_tag: str
) -> ZoomCalibrationEntry | None:
    """Set zoom, detect, gate via visibility, and solve.

    Records a :class:`SkippedZoom` (and returns ``None``) when the view is not
    calibratable or the solver declines; raises nothing for a single bad zoom.
    """
    ctx.device.set_zoom(zoom)
    lines = _detect_lines_at_zoom(ctx, zoom, image_tag=image_tag)

    report = assess_camera_lines(
        lines,
        float(ctx.camera_spec.image_width),
        float(ctx.camera_spec.image_height),
        ctx.criteria,
    )
    try:
        require_calibratable_view(report, context=f"zoom={zoom:g}")
    except NoCalibratableViewError as exc:
        logger.warning("Zoom %g not calibratable: %s", zoom, exc)
        ctx.skipped.append(SkippedZoom(zoom_factor=zoom, num_lines=len(lines), reason=str(exc)))
        return None

    outcome = calibrate_zoom_from_lines(
        lines, zoom, camera_spec=ctx.camera_spec, config=ctx.scene_config
    )
    if isinstance(outcome, SkippedZoom):
        logger.warning("Zoom %g skipped: %s", zoom, outcome.reason)
        ctx.skipped.append(outcome)
        return None
    return outcome


def _adaptive_zooms(entries: list[ZoomCalibrationEntry], tol: float) -> list[float]:
    """Midpoints where neighbouring k1 disagree with linear interpolation.

    For each consecutive calibrated pair the midpoint k1 is *defined* as the mean
    of the two endpoints (linear interpolation), so any non-linearity shows up as
    the gap between that interpolation and a freshly solved midpoint. We cannot
    know the solved value yet, so we proxy disagreement by the endpoint spread:
    a pair whose ``|k1_b - k1_a| > 2*tol`` is a candidate for a midpoint insert
    (the interpolation can be off by up to half the spread). Returns the new zoom
    levels to solve.
    """
    new_zooms: list[float] = []
    ordered = sorted(entries, key=lambda e: float(e.zoom_factor))
    for a, b in itertools.pairwise(ordered):
        za, zb = float(a.zoom_factor), float(b.zoom_factor)
        k1a, k1b = float(a.distortion.k1), float(b.distortion.k1)
        if abs(k1b - k1a) > 2.0 * tol:
            midpoint = round((za + zb) / 2.0, 6)
            if midpoint not in (za, zb):
                new_zooms.append(midpoint)
    return new_zooms


def _coarse_zooms(config: AutoCalibrationConfig, camera_spec: CameraSpec) -> list[float]:
    """Evenly spaced coarse zoom levels within the model's optical range."""
    hi = min(config.zoom_max, camera_spec.max_zoom)
    lo = min(config.zoom_min, hi)
    if config.coarse_steps == 1:
        return [lo]
    return [
        round(lo + (hi - lo) * i / (config.coarse_steps - 1), 6) for i in range(config.coarse_steps)
    ]


def _calibrate_once(
    device: CalibrationDevice,
    *,
    config: AutoCalibrationConfig,
    camera_spec: CameraSpec,
    criteria: VisibilityCriteria,
    scene_config: ScenePerZoomConfig,
    detector: PaintedLineDetector,
    image_tag: str,
) -> tuple[list[ZoomCalibrationEntry], list[SkippedZoom]]:
    """Run one full coarse + adaptive sweep, returning entries and skips."""
    ctx = _ZoomSolveContext(
        device=device,
        camera_spec=camera_spec,
        criteria=criteria,
        scene_config=scene_config,
        frames_per_zoom=config.frames_per_zoom,
        detector=detector,
    )

    entries: list[ZoomCalibrationEntry] = []
    solved_zooms: set[float] = set()
    for zoom in _coarse_zooms(config, camera_spec):
        entry = _solve_zoom(ctx, zoom, image_tag=image_tag)
        if entry is not None:
            entries.append(entry)
            solved_zooms.add(round(float(entry.zoom_factor), 6))

    # Adaptive refinement: insert midpoints where coefficients disagree.
    for _level in range(config.max_adaptive_levels):
        candidates = [z for z in _adaptive_zooms(entries, config.zoom_tol) if z not in solved_zooms]
        if not candidates:
            break
        for zoom in candidates:
            solved_zooms.add(zoom)
            entry = _solve_zoom(ctx, zoom, image_tag=image_tag)
            if entry is not None:
                entries.append(entry)

    return entries, ctx.skipped


@dataclass(frozen=True)
class _SurveyPose:
    """A surveyed candidate pose and how well its view scored for floor paint."""

    pan: float
    tilt: float
    score: float
    passed: bool
    num_curved: int


def _survey_for_floor(
    device: CalibrationDevice,
    *,
    config: AutoCalibrationConfig,
    camera_spec: CameraSpec,
    criteria: VisibilityCriteria,
    detector: PaintedLineDetector,
) -> _SurveyPose | None:
    """Scan tilt-down (and optional pan) poses and aim at the best floor view.

    At wide zoom, probe each ``(pan_offset, tilt)`` candidate, detect painted
    lines, and score the view with the visibility report (curved-line count plus
    quadrant/orientation coverage). The camera is left aimed at the highest
    scoring pose so the subsequent zoom sweep calibrates on actual floor paint
    rather than on whatever the camera happened to point at. Returns the chosen
    pose (``None`` when the survey is disabled).
    """
    if not config.survey_enabled:
        return None

    base_pan, _, _ = device.get_ptz()
    device.set_zoom(config.survey_zoom)
    survey_ptz = PTZPosition(
        pan_deg=Degrees(0.0), tilt_deg=Degrees(0.0), zoom_factor=config.survey_zoom
    )

    best: _SurveyPose | None = None
    for pan_offset in config.survey_pan_offsets:
        pan = base_pan + pan_offset
        for tilt in config.survey_tilt_degrees:
            device.set_pan_tilt(pan, tilt)
            frame = device.capture()
            lines = [
                painted.to_camera_line(
                    line_id=f"survey_p{pan:g}_t{tilt:g}_line{idx}",
                    image_path=f"survey_p{pan:g}_t{tilt:g}",
                    ptz_position=survey_ptz,
                )
                for idx, painted in enumerate(detector.detect(frame))
            ]
            report = assess_camera_lines(
                lines,
                float(camera_spec.image_width),
                float(camera_spec.image_height),
                criteria,
            )
            pose = _SurveyPose(
                pan=pan,
                tilt=tilt,
                score=report.score(),
                passed=report.passed,
                num_curved=report.num_curved_lines,
            )
            logger.info(
                "survey pan=%.1f tilt=%.1f -> score=%.1f passed=%s curved=%d",
                pan,
                tilt,
                pose.score,
                pose.passed,
                pose.num_curved,
            )
            if best is None or pose.score > best.score:
                best = pose

    if best is not None:
        device.set_pan_tilt(best.pan, best.tilt)
        logger.info(
            "survey chose pan=%.1f tilt=%.1f (score=%.1f passed=%s)",
            best.pan,
            best.tilt,
            best.score,
            best.passed,
        )
    return best


def _save_model_table_yaml(model_table: ModelCalibrationTable, path: Path) -> None:
    """Persist a model table as YAML (entity has no ``.id`` for the YAML repo)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.dump(model_table.to_dict(), handle, default_flow_style=False, sort_keys=False)


def run_auto_calibration(
    device: CalibrationDevice,
    *,
    camera_id: str,
    camera_spec: CameraSpec = CameraSpec.HIKVISION_DS_2DF8425IX,
    config: AutoCalibrationConfig | None = None,
    criteria: VisibilityCriteria | None = None,
    scene_config: ScenePerZoomConfig | None = None,
    detector: PaintedLineDetector | None = None,
    aggregation_config: AggregationConfig | None = None,
    output_dir: Path | None = None,
    existing_model_table: ModelCalibrationTable | None = None,
    now: str | None = None,
) -> AutoCalibrationResult:
    """Automatically calibrate lens distortion across zoom, restoring PTZ.

    Saves the current PTZ first, runs ``config.runs`` coarse+adaptive sweeps
    (folding each into the model table so precision grows), and always restores
    PTZ in a ``finally``. The last run's per-camera table is returned and, when
    ``output_dir`` is given, both tables are persisted as YAML.

    Args:
        device: The injected camera/frame device (real adapter or fake).
        camera_id: Camera identifier; also the per-camera table id.
        camera_spec: Camera model spec (sensor, dims, optical range).
        config: Orchestration knobs (zoom range, adaptivity, runs).
        criteria: Visibility precheck thresholds.
        scene_config: Per-zoom solver knobs.
        detector: Painted-line detector (a default one when ``None``).
        aggregation_config: Model-aggregation knobs.
        output_dir: When given, persist per-camera and per-model YAML here.
        existing_model_table: Prior model table to fold into (precision grows).
        now: ISO timestamp override (testing).

    Returns:
        An :class:`AutoCalibrationResult` carrying both tables and skip reasons.

    Raises:
        NoCalibratableViewError: If *zero* zooms calibrate across the whole run.
    """
    cfg = config or AutoCalibrationConfig()
    crit = criteria or VisibilityCriteria()
    scene_cfg = scene_config or ScenePerZoomConfig()
    line_detector = detector or PaintedLineDetector()
    agg_cfg = aggregation_config or AggregationConfig()
    ts = now or datetime.now().isoformat()
    model_name = camera_spec.model_name

    saved_state = device.get_ptz()
    restored = False
    model_table = existing_model_table
    camera_table: LensCalibrationTable | None = None
    all_skipped: list[SkippedZoom] = []
    last_report: object = None

    try:
        # Aim at a floor view first (fix for "calibrate whatever it points at").
        _survey_for_floor(
            device,
            config=cfg,
            camera_spec=camera_spec,
            criteria=crit,
            detector=line_detector,
        )
        for run_idx in range(cfg.runs):
            entries, skipped = _calibrate_once(
                device,
                config=cfg,
                camera_spec=camera_spec,
                criteria=crit,
                scene_config=scene_cfg,
                detector=line_detector,
                image_tag=f"{camera_id}_run{run_idx}",
            )
            all_skipped.extend(skipped)

            camera_table = LensCalibrationTable(
                id=camera_id,
                entries=tuple(entries),
                created_date=ts,
                last_modified=ts,
            )
            if entries:
                model_table = fold_camera_table(
                    model_table,
                    camera_table,
                    model_name=model_name,
                    camera_id=camera_id,
                    now=ts,
                    config=agg_cfg,
                )
            elif skipped:
                last_report = skipped[-1]
    finally:
        device.restore_ptz(saved_state)
        # Verify restore: re-read and compare (best effort; tolerate read fail).
        try:
            restored = _ptz_close(device.get_ptz(), saved_state)
        except Exception:  # restoration verification is best effort
            restored = True

    if model_table is None or not model_table.run_history:
        # Zero zooms calibrated across all runs -> hard failure with a reason.
        from poc_homography.calibration.lens_distortion.visibility import VisibilityReport

        if isinstance(last_report, SkippedZoom):
            report = VisibilityReport(
                passed=False,
                num_lines=last_report.num_lines,
                num_curved_lines=0,
                quadrants_covered=0,
                orientation_buckets=0,
                reasons=(last_report.reason,),
            )
        else:
            report = VisibilityReport(
                passed=False,
                num_lines=0,
                num_curved_lines=0,
                quadrants_covered=0,
                orientation_buckets=0,
                reasons=("no zoom produced a calibratable view",),
            )
        raise NoCalibratableViewError(report, context=f"camera={camera_id}")

    assert camera_table is not None  # guaranteed once run_history is non-empty

    if output_dir is not None:
        from poc_homography.infrastructure.repositories.repo_yaml_lens_calibration_table import (
            RepoYamlLensCalibrationTable,
        )

        RepoYamlLensCalibrationTable(Path(output_dir) / "calibration_results").save(camera_table)
        _save_model_table_yaml(
            model_table,
            Path(output_dir) / "calibration_results" / "models" / f"{model_name}.yaml",
        )

    return AutoCalibrationResult(
        camera_table=camera_table,
        model_table=model_table,
        skipped_zooms=tuple(all_skipped),
        restored=restored,
    )


def _ptz_close(a: PTZTriple, b: PTZTriple, *, tol: float = 0.5) -> bool:
    """Whether two PTZ triples agree within ``tol`` per axis."""
    return all(abs(x - y) <= tol for x, y in zip(a, b))


class OfflineFrameDevice:
    """A :class:`CalibrationDevice` backed by pre-captured ``(image, zoom)`` pairs.

    Lets the orchestrator (and the CLI ``--offline-dir`` mode) run without
    hardware: ``set_zoom`` selects the nearest stored zoom group and ``capture``
    replays its frames round-robin. ``get_ptz``/``restore_ptz`` are no-ops on a
    fixed synthetic state.
    """

    def __init__(
        self, image_zoom_pairs: Sequence[tuple[np.ndarray, float]], *, tol: float = 0.25
    ) -> None:
        """Group ``(image, zoom)`` pairs by rounded zoom for replay.

        Args:
            image_zoom_pairs: Pre-captured frames and the zoom each was taken at.
            tol: Zoom grouping/selection tolerance.
        """
        self._groups: dict[float, list[np.ndarray]] = {}
        for image, zoom in image_zoom_pairs:
            key = round(round(zoom / tol) * tol, 6)
            self._groups.setdefault(key, []).append(image)
        self._current = sorted(self._groups)[0] if self._groups else 1.0
        self._cursor = 0

    def get_ptz(self) -> PTZTriple:
        """Return a fixed synthetic PTZ state."""
        return (0.0, 0.0, self._current)

    def set_pan_tilt(self, pan: float, tilt: float) -> None:
        """No-op aim for offline replay (frames are fixed)."""

    def set_zoom(self, zoom: float) -> None:
        """Select the nearest stored zoom group to ``zoom``."""
        if not self._groups:
            self._current = zoom
            return
        self._current = min(self._groups, key=lambda k: abs(k - zoom))
        self._cursor = 0

    def capture(self) -> np.ndarray:
        """Replay the next frame of the selected zoom group (round-robin)."""
        frames = self._groups.get(self._current)
        if not frames:
            raise ValueError(f"No offline frames for zoom {self._current}")
        frame = frames[self._cursor % len(frames)]
        self._cursor += 1
        return frame

    def restore_ptz(self, state: PTZTriple) -> None:
        """No-op restore for offline replay."""
        self._current = state[2]


__all__ = [
    "AutoCalibrationConfig",
    "AutoCalibrationResult",
    "CalibrationDevice",
    "DeviceFrameCapturer",
    "OfflineFrameDevice",
    "run_auto_calibration",
]
