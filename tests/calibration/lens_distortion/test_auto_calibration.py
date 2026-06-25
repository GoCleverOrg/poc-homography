"""Offline tests for the automatic lens-distortion orchestrator (fake device)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from poc_homography.calibration.lens_distortion.auto_calibration import (
    AutoCalibrationConfig,
    CalibrationDevice,
    OfflineFrameDevice,
    PTZTriple,
    run_auto_calibration,
)
from poc_homography.calibration.lens_distortion.scene_self_calibration import (
    ScenePerZoomConfig,
)
from poc_homography.calibration.lens_distortion.visibility import (
    NoCalibratableViewError,
    VisibilityCriteria,
)
from poc_homography.domain.enums.camera_spec import CameraSpec

_SPEC = CameraSpec.HIKVISION_DS_2DF8425IX
_W = int(_SPEC.image_width)
_H = int(_SPEC.image_height)

# Lenient gates: synthetic frames carry plenty of curved structure.
_CRIT = VisibilityCriteria(min_lines=4, min_curved_lines=2, min_quadrants=2, min_orientations=2)
# Tight solver budget keeps the offline suite fast (full-sensor frames).
_SCENE = ScenePerZoomConfig(min_lines=2, num_samples_per_line=12, max_iterations=200)


def _rich_frame(bow: int = 30) -> np.ndarray:
    """A full-sensor gray frame with many bowed white lines in both axes."""
    img = np.full((_H, _W, 3), 110, dtype=np.uint8)
    for y in range(120, _H - 120, 140):
        xs = np.linspace(120, _W - 120, 120)
        ys = y + bow * np.sin(np.linspace(0, np.pi, 120))
        cv2.polylines(img, [np.column_stack([xs, ys]).astype(np.int32)], False, (255, 255, 255), 8)
    for x in range(200, _W - 200, 240):
        ys = np.linspace(120, _H - 120, 120)
        xs = x + bow * np.sin(np.linspace(0, np.pi, 120))
        cv2.polylines(img, [np.column_stack([xs, ys]).astype(np.int32)], False, (255, 255, 255), 8)
    return img


def _blank_frame() -> np.ndarray:
    """A featureless gray frame (no calibratable structure)."""
    return np.full((_H, _W, 3), 110, dtype=np.uint8)


class FakeDevice:
    """Records the call order and returns a configurable frame per capture."""

    def __init__(
        self,
        frame_fn: object = None,
        *,
        raise_on_set_zoom: float | None = None,
    ) -> None:
        self.log: list[object] = []
        self._frame_fn = frame_fn or _rich_frame
        self._raise_on = raise_on_set_zoom
        self.z = 5.0

    def get_ptz(self) -> PTZTriple:
        self.log.append("get")
        return (10.0, -5.0, self.z)

    def set_zoom(self, zoom: float) -> None:
        self.log.append(("set", zoom))
        if self._raise_on is not None and abs(zoom - self._raise_on) < 1e-6:
            raise RuntimeError("simulated camera fault")
        self.z = zoom

    def capture(self) -> np.ndarray:
        self.log.append("cap")
        return self._frame_fn()  # type: ignore[operator]

    def restore_ptz(self, state: PTZTriple) -> None:
        self.log.append(("restore", state))
        self.z = state[2]


def _run(device: CalibrationDevice, **kwargs: object):
    return run_auto_calibration(
        device,
        camera_id="cam1",
        camera_spec=_SPEC,
        criteria=_CRIT,
        scene_config=_SCENE,
        now="2026-01-01T00:00:00",
        **kwargs,  # type: ignore[arg-type]
    )


def test_save_capture_restore_order() -> None:
    device = FakeDevice()
    cfg = AutoCalibrationConfig(zoom_min=2.0, zoom_max=10.0, coarse_steps=2, runs=1)
    result = _run(device, config=cfg)

    assert result.camera_table.entries  # at least one zoom calibrated
    # First op is the save (get_ptz).
    assert device.log[0] == "get"
    # Last op is the restore-verification re-read (get_ptz after restore_ptz).
    assert device.log[-1] == "get"
    restore_ops = [op for op in device.log if isinstance(op, tuple) and op[0] == "restore"]
    assert len(restore_ops) == 1
    assert restore_ops[0][1] == (10.0, -5.0, 5.0)
    assert result.restored is True
    # A capture happens after the first set_zoom.
    first_set = next(
        i for i, op in enumerate(device.log) if isinstance(op, tuple) and op[0] == "set"
    )
    assert "cap" in device.log[first_set:]


def test_restore_happens_even_when_zoom_raises() -> None:
    device = FakeDevice(raise_on_set_zoom=10.0)
    cfg = AutoCalibrationConfig(zoom_min=10.0, zoom_max=10.0, coarse_steps=1, runs=1)

    with pytest.raises((RuntimeError, NoCalibratableViewError)):
        _run(device, config=cfg)

    restore_ops = [op for op in device.log if isinstance(op, tuple) and op[0] == "restore"]
    assert len(restore_ops) == 1
    assert restore_ops[0][1] == (10.0, -5.0, 5.0)


def test_all_zoom_fail_raises_no_calibratable_view() -> None:
    device = FakeDevice(frame_fn=_blank_frame)
    cfg = AutoCalibrationConfig(zoom_min=2.0, zoom_max=10.0, coarse_steps=3, runs=1)

    with pytest.raises(NoCalibratableViewError):
        _run(device, config=cfg)

    # PTZ is still restored on the failure path.
    assert any(isinstance(op, tuple) and op[0] == "restore" for op in device.log)


def test_adaptive_zoom_inserts_level_when_coeffs_disagree() -> None:
    # bow varies with zoom -> k1 differs across coarse zooms -> a midpoint insert.
    def varying_frame_device() -> FakeDevice:
        dev = FakeDevice()

        def frame_fn() -> np.ndarray:
            # bow scales with current commanded zoom, so distortion signal differs.
            return _rich_frame(bow=int(20 + dev.z * 4))

        dev._frame_fn = frame_fn  # type: ignore[attr-defined]
        return dev

    device = varying_frame_device()
    cfg = AutoCalibrationConfig(
        zoom_min=2.0,
        zoom_max=18.0,
        coarse_steps=2,
        zoom_tol=1e-6,  # force disagreement -> insert
        max_adaptive_levels=1,
        runs=1,
    )
    result = _run(device, config=cfg)

    solved_zooms = {round(float(e.zoom_factor), 3) for e in result.camera_table.entries}
    # A midpoint between the two coarse zooms (2 and 18 -> 10) must appear.
    assert 10.0 in solved_zooms


def test_runs_grow_model_run_history() -> None:
    cfg1 = AutoCalibrationConfig(zoom_min=2.0, zoom_max=10.0, coarse_steps=2, runs=1)
    r1 = _run(FakeDevice(), config=cfg1)

    cfg3 = AutoCalibrationConfig(zoom_min=2.0, zoom_max=10.0, coarse_steps=2, runs=3)
    r3 = _run(FakeDevice(), config=cfg3)

    assert len(r3.model_table.run_history) > len(r1.model_table.run_history)
    assert len(r3.model_table.run_history) == 3 * len(r1.model_table.run_history)


def test_persists_yaml_outputs(tmp_path) -> None:
    cfg = AutoCalibrationConfig(zoom_min=2.0, zoom_max=10.0, coarse_steps=2, runs=1)
    _run(FakeDevice(), config=cfg, output_dir=tmp_path)

    cam_yaml = tmp_path / "calibration_results" / "cam1.yaml"
    model_yaml = tmp_path / "calibration_results" / "models" / f"{_SPEC.model_name}.yaml"
    assert cam_yaml.exists()
    assert model_yaml.exists()


def test_offline_frame_device_replays_by_zoom() -> None:
    frame_a = _rich_frame(bow=25)
    frame_b = _rich_frame(bow=35)
    device = OfflineFrameDevice([(frame_a, 2.0), (frame_b, 10.0)])
    cfg = AutoCalibrationConfig(zoom_min=2.0, zoom_max=10.0, coarse_steps=2, runs=1)

    result = run_auto_calibration(
        device,
        camera_id="cam_off",
        camera_spec=_SPEC,
        config=cfg,
        criteria=_CRIT,
        scene_config=_SCENE,
        now="2026-01-01T00:00:00",
    )
    assert result.camera_table.entries
    assert result.restored is True
