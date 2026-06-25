"""Offline tests for the ``hom calibrate lens-auto`` CLI command."""

from __future__ import annotations

import cv2
import numpy as np
from typer.testing import CliRunner

from poc_homography.cli.main import app
from poc_homography.domain.enums.camera_spec import CameraSpec

runner = CliRunner()

_SPEC = CameraSpec.HIKVISION_DS_2DF8425IX
_W = int(_SPEC.image_width)
_H = int(_SPEC.image_height)


def _rich_frame(bow: int = 30) -> np.ndarray:
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


def _write_frames(folder, zooms_bows) -> None:
    for idx, (zoom, bow) in enumerate(zooms_bows):
        cv2.imwrite(str(folder / f"frame_zoom{zoom}_{idx}.png"), _rich_frame(bow))


def test_offline_mode_calibrates_and_persists(tmp_path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_frames(frames, [(2.0, 25), (10.0, 35)])
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "calibrate",
            "lens-auto",
            "--camera-id",
            "camX",
            "--offline-dir",
            str(frames),
            "--output-dir",
            str(out),
            "--zoom-min",
            "2",
            "--zoom-max",
            "10",
            "--coarse-steps",
            "2",
            "--min-lines",
            "4",
            "--min-orientations",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Done:" in result.output
    assert (out / "calibration_results" / "camX.yaml").exists()
    assert (out / "calibration_results" / "models" / f"{_SPEC.model_name}.yaml").exists()


def test_blank_frames_exit_nonzero(tmp_path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    # Featureless frames -> no calibratable view.
    for idx, zoom in enumerate([2.0, 6.0, 10.0]):
        cv2.imwrite(
            str(frames / f"frame_zoom{zoom}_{idx}.png"),
            np.full((_H, _W, 3), 110, dtype=np.uint8),
        )

    result = runner.invoke(
        app,
        [
            "calibrate",
            "lens-auto",
            "--camera-id",
            "camBlank",
            "--offline-dir",
            str(frames),
            "--output-dir",
            str(tmp_path / "out"),
            "--coarse-steps",
            "2",
        ],
    )

    assert result.exit_code == 1
    assert "No calibratable view" in result.output


def test_missing_offline_dir_flag_exits(tmp_path) -> None:
    result = runner.invoke(
        app,
        ["calibrate", "lens-auto", "--camera-id", "cam0"],
    )
    assert result.exit_code == 2
    assert "offline-dir" in result.output


def test_unknown_model_rejected(tmp_path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_frames(frames, [(2.0, 25)])
    result = runner.invoke(
        app,
        [
            "calibrate",
            "lens-auto",
            "--camera-id",
            "cam0",
            "--model",
            "NOPE",
            "--offline-dir",
            str(frames),
        ],
    )
    assert result.exit_code != 0
