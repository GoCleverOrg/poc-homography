"""Tests for the ``hom cleanplate`` CLI sub-app."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import yaml
from tests.domain.survey.builders import make_frame_record
from typer.testing import CliRunner

from poc_homography.cli.main import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()

RUN_ID = "run-0001"
CAMERA_ID = "cam01"
POSE_A = "p+0120.0_t-0015.0_z004.00"
POSE_B = "p+0090.0_t-0010.0_z002.00"


def test_cleanplate_help_lists_commands() -> None:
    """`hom cleanplate --help` lists both `reconstruct` and `synth`."""
    result = runner.invoke(app, ["cleanplate", "--help"])

    assert result.exit_code == 0, result.output
    assert "reconstruct" in result.output
    assert "synth" in result.output


def test_synth_produces_orthophoto(tmp_path: Path) -> None:
    """`cleanplate synth` writes a readable orthophoto and reports a small MAE."""
    ortho_path = tmp_path / "ortho.png"
    coverage_path = tmp_path / "coverage.tif"
    truth_path = tmp_path / "truth.png"

    result = runner.invoke(
        app,
        [
            "cleanplate",
            "synth",
            "--output",
            str(ortho_path),
            "--coverage-output",
            str(coverage_path),
            "--truth-output",
            str(truth_path),
            "--n-visits",
            "4",
            "--seed",
            "0",
            "--x-min",
            "0",
            "--x-max",
            "6",
            "--y-min",
            "0",
            "--y-max",
            "6",
            "--pixels-per-meter",
            "16",
        ],
    )

    assert result.exit_code == 0, result.output
    assert ortho_path.exists()
    assert coverage_path.exists()
    assert truth_path.exists()

    image = cv2.imread(str(ortho_path), cv2.IMREAD_COLOR)
    assert image is not None
    # 6m * 16 ppm = 96 cells square.
    assert image.shape == (96, 96, 3)
    assert "MAE" in result.output


def test_synth_is_deterministic(tmp_path: Path) -> None:
    """Two `synth` runs with the same seed produce byte-identical orthophotos."""
    out_a = tmp_path / "a.png"
    out_b = tmp_path / "b.png"
    args = [
        "cleanplate",
        "synth",
        "--seed",
        "7",
        "--output",
    ]
    assert runner.invoke(app, [*args, str(out_a)]).exit_code == 0
    assert runner.invoke(app, [*args, str(out_b)]).exit_code == 0
    assert out_a.read_bytes() == out_b.read_bytes()


def _write_frame_yaml(frames_root: Path, record_dict: dict) -> None:
    """Write a FrameRecord dict to its survey layout location."""
    cam_dir = frames_root / RUN_ID / CAMERA_ID / "frames"
    cam_dir.mkdir(parents=True, exist_ok=True)
    capture_id = record_dict["capture"]["capture_id"]
    path = cam_dir / f"{capture_id}.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(record_dict, handle)


def _write_image(frames_root: Path, rel_path: str, *, color: int = 100) -> None:
    """Write a small solid-color BGR image at the survey-relative path."""
    abs_path = frames_root / RUN_ID / CAMERA_ID / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((8, 12, 3), color, dtype=np.uint8)
    cv2.imwrite(str(abs_path), img)


def _make_record_dict(
    *, capture_id: str, image_path: str, frame_index: int, pose_id: str = POSE_A
) -> dict:
    """Build a serialised FrameRecord dict with a usable cached homography."""
    record = make_frame_record(
        capture_id=capture_id,
        run_id=RUN_ID,
        camera_id=CAMERA_ID,
        frame_index=frame_index,
        image_path=image_path,
        with_clean_plate=True,
    )
    data = record.to_dict()
    data["survey_context"]["pose_id"] = pose_id
    data["floor_mask_reference"]["mask_ref"] = None
    return data


def _build_run(tmp_path: Path) -> Path:
    """Create a tiny survey run with two frames in a single pose."""
    frames_root = tmp_path / "survey"
    for idx, capture_id in enumerate(("cap-0001", "cap-0002")):
        _write_frame_yaml(
            frames_root,
            _make_record_dict(
                capture_id=capture_id,
                image_path=f"frames/{capture_id}.png",
                frame_index=idx,
            ),
        )
        _write_image(frames_root, f"frames/{capture_id}.png", color=100 + 40 * idx)
    return frames_root


def test_reconstruct_on_tiny_run(tmp_path: Path) -> None:
    """`cleanplate reconstruct` on a tiny on-disk run writes an orthophoto."""
    frames_root = _build_run(tmp_path)
    ortho_path = tmp_path / "plate.png"

    result = runner.invoke(
        app,
        [
            "cleanplate",
            "reconstruct",
            "--run-dir",
            str(frames_root),
            "--run-id",
            RUN_ID,
            "--camera-id",
            CAMERA_ID,
            "--pose-id",
            POSE_A,
            "--x-min",
            "0",
            "--x-max",
            "4",
            "--y-min",
            "0",
            "--y-max",
            "4",
            "--pixels-per-meter",
            "4",
            "--output",
            str(ortho_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert ortho_path.exists()
    image = cv2.imread(str(ortho_path), cv2.IMREAD_COLOR)
    assert image is not None
    assert "frames" in result.output


def test_reconstruct_all_groups_uses_suffix(tmp_path: Path) -> None:
    """Without `--pose-id`, one suffixed orthophoto is written per group."""
    frames_root = _build_run(tmp_path)
    # Add a second pose so more than one group is processed.
    _write_frame_yaml(
        frames_root,
        _make_record_dict(
            capture_id="cap-0003",
            image_path="frames/cap-0003.png",
            frame_index=2,
            pose_id=POSE_B,
        ),
    )
    _write_image(frames_root, "frames/cap-0003.png", color=180)
    ortho_path = tmp_path / "plate.png"

    result = runner.invoke(
        app,
        [
            "cleanplate",
            "reconstruct",
            "--run-dir",
            str(frames_root),
            "--run-id",
            RUN_ID,
            "--x-min",
            "0",
            "--x-max",
            "4",
            "--y-min",
            "0",
            "--y-max",
            "4",
            "--pixels-per-meter",
            "4",
            "--output",
            str(ortho_path),
        ],
    )

    assert result.exit_code == 0, result.output
    # Suffixed by (camera_id, pose_id); the bare --output path is NOT used directly.
    suffixed = tmp_path / f"plate__{CAMERA_ID}__{POSE_A}.png"
    assert suffixed.exists()


def test_reconstruct_unknown_run_exits_nonzero(tmp_path: Path) -> None:
    """`cleanplate reconstruct` exits 1 when no groups are found."""
    frames_root = tmp_path / "empty"
    frames_root.mkdir()
    result = runner.invoke(
        app,
        [
            "cleanplate",
            "reconstruct",
            "--run-dir",
            str(frames_root),
            "--run-id",
            "nope",
            "--x-min",
            "0",
            "--x-max",
            "4",
            "--y-min",
            "0",
            "--y-max",
            "4",
            "--pixels-per-meter",
            "4",
            "--output",
            str(tmp_path / "out.png"),
        ],
    )

    assert result.exit_code == 1
    assert "Error" in result.output
