"""Tests for the survey-dataset loader feeding clean-plate reconstruction.

These tests build a TINY on-disk survey run under ``tmp_path`` (a couple of
:class:`FrameRecord` YAMLs plus generated images / masks), then exercise
:class:`CleanPlateDataset` grouping and per-group frame loading.

Fast & deterministic: no DVC, no DB, no network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import yaml
from tests.domain.survey.builders import make_frame_record

from poc_homography.cleanplate.dataset import CleanPlateDataset
from poc_homography.cleanplate.reconstruct import CleanPlateFrame

if TYPE_CHECKING:
    from pathlib import Path

RUN_ID = "run-0001"
CAMERA_ID = "cam01"
POSE_A = "p+0120.0_t-0015.0_z004.00"
POSE_B = "p+0090.0_t-0010.0_z002.00"


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
    *,
    capture_id: str,
    pose_id: str | None,
    image_path: str,
    frame_index: int = 0,
    mask_ref: str | None = None,
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
    if mask_ref is not None:
        data["floor_mask_reference"]["mask_ref"] = mask_ref
    else:
        data["floor_mask_reference"]["mask_ref"] = None
    return data


def _build_run(tmp_path: Path) -> Path:
    """Create a tiny survey run with three frames (2 in pose A, 1 in pose B)."""
    frames_root = tmp_path / "survey"
    # Two frames sharing pose A.
    _write_frame_yaml(
        frames_root,
        _make_record_dict(capture_id="cap-0001", pose_id=POSE_A, image_path="frames/cap-0001.png"),
    )
    _write_image(frames_root, "frames/cap-0001.png")
    _write_frame_yaml(
        frames_root,
        _make_record_dict(
            capture_id="cap-0002",
            pose_id=POSE_A,
            image_path="frames/cap-0002.png",
            frame_index=1,
        ),
    )
    _write_image(frames_root, "frames/cap-0002.png", color=150)
    # One frame in pose B.
    _write_frame_yaml(
        frames_root,
        _make_record_dict(
            capture_id="cap-0003",
            pose_id=POSE_B,
            image_path="frames/cap-0003.png",
            frame_index=2,
        ),
    )
    _write_image(frames_root, "frames/cap-0003.png", color=200)
    return frames_root


def test_groups_by_camera_and_pose(tmp_path: Path) -> None:
    """Frames are grouped by ``(camera_id, pose_id)``."""
    frames_root = _build_run(tmp_path)
    dataset = CleanPlateDataset.from_survey_run(frames_root, RUN_ID)

    groups = dataset.groups()
    assert (CAMERA_ID, POSE_A) in groups
    assert (CAMERA_ID, POSE_B) in groups
    assert len(groups[(CAMERA_ID, POSE_A)]) == 2
    assert len(groups[(CAMERA_ID, POSE_B)]) == 1


def test_frames_for_loads_clean_plate_frames(tmp_path: Path) -> None:
    """``frames_for`` returns CleanPlateFrames with image, mask and 3x3 H."""
    frames_root = _build_run(tmp_path)
    dataset = CleanPlateDataset.from_survey_run(frames_root, RUN_ID)

    frames = dataset.frames_for(CAMERA_ID, POSE_A)
    assert len(frames) == 2
    for frame in frames:
        assert isinstance(frame, CleanPlateFrame)
        assert frame.image.shape == (8, 12, 3)
        # No mask_ref present -> mask is None, the canonical all-floor sentinel
        # (ortho treats None as all-floor via its cheaper no-mask path).
        assert frame.floor_mask is None
        assert frame.ground_homography.shape == (3, 3)
        assert frame.gain == 6.0


def test_ground_homography_orientation_world_to_image(tmp_path: Path) -> None:
    """The loaded H is used as-is (world meters -> image pixels), not inverted.

    The survey builder stores ``h_matrix = [[1,0,3],[0,1,4],[0,0,1]]`` (a pure
    translation by (3, 4)). Mapping world origin ``(0, 0)`` must land at image
    pixel ``(3, 4)`` — confirming the matrix is applied directly (world->image),
    matching :attr:`CleanPlateFrame.ground_homography`'s documented convention.
    """
    frames_root = _build_run(tmp_path)
    dataset = CleanPlateDataset.from_survey_run(frames_root, RUN_ID)

    frame = dataset.frames_for(CAMERA_ID, POSE_A)[0]
    world_origin = np.array([0.0, 0.0, 1.0])
    image_pt = frame.ground_homography @ world_origin
    image_pt = image_pt / image_pt[2]
    np.testing.assert_allclose(image_pt[:2], [3.0, 4.0])


def test_missing_image_frames_are_skipped(tmp_path: Path) -> None:
    """A frame whose image file is absent is dropped from ``frames_for``."""
    frames_root = _build_run(tmp_path)
    # Add a fourth frame in pose B whose image is never written to disk.
    _write_frame_yaml(
        frames_root,
        _make_record_dict(
            capture_id="cap-0004",
            pose_id=POSE_B,
            image_path="frames/missing.png",
            frame_index=3,
        ),
    )
    dataset = CleanPlateDataset.from_survey_run(frames_root, RUN_ID)

    # Group still lists both records...
    assert len(dataset.groups()[(CAMERA_ID, POSE_B)]) == 2
    # ...but only the one with an on-disk image yields a CleanPlateFrame.
    frames = dataset.frames_for(CAMERA_ID, POSE_B)
    assert len(frames) == 1


def test_mask_loaded_from_reference_when_present(tmp_path: Path) -> None:
    """When ``mask_ref`` points to a real file, it is loaded (not synthesized)."""
    frames_root = _build_run(tmp_path)
    _write_frame_yaml(
        frames_root,
        _make_record_dict(
            capture_id="cap-0005",
            pose_id=POSE_B,
            image_path="frames/cap-0005.png",
            frame_index=4,
            mask_ref="masks/cap-0005.png",
        ),
    )
    _write_image(frames_root, "frames/cap-0005.png", color=120)
    # Mask: top half occluder (0), bottom half floor (255).
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[4:, :] = 255
    mask_abs = frames_root / RUN_ID / CAMERA_ID / "masks/cap-0005.png"
    mask_abs.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_abs), mask)

    dataset = CleanPlateDataset.from_survey_run(frames_root, RUN_ID)
    frames = dataset.frames_for(CAMERA_ID, POSE_B)
    # Frames without a mask_ref now carry None (all-floor); the one with a real
    # mask_ref carries the loaded array.
    loaded = next(f for f in frames if f.floor_mask is not None)
    assert loaded.floor_mask.shape == (8, 12)
    assert not loaded.floor_mask[0, 0]
    assert loaded.floor_mask[7, 0]
