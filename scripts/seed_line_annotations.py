"""Seed LineAnnotation repo from line annotation YAML files.

Reads both the monolithic valte_line_annotations.yaml and per-image
*_line_annotations.yaml files, maps each to its CapturedFrame, and saves
LineAnnotation entities via RepoYamlLineAnnotation.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from poc_homography.domain.entities.line_annotation import LineAnnotation
from poc_homography.domain.vo.pixel_point import PixelPoint
from poc_homography.domain.vo.ptz_state import PTZState
from poc_homography.infrastructure.repositories import (
    RepoYamlCapturedFrame,
    RepoYamlLineAnnotation,
)
from poc_homography.types import Degrees, Unitless

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "homography" / "test_data"
FRAMES_DIR = PROJECT_ROOT / "data" / "captured_frames"
LINE_ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "line_annotations"


def _build_image_to_frame_id(repo: RepoYamlCapturedFrame) -> dict[str, str]:
    """Build a mapping from image filename -> CapturedFrame.id."""
    mapping: dict[str, str] = {}
    for frame in repo.get_all():
        mapping[frame.image_path.name] = frame.id
    return mapping


def _collect_annotation_files() -> list[Path]:
    """Collect all per-image line annotation YAML files from test data directory."""
    return sorted(TEST_DATA_DIR.glob("*_line_annotations.yaml"))


def _seed_from_file(
    path: Path,
    frame_repo: RepoYamlCapturedFrame,
    line_ann_repo: RepoYamlLineAnnotation,
    image_to_frame: dict[str, str],
) -> int:
    """Seed line annotations from a single YAML file. Returns count."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "test_cases" not in data:
        return 0

    total = 0
    for tc in data["test_cases"]:
        image = tc.get("image", "")
        frame_id = image_to_frame.get(image)
        if not frame_id:
            print(f"  SKIP: no CapturedFrame for image '{image}'")
            continue

        # Build camera_pose from camera_status or from frame
        cs = tc.get("camera_status")
        if cs:
            camera_pose = PTZState(
                pan_raw=Degrees(float(cs["pan"])),
                tilt_deg=Degrees(float(cs["tilt"])),
                zoom=Unitless(float(cs["zoom"])),
            )
        else:
            frame = frame_repo.get(frame_id)
            camera_pose = frame.ptz_state if frame else PTZState(
                pan_raw=Degrees(0.0), tilt_deg=Degrees(0.0), zoom=Unitless(1.0)
            )

        for ann in tc.get("line_annotations", []):
            line_ann = LineAnnotation(
                line_id=ann["line_id"],
                frame_id=frame_id,
                camera_pose=camera_pose,
                start_pixel=PixelPoint.create(
                    float(ann["start_pixel_x"]),
                    float(ann["start_pixel_y"]),
                ),
                end_pixel=PixelPoint.create(
                    float(ann["end_pixel_x"]),
                    float(ann["end_pixel_y"]),
                ),
            )
            line_ann_repo.save(line_ann)
            total += 1
            print(f"  {line_ann.id}")

    return total


def main() -> None:
    files = _collect_annotation_files()
    if not files:
        print(f"No line annotation files found in {TEST_DATA_DIR}")
        return

    frame_repo = RepoYamlCapturedFrame(FRAMES_DIR)
    line_ann_repo = RepoYamlLineAnnotation(LINE_ANNOTATIONS_DIR)
    image_to_frame = _build_image_to_frame_id(frame_repo)

    total = 0
    for path in files:
        print(f"\nProcessing {path.name}:")
        total += _seed_from_file(path, frame_repo, line_ann_repo, image_to_frame)

    print(f"\nSeeded {total} line annotations in {LINE_ANNOTATIONS_DIR}")


if __name__ == "__main__":
    main()
