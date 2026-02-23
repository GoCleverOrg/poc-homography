"""Seed per-frame annotations from the monolithic valte_annotations.yaml.

Reads test cases, maps each to its CapturedFrame, and saves annotations
via RepoYamlCapturedFrame.save_annotations() — co-located alongside the
frame YAML files.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from poc_homography.domain.entities.annotation import Annotation
from poc_homography.domain.vo.pixel_point import PixelPoint
from poc_homography.domain.vo.ptz_state import PTZState
from poc_homography.infrastructure.repositories import RepoYamlCapturedFrame
from poc_homography.types import Degrees, Unitless

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "homography" / "test_data"
ANNOTATIONS_FILE = TEST_DATA_DIR / "valte_annotations.yaml"
FRAMES_DIR = PROJECT_ROOT / "data" / "captured_frames"


def _build_image_to_frame_id(repo: RepoYamlCapturedFrame) -> dict[str, str]:
    """Build a mapping from image filename -> CapturedFrame.id."""
    mapping: dict[str, str] = {}
    for frame in repo.get_all():
        mapping[frame.image_path.name] = frame.id
    return mapping


def main() -> None:
    if not ANNOTATIONS_FILE.exists():
        print(f"Annotations file not found: {ANNOTATIONS_FILE}")
        return

    with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "test_cases" not in data:
        print("No test_cases found in annotations file")
        return

    repo = RepoYamlCapturedFrame(FRAMES_DIR)
    image_to_frame = _build_image_to_frame_id(repo)

    total_annotations = 0
    for tc in data["test_cases"]:
        image = tc.get("image", "")
        frame_id = image_to_frame.get(image)
        if not frame_id:
            print(f"  SKIP: no CapturedFrame for image '{image}'")
            continue

        # Build camera_pose from camera_status or from frame PTZ
        cs = tc.get("camera_status")
        if cs:
            camera_pose = PTZState(
                pan_raw=Degrees(float(cs["pan"])),
                tilt_deg=Degrees(float(cs["tilt"])),
                zoom=Unitless(float(cs["zoom"])),
            )
        else:
            frame = repo.get(frame_id)
            camera_pose = frame.ptz_state if frame else PTZState(
                pan_raw=Degrees(0.0), tilt_deg=Degrees(0.0), zoom=Unitless(1.0)
            )

        raw_annotations = tc.get("annotations", [])
        annotations = [
            Annotation(
                gcp_id=ann["gcp_id"],
                frame_id=frame_id,
                camera_pose=camera_pose,
                pixel=PixelPoint.create(float(ann["pixel_x"]), float(ann["pixel_y"])),
            )
            for ann in raw_annotations
        ]

        repo.save_annotations(frame_id, annotations)
        total_annotations += len(annotations)
        print(f"  {frame_id}: {len(annotations)} annotations")

    print(f"\nSeeded {total_annotations} annotations across {len(data['test_cases'])} frames")


if __name__ == "__main__":
    main()
