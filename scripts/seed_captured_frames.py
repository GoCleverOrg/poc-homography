"""Seed CapturedFrame repo from test images and annotation YAML files.

Parses 14 images from tests/homography/test_data/, creates CapturedFrame
entities with PTZ metadata, and copies images to data/captured_frames/.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from poc_homography.domain.entities.captured_frame import CapturedFrame
from poc_homography.domain.vo.ptz_state import PTZState
from poc_homography.infrastructure.repositories import RepoYamlCapturedFrame
from poc_homography.types import Degrees, Unitless

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "homography" / "test_data"
ANNOTATIONS_FILE = TEST_DATA_DIR / "valte_annotations.yaml"
FRAMES_DIR = PROJECT_ROOT / "data" / "captured_frames"

MAP_ID = "Cartografia_valencia"

# Regex patterns for filename parsing
SURVEY_PATTERN = re.compile(
    r"valte_valte_cam01_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_"
    r"([\d.]+)_([\d.]+)_([\d.]+)\.\w+$"
)
LEGACY_PATTERN = re.compile(
    r"valte_([\d.]+)_([\d.]+)_(\d+)_(\d{4})(\d{2})(\d{2})(?:_(\d{6}))?\.\w+$"
)


def _parse_survey_filename(filename: str) -> tuple[str, datetime, PTZState] | None:
    """Parse survey-pattern filename: valte_valte_cam01_YYYYMMDD_HHMMSS_pan_tilt_zoom."""
    m = SURVEY_PATTERN.search(filename)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups()[:6])
    pan, tilt, zoom = float(m.group(7)), float(m.group(8)), float(m.group(9))
    ts = datetime(y, mo, d, h, mi, s)
    ptz = PTZState(pan_raw=Degrees(pan), tilt_deg=Degrees(tilt), zoom=Unitless(zoom))
    return "cam01", ts, ptz


def _parse_legacy_filename(
    filename: str,
    annotations_by_image: dict[str, dict],
) -> tuple[str, datetime, PTZState] | None:
    """Parse legacy-pattern filename: valte_pan_tilt_zoom_YYYYMMDD[_HHMMSS]."""
    m = LEGACY_PATTERN.search(filename)
    if not m:
        return None
    pan, tilt, zoom_str = m.group(1), m.group(2), m.group(3)
    y, mo, d = int(m.group(4)), int(m.group(5)), int(m.group(6))
    time_part = m.group(7)
    if time_part:
        h, mi, s = int(time_part[:2]), int(time_part[2:4]), int(time_part[4:6])
    else:
        h, mi, s = 0, 0, 0
    ts = datetime(y, mo, d, h, mi, s)

    # Prefer camera_status from annotations YAML if available
    tc = annotations_by_image.get(filename)
    if tc and "camera_status" in tc:
        cs = tc["camera_status"]
        ptz = PTZState(
            pan_raw=Degrees(float(cs["pan"])),
            tilt_deg=Degrees(float(cs["tilt"])),
            zoom=Unitless(float(cs["zoom"])),
        )
    else:
        ptz = PTZState(
            pan_raw=Degrees(float(pan)),
            tilt_deg=Degrees(float(tilt)),
            zoom=Unitless(float(zoom_str)),
        )
    return "Valte", ts, ptz


def _load_annotations_by_image() -> dict[str, dict]:
    """Load annotations YAML and index test cases by image filename."""
    if not ANNOTATIONS_FILE.exists():
        return {}
    with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "test_cases" not in data:
        return {}
    return {tc["image"]: tc for tc in data["test_cases"] if "image" in tc}


def main() -> None:
    repo = RepoYamlCapturedFrame(FRAMES_DIR)
    annotations_by_image = _load_annotations_by_image()

    image_extensions = {".jpg", ".jpeg", ".png"}
    images = sorted(
        p
        for p in TEST_DATA_DIR.iterdir()
        if p.suffix.lower() in image_extensions
    )

    created = 0
    for img_path in images:
        filename = img_path.name

        # Try survey pattern first
        parsed = _parse_survey_filename(filename)
        if parsed:
            camera_name, ts, ptz = parsed
        else:
            parsed = _parse_legacy_filename(filename, annotations_by_image)
            if parsed:
                camera_name, ts, ptz = parsed
            else:
                print(f"  SKIP: could not parse '{filename}'")
                continue

        frame = CapturedFrame.create(
            map_id=MAP_ID,
            camera_name=camera_name,
            timestamp=ts,
            ptz_state=ptz,
            image_path=Path(filename),
        )

        # Save YAML metadata
        repo.save(frame)

        # Copy image alongside YAML
        dest_dir = FRAMES_DIR / MAP_ID / camera_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_image = dest_dir / filename
        if not dest_image.exists():
            shutil.copy2(img_path, dest_image)

        created += 1
        print(f"  {frame.id} <- {filename}")

    print(f"\nSeeded {created} captured frames in {FRAMES_DIR}")


if __name__ == "__main__":
    main()
