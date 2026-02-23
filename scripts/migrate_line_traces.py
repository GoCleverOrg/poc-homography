"""One-shot migration: seed CalibrationLineTraceSet repo from existing test YAMLs.

Reads each *_line_annotations.yaml from tests/homography/test_data/,
extracts test cases that contain N-point line traces, creates
CalibrationLineTraceSet entities, and saves them to data/calibration_line_traces/.

Usage:
    uv run python scripts/migrate_line_traces.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

from poc_homography.domain.entities.calibration_line_trace_set import (
    CalibrationLineTraceSet,
)
from poc_homography.domain.vo.line_trace import LineTrace
from poc_homography.domain.vo.ptz_state import PTZState
from poc_homography.infrastructure.repositories.repo_yaml_calibration_line_trace_set import (
    RepoYamlCalibrationLineTraceSet,
)
from poc_homography.types import Degrees, Unitless

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "homography" / "test_data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "calibration_line_traces"


def migrate() -> None:
    repo = RepoYamlCalibrationLineTraceSet(OUTPUT_DIR)

    yaml_files = sorted(TEST_DATA_DIR.glob("*_line_annotations.yaml"))
    if not yaml_files:
        print(f"No *_line_annotations.yaml files found in {TEST_DATA_DIR}")
        return

    total = 0
    for yaml_path in yaml_files:
        print(f"Processing {yaml_path.name}...")
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "test_cases" not in data:
            print("  Skipping (no test_cases)")
            continue

        for tc in data["test_cases"]:
            # Only migrate test cases that have N-point traces
            line_annotations = tc.get("line_annotations", [])
            traces = []
            for la in line_annotations:
                points = la.get("points")
                if not points or len(points) < 2:
                    continue
                traces.append(
                    LineTrace(
                        line_id=la["line_id"],
                        points=tuple(tuple(p) for p in points),
                    )
                )

            if not traces:
                print(f"  Skipping test case '{tc.get('name')}' (no N-point traces)")
                continue

            # Map camera_status fields to PTZState fields
            cs = tc.get("camera_status", {})
            camera_pose = PTZState(
                pan_raw=Degrees(float(cs.get("pan", 0.0))),
                tilt_deg=Degrees(float(cs.get("tilt", 0.0))),
                zoom=Unitless(float(cs.get("zoom", 1.0))),
            )

            entity = CalibrationLineTraceSet(
                name=tc["name"],
                image=tc.get("image", ""),
                camera_pose=camera_pose,
                line_traces=tuple(traces),
            )

            repo.save(entity)
            total += 1
            print(f"  Saved entity '{entity.name}' ({len(traces)} traces)")

    print(f"\nDone! {total} entities saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    migrate()
