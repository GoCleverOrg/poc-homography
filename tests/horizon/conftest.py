"""Shared fixtures and the live-calibration ground truth for horizon tests.

All values come from the live measurement on ``icozee-camptz-04`` at zoom 1.0×
recorded in issue #274. The seven frames live under ``tests/fixtures/horizon``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "horizon"

# DS-2DF8425IX optics (matches CameraSpec.HIKVISION_DS_2DF8425IX).
IMAGE_WIDTH = 2560
IMAGE_HEIGHT = 1440
SENSOR_WIDTH_MM = 6.78
BASE_FOCAL_LENGTH_MM = 5.9
ZOOM = 1.0

# Measured horizon position as a fraction of frame height from the top.
# (reported_tilt_deg, fraction). ``None`` fraction == horizon not in frame
# (100% ground). Source: issue #274 calibration table.
MEASURED_TABLE: list[tuple[float, float | None]] = [
    (-10.0, None),
    (-20.0, 0.20),
    (-30.0, 0.48),
    (-40.0, 0.78),
    (-48.0, 0.95),
]

# Fixture filename ↔ reported tilt (positive = down).
FIXTURE_TILTS: dict[str, float] = {
    "tilt_-10.jpg": -10.0,
    "tilt_+00.jpg": 0.0,
    "tilt_+10.jpg": 10.0,
    "up_-20.jpg": -20.0,
    "up_-30.jpg": -30.0,
    "up_-40.jpg": -40.0,
    "up_-48.jpg": -48.0,
}

# In-frame fixtures where the CV detector reliably finds the horizon.
IN_FRAME_FIXTURES: list[str] = ["up_-20.jpg", "up_-30.jpg", "up_-40.jpg"]
# Fixtures where the whole frame is ground (horizon above the top edge).
ALL_GROUND_FIXTURES: list[str] = ["tilt_-10.jpg", "tilt_+00.jpg", "tilt_+10.jpg"]


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    """Directory holding the seven committed calibration frames."""
    return FIXTURE_DIR
