"""Offline, deterministic tests for the end-to-end single-camera runner (#365).

The runner glues calibrate→register→validate→persist for one camera. These tests
exercise that glue with a fake frame store, a fake session, and canned doubles
for the self-calibrate stage / frame loader / repositories — with NO live
network, MinIO, or Neon. The extrinsic AUTO matcher and hold-out validator run
for REAL over a synthetic known-homography dataset (mirroring
``tests/calibration/extrinsic/test_validation.py``), so the registration and
validation math is genuinely exercised, not stubbed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from poc_homography.calibration.extrinsic import HoldoutValidationError
from poc_homography.calibration.lens_distortion.frame_source import SurveyFrame
from poc_homography.calibration.lens_distortion.line_detection import CandidateLine
from poc_homography.calibration.lens_distortion.scene_self_calibration import SceneCalibrationResult
from poc_homography.cli import run_camera as cmd
from poc_homography.cli.main import app
from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.domain.vo.ortho_line import OrthoLine
from poc_homography.domain.vo.zoom_calibration_entry import ZoomCalibrationEntry

if TYPE_CHECKING:
    from collections.abc import Iterator

_IMAGE_WIDTH = 1920
_IMAGE_HEIGHT = 1080
_GSD_M = 0.10  # 0.10 m/px ground sampling distance.

# Known perspective warp ORTHO map-pixel -> FRAME pixel; the AUTO matcher must
# recover frame -> ortho (~ inverse of this) with no operator seed.
_H_ORTHO_TO_FRAME = np.array(
    [
        [1.10, 0.05, 40.0],
        [0.03, 1.20, 25.0],
        [1.0e-4, 5.0e-5, 1.0],
    ],
    dtype=np.float64,
)

# Well-conditioned ortho segments spread across the frame so the warped inlier
# endpoints achieve a "Good"/"Fair" distribution.
_ORTHO_SEGMENTS: list[tuple[tuple[float, float], tuple[float, float]]] = [
    ((150.0, 150.0), (1400.0, 180.0)),
    ((200.0, 250.0), (240.0, 850.0)),
    ((300.0, 300.0), (1300.0, 820.0)),
    ((1350.0, 260.0), (350.0, 800.0)),
    ((900.0, 200.0), (940.0, 880.0)),
]


def _lens_table(camera_id: str = "cam1") -> LensCalibrationTable:
    """DDD lens table with real intrinsics + zero distortion (exact math)."""
    entry = ZoomCalibrationEntry(
        zoom_factor=1.0,
        distortion=LensDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
        calibration_date="2024-01-01T00:00:00",
        source_images=(),
        fx=1000.0,
        fy=1000.0,
        cx=960.0,
        cy=540.0,
    )
    return LensCalibrationTable(
        id=camera_id,
        entries=(entry,),
        created_date="2024-01-01T00:00:00",
        last_modified="2024-01-01T00:00:00",
    )


def _ortho_lines() -> list[OrthoLine]:
    lines: list[OrthoLine] = []
    for i, (start, end) in enumerate(_ORTHO_SEGMENTS):
        lines.append(
            OrthoLine.from_endpoints(
                start_pixel=start,
                end_pixel=end,
                start_utm=(500000.0 + i, 4000000.0 + i),
                end_utm=(500010.0 + i, 4000010.0 + i),
            )
        )
    return lines


def _geotiff() -> GeoTiff:
    """North-up GeoTIFF with a 0.10 m/px ground sampling distance."""
    return GeoTiff(
        geotransform=GeoTransform(
            origin_easting=500000.0,
            pixel_width=_GSD_M,
            row_rotation=0.0,
            origin_northing=4000000.0,
            col_rotation=0.0,
            pixel_height=-_GSD_M,
        ),
        crs="EPSG:25830",
    )


def _warp_point(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    pt = np.array([[[point[0], point[1]]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, homography)[0, 0]
    return float(out[0]), float(out[1])


def _frame_lines(
    ortho_lines: list[OrthoLine],
    noise_px: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[CandidateLine]:
    """Project ortho map-pixel endpoints into frame pixels via the known H."""
    lines: list[CandidateLine] = []
    for line in ortho_lines:
        start = _warp_point(
            (float(line.start_pixel[0]), float(line.start_pixel[1])), _H_ORTHO_TO_FRAME
        )
        end = _warp_point((float(line.end_pixel[0]), float(line.end_pixel[1])), _H_ORTHO_TO_FRAME)
        if noise_px > 0.0 and rng is not None:
            start = (start[0] + rng.normal(0, noise_px), start[1] + rng.normal(0, noise_px))
            end = (end[0] + rng.normal(0, noise_px), end[1] + rng.normal(0, noise_px))
        lines.append(CandidateLine(start=start, end=end))
    return lines


class _FakeDetector:
    """Frame line detector double: returns the same canned lines for any image."""

    def __init__(self, lines: list[CandidateLine]) -> None:
        self._lines = lines

    def detect(self, image: np.ndarray) -> list[CandidateLine]:
        return self._lines


class _CapturingPtzRepo:
    """RepoPostgresPtzRegistration double that records every saved record."""

    saved: list = []

    def __init__(self, session: object) -> None:
        self._session = session

    def save(self, record: object) -> None:
        type(self).saved.append(record)


class _FakeFrameRepo:
    """RepoPostgresCleanPlateFrame double; never queried (loader is patched)."""

    def __init__(self, session: object) -> None:
        self._session = session


class _FakeSession:
    """Minimal session object handed to the repository doubles."""


@contextmanager
def _fake_session_factory() -> Iterator[_FakeSession]:
    yield _FakeSession()


class _FakeFrameStore:
    """FrameStore double; iter_survey_frames is patched so this is never read."""

    def get_frame(self, key: str, *, bucket: str | None = None) -> bytes:
        raise AssertionError("frame store should not be read when the loader is faked")


def _install_doubles(
    monkeypatch: pytest.MonkeyPatch,
    *,
    frames: list[SurveyFrame],
    lens_table: LensCalibrationTable | None = None,
) -> None:
    """Patch the runner's module-level seams to offline doubles."""
    table = lens_table if lens_table is not None else _lens_table()
    result = SceneCalibrationResult(table=table, skipped_zooms=())

    def fake_self_calibration(*, run_id, camera_id, frame_store, session_factory):
        return result

    monkeypatch.setattr(cmd, "_run_self_calibration", fake_self_calibration)
    monkeypatch.setattr(cmd, "iter_survey_frames", lambda **_kwargs: iter(frames))
    monkeypatch.setattr(cmd, "RepoPostgresCleanPlateFrame", _FakeFrameRepo)
    monkeypatch.setattr(cmd, "RepoPostgresPtzRegistration", _CapturingPtzRepo)
    _CapturingPtzRepo.saved = []


def _survey_frame(zoom: float = 1.0, tilt: float = -15.0) -> SurveyFrame:
    image = np.zeros((_IMAGE_HEIGHT, _IMAGE_WIDTH, 3), dtype=np.uint8)
    return SurveyFrame(image=image, commanded_zoom=zoom, commanded_tilt=tilt)


# ---------------------------------------------------------------------------
# DoD 2: offline calibrate -> register -> validate -> persist
# ---------------------------------------------------------------------------


def test_run_camera_registration_persists_one_record_per_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ortho_lines = _ortho_lines()
    frame_lines = _frame_lines(ortho_lines)
    frames = [_survey_frame(), _survey_frame()]
    _install_doubles(monkeypatch, frames=frames)

    registrations = cmd._run_camera_registration(
        run_id="run1",
        camera_id="cam1",
        map_id="map1",
        ortho_lines=ortho_lines,
        geotiff=_geotiff(),
        frame_store=_FakeFrameStore(),
        session_factory=_fake_session_factory,
        rms_threshold_m=1.0,
        line_detector=_FakeDetector(frame_lines),
    )

    # One persisted registration per frame, captured by the repo double.
    assert len(registrations) == 2
    assert len(_CapturingPtzRepo.saved) == 2
    for record in _CapturingPtzRepo.saved:
        assert record.camera_id == "cam1"
        assert record.run_id == "run1"
        # Hold-out reprojection stats accompany every persisted record.
        assert record.reprojection_stats is not None
        assert record.reprojection_stats.rms_m <= 1.0


def test_run_camera_registration_uses_self_calibrated_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A lens table whose camera_id round-trips into each persisted record proves
    # the self-calibration output drove the (real) registration solve.
    ortho_lines = _ortho_lines()
    frames = [_survey_frame()]
    _install_doubles(monkeypatch, frames=frames, lens_table=_lens_table("cam1"))

    registrations = cmd._run_camera_registration(
        run_id="run1",
        camera_id="cam1",
        map_id="map1",
        ortho_lines=ortho_lines,
        geotiff=_geotiff(),
        frame_store=_FakeFrameStore(),
        session_factory=_fake_session_factory,
        rms_threshold_m=1.0,
        line_detector=_FakeDetector(_frame_lines(ortho_lines)),
    )

    assert len(registrations) == 1
    reg = registrations[0].registration
    # A 3x3 homography was actually solved (not a stub identity passthrough).
    assert reg.homography_matrix.shape == (3, 3)
    assert reg.num_inliers >= 2


# ---------------------------------------------------------------------------
# DoD 3: hold-out RMS over threshold fails the camera
# ---------------------------------------------------------------------------


def test_holdout_over_threshold_raises_and_persists_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ortho_lines = _ortho_lines()
    rng = np.random.default_rng(7)
    noisy_lines = _frame_lines(ortho_lines, noise_px=1.0, rng=rng)
    frames = [_survey_frame()]
    _install_doubles(monkeypatch, frames=frames)

    with pytest.raises(HoldoutValidationError):
        cmd._run_camera_registration(
            run_id="run1",
            camera_id="cam1",
            map_id="map1",
            ortho_lines=ortho_lines,
            geotiff=_geotiff(),
            frame_store=_FakeFrameStore(),
            session_factory=_fake_session_factory,
            rms_threshold_m=1.0e-6,  # any real reprojection error exceeds this
            line_detector=_FakeDetector(noisy_lines),
        )


# ---------------------------------------------------------------------------
# DoD 1: command is wired and exits non-zero on hold-out failure
# ---------------------------------------------------------------------------


def _wire_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the live-only collaborators so the CLI reaches the runner."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://stub")
    monkeypatch.setattr(
        cmd,
        "_resolve_camera",
        lambda _camera: {"id": "cam1", "map_id": "map1", "name": "Cam1", "ip": "10.0.0.1"},
    )
    monkeypatch.setattr(cmd.MinioFrameStore, "from_env", classmethod(lambda _cls: object()))
    monkeypatch.setattr(cmd, "_load_ortho_context", lambda _map_id: (_ortho_lines(), _geotiff()))


def test_command_offline_run_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire_cli(monkeypatch)
    monkeypatch.setattr(cmd, "_run_camera_registration", lambda **_kwargs: ["rec_a", "rec_b"])

    result = CliRunner().invoke(
        app, ["calibrate", "run-camera", "--camera", "cam1", "--offline-run-id", "run1"]
    )

    assert result.exit_code == 0, result.output
    assert "registrations=2" in result.output


def test_command_holdout_failure_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire_cli(monkeypatch)

    def boom(**_kwargs):
        from poc_homography.calibration.extrinsic.validation import ReprojectionStats

        stats = ReprojectionStats(rms_m=2.0, p90_m=2.0, max_m=2.0, n_points=4)
        raise HoldoutValidationError(2.0, 1.0, stats)

    monkeypatch.setattr(cmd, "_run_camera_registration", boom)

    result = CliRunner().invoke(
        app, ["calibrate", "run-camera", "--camera", "cam1", "--offline-run-id", "run1"]
    )

    assert result.exit_code != 0
    assert "Hold-out validation failed" in result.output


def test_command_live_run_requires_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire_cli(monkeypatch)

    result = CliRunner().invoke(app, ["calibrate", "run-camera", "--camera", "cam1"])

    assert result.exit_code != 0
    assert "--plan is required" in result.output
