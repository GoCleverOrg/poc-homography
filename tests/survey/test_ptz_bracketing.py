"""Unit tests for the PTZ save/restore sweep bracketing (#342).

A minimal recording :class:`CameraDevice` double asserts the exact pre-sweep
``(pan, tilt, zoom)`` is re-issued via ``move_absolute`` exactly once, on both
the normal-completion and the exception/abort paths.
"""

from __future__ import annotations

import pytest

from poc_homography.domain.vo.ptz_state import PTZState
from poc_homography.survey.ptz_bracketing import preserve_ptz_position
from poc_homography.types import Degrees, FocusSteps, Unitless

_ORIGINAL = PTZState(
    pan_raw=Degrees(42.5),
    tilt_deg=Degrees(-17.0),
    zoom=Unitless(8.0),
    focus=FocusSteps(300),
)


class _RecordingCamera:
    """A :class:`CameraDevice` double recording PTZ reads and absolute moves.

    ``get_ptz_status`` answers a fixed original position; ``move_absolute``
    records every ``(pan, tilt, zoom)`` it is asked to issue so the test can
    assert the restore happened exactly once with the snapshotted values.
    """

    def __init__(self, original: PTZState) -> None:
        self._original = original
        self.status_reads = 0
        self.absolute_moves: list[tuple[float | None, float | None, float | None]] = []

    def get_ptz_status(self) -> PTZState:
        self.status_reads += 1
        return self._original

    def move_absolute(
        self,
        pan: float | None = None,
        tilt: float | None = None,
        zoom: float | None = None,
    ) -> PTZState:
        self.absolute_moves.append((pan, tilt, zoom))
        return self._original


def test_restores_original_position_on_normal_completion() -> None:
    camera = _RecordingCamera(_ORIGINAL)

    with preserve_ptz_position(camera) as original:  # type: ignore[arg-type]
        assert original == _ORIGINAL
        # Simulate a sweep moving the camera elsewhere.
        camera.move_absolute(pan=180.0, tilt=-80.0, zoom=1.0)

    # Last move is the restore; the original (pan, tilt, zoom) is re-issued once.
    assert camera.status_reads == 1
    assert camera.absolute_moves[-1] == (
        _ORIGINAL.pan_raw,
        _ORIGINAL.tilt_deg,
        _ORIGINAL.zoom,
    )
    assert camera.absolute_moves.count((_ORIGINAL.pan_raw, _ORIGINAL.tilt_deg, _ORIGINAL.zoom)) == 1


def test_restores_original_position_on_exception() -> None:
    camera = _RecordingCamera(_ORIGINAL)

    class _SweepAborted(RuntimeError):
        """Stand-in for a mid-sweep failure or operator abort."""

    with pytest.raises(_SweepAborted):
        with preserve_ptz_position(camera):  # type: ignore[arg-type]
            raise _SweepAborted("aborted mid-sweep")

    # Restore still ran exactly once despite the propagating exception.
    assert camera.absolute_moves == [(_ORIGINAL.pan_raw, _ORIGINAL.tilt_deg, _ORIGINAL.zoom)]


def test_restore_failure_does_not_mask_sweep_error() -> None:
    class _SweepAborted(RuntimeError):
        """Stand-in for a mid-sweep failure or operator abort."""

    class _RestoreFailingCamera(_RecordingCamera):
        """A camera whose restore move also fails (e.g. lost connection)."""

        def move_absolute(
            self,
            pan: float | None = None,
            tilt: float | None = None,
            zoom: float | None = None,
        ) -> PTZState:
            raise ConnectionError("camera unreachable during restore")

    camera = _RestoreFailingCamera(_ORIGINAL)

    # The original sweep error wins; the restore failure is logged, not raised.
    with pytest.raises(_SweepAborted):
        with preserve_ptz_position(camera):  # type: ignore[arg-type]
            raise _SweepAborted("aborted mid-sweep")


def test_restore_failure_propagates_on_normal_completion() -> None:
    class _RestoreFailingCamera(_RecordingCamera):
        def move_absolute(
            self,
            pan: float | None = None,
            tilt: float | None = None,
            zoom: float | None = None,
        ) -> PTZState:
            raise ConnectionError("camera unreachable during restore")

    camera = _RestoreFailingCamera(_ORIGINAL)

    # No sweep error to preserve, so a restore failure surfaces to the caller.
    with pytest.raises(ConnectionError):
        with preserve_ptz_position(camera):  # type: ignore[arg-type]
            pass
