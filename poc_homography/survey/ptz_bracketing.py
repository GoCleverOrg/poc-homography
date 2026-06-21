"""PTZ save/restore bracketing for capture sweeps (#342).

A survey sweep repositions a PTZ camera across many poses but leaves it parked
at the last commanded pose when it finishes. :func:`preserve_ptz_position`
brackets a sweep so the camera is returned to exactly where it started — on
normal completion *and* on exception or operator abort.

This is a thin orchestration concern: it composes the existing
:class:`~poc_homography.domain.protocols.camera_device.CameraDevice` surface
(``get_ptz_status`` to snapshot, ``move_absolute`` to restore) and deliberately
adds **no** new method to the Hikvision ISAPI adapter — the try/finally lives
here, not in the client.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from poc_homography.domain.protocols.camera_device import CameraDevice
    from poc_homography.domain.vo.ptz_state import PTZState


@contextmanager
def preserve_ptz_position(camera: CameraDevice) -> Iterator[PTZState]:
    """Snapshot the camera's PTZ position and restore it when the block exits.

    On entry the current ``(pan, tilt, zoom)`` is read via
    :meth:`CameraDevice.get_ptz_status`. The position is restored via a single
    :meth:`CameraDevice.move_absolute` call in a ``finally`` block, so it runs
    whether the wrapped sweep completes normally, raises, or is aborted.

    Args:
        camera: The PTZ device to bracket. Must satisfy :class:`CameraDevice`.

    Yields:
        The original :class:`PTZState` captured before the sweep, for callers
        that want to inspect or log the parked position.

    Raises:
        CameraControllerError: If reading the original position fails (the
            sweep never starts). Errors raised by the wrapped block propagate
            after the restore is attempted.
    """
    original = camera.get_ptz_status()
    try:
        yield original
    finally:
        camera.move_absolute(
            pan=original.pan_raw,
            tilt=original.tilt_deg,
            zoom=original.zoom,
        )
