"""Per-camera lock registry for survey concurrency control.

A :class:`CameraLockRegistry` hands out one :class:`threading.Lock` per camera
id, created on first access. Locking camera A must never block camera B.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


class CameraLockRegistry:
    """Registry of per-camera locks, created on first access.

    The registry guards its internal mapping with a meta-lock so concurrent
    ``get`` calls are safe. Each camera id maps to a stable
    :class:`threading.Lock` reused on subsequent calls.
    """

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()

    def get(self, camera_id: str) -> threading.Lock:
        """Return the lock for ``camera_id``, creating it on first access.

        Args:
            camera_id: The camera identifier.

        Returns:
            The stable :class:`threading.Lock` for that camera.
        """
        with self._meta_lock:
            lock = self._locks.get(camera_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[camera_id] = lock
            return lock

    @contextmanager
    def acquire(self, camera_id: str, *, blocking: bool = False) -> Iterator[bool]:
        """Context manager acquiring a camera's lock.

        Args:
            camera_id: The camera identifier.
            blocking: Whether to block until the lock is available.

        Yields:
            ``True`` if the lock was acquired, ``False`` otherwise. The lock is
            released on exit only if it was acquired.
        """
        lock = self.get(camera_id)
        acquired = lock.acquire(blocking=blocking)
        try:
            yield acquired
        finally:
            if acquired:
                lock.release()
