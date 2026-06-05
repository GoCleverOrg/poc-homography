"""Per-camera lock registry tests for the survey planner."""

import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.survey.planner import CameraLockRegistry


class TestCameraLockRegistry:
    def test_same_camera_returns_same_lock(self):
        reg = CameraLockRegistry()
        assert reg.get("A") is reg.get("A")

    def test_different_cameras_distinct_locks(self):
        reg = CameraLockRegistry()
        assert reg.get("A") is not reg.get("B")

    def test_double_acquire_same_camera_fails(self):
        reg = CameraLockRegistry()
        lock = reg.get("A")
        assert lock.acquire(blocking=False) is True
        try:
            assert lock.acquire(blocking=False) is False
        finally:
            lock.release()

    def test_camera_a_does_not_block_camera_b(self):
        reg = CameraLockRegistry()
        # Hold camera A.
        lock_a = reg.get("A")
        assert lock_a.acquire(blocking=False) is True

        acquired_b = threading.Event()

        def grab_b() -> None:
            lock_b = reg.get("B")
            if lock_b.acquire(blocking=True):
                acquired_b.set()
                lock_b.release()

        thread = threading.Thread(target=grab_b)
        thread.start()
        try:
            assert acquired_b.wait(timeout=2.0), "camera B blocked by camera A lock"
        finally:
            thread.join(timeout=2.0)
            lock_a.release()

    def test_acquire_context_manager(self):
        reg = CameraLockRegistry()
        with reg.acquire("A", blocking=False) as got:
            assert got is True
            # nested non-blocking acquire of same camera must fail
            with reg.acquire("A", blocking=False) as got2:
                assert got2 is False
        # released after context; can acquire again
        with reg.acquire("A", blocking=False) as got3:
            assert got3 is True
