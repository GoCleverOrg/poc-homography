"""Lifecycle status enum for a :class:`SurveyRun`."""

from __future__ import annotations

from enum import Enum


class SurveyRunStatus(str, Enum):
    """Lifecycle status of a survey run.

    Members subclass :class:`str` so their ``.value`` is a stable, plain
    string safe for YAML/JSON serialisation. The ``status`` field enables a
    planner to resume interrupted runs.

    Members:
        PENDING: Run created but not yet started.
        RUNNING: Run is actively capturing.
        COMPLETED: Run finished successfully.
        FAILED: Run terminated due to an error.
        ABORTED: Run was deliberately aborted.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
