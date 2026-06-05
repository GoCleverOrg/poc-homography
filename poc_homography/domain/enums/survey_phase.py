"""Survey phase enum for the multi-phase, multi-camera survey epic."""

from __future__ import annotations

from enum import Enum


class SurveyPhase(str, Enum):
    """The nine phases of a multi-camera survey run.

    Members subclass :class:`str` so their ``.value`` is a stable, plain
    string safe for YAML/JSON serialisation. Always serialise via ``.value``
    and reconstruct via ``SurveyPhase(value)``.

    Members:
        CAMERA_INVENTORY: Phase 1 — enumerate cameras and capabilities.
        PTZ_CHARACTERIZATION: Phase 2 — characterise pan/tilt behaviour.
        ZOOM_CHARACTERIZATION: Phase 3 — characterise zoom/optics behaviour.
        DENSE_NADIR: Phase 4 — dense nadir-pointing capture.
        MAIN_SURVEY: Phase 5 — the main survey sweep.
        CROSS_ZOOM: Phase 6 — cross-zoom consistency capture.
        REPEATABILITY: Phase 7 — repeatability sequence.
        STATIC_JITTER: Phase 8 — static jitter video bursts.
        VALIDATION: Phase 9 — validation / repeatability re-check.
    """

    CAMERA_INVENTORY = "camera_inventory"
    PTZ_CHARACTERIZATION = "ptz_characterization"
    ZOOM_CHARACTERIZATION = "zoom_characterization"
    DENSE_NADIR = "dense_nadir"
    MAIN_SURVEY = "main_survey"
    CROSS_ZOOM = "cross_zoom"
    REPEATABILITY = "repeatability"
    STATIC_JITTER = "static_jitter"
    VALIDATION = "validation"
