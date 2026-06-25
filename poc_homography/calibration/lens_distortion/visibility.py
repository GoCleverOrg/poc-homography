"""Visibility precheck: can this scene support lens-distortion calibration?

Distortion is only observable when the frame shows enough straight-line
structure, spread across the image, in diverse orientations, with real edge
curvature (the distortion signal). This module assesses a candidate view and —
when it fails — raises a structured :class:`NoCalibratableViewError` naming the
criterion that failed, so the survey can reject the view and the product fails
with a clear message instead of silently producing a bad calibration.

Thresholds mirror ``docs/lens_calibration_requirements.md`` and are tunable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from poc_homography.calibration.lens_distortion.models import CameraLine

# A single line for assessment: (start_xy, end_xy, has_edge_curvature).
LineSpec = tuple[tuple[float, float], tuple[float, float], bool]


@dataclass(frozen=True)
class VisibilityCriteria:
    """Tunable gate for a calibratable view.

    Attributes:
        min_lines: Minimum detected lines.
        min_curved_lines: Minimum lines carrying edge curvature (distortion signal).
        min_quadrants: Minimum image quadrants that must contain a line.
        min_orientations: Minimum distinct 30-degree orientation buckets.
    """

    min_lines: int = 8
    min_curved_lines: int = 3
    min_quadrants: int = 2
    min_orientations: int = 2


@dataclass(frozen=True)
class VisibilityReport:
    """Outcome of a visibility assessment.

    Attributes:
        passed: Whether all criteria were met.
        num_lines: Total lines assessed.
        num_curved_lines: Lines with edge curvature.
        quadrants_covered: Distinct image quadrants with at least one line.
        orientation_buckets: Distinct 30-degree orientation buckets covered.
        reasons: One human-readable string per failed criterion (empty if passed).
    """

    passed: bool
    num_lines: int
    num_curved_lines: int
    quadrants_covered: int
    orientation_buckets: int
    reasons: tuple[str, ...]

    def score(self) -> float:
        """A coarse desirability score for ranking competing views."""
        return (
            self.num_lines
            + 2.0 * self.num_curved_lines
            + 3.0 * self.quadrants_covered
            + 2.0 * self.orientation_buckets
        )


class NoCalibratableViewError(RuntimeError):
    """Raised when no view meets the visibility criteria for calibration."""

    def __init__(self, report: VisibilityReport, *, context: str = "") -> None:
        self.report = report
        where = f" ({context})" if context else ""
        detail = "; ".join(report.reasons) or "unknown"
        super().__init__(f"No calibratable view{where}: {detail}")


def _quadrant(mid: tuple[float, float], width: float, height: float) -> int:
    cx, cy = width / 2.0, height / 2.0
    return (1 if mid[0] >= cx else 0) + (2 if mid[1] >= cy else 0)


def _orientation_bucket(start: tuple[float, float], end: tuple[float, float]) -> int:
    angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180.0
    return int(angle // 30.0)


def assess_lines(
    lines: Sequence[LineSpec],
    image_width: float,
    image_height: float,
    criteria: VisibilityCriteria | None = None,
) -> VisibilityReport:
    """Assess whether a set of detected lines supports calibration."""
    crit = criteria or VisibilityCriteria()
    num_lines = len(lines)
    num_curved = sum(1 for _, _, curved in lines if curved)
    quadrants = {
        _quadrant(((s[0] + e[0]) / 2.0, (s[1] + e[1]) / 2.0), image_width, image_height)
        for s, e, _ in lines
    }
    buckets = {_orientation_bucket(s, e) for s, e, _ in lines}

    reasons: list[str] = []
    if num_lines < crit.min_lines:
        reasons.append(f"too few lines ({num_lines} < {crit.min_lines})")
    if num_curved < crit.min_curved_lines:
        reasons.append(
            f"too few curved lines carrying distortion signal "
            f"({num_curved} < {crit.min_curved_lines})"
        )
    if len(quadrants) < crit.min_quadrants:
        reasons.append(f"insufficient quadrant coverage ({len(quadrants)} < {crit.min_quadrants})")
    if len(buckets) < crit.min_orientations:
        reasons.append(
            f"insufficient orientation diversity ({len(buckets)} < {crit.min_orientations})"
        )

    return VisibilityReport(
        passed=not reasons,
        num_lines=num_lines,
        num_curved_lines=num_curved,
        quadrants_covered=len(quadrants),
        orientation_buckets=len(buckets),
        reasons=tuple(reasons),
    )


def assess_camera_lines(
    lines: Sequence[CameraLine],
    image_width: float,
    image_height: float,
    criteria: VisibilityCriteria | None = None,
) -> VisibilityReport:
    """Assess :class:`CameraLine` objects (uses endpoints + edge curvature)."""
    specs: list[LineSpec] = [
        (line.start_pixel, line.end_pixel, line.has_edge_curvature()) for line in lines
    ]
    return assess_lines(specs, image_width, image_height, criteria)


def require_calibratable_view(report: VisibilityReport, *, context: str = "") -> None:
    """Raise :class:`NoCalibratableViewError` if the report did not pass."""
    if not report.passed:
        raise NoCalibratableViewError(report, context=context)
