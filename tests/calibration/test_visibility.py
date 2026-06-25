"""Tests for the lens-distortion visibility precheck (offline)."""

from __future__ import annotations

import pytest

from poc_homography.calibration.lens_distortion.visibility import (
    NoCalibratableViewError,
    VisibilityCriteria,
    assess_lines,
    require_calibratable_view,
)

W, H = 1920.0, 1080.0


def _diverse_lines(n: int, curved: int) -> list:
    """n lines spread across all 4 quadrants and 3+ orientations."""
    specs = []
    for i in range(n):
        # spread midpoints across quadrants
        qx = 400.0 if i % 2 == 0 else 1500.0
        qy = 300.0 if (i // 2) % 2 == 0 else 800.0
        # vary orientation across buckets (0,45,90,135 deg)
        ang = [(40, 0), (0, 40), (40, 40), (40, -40)][i % 4]
        start = (qx, qy)
        end = (qx + ang[0], qy + ang[1])
        specs.append((start, end, i < curved))
    return specs


def test_good_view_passes() -> None:
    report = assess_lines(_diverse_lines(12, 5), W, H)
    assert report.passed
    assert report.quadrants_covered == 4
    assert report.orientation_buckets >= 3
    require_calibratable_view(report)  # must not raise


def test_too_few_lines_fails_with_reason() -> None:
    report = assess_lines(_diverse_lines(3, 3), W, H)
    assert not report.passed
    assert any("too few lines" in r for r in report.reasons)


def test_no_curvature_fails() -> None:
    report = assess_lines(_diverse_lines(12, 0), W, H)
    assert not report.passed
    assert any("curved" in r for r in report.reasons)


def test_single_quadrant_fails() -> None:
    specs = [((100.0, 100.0), (140.0, 100.0 + 10 * i), True) for i in range(12)]
    report = assess_lines(specs, W, H)
    assert not report.passed
    assert any("quadrant" in r for r in report.reasons)


def test_require_raises_structured_error() -> None:
    report = assess_lines(_diverse_lines(2, 0), W, H)
    with pytest.raises(NoCalibratableViewError) as exc:
        require_calibratable_view(report, context="zoom=2.0")
    assert "zoom=2.0" in str(exc.value)
    assert exc.value.report is report


def test_criteria_are_tunable() -> None:
    lines = _diverse_lines(6, 3)
    assert not assess_lines(lines, W, H).passed  # default min_lines=8
    relaxed = VisibilityCriteria(min_lines=6)
    assert assess_lines(lines, W, H, relaxed).passed
