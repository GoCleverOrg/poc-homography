"""Offline tests for the painted-line detector (synthetic floor markings)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from poc_homography.calibration.lens_distortion.models import PTZPosition
from poc_homography.calibration.lens_distortion.painted_line_detection import (
    HAS_XIMGPROC,
    PaintedLineConfig,
    PaintedLineDetector,
)
from poc_homography.types import Degrees


def _gray_canvas(width: int = 640, height: int = 480) -> np.ndarray:
    """A mid-gray BGR canvas (like an apron surface)."""
    return np.full((height, width, 3), 110, dtype=np.uint8)


def _draw_straight_white_line(
    img: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], thickness: int = 4
) -> None:
    cv2.line(img, p0, p1, (255, 255, 255), thickness)


def _draw_curved_white_line(
    img: np.ndarray, x0: int, x1: int, y: int, bow: int = 14, thickness: int = 4
) -> None:
    """Draw a slightly bowed bright line (carries a distortion-like signal)."""
    xs = np.linspace(x0, x1, 60)
    ys = y + bow * np.sin(np.linspace(0, np.pi, 60))
    pts = np.column_stack([xs, ys]).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=(255, 255, 255), thickness=thickness)


def _draw_yellow_line(
    img: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], thickness: int = 4
) -> None:
    # BGR yellow = (0, 255, 255).
    cv2.line(img, p0, p1, (0, 255, 255), thickness)


def test_detects_painted_lines_with_edge_pixels() -> None:
    img = _gray_canvas()
    _draw_straight_white_line(img, (60, 120), (560, 130))
    _draw_straight_white_line(img, (60, 240), (560, 250))
    _draw_yellow_line(img, (60, 360), (560, 370))

    detector = PaintedLineDetector(PaintedLineConfig(min_component_length=40.0))
    lines = detector.detect(img)

    assert len(lines) >= 2
    for line in lines:
        assert line.edge_pixels is not None
        assert len(line.edge_pixels) >= 3
        assert line.length > 40.0


def test_to_camera_line_populates_edge_pixels() -> None:
    img = _gray_canvas()
    _draw_straight_white_line(img, (40, 200), (600, 205))

    lines = PaintedLineDetector(PaintedLineConfig(min_component_length=40.0)).detect(img)
    assert lines

    ptz = PTZPosition(pan_deg=Degrees(0.0), tilt_deg=Degrees(0.0), zoom_factor=3.0)
    cam = lines[0].to_camera_line("l0", "img0", ptz)
    assert cam.edge_pixels is not None
    assert len(cam.edge_pixels) >= 3
    assert cam.ptz_position.zoom_factor == 3.0


def test_curved_line_has_edge_curvature() -> None:
    img = _gray_canvas()
    _draw_curved_white_line(img, 60, 580, 240, bow=18, thickness=4)

    lines = PaintedLineDetector(
        PaintedLineConfig(min_component_length=40.0, min_elongation=2.0)
    ).detect(img)
    assert lines

    ptz = PTZPosition(pan_deg=Degrees(0.0), tilt_deg=Degrees(0.0), zoom_factor=2.0)
    cams = [line.to_camera_line(f"l{i}", "img0", ptz) for i, line in enumerate(lines)]
    assert any(c.has_edge_curvature(tolerance=1.0) for c in cams)


def test_empty_image_yields_no_lines() -> None:
    img = _gray_canvas()
    assert PaintedLineDetector().detect(img) == []


@pytest.mark.skipif(not HAS_XIMGPROC, reason="cv2.ximgproc not available")
def test_thinning_backend_available() -> None:  # pragma: no cover - env dependent
    img = _gray_canvas()
    _draw_straight_white_line(img, (60, 120), (560, 130))
    assert PaintedLineDetector(PaintedLineConfig(min_component_length=40.0)).detect(img)
