"""Tests for the presentation DTOs that replaced raw-dict annotation returns (issue #296, L2).

These lock in the byte-identical legacy JSON shape produced by ``to_dict()`` —
in particular the line DTO's ``points`` key, which must appear ONLY when the
annotation carries polyline points.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add webapp to path so the presentation DTOs are importable.
PROJECT_ROOT = Path(__file__).parent.parent
WEBAPP_DIR = PROJECT_ROOT / "webapp"
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

from homography_web.view_models import FrameAnnotation, LineAnnotationView


def test_frame_annotation_to_dict_exact_shape() -> None:
    dto = FrameAnnotation(gcp_id="PS1", pixel_x=1.0, pixel_y=2.0)
    assert dto.to_dict() == {"gcp_id": "PS1", "pixel_x": 1.0, "pixel_y": 2.0}


def test_line_annotation_to_dict_omits_points_when_none() -> None:
    dto = LineAnnotationView(
        line_id="L1",
        start_pixel_x=1.0,
        start_pixel_y=2.0,
        end_pixel_x=3.0,
        end_pixel_y=4.0,
    )
    payload = dto.to_dict()
    assert "points" not in payload
    assert payload == {
        "line_id": "L1",
        "start_pixel_x": 1.0,
        "start_pixel_y": 2.0,
        "end_pixel_x": 3.0,
        "end_pixel_y": 4.0,
    }


def test_line_annotation_to_dict_includes_points_when_present() -> None:
    dto = LineAnnotationView(
        line_id="L2",
        start_pixel_x=1.0,
        start_pixel_y=2.0,
        end_pixel_x=3.0,
        end_pixel_y=4.0,
        points=[[5.0, 6.0], [7.0, 8.0]],
    )
    assert dto.to_dict()["points"] == [[5.0, 6.0], [7.0, 8.0]]
