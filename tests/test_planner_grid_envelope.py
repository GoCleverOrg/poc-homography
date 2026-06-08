"""Horizon-envelope constraint tests for the FOV grid generator."""

from __future__ import annotations

from poc_homography.domain.enums.camera_spec import CameraSpec
from poc_homography.domain.vo.tilt_envelope import TiltEnvelope
from poc_homography.survey.planner import fov_grid

SPEC = CameraSpec.HIKVISION_DS_2DF8425IX
PAN_RANGE = (0.0, 90.0)
# Tilt range spans well above and below the bound (positive tilt = down, so the
# low (negative) end points up at the sky).
TILT_RANGE = (-30.0, 0.0)
ZOOM_LEVELS = [2.0, 5.0]
OVERLAP = 0.2


def _envelope() -> TiltEnvelope:
    """A bound near -13° across the swept pan range (sky below it)."""
    return TiltEnvelope(
        bounds={0.0: -13.0, 90.0: -13.0},
        tilt_offset_deg=-31.0,
        vfov_deg=36.0,
        zoom=1.0,
    )


class TestEnvelopeConstrainedGrid:
    def test_none_envelope_is_byte_for_byte_identical(self) -> None:
        baseline = fov_grid(SPEC, PAN_RANGE, TILT_RANGE, ZOOM_LEVELS, OVERLAP)
        explicit_none = fov_grid(
            SPEC, PAN_RANGE, TILT_RANGE, ZOOM_LEVELS, OVERLAP, tilt_envelope=None
        )
        assert [p.to_dict() for p in explicit_none] == [p.to_dict() for p in baseline]

    def test_constrained_generates_strictly_fewer_poses(self) -> None:
        unconstrained = fov_grid(SPEC, PAN_RANGE, TILT_RANGE, ZOOM_LEVELS, OVERLAP)
        constrained = fov_grid(
            SPEC, PAN_RANGE, TILT_RANGE, ZOOM_LEVELS, OVERLAP, tilt_envelope=_envelope()
        )
        assert 0 < len(constrained) < len(unconstrained)

    def test_no_pose_points_above_the_bound(self) -> None:
        env = _envelope()
        constrained = fov_grid(SPEC, PAN_RANGE, TILT_RANGE, ZOOM_LEVELS, OVERLAP, tilt_envelope=env)
        for pose in constrained:
            assert pose.tilt >= env.upper_bound(pose.pan) - 1e-6

    def test_constrained_is_subset_of_unconstrained(self) -> None:
        unconstrained = {
            (p.pan, p.tilt, p.zoom)
            for p in fov_grid(SPEC, PAN_RANGE, TILT_RANGE, ZOOM_LEVELS, OVERLAP)
        }
        constrained = {
            (p.pan, p.tilt, p.zoom)
            for p in fov_grid(
                SPEC, PAN_RANGE, TILT_RANGE, ZOOM_LEVELS, OVERLAP, tilt_envelope=_envelope()
            )
        }
        assert constrained.issubset(unconstrained)
