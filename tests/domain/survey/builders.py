"""Builders for fully-populated survey entities used across the survey tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from poc_homography.domain.entities.survey.frame_record import (
    CameraIdentity,
    CaptureIdentity,
    CommandedState,
    FloorMaskReference,
    FrameRecord,
    FullOptics,
    GroundHomography,
    ImageData,
    ImagePipelineState,
    Intrinsics,
    MovementContext,
    ReportedState,
    SurveyContext,
)
from poc_homography.domain.entities.survey.pose_catalog import PoseCatalog
from poc_homography.domain.entities.survey.survey_run import SurveyRun
from poc_homography.domain.entities.survey.video_burst_record import (
    FrameRef,
    VideoBurstRecord,
)
from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.domain.enums.survey_run_status import SurveyRunStatus
from poc_homography.survey.planner.poses import Pose
from poc_homography.types import (
    FPS,
    Degrees,
    Meters,
    Millimeters,
    Pixels,
    Seconds,
    Unitless,
)

_TS = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def make_camera_identity(camera_id: str = "cam01") -> CameraIdentity:
    """Build a :class:`CameraIdentity`."""
    return CameraIdentity(
        camera_id=camera_id,
        brand="Hikvision",
        model="DS-2DF8425IX-AELW",
        serial="SN123456",
        firmware="V5.7.3",
        channel_id="1",
        stream_id="101",
    )


def make_capture_identity(
    *,
    capture_id: str = "cap-0001",
    run_id: str = "run-0001",
    phase: SurveyPhase = SurveyPhase.MAIN_SURVEY,
    burst_id: str | None = None,
    frame_index: int = 0,
) -> CaptureIdentity:
    """Build a :class:`CaptureIdentity`."""
    return CaptureIdentity(
        capture_id=capture_id,
        run_id=run_id,
        phase=phase,
        burst_id=burst_id,
        frame_index=frame_index,
        timestamp_before_move=_TS,
        timestamp_after_move=_TS,
        timestamp_at_capture=_TS,
    )


def make_commanded_state(*, commanded_focus: int | None = 512) -> CommandedState:
    """Build a :class:`CommandedState`."""
    return CommandedState(
        commanded_pan=Degrees(120.0),
        commanded_tilt=Degrees(-15.0),
        commanded_zoom=Unitless(4.0),
        commanded_focus=commanded_focus,
    )


def make_reported_state(
    *,
    reported_zoom: float = 4.0,
    reported_azimuth: float | None = 121.0,
    reported_focal_length_mm: float | None = 23.6,
    reported_focus: int | None = 510,
) -> ReportedState:
    """Build a :class:`ReportedState`."""
    return ReportedState(
        reported_pan=Degrees(120.1),
        reported_azimuth=Degrees(reported_azimuth) if reported_azimuth is not None else None,
        reported_tilt=Degrees(-15.1),
        reported_elevation=Degrees(15.1),
        reported_zoom=Unitless(reported_zoom),
        reported_focal_length_mm=(
            Millimeters(reported_focal_length_mm) if reported_focal_length_mm is not None else None
        ),
        reported_focus=reported_focus,
        ptz_settled=True,
    )


def make_movement_context() -> MovementContext:
    """Build a :class:`MovementContext`."""
    return MovementContext(
        prev_pan=Degrees(110.0),
        prev_tilt=Degrees(-10.0),
        prev_zoom=Unitless(2.0),
        direction_pan="cw",
        direction_tilt="down",
        direction_zoom="tele",
        settling_delay_s=Seconds(1.5),
        is_repeatability_sequence=False,
    )


def make_pipeline_state() -> ImagePipelineState:
    """Build an :class:`ImagePipelineState`."""
    return ImagePipelineState(
        resolution_width=Pixels(2560),
        resolution_height=Pixels(1440),
        codec="H.264",
        profile="Main",
        fps=FPS(25.0),
        eis_enabled=True,
        eptz_enabled=False,
        digital_zoom=Unitless(1.0),
        digital_zoom_limit=Unitless(16.0),
        mirror=False,
        flip=False,
        corridor_mode=False,
        day_night_mode="auto",
        crop_enabled=False,
        stabilization_enabled=True,
        exposure_mode="auto",
        focus_mode="SEMIAUTOMATIC",
    )


def make_image_data(image_path: str = "frames/cap-0001.jpg") -> ImageData:
    """Build an :class:`ImageData`."""
    return ImageData(
        image_path=Path(image_path),
        checksum="a" * 64,
        width=Pixels(2560),
        height=Pixels(1440),
        capture_format="jpeg",
    )


def make_intrinsics() -> Intrinsics:
    """Build an :class:`Intrinsics` with a cached K matrix."""
    return Intrinsics(
        zoom=Unitless(4.0),
        image_width=Pixels(2560),
        image_height=Pixels(1440),
        sensor_width_mm=Millimeters(6.78),
        base_focal_length_mm=Millimeters(5.9),
        k_matrix=[[2200.0, 0.0, 1280.0], [0.0, 2200.0, 720.0], [0.0, 0.0, 1.0]],
    )


def make_ground_homography() -> GroundHomography:
    """Build a :class:`GroundHomography` with a cached H matrix."""
    return GroundHomography(
        camera_height_m=Meters(8.5),
        pan_deg=Degrees(120.0),
        tilt_deg=Degrees(-15.0),
        roll_deg=Degrees(0.0),
        pixels_per_meter=12.5,
        map_origin=(100.0, 200.0),
        h_matrix=[[1.0, 0.0, 3.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
    )


def make_full_optics() -> FullOptics:
    """Build a :class:`FullOptics`."""
    return FullOptics(
        exposure_type="auto",
        shutter="1/1000",
        gain=6.0,
        iris=42,
        white_balance="auto",
        focus=510,
    )


def make_floor_mask_reference() -> FloorMaskReference:
    """Build a :class:`FloorMaskReference`."""
    return FloorMaskReference(mask_ref="masks/cap-0001.png", checksum="b" * 64)


def make_frame_record(
    *,
    capture_id: str = "cap-0001",
    run_id: str = "run-0001",
    camera_id: str = "cam01",
    phase: SurveyPhase = SurveyPhase.MAIN_SURVEY,
    burst_id: str | None = None,
    frame_index: int = 0,
    reported_zoom: float = 4.0,
    image_path: str = "frames/cap-0001.jpg",
    with_clean_plate: bool = False,
) -> FrameRecord:
    """Build a fully-populated :class:`FrameRecord`.

    When ``with_clean_plate`` is set, the four #276 optional sub-VOs plus
    ``survey_context.pose_id`` are populated.
    """
    return FrameRecord(
        camera=make_camera_identity(camera_id=camera_id),
        capture=make_capture_identity(
            capture_id=capture_id,
            run_id=run_id,
            phase=phase,
            burst_id=burst_id,
            frame_index=frame_index,
        ),
        commanded=make_commanded_state(),
        reported=make_reported_state(reported_zoom=reported_zoom),
        movement=make_movement_context(),
        pipeline=make_pipeline_state(),
        image_data=make_image_data(image_path=image_path),
        survey_context=(
            SurveyContext(pose_id="p+0120.0_t-0015.0_z004.00")
            if with_clean_plate
            else SurveyContext()
        ),
        intrinsics=make_intrinsics() if with_clean_plate else None,
        ground_homography=make_ground_homography() if with_clean_plate else None,
        full_optics=make_full_optics() if with_clean_plate else None,
        floor_mask_reference=make_floor_mask_reference() if with_clean_plate else None,
    )


def make_pose(
    *,
    pan: float = 120.0,
    tilt: float = -15.0,
    zoom: float = 4.0,
) -> Pose:
    """Build a :class:`Pose`."""
    return Pose(pan=Degrees(pan), tilt=Degrees(tilt), zoom=Unitless(zoom))


def make_pose_catalog(catalog_id: str = "cat-0001", camera_id: str = "cam01") -> PoseCatalog:
    """Build a :class:`PoseCatalog` from a few poses."""
    return PoseCatalog.from_poses(
        catalog_id,
        camera_id,
        [
            make_pose(pan=120.0, tilt=-15.0, zoom=4.0),
            make_pose(pan=90.0, tilt=-10.0, zoom=2.0),
        ],
    )


def make_frame_ref(frame_index: int = 0) -> FrameRef:
    """Build a :class:`FrameRef`."""
    return FrameRef(
        capture_id=f"cap-{frame_index:04d}",
        frame_index=frame_index,
        timestamp_at_capture=_TS,
        image_path=Path(f"frames/cap-{frame_index:04d}.jpg"),
    )


def make_video_burst_record(*, n_frames: int = 3) -> VideoBurstRecord:
    """Build a :class:`VideoBurstRecord` with ``n_frames`` frame refs."""
    return VideoBurstRecord(
        burst_id="burst-0001",
        run_id="run-0001",
        camera_id="cam01",
        phase=SurveyPhase.STATIC_JITTER,
        segment_path=Path("bursts/burst-0001.mp4"),
        duration_s=Seconds(10.0),
        fps=FPS(25.0),
        codec="H.265",
        commanded_state=make_commanded_state(),
        frame_refs=tuple(make_frame_ref(i) for i in range(n_frames)),
    )


def make_survey_run(
    *,
    run_id: str = "run-0001",
    camera_id: str = "cam01",
    status: SurveyRunStatus = SurveyRunStatus.RUNNING,
    finished: bool = False,
    pose_catalog_id: str | None = None,
) -> SurveyRun:
    """Build a :class:`SurveyRun`."""
    return SurveyRun(
        run_id=run_id,
        camera_id=camera_id,
        phases=frozenset({SurveyPhase.MAIN_SURVEY, SurveyPhase.VALIDATION}),
        started_at=_TS,
        finished_at=_TS if finished else None,
        status=status,
        pose_catalog_id=pose_catalog_id,
    )
