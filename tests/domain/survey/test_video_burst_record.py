"""Round-trip and addressability tests for VideoBurstRecord."""

from __future__ import annotations

from tests.domain.survey.builders import make_video_burst_record

from poc_homography.domain.entities.survey.video_burst_record import VideoBurstRecord


class TestVideoBurstRecordRoundTrip:
    def test_round_trip(self) -> None:
        burst = make_video_burst_record(n_frames=4)
        restored = VideoBurstRecord.from_dict(burst.to_dict())
        assert restored == burst
        assert restored.to_dict() == burst.to_dict()

    def test_id_is_burst_id(self) -> None:
        burst = make_video_burst_record()
        assert burst.id == "burst-0001"

    def test_frame_refs_preserved(self) -> None:
        burst = make_video_burst_record(n_frames=3)
        restored = VideoBurstRecord.from_dict(burst.to_dict())
        assert len(restored.frame_refs) == 3


class TestVideoBurstRecordAddressability:
    def test_frame_by_index_returns_correct_ref(self) -> None:
        burst = make_video_burst_record(n_frames=5)
        ref = burst.frame_by_index(3)
        assert ref is not None
        assert ref.frame_index == 3
        assert ref.capture_id == "cap-0003"

    def test_frame_by_index_missing_returns_none(self) -> None:
        burst = make_video_burst_record(n_frames=2)
        assert burst.frame_by_index(99) is None

    def test_addressable_after_round_trip(self) -> None:
        burst = VideoBurstRecord.from_dict(make_video_burst_record(n_frames=4).to_dict())
        ref = burst.frame_by_index(2)
        assert ref is not None
        assert ref.frame_index == 2
