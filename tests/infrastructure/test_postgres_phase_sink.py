"""Unit tests for :class:`PostgresPhaseSink` (mocked store + session)."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import TYPE_CHECKING

from tests.domain.survey.builders import make_frame_record

from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.infrastructure.clients.minio_frame_store import PutResult
from poc_homography.infrastructure.survey.postgres_phase_sink import PostgresPhaseSink

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from poc_homography.infrastructure.models.clean_plate_frame import CleanPlateFrameModel


class FakeStore:
    """Records uploads; returns a real sha256 like the production store."""

    bucket = "cleanplate-frames"

    def __init__(self) -> None:
        self.ensured = False
        self.uploads: list[tuple[bytes, str]] = []

    def ensure_bucket(self) -> None:
        self.ensured = True

    def put_frame(self, data: bytes, key: str, content_type: str = "image/jpeg") -> PutResult:
        self.uploads.append((data, key))
        return PutResult(self.bucket, key, hashlib.sha256(data).hexdigest())


class FakeSession:
    """Minimal SQLAlchemy-Session stand-in keyed by primary key."""

    def __init__(self) -> None:
        self.rows: dict[str, CleanPlateFrameModel] = {}
        self.flushes = 0

    def get(self, _model: type, pk: str) -> CleanPlateFrameModel | None:
        return self.rows.get(pk)

    def add(self, row: CleanPlateFrameModel) -> None:
        self.rows[row.id] = row

    def flush(self) -> None:
        self.flushes += 1


def _factory(session: FakeSession):
    @contextmanager
    def factory() -> Iterator[FakeSession]:
        yield session

    return factory


def _frame_with_image(tmp_path: Path):
    img = tmp_path / "cap-0001.jpg"
    data = b"\xff\xd8\xff floor pixels"
    img.write_bytes(data)
    record = make_frame_record(
        capture_id="cap-0001",
        run_id="run-42",
        camera_id="cam04",
        phase=SurveyPhase.MAIN_SURVEY,
        frame_index=2,
        image_path=str(img),
        with_clean_plate=True,
    )
    return record, data


def test_save_frame_uploads_and_upserts(tmp_path: Path) -> None:
    store = FakeStore()
    session = FakeSession()
    record, data = _frame_with_image(tmp_path)

    sink = PostgresPhaseSink(store, session_factory=_factory(session))  # type: ignore[arg-type]
    sink.save_frame(record)

    # image uploaded under the deterministic key
    assert store.ensured is True
    assert len(store.uploads) == 1
    up_data, up_key = store.uploads[0]
    assert up_data == data
    assert up_key == "run-42/main_survey/p+0120.0_t-0015.0_z004.00/cap-0001.jpg"

    # row upserted with the right projection columns + linkage
    row = session.rows["cap-0001"]
    assert row.run_id == "run-42"
    assert row.camera_id == "cam04"
    assert row.phase == "main_survey"
    assert row.pose_id == "p+0120.0_t-0015.0_z004.00"
    assert row.frame_index == 2
    assert row.minio_bucket == "cleanplate-frames"
    assert row.minio_object_key == up_key
    assert row.checksum_sha256 == hashlib.sha256(data).hexdigest()
    # lossless full record in JSONB
    assert row.record["capture"]["capture_id"] == "cap-0001"
    assert row.record["schema_version"]


def test_save_frame_is_idempotent(tmp_path: Path) -> None:
    store = FakeStore()
    session = FakeSession()
    record, _ = _frame_with_image(tmp_path)

    sink = PostgresPhaseSink(store, session_factory=_factory(session))  # type: ignore[arg-type]
    sink.save_frame(record)
    sink.save_frame(record)

    assert len(session.rows) == 1  # same id → updated, not duplicated
    assert len(store.uploads) == 2  # re-upload to the same key (overwrite)


class _RecordingFallback:
    def __init__(self) -> None:
        self.inventory: list[object] = []
        self.bursts: list[object] = []

    def save_inventory(self, record: object) -> None:
        self.inventory.append(record)

    def save_frame(self, record: object) -> None:  # pragma: no cover - unused
        raise AssertionError("frames must not go to the fallback")

    def save_burst(self, record: object) -> None:
        self.bursts.append(record)


def test_inventory_and_burst_forwarded_to_fallback(tmp_path: Path) -> None:
    store = FakeStore()
    session = FakeSession()
    fallback = _RecordingFallback()
    sink = PostgresPhaseSink(
        store,
        session_factory=_factory(session),  # type: ignore[arg-type]
        fallback=fallback,
    )

    sentinel_inv = object()
    sentinel_burst = object()
    sink.save_inventory(sentinel_inv)  # type: ignore[arg-type]
    sink.save_burst(sentinel_burst)  # type: ignore[arg-type]

    assert fallback.inventory == [sentinel_inv]
    assert fallback.bursts == [sentinel_burst]


def test_inventory_without_fallback_is_skipped_not_raised() -> None:
    class _Rec:
        id = "inv-1"

    sink = PostgresPhaseSink(FakeStore())  # type: ignore[arg-type]
    # no fallback → logged + skipped, must not raise
    sink.save_inventory(_Rec())  # type: ignore[arg-type]
