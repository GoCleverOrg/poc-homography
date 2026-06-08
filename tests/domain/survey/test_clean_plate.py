"""Tests for the #276 clean-plate additive domain layer.

Covers the new FrameRecord sub-VOs, legacy 1.0 load tolerance, stable pose ids,
PoseCatalog round-trip, and SurveyRun.pose_catalog_id.
"""

from __future__ import annotations

from tests.domain.survey.builders import (
    make_frame_record,
    make_pose,
    make_pose_catalog,
    make_survey_run,
)

from poc_homography.domain.entities.survey import (
    SUPPORTED_SURVEY_SCHEMA_VERSIONS,
    SURVEY_SCHEMA_VERSION,
)
from poc_homography.domain.entities.survey.frame_record import FrameRecord
from poc_homography.domain.entities.survey.pose_catalog import PoseCatalog
from poc_homography.domain.entities.survey.survey_run import SurveyRun
from poc_homography.survey.planner.poses import Pose, canonical_pose_key


class TestSchemaVersion:
    def test_current_is_one_one(self) -> None:
        assert SURVEY_SCHEMA_VERSION == "1.1"

    def test_supported_set(self) -> None:
        assert frozenset({"1.0", "1.1"}) == SUPPORTED_SURVEY_SCHEMA_VERSIONS


class TestFrameRecordCleanPlateRoundTrip:
    def test_round_trip_with_all_new_sub_vos(self) -> None:
        record = make_frame_record(with_clean_plate=True)
        restored = FrameRecord.from_dict(record.to_dict())
        assert restored == record
        assert restored.to_dict() == record.to_dict()

    def test_new_sub_vos_populated(self) -> None:
        record = make_frame_record(with_clean_plate=True)
        restored = FrameRecord.from_dict(record.to_dict())
        assert restored.intrinsics == record.intrinsics
        assert restored.ground_homography == record.ground_homography
        assert restored.full_optics == record.full_optics
        assert restored.floor_mask_reference == record.floor_mask_reference
        assert restored.survey_context.pose_id == "p+0120.0_t-0015.0_z004.00"

    def test_optional_fields_default_none(self) -> None:
        record = make_frame_record(with_clean_plate=False)
        assert record.intrinsics is None
        assert record.ground_homography is None
        assert record.full_optics is None
        assert record.floor_mask_reference is None
        assert record.survey_context.pose_id is None
        restored = FrameRecord.from_dict(record.to_dict())
        assert restored == record
        assert restored.to_dict() == record.to_dict()


class TestFrameRecordLegacyLoad:
    def test_legacy_1_0_without_new_keys_loads(self) -> None:
        record = make_frame_record(with_clean_plate=True)
        payload = record.to_dict()
        # Simulate an old 1.0 record: bump version down and strip the new keys.
        payload["schema_version"] = "1.0"
        del payload["intrinsics"]
        del payload["ground_homography"]
        del payload["full_optics"]
        del payload["floor_mask_reference"]
        del payload["survey_context"]["pose_id"]

        restored = FrameRecord.from_dict(payload)

        assert restored.schema_version == "1.0"
        assert restored.intrinsics is None
        assert restored.ground_homography is None
        assert restored.full_optics is None
        assert restored.floor_mask_reference is None
        assert restored.survey_context.pose_id is None


class TestStablePoseId:
    def test_same_pose_same_id_across_runs(self) -> None:
        # Two independent computations simulate two runs.
        first = canonical_pose_key(120.0, -15.0, 4.0)
        second = canonical_pose_key(120.0, -15.0, 4.0)
        assert first == second

    def test_quantisation_collapses_tiny_differences(self) -> None:
        base = canonical_pose_key(120.0, -15.0, 4.0)
        within = canonical_pose_key(120.04, -14.96, 4.004)
        assert within == base

    def test_distinct_poses_differ(self) -> None:
        base = canonical_pose_key(120.0, -15.0, 4.0)
        other = canonical_pose_key(120.5, -15.0, 4.0)
        assert other != base

    def test_negative_zero_normalised(self) -> None:
        assert canonical_pose_key(-0.0, 0.0, 1.0) == canonical_pose_key(0.0, 0.0, 1.0)

    def test_catalog_assign_matches_canonical_key(self) -> None:
        catalog = make_pose_catalog()
        assert catalog.assign(120.0, -15.0, 4.0) == canonical_pose_key(120.0, -15.0, 4.0)


class TestPoseCatalog:
    def test_round_trip(self) -> None:
        catalog = make_pose_catalog()
        restored = PoseCatalog.from_dict(catalog.to_dict())
        assert restored == catalog
        assert restored.entries == catalog.entries
        assert restored.to_dict() == catalog.to_dict()

    def test_deterministic_key_ordering(self) -> None:
        catalog = make_pose_catalog()
        keys = list(catalog.to_dict()["entries"].keys())
        assert keys == sorted(keys)

    def test_id_is_catalog_id(self) -> None:
        catalog = make_pose_catalog(catalog_id="cat-xyz")
        assert catalog.id == "cat-xyz"

    def test_with_pose_is_immutable(self) -> None:
        catalog = PoseCatalog(catalog_id="c1", camera_id="cam01")
        updated = catalog.with_pose(10.0, 5.0, 1.0)
        assert catalog.entries == {}
        pose_id = canonical_pose_key(10.0, 5.0, 1.0)
        assert updated.entries[pose_id] == (10.0, 5.0, 1.0)

    def test_from_poses_order_independent(self) -> None:
        poses_a = [make_pose(pan=10.0), make_pose(pan=20.0)]
        poses_b = [make_pose(pan=20.0), make_pose(pan=10.0)]
        cat_a = PoseCatalog.from_poses("c", "cam01", poses_a)
        cat_b = PoseCatalog.from_poses("c", "cam01", poses_b)
        assert cat_a.to_dict() == cat_b.to_dict()

    def test_schema_version_stamped(self) -> None:
        catalog = make_pose_catalog()
        assert catalog.schema_version == SURVEY_SCHEMA_VERSION
        assert catalog.to_dict()["schema_version"] == SURVEY_SCHEMA_VERSION


class TestPosePoseId:
    def test_pose_round_trip_with_pose_id(self) -> None:
        pose = make_pose()
        with_id = Pose(
            pan=pose.pan,
            tilt=pose.tilt,
            zoom=pose.zoom,
            pose_id="p+0120.0_t-0015.0_z004.00",
        )
        restored = Pose.from_dict(with_id.to_dict())
        assert restored == with_id

    def test_pose_legacy_without_pose_id(self) -> None:
        payload = make_pose().to_dict()
        del payload["pose_id"]
        restored = Pose.from_dict(payload)
        assert restored.pose_id is None


class TestSurveyRunPoseCatalogId:
    def test_round_trip_with_pose_catalog_id(self) -> None:
        run = make_survey_run(pose_catalog_id="cat-0001")
        restored = SurveyRun.from_dict(run.to_dict())
        assert restored.pose_catalog_id == "cat-0001"
        assert restored == run
        assert restored.to_dict() == run.to_dict()

    def test_round_trip_without_pose_catalog_id(self) -> None:
        run = make_survey_run(pose_catalog_id=None)
        assert run.to_dict()["pose_catalog_id"] is None
        restored = SurveyRun.from_dict(run.to_dict())
        assert restored.pose_catalog_id is None

    def test_legacy_without_key_loads(self) -> None:
        payload = make_survey_run().to_dict()
        payload["schema_version"] = "1.0"
        del payload["pose_catalog_id"]
        restored = SurveyRun.from_dict(payload)
        assert restored.pose_catalog_id is None
        assert restored.schema_version == "1.0"
