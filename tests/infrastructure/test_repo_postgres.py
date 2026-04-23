"""Comprehensive integration tests for all PostgreSQL repository implementations.

These tests exercise the same CRUD operations as the YAML repo tests
but against a real PostgreSQL database. Every test uses the db_session
fixture which rolls back after each test — no permanent data is written.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from poc_homography.domain.entities.annotation import Annotation
from poc_homography.domain.entities.calibration_line_trace_set import (
    CalibrationLineTraceSet,
)
from poc_homography.domain.entities.camera_calibration import CameraCalibration
from poc_homography.domain.entities.camera_config import CameraConfig
from poc_homography.domain.entities.captured_frame import CapturedFrame
from poc_homography.domain.entities.ground_control_point import GroundControlPoint
from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
from poc_homography.domain.entities.line import Line
from poc_homography.domain.entities.line_annotation import LineAnnotation
from poc_homography.domain.entities.map import Map
from poc_homography.domain.entities.tenant import Tenant
from poc_homography.domain.enums import CameraSpec
from poc_homography.domain.vo.credential import Credential
from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.domain.vo.line_trace import LineTrace
from poc_homography.domain.vo.map_point import MapPoint
from poc_homography.domain.vo.orientation import Orientation
from poc_homography.domain.vo.photo import Photo
from poc_homography.domain.vo.pixel_point import PixelPoint
from poc_homography.domain.vo.ptz_state import PTZState
from poc_homography.domain.vo.zoom_calibration_entry import ZoomCalibrationEntry
from poc_homography.infrastructure.repositories import (
    RepoPostgresAnnotation,
    RepoPostgresCalibrationLineTraceSet,
    RepoPostgresCameraCalibration,
    RepoPostgresCameraConfig,
    RepoPostgresCapturedFrame,
    RepoPostgresGroundControlPoint,
    RepoPostgresLensCalibrationTable,
    RepoPostgresLine,
    RepoPostgresLineAnnotation,
    RepoPostgresMap,
    RepoPostgresTenant,
)
from poc_homography.types import (
    Degrees,
    Easting,
    Meters,
    Northing,
    Pixels,
    PixelsFloat,
    Unitless,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# Entity construction helpers
# ---------------------------------------------------------------------------

def _uid(prefix: str = "") -> str:
    """Return a short unique id to avoid collisions between tests."""
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def _make_tenant(tenant_id: str | None = None) -> Tenant:
    tid = tenant_id or _uid("t_")
    return Tenant(
        id=tid,
        name=f"Tenant {tid}",
        description="test tenant",
        location_lat="39d38'25.72\"N",
        location_lon="0d13'48.63\"W",
    )


def _make_map(tenant_id: str, map_id: str | None = None) -> Map:
    mid = map_id or _uid("m_")
    return Map(
        id=mid,
        tenant_id=tenant_id,
        photo=Photo(path=Path("map.png"), width=Pixels(1000), height=Pixels(800)),
        geotiff=GeoTiff(
            geotransform=GeoTransform(
                origin_easting=Easting(500000.0),
                pixel_width=Meters(0.5),
                row_rotation=Unitless(0.0),
                origin_northing=Northing(4400000.0),
                col_rotation=Unitless(0.0),
                pixel_height=Meters(-0.5),
            ),
            crs="EPSG:25830",
        ),
    )


def _make_camera_config(
    tenant_id: str,
    map_id: str,
    name: str | None = None,
) -> CameraConfig:
    cam_name = name or _uid("cam_")
    return CameraConfig(
        id=f"{map_id}/{cam_name}",
        tenant_id=tenant_id,
        map_id=map_id,
        name=cam_name,
        spec=CameraSpec.HIKVISION_DS_2DF8425IX,
        credential=Credential(username="admin", password="secret"),
        ip_address="192.168.1.100",
    )


def _make_ptz() -> PTZState:
    return PTZState(pan_raw=Degrees(30.0), tilt_deg=Degrees(15.0), zoom=Unitless(2.0))


def _make_camera_calibration(camera_config_id: str) -> CameraCalibration:
    return CameraCalibration(
        id=camera_config_id,
        position=PixelPoint.create(500.0, 400.0),
        height=Meters(12.5),
        base_orientation=Orientation.create(
            yaw=Degrees(90.0), pitch=Degrees(-10.0), roll=Degrees(0.0)
        ),
        distortion=LensDistortion(
            k1=Unitless(-0.1),
            k2=Unitless(0.01),
            p1=Unitless(0.001),
            p2=Unitless(-0.001),
            k3=Unitless(0.0),
        ),
    )


def _make_captured_frame(
    map_id: str,
    camera_name: str = "cam01",
    ts: datetime | None = None,
) -> CapturedFrame:
    timestamp = ts or datetime(2024, 1, 15, 10, 30, 45)
    return CapturedFrame.create(
        map_id=map_id,
        camera_name=camera_name,
        timestamp=timestamp,
        ptz_state=_make_ptz(),
        image_path=Path("frame.jpg"),
    )


def _make_annotation(frame_id: str, gcp_name: str = "GCP1") -> Annotation:
    return Annotation(
        gcp_id=gcp_name,
        frame_id=frame_id,
        camera_pose=_make_ptz(),
        pixel=PixelPoint.create(123.4, 567.8),
    )


def _make_gcp(map_id: str, name: str = "GCP1") -> GroundControlPoint:
    return GroundControlPoint(
        name=name,
        map_point=MapPoint(
            map_id=map_id,
            pixel_point=PixelPoint.create(100.5, 200.5),
        ),
    )


def _make_line(map_id: str, name: str = "L1") -> Line:
    return Line(
        name=name,
        map_id=map_id,
        start=PixelPoint.create(10.0, 20.0),
        end=PixelPoint.create(300.0, 400.0),
    )


def _make_line_annotation(frame_id: str, line_id: str = "L1") -> LineAnnotation:
    return LineAnnotation(
        line_id=line_id,
        frame_id=frame_id,
        camera_pose=_make_ptz(),
        start_pixel=PixelPoint.create(50.0, 60.0),
        end_pixel=PixelPoint.create(250.0, 350.0),
        points=(
            PixelPoint.create(50.0, 60.0),
            PixelPoint.create(150.0, 200.0),
            PixelPoint.create(250.0, 350.0),
        ),
    )


def _make_lens_calibration_table(camera_config_id: str) -> LensCalibrationTable:
    return LensCalibrationTable(
        id=camera_config_id,
        entries=(
            ZoomCalibrationEntry(
                zoom_factor=Unitless(1.0),
                distortion=LensDistortion(
                    k1=Unitless(-0.1), k2=Unitless(0.01),
                ),
                calibration_date="2024-01-15",
                source_images=("img1.jpg", "img2.jpg"),
                validation_rmse=0.5,
                num_lines_used=10,
                fx=PixelsFloat(1000.0),
                fy=PixelsFloat(1000.0),
                cx=PixelsFloat(640.0),
                cy=PixelsFloat(360.0),
                reprojection_error_px=0.3,
            ),
        ),
        created_date="2024-01-15T10:00:00",
        last_modified="2024-01-15T12:00:00",
    )


def _make_calibration_line_trace_set(name: str | None = None) -> CalibrationLineTraceSet:
    return CalibrationLineTraceSet(
        name=name or _uid("clts_"),
        image="trace_image.jpg",
        camera_pose=_make_ptz(),
        line_traces=(
            LineTrace(line_id="L1", points=((10.0, 20.0), (30.0, 40.0), (50.0, 60.0))),
            LineTrace(line_id="L2", points=((100.0, 200.0), (300.0, 400.0))),
        ),
    )


# ---------------------------------------------------------------------------
# Helpers to persist FK parents
# ---------------------------------------------------------------------------

def _persist_tenant(session: Session, tenant_id: str | None = None) -> Tenant:
    tenant = _make_tenant(tenant_id)
    RepoPostgresTenant(session).save(tenant)
    return tenant


def _persist_map(session: Session, tenant_id: str, map_id: str | None = None) -> Map:
    m = _make_map(tenant_id, map_id)
    RepoPostgresMap(session).save(m)
    return m


def _persist_camera_config(
    session: Session,
    tenant_id: str,
    map_id: str,
    name: str | None = None,
) -> CameraConfig:
    cc = _make_camera_config(tenant_id, map_id, name)
    RepoPostgresCameraConfig(session).save(cc)
    return cc


def _persist_frame(
    session: Session,
    map_id: str,
    camera_name: str = "cam01",
    ts: datetime | None = None,
) -> CapturedFrame:
    frame = _make_captured_frame(map_id, camera_name, ts)
    RepoPostgresCapturedFrame(session).save(frame)
    return frame


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRepoPostgresTenant:
    def test_save_and_get(self, db_session: Session) -> None:
        repo = RepoPostgresTenant(db_session)
        tenant = _make_tenant()
        repo.save(tenant)

        loaded = repo.get(tenant.id)
        assert loaded is not None
        assert loaded.id == tenant.id
        assert loaded.name == tenant.name
        assert loaded.description == tenant.description
        assert loaded.location_lat == tenant.location_lat
        assert loaded.location_lon == tenant.location_lon

    def test_get_nonexistent(self, db_session: Session) -> None:
        repo = RepoPostgresTenant(db_session)
        assert repo.get("nonexistent_tenant_id") is None

    def test_exists(self, db_session: Session) -> None:
        repo = RepoPostgresTenant(db_session)
        tenant = _make_tenant()
        repo.save(tenant)
        assert repo.exists(tenant.id) is True
        assert repo.exists("nonexistent") is False

    def test_delete(self, db_session: Session) -> None:
        repo = RepoPostgresTenant(db_session)
        tenant = _make_tenant()
        repo.save(tenant)
        assert repo.exists(tenant.id)

        assert repo.delete(tenant.id) is True
        assert repo.get(tenant.id) is None
        assert repo.exists(tenant.id) is False

    def test_delete_nonexistent(self, db_session: Session) -> None:
        repo = RepoPostgresTenant(db_session)
        assert repo.delete("nonexistent") is False

    def test_get_all(self, db_session: Session) -> None:
        repo = RepoPostgresTenant(db_session)
        t1 = _make_tenant()
        t2 = _make_tenant()
        repo.save(t1)
        repo.save(t2)

        all_tenants = repo.get_all()
        ids = {t.id for t in all_tenants}
        assert t1.id in ids
        assert t2.id in ids

    def test_save_update(self, db_session: Session) -> None:
        repo = RepoPostgresTenant(db_session)
        tenant = _make_tenant()
        repo.save(tenant)

        updated = Tenant(
            id=tenant.id,
            name="Updated Name",
            description="Updated desc",
        )
        repo.save(updated)

        loaded = repo.get(tenant.id)
        assert loaded is not None
        assert loaded.name == "Updated Name"
        assert loaded.description == "Updated desc"


@pytest.mark.integration
class TestRepoPostgresMap:
    def test_save_and_get(self, db_session: Session, test_tenant: Tenant) -> None:
        repo = RepoPostgresMap(db_session)
        m = _make_map(test_tenant.id)
        repo.save(m)

        loaded = repo.get(m.id)
        assert loaded is not None
        assert loaded.id == m.id
        assert loaded.tenant_id == test_tenant.id
        assert loaded.photo.width == Pixels(1000)
        assert loaded.photo.height == Pixels(800)
        assert loaded.geotiff.crs == "EPSG:25830"
        assert float(loaded.geotiff.geotransform.pixel_width) == 0.5

    def test_get_nonexistent(self, db_session: Session) -> None:
        repo = RepoPostgresMap(db_session)
        assert repo.get("nonexistent_map") is None

    def test_exists(self, db_session: Session, test_tenant: Tenant) -> None:
        repo = RepoPostgresMap(db_session)
        m = _make_map(test_tenant.id)
        repo.save(m)
        assert repo.exists(m.id) is True
        assert repo.exists("nonexistent") is False

    def test_delete(self, db_session: Session, test_tenant: Tenant) -> None:
        repo = RepoPostgresMap(db_session)
        m = _make_map(test_tenant.id)
        repo.save(m)

        assert repo.delete(m.id) is True
        assert repo.get(m.id) is None

    def test_get_all(self, db_session: Session, test_tenant: Tenant) -> None:
        repo = RepoPostgresMap(db_session)
        m1 = _make_map(test_tenant.id)
        m2 = _make_map(test_tenant.id)
        repo.save(m1)
        repo.save(m2)

        all_maps = repo.get_all()
        ids = {m.id for m in all_maps}
        assert m1.id in ids
        assert m2.id in ids


    def test_get_by_tenant(self, db_session: Session) -> None:
        t1 = _persist_tenant(db_session)
        t2 = _persist_tenant(db_session)

        repo = RepoPostgresMap(db_session)
        m1 = _make_map(t1.id)
        m2 = _make_map(t2.id)
        repo.save(m1)
        repo.save(m2)

        result = repo.get_by_tenant(t1.id)
        assert m1.id in result
        assert m2.id not in result

    def test_geotiff_round_trip(self, db_session: Session, test_tenant: Tenant) -> None:
        repo = RepoPostgresMap(db_session)
        m = _make_map(test_tenant.id)
        repo.save(m)

        loaded = repo.get(m.id)
        assert loaded is not None
        gt = loaded.geotiff.geotransform
        assert float(gt.origin_easting) == 500000.0
        assert float(gt.origin_northing) == 4400000.0
        assert float(gt.pixel_height) == -0.5


@pytest.mark.integration
class TestRepoPostgresCameraConfig:
    def test_save_and_get(
        self, db_session: Session, test_tenant: Tenant, test_map: Map
    ) -> None:
        repo = RepoPostgresCameraConfig(db_session)
        cc = _make_camera_config(test_tenant.id, test_map.id)
        repo.save(cc)

        loaded = repo.get(cc.id)
        assert loaded is not None
        assert loaded.id == cc.id
        assert loaded.tenant_id == test_tenant.id
        assert loaded.map_id == test_map.id
        assert loaded.name == cc.name
        assert loaded.spec == CameraSpec.HIKVISION_DS_2DF8425IX
        assert loaded.credential.username == "admin"
        assert loaded.credential.password == "secret"
        assert loaded.ip_address == "192.168.1.100"

    def test_get_nonexistent(self, db_session: Session) -> None:
        repo = RepoPostgresCameraConfig(db_session)
        assert repo.get("nonexistent/cam") is None

    def test_exists(
        self, db_session: Session, test_tenant: Tenant, test_map: Map
    ) -> None:
        repo = RepoPostgresCameraConfig(db_session)
        cc = _make_camera_config(test_tenant.id, test_map.id)
        repo.save(cc)
        assert repo.exists(cc.id) is True
        assert repo.exists("nonexistent/cam") is False

    def test_delete(
        self, db_session: Session, test_tenant: Tenant, test_map: Map
    ) -> None:
        repo = RepoPostgresCameraConfig(db_session)
        cc = _make_camera_config(test_tenant.id, test_map.id)
        repo.save(cc)

        assert repo.delete(cc.id) is True
        assert repo.get(cc.id) is None

    def test_get_all(
        self, db_session: Session, test_tenant: Tenant, test_map: Map
    ) -> None:
        repo = RepoPostgresCameraConfig(db_session)
        cc1 = _make_camera_config(test_tenant.id, test_map.id)
        cc2 = _make_camera_config(test_tenant.id, test_map.id)
        repo.save(cc1)
        repo.save(cc2)

        all_ccs = repo.get_all()
        ids = {c.id for c in all_ccs}
        assert cc1.id in ids
        assert cc2.id in ids


    def test_get_by_tenant(self, db_session: Session) -> None:
        t1 = _persist_tenant(db_session)
        t2 = _persist_tenant(db_session)
        m1 = _persist_map(db_session, t1.id)
        m2 = _persist_map(db_session, t2.id)

        repo = RepoPostgresCameraConfig(db_session)
        cc1 = _make_camera_config(t1.id, m1.id)
        cc2 = _make_camera_config(t2.id, m2.id)
        repo.save(cc1)
        repo.save(cc2)

        result = repo.get_by_tenant(t1.id)
        assert cc1.id in result
        assert cc2.id not in result


    def test_get_by_map(self, db_session: Session) -> None:
        t = _persist_tenant(db_session)
        m1 = _persist_map(db_session, t.id)
        m2 = _persist_map(db_session, t.id)

        repo = RepoPostgresCameraConfig(db_session)
        cc1 = _make_camera_config(t.id, m1.id)
        cc2 = _make_camera_config(t.id, m2.id)
        repo.save(cc1)
        repo.save(cc2)

        result = repo.get_by_map(m1.id)
        assert cc1.id in result
        assert cc2.id not in result


@pytest.mark.integration
class TestRepoPostgresCameraCalibration:
    def test_save_and_get(
        self,
        db_session: Session,
        test_tenant: Tenant,
        test_map: Map,
        test_camera_config: CameraConfig,
    ) -> None:
        repo = RepoPostgresCameraCalibration(db_session)
        cal = _make_camera_calibration(test_camera_config.id)
        repo.save(cal)

        loaded = repo.get(cal.id)
        assert loaded is not None
        assert loaded.id == cal.id
        assert float(loaded.position.x) == 500.0
        assert float(loaded.position.y) == 400.0
        assert float(loaded.height) == 12.5
        assert float(loaded.base_orientation.yaw) == 90.0
        assert float(loaded.base_orientation.pitch) == -10.0
        assert float(loaded.distortion.k1) == pytest.approx(-0.1)
        assert float(loaded.distortion.k2) == pytest.approx(0.01)

    def test_get_nonexistent(self, db_session: Session) -> None:
        repo = RepoPostgresCameraCalibration(db_session)
        assert repo.get("nonexistent/cam") is None

    def test_exists(
        self,
        db_session: Session,
        test_camera_config: CameraConfig,
    ) -> None:
        repo = RepoPostgresCameraCalibration(db_session)
        cal = _make_camera_calibration(test_camera_config.id)
        repo.save(cal)
        assert repo.exists(cal.id) is True
        assert repo.exists("nonexistent/cam") is False

    def test_delete(
        self,
        db_session: Session,
        test_camera_config: CameraConfig,
    ) -> None:
        repo = RepoPostgresCameraCalibration(db_session)
        cal = _make_camera_calibration(test_camera_config.id)
        repo.save(cal)

        assert repo.delete(cal.id) is True
        assert repo.get(cal.id) is None

    def test_get_all(
        self,
        db_session: Session,
        test_tenant: Tenant,
        test_map: Map,
        test_camera_config: CameraConfig,
    ) -> None:
        repo = RepoPostgresCameraCalibration(db_session)
        cal = _make_camera_calibration(test_camera_config.id)
        repo.save(cal)

        all_cals = repo.get_all()
        ids = {c.id for c in all_cals}
        assert cal.id in ids

    def test_value_object_round_trip(
        self,
        db_session: Session,
        test_camera_config: CameraConfig,
    ) -> None:
        """Orientation, LensDistortion, PixelPoint survive save/get cycle."""
        repo = RepoPostgresCameraCalibration(db_session)
        cal = _make_camera_calibration(test_camera_config.id)
        repo.save(cal)

        loaded = repo.get(cal.id)
        assert loaded is not None
        # Orientation
        assert float(loaded.base_orientation.yaw) == float(cal.base_orientation.yaw)
        assert float(loaded.base_orientation.pitch) == float(cal.base_orientation.pitch)
        assert float(loaded.base_orientation.roll) == float(cal.base_orientation.roll)
        # LensDistortion
        assert float(loaded.distortion.p1) == pytest.approx(float(cal.distortion.p1))
        assert float(loaded.distortion.p2) == pytest.approx(float(cal.distortion.p2))
        # PixelPoint
        assert float(loaded.position.x) == float(cal.position.x)
        assert float(loaded.position.y) == float(cal.position.y)


@pytest.mark.integration
class TestRepoPostgresCapturedFrame:
    def test_save_and_get(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresCapturedFrame(db_session)
        frame = _make_captured_frame(test_map.id)
        repo.save(frame)

        loaded = repo.get(frame.id)
        assert loaded is not None
        assert loaded.id == frame.id
        assert loaded.map_id == test_map.id
        assert loaded.camera_name == "cam01"
        # PostgreSQL stores timezone-aware; compare the wall-clock time
        assert loaded.timestamp.replace(tzinfo=None) == frame.timestamp.replace(tzinfo=None)
        assert float(loaded.ptz_state.pan_raw) == 30.0
        assert float(loaded.ptz_state.zoom) == 2.0
        assert str(loaded.image_path) == "frame.jpg"

    def test_get_nonexistent(self, db_session: Session) -> None:
        repo = RepoPostgresCapturedFrame(db_session)
        assert repo.get("map/cam/nonexistent") is None

    def test_exists(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresCapturedFrame(db_session)
        frame = _make_captured_frame(test_map.id)
        repo.save(frame)
        assert repo.exists(frame.id) is True
        assert repo.exists("nonexistent/frame/id") is False

    def test_delete(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresCapturedFrame(db_session)
        frame = _make_captured_frame(test_map.id)
        repo.save(frame)

        assert repo.delete(frame.id) is True
        assert repo.get(frame.id) is None

    def test_get_all(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresCapturedFrame(db_session)
        f1 = _make_captured_frame(test_map.id, ts=datetime(2024, 1, 15, 10, 0, 0))
        f2 = _make_captured_frame(test_map.id, ts=datetime(2024, 1, 15, 11, 0, 0))
        repo.save(f1)
        repo.save(f2)

        all_frames = repo.get_all()
        ids = {f.id for f in all_frames}
        assert f1.id in ids
        assert f2.id in ids

    def test_get_by_map(self, db_session: Session) -> None:
        t = _persist_tenant(db_session)
        m1 = _persist_map(db_session, t.id)
        m2 = _persist_map(db_session, t.id)

        repo = RepoPostgresCapturedFrame(db_session)
        f1 = _make_captured_frame(m1.id, ts=datetime(2024, 1, 15, 10, 0, 0))
        f2 = _make_captured_frame(m2.id, ts=datetime(2024, 1, 15, 11, 0, 0))
        repo.save(f1)
        repo.save(f2)

        result = repo.get_by_map(m1.id)
        result_ids = {f.id for f in result}
        assert f1.id in result_ids
        assert f2.id not in result_ids

    def test_get_by_camera(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresCapturedFrame(db_session)
        f1 = _make_captured_frame(
            test_map.id, camera_name="cam_a", ts=datetime(2024, 1, 15, 10, 0, 0)
        )
        f2 = _make_captured_frame(
            test_map.id, camera_name="cam_b", ts=datetime(2024, 1, 15, 11, 0, 0)
        )
        repo.save(f1)
        repo.save(f2)

        result = repo.get_by_camera(test_map.id, "cam_a")
        result_ids = {f.id for f in result}
        assert f1.id in result_ids
        assert f2.id not in result_ids

    def test_ptz_state_round_trip(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresCapturedFrame(db_session)
        frame = _make_captured_frame(test_map.id)
        repo.save(frame)

        loaded = repo.get(frame.id)
        assert loaded is not None
        assert float(loaded.ptz_state.pan_raw) == float(frame.ptz_state.pan_raw)
        assert float(loaded.ptz_state.tilt_deg) == float(frame.ptz_state.tilt_deg)
        assert float(loaded.ptz_state.zoom) == float(frame.ptz_state.zoom)


@pytest.mark.integration
class TestRepoPostgresAnnotation:
    def test_save_and_get(self, db_session: Session, test_map: Map) -> None:
        frame = _persist_frame(db_session, test_map.id)
        repo = RepoPostgresAnnotation(db_session)
        ann = _make_annotation(frame.id, "GCP_A")
        repo.save(ann)

        loaded = repo.get(ann.id)
        assert loaded is not None
        assert loaded.id == ann.id
        assert loaded.gcp_id == "GCP_A"
        assert loaded.frame_id == frame.id
        assert float(loaded.pixel.x) == pytest.approx(123.4)
        assert float(loaded.pixel.y) == pytest.approx(567.8)
        assert float(loaded.camera_pose.pan_raw) == 30.0

    def test_get_nonexistent(self, db_session: Session) -> None:
        repo = RepoPostgresAnnotation(db_session)
        assert repo.get("frame/GCP_nonexistent") is None

    def test_exists(self, db_session: Session, test_map: Map) -> None:
        frame = _persist_frame(db_session, test_map.id)
        repo = RepoPostgresAnnotation(db_session)
        ann = _make_annotation(frame.id)
        repo.save(ann)
        assert repo.exists(ann.id) is True
        assert repo.exists("nonexistent/id") is False

    def test_delete(self, db_session: Session, test_map: Map) -> None:
        frame = _persist_frame(db_session, test_map.id)
        repo = RepoPostgresAnnotation(db_session)
        ann = _make_annotation(frame.id)
        repo.save(ann)

        assert repo.delete(ann.id) is True
        assert repo.get(ann.id) is None

    def test_get_all(self, db_session: Session, test_map: Map) -> None:
        frame = _persist_frame(db_session, test_map.id)
        repo = RepoPostgresAnnotation(db_session)
        a1 = _make_annotation(frame.id, "GCP_X")
        a2 = _make_annotation(frame.id, "GCP_Y")
        repo.save(a1)
        repo.save(a2)

        all_anns = repo.get_all()
        ids = {a.id for a in all_anns}
        assert a1.id in ids
        assert a2.id in ids

    def test_composite_id(self, db_session: Session, test_map: Map) -> None:
        """Annotation uses composite id: frame_id/gcp_id."""
        frame = _persist_frame(db_session, test_map.id)
        ann = _make_annotation(frame.id, "GCP_Z")
        assert ann.id == f"{frame.id}/GCP_Z"

    def test_get_by_frame_id(self, db_session: Session, test_map: Map) -> None:
        f1 = _persist_frame(db_session, test_map.id, ts=datetime(2024, 1, 15, 10, 0, 0))
        f2 = _persist_frame(db_session, test_map.id, ts=datetime(2024, 1, 15, 11, 0, 0))

        repo = RepoPostgresAnnotation(db_session)
        a1 = _make_annotation(f1.id, "GCP_A")
        a2 = _make_annotation(f2.id, "GCP_B")
        repo.save(a1)
        repo.save(a2)

        result = repo.get_by_frame_id(f1.id)
        result_ids = {a.id for a in result}
        assert a1.id in result_ids
        assert a2.id not in result_ids

    def test_pixel_point_round_trip(self, db_session: Session, test_map: Map) -> None:
        frame = _persist_frame(db_session, test_map.id)
        repo = RepoPostgresAnnotation(db_session)
        ann = _make_annotation(frame.id)
        repo.save(ann)

        loaded = repo.get(ann.id)
        assert loaded is not None
        assert float(loaded.pixel.x) == float(ann.pixel.x)
        assert float(loaded.pixel.y) == float(ann.pixel.y)


@pytest.mark.integration
class TestRepoPostgresGroundControlPoint:
    def test_save_and_get(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresGroundControlPoint(db_session)
        gcp = _make_gcp(test_map.id, "GCP_1")
        repo.save(gcp)

        loaded = repo.get(gcp.id)
        assert loaded is not None
        assert loaded.id == gcp.id
        assert loaded.name == "GCP_1"
        assert loaded.map_id == test_map.id
        assert float(loaded.map_point.pixel_point.x) == pytest.approx(100.5)
        assert float(loaded.map_point.pixel_point.y) == pytest.approx(200.5)

    def test_get_nonexistent(self, db_session: Session) -> None:
        repo = RepoPostgresGroundControlPoint(db_session)
        assert repo.get("nonexistent/GCP") is None

    def test_exists(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresGroundControlPoint(db_session)
        gcp = _make_gcp(test_map.id)
        repo.save(gcp)
        assert repo.exists(gcp.id) is True
        assert repo.exists("nonexistent/GCP") is False

    def test_delete(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresGroundControlPoint(db_session)
        gcp = _make_gcp(test_map.id)
        repo.save(gcp)

        assert repo.delete(gcp.id) is True
        assert repo.get(gcp.id) is None

    def test_get_all(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresGroundControlPoint(db_session)
        g1 = _make_gcp(test_map.id, "GCP_A")
        g2 = _make_gcp(test_map.id, "GCP_B")
        repo.save(g1)
        repo.save(g2)

        all_gcps = repo.get_all()
        ids = {g.id for g in all_gcps}
        assert g1.id in ids
        assert g2.id in ids

    def test_composite_id(self, db_session: Session, test_map: Map) -> None:
        gcp = _make_gcp(test_map.id, "GCP_X")
        assert gcp.id == f"{test_map.id}/GCP_X"


    def test_get_by_map(self, db_session: Session) -> None:
        t = _persist_tenant(db_session)
        m1 = _persist_map(db_session, t.id)
        m2 = _persist_map(db_session, t.id)

        repo = RepoPostgresGroundControlPoint(db_session)
        g1 = _make_gcp(m1.id, "GCP_A")
        g2 = _make_gcp(m2.id, "GCP_B")
        repo.save(g1)
        repo.save(g2)

        result = repo.get_by_map(m1.id)
        assert g1.id in result
        assert g2.id not in result

    def test_map_point_round_trip(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresGroundControlPoint(db_session)
        gcp = _make_gcp(test_map.id)
        repo.save(gcp)

        loaded = repo.get(gcp.id)
        assert loaded is not None
        assert loaded.map_point.map_id == gcp.map_point.map_id
        assert float(loaded.map_point.pixel_point.x) == float(gcp.map_point.pixel_point.x)
        assert float(loaded.map_point.pixel_point.y) == float(gcp.map_point.pixel_point.y)


@pytest.mark.integration
class TestRepoPostgresLine:
    def test_save_and_get(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresLine(db_session)
        line = _make_line(test_map.id, "L1")
        repo.save(line)

        loaded = repo.get(line.id)
        assert loaded is not None
        assert loaded.id == line.id
        assert loaded.name == "L1"
        assert loaded.map_id == test_map.id
        assert float(loaded.start.x) == 10.0
        assert float(loaded.end.y) == 400.0

    def test_get_nonexistent(self, db_session: Session) -> None:
        repo = RepoPostgresLine(db_session)
        assert repo.get("nonexistent/L1") is None

    def test_exists(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresLine(db_session)
        line = _make_line(test_map.id)
        repo.save(line)
        assert repo.exists(line.id) is True
        assert repo.exists("nonexistent/L99") is False

    def test_delete(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresLine(db_session)
        line = _make_line(test_map.id)
        repo.save(line)

        assert repo.delete(line.id) is True
        assert repo.get(line.id) is None

    def test_get_all(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresLine(db_session)
        l1 = _make_line(test_map.id, "L1")
        l2 = _make_line(test_map.id, "L2")
        repo.save(l1)
        repo.save(l2)

        all_lines = repo.get_all()
        ids = {l.id for l in all_lines}
        assert l1.id in ids
        assert l2.id in ids

    def test_composite_id(self, db_session: Session, test_map: Map) -> None:
        line = _make_line(test_map.id, "L5")
        assert line.id == f"{test_map.id}/L5"

    def test_get_by_map_id(self, db_session: Session) -> None:
        t = _persist_tenant(db_session)
        m1 = _persist_map(db_session, t.id)
        m2 = _persist_map(db_session, t.id)

        repo = RepoPostgresLine(db_session)
        l1 = _make_line(m1.id, "L1")
        l2 = _make_line(m2.id, "L2")
        repo.save(l1)
        repo.save(l2)

        result = repo.get_by_map_id(m1.id)
        result_ids = {l.id for l in result}
        assert l1.id in result_ids
        assert l2.id not in result_ids


    def test_get_by_map(self, db_session: Session) -> None:
        t = _persist_tenant(db_session)
        m1 = _persist_map(db_session, t.id)
        m2 = _persist_map(db_session, t.id)

        repo = RepoPostgresLine(db_session)
        l1 = _make_line(m1.id, "L1")
        l2 = _make_line(m2.id, "L2")
        repo.save(l1)
        repo.save(l2)

        result = repo.get_by_map(m1.id)
        assert l1.id in result
        assert l2.id not in result

    def test_pixel_point_round_trip(self, db_session: Session, test_map: Map) -> None:
        repo = RepoPostgresLine(db_session)
        line = _make_line(test_map.id)
        repo.save(line)

        loaded = repo.get(line.id)
        assert loaded is not None
        assert float(loaded.start.x) == float(line.start.x)
        assert float(loaded.start.y) == float(line.start.y)
        assert float(loaded.end.x) == float(line.end.x)
        assert float(loaded.end.y) == float(line.end.y)


@pytest.mark.integration
class TestRepoPostgresLineAnnotation:
    def test_save_and_get(self, db_session: Session, test_map: Map) -> None:
        frame = _persist_frame(db_session, test_map.id)
        repo = RepoPostgresLineAnnotation(db_session)
        la = _make_line_annotation(frame.id, "L1")
        repo.save(la)

        loaded = repo.get(la.id)
        assert loaded is not None
        assert loaded.id == la.id
        assert loaded.line_id == "L1"
        assert loaded.frame_id == frame.id
        assert float(loaded.start_pixel.x) == 50.0
        assert float(loaded.end_pixel.y) == 350.0
        assert loaded.points is not None
        assert len(loaded.points) == 3

    def test_get_nonexistent(self, db_session: Session) -> None:
        repo = RepoPostgresLineAnnotation(db_session)
        assert repo.get("frame/L99") is None

    def test_exists(self, db_session: Session, test_map: Map) -> None:
        frame = _persist_frame(db_session, test_map.id)
        repo = RepoPostgresLineAnnotation(db_session)
        la = _make_line_annotation(frame.id)
        repo.save(la)
        assert repo.exists(la.id) is True
        assert repo.exists("nonexistent/L1") is False

    def test_delete(self, db_session: Session, test_map: Map) -> None:
        frame = _persist_frame(db_session, test_map.id)
        repo = RepoPostgresLineAnnotation(db_session)
        la = _make_line_annotation(frame.id)
        repo.save(la)

        assert repo.delete(la.id) is True
        assert repo.get(la.id) is None

    def test_get_all(self, db_session: Session, test_map: Map) -> None:
        frame = _persist_frame(db_session, test_map.id)
        repo = RepoPostgresLineAnnotation(db_session)
        la1 = _make_line_annotation(frame.id, "L1")
        la2 = _make_line_annotation(frame.id, "L2")
        repo.save(la1)
        repo.save(la2)

        all_las = repo.get_all()
        ids = {la.id for la in all_las}
        assert la1.id in ids
        assert la2.id in ids

    def test_composite_id(self, db_session: Session, test_map: Map) -> None:
        frame = _persist_frame(db_session, test_map.id)
        la = _make_line_annotation(frame.id, "L3")
        assert la.id == f"{frame.id}/L3"

    def test_points_round_trip(self, db_session: Session, test_map: Map) -> None:
        """Points polyline survives save/get cycle."""
        frame = _persist_frame(db_session, test_map.id)
        repo = RepoPostgresLineAnnotation(db_session)
        la = _make_line_annotation(frame.id)
        repo.save(la)

        loaded = repo.get(la.id)
        assert loaded is not None
        assert loaded.points is not None
        assert len(loaded.points) == len(la.points)  # type: ignore[arg-type]
        for orig, load in zip(la.points, loaded.points):  # type: ignore[arg-type]
            assert float(load.x) == pytest.approx(float(orig.x))
            assert float(load.y) == pytest.approx(float(orig.y))

    def test_null_points(self, db_session: Session, test_map: Map) -> None:
        """LineAnnotation with points=None saves and loads correctly."""
        frame = _persist_frame(db_session, test_map.id)
        repo = RepoPostgresLineAnnotation(db_session)
        la = LineAnnotation(
            line_id="L_no_pts",
            frame_id=frame.id,
            camera_pose=_make_ptz(),
            start_pixel=PixelPoint.create(10.0, 20.0),
            end_pixel=PixelPoint.create(30.0, 40.0),
            points=None,
        )
        repo.save(la)

        loaded = repo.get(la.id)
        assert loaded is not None
        assert loaded.points is None

    def test_ptz_round_trip(self, db_session: Session, test_map: Map) -> None:
        frame = _persist_frame(db_session, test_map.id)
        repo = RepoPostgresLineAnnotation(db_session)
        la = _make_line_annotation(frame.id)
        repo.save(la)

        loaded = repo.get(la.id)
        assert loaded is not None
        assert float(loaded.camera_pose.pan_raw) == float(la.camera_pose.pan_raw)
        assert float(loaded.camera_pose.tilt_deg) == float(la.camera_pose.tilt_deg)
        assert float(loaded.camera_pose.zoom) == float(la.camera_pose.zoom)


@pytest.mark.integration
class TestRepoPostgresLensCalibrationTable:
    def test_save_and_get(
        self,
        db_session: Session,
        test_camera_config: CameraConfig,
    ) -> None:
        repo = RepoPostgresLensCalibrationTable(db_session)
        lct = _make_lens_calibration_table(test_camera_config.id)
        repo.save(lct)

        loaded = repo.get(lct.id)
        assert loaded is not None
        assert loaded.id == lct.id
        assert loaded.created_date == "2024-01-15T10:00:00"
        assert loaded.last_modified == "2024-01-15T12:00:00"
        assert len(loaded.entries) == 1
        assert float(loaded.entries[0].zoom_factor) == 1.0
        assert float(loaded.entries[0].distortion.k1) == pytest.approx(-0.1)

    def test_get_nonexistent(self, db_session: Session) -> None:
        repo = RepoPostgresLensCalibrationTable(db_session)
        assert repo.get("nonexistent/cam") is None

    def test_exists(
        self,
        db_session: Session,
        test_camera_config: CameraConfig,
    ) -> None:
        repo = RepoPostgresLensCalibrationTable(db_session)
        lct = _make_lens_calibration_table(test_camera_config.id)
        repo.save(lct)
        assert repo.exists(lct.id) is True
        assert repo.exists("nonexistent") is False

    def test_delete(
        self,
        db_session: Session,
        test_camera_config: CameraConfig,
    ) -> None:
        repo = RepoPostgresLensCalibrationTable(db_session)
        lct = _make_lens_calibration_table(test_camera_config.id)
        repo.save(lct)

        assert repo.delete(lct.id) is True
        assert repo.get(lct.id) is None

    def test_get_all(
        self,
        db_session: Session,
        test_camera_config: CameraConfig,
    ) -> None:
        repo = RepoPostgresLensCalibrationTable(db_session)
        lct = _make_lens_calibration_table(test_camera_config.id)
        repo.save(lct)

        all_lcts = repo.get_all()
        ids = {l.id for l in all_lcts}
        assert lct.id in ids

    def test_entries_round_trip(
        self,
        db_session: Session,
        test_camera_config: CameraConfig,
    ) -> None:
        """ZoomCalibrationEntry fields survive the round-trip."""
        repo = RepoPostgresLensCalibrationTable(db_session)
        lct = _make_lens_calibration_table(test_camera_config.id)
        repo.save(lct)

        loaded = repo.get(lct.id)
        assert loaded is not None
        entry = loaded.entries[0]
        assert entry.calibration_date == "2024-01-15"
        assert list(entry.source_images) == ["img1.jpg", "img2.jpg"]
        assert entry.validation_rmse == pytest.approx(0.5)
        assert entry.num_lines_used == 10
        assert float(entry.fx) == pytest.approx(1000.0)
        assert float(entry.fy) == pytest.approx(1000.0)
        assert float(entry.cx) == pytest.approx(640.0)
        assert float(entry.cy) == pytest.approx(360.0)
        assert entry.reprojection_error_px == pytest.approx(0.3)


@pytest.mark.integration
class TestRepoPostgresCalibrationLineTraceSet:
    def test_save_and_get(self, db_session: Session) -> None:
        repo = RepoPostgresCalibrationLineTraceSet(db_session)
        clts = _make_calibration_line_trace_set()
        repo.save(clts)

        loaded = repo.get(clts.id)
        assert loaded is not None
        assert loaded.id == clts.id
        assert loaded.name == clts.name
        assert loaded.image == "trace_image.jpg"
        assert float(loaded.camera_pose.pan_raw) == 30.0
        assert len(loaded.line_traces) == 2

    def test_get_nonexistent(self, db_session: Session) -> None:
        repo = RepoPostgresCalibrationLineTraceSet(db_session)
        assert repo.get("nonexistent_clts") is None

    def test_exists(self, db_session: Session) -> None:
        repo = RepoPostgresCalibrationLineTraceSet(db_session)
        clts = _make_calibration_line_trace_set()
        repo.save(clts)
        assert repo.exists(clts.id) is True
        assert repo.exists("nonexistent") is False

    def test_delete(self, db_session: Session) -> None:
        repo = RepoPostgresCalibrationLineTraceSet(db_session)
        clts = _make_calibration_line_trace_set()
        repo.save(clts)

        assert repo.delete(clts.id) is True
        assert repo.get(clts.id) is None

    def test_get_all(self, db_session: Session) -> None:
        repo = RepoPostgresCalibrationLineTraceSet(db_session)
        c1 = _make_calibration_line_trace_set()
        c2 = _make_calibration_line_trace_set()
        repo.save(c1)
        repo.save(c2)

        all_clts = repo.get_all()
        ids = {c.id for c in all_clts}
        assert c1.id in ids
        assert c2.id in ids

    def test_line_traces_round_trip(self, db_session: Session) -> None:
        """LineTrace value objects survive save/get cycle."""
        repo = RepoPostgresCalibrationLineTraceSet(db_session)
        clts = _make_calibration_line_trace_set()
        repo.save(clts)

        loaded = repo.get(clts.id)
        assert loaded is not None
        assert len(loaded.line_traces) == 2
        lt1 = loaded.line_traces[0]
        assert lt1.line_id == "L1"
        assert len(lt1.points) == 3
        assert lt1.points[0] == (10.0, 20.0)
        assert lt1.points[2] == (50.0, 60.0)

        lt2 = loaded.line_traces[1]
        assert lt2.line_id == "L2"
        assert len(lt2.points) == 2
