"""Tests for build_legacy_camera_dict in poc_homography.camera_config.

Verifies that build_legacy_camera_dict correctly converts DDD entities
(CameraConfig + CameraCalibration) into the legacy dict format expected
by older CLI code.

Run with: python -m pytest tests/test_build_legacy_camera_dict.py -v
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from poc_homography.camera_config import build_legacy_camera_dict
from poc_homography.domain.entities.camera_calibration import CameraCalibration
from poc_homography.domain.entities.camera_config import CameraConfig
from poc_homography.domain.enums import CameraSpec
from poc_homography.domain.vo.credential import Credential
from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.domain.vo.orientation import Orientation
from poc_homography.domain.vo.pixel_point import PixelPoint
from poc_homography.types import Degrees, Meters, Unitless


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def camera() -> CameraConfig:
    """A minimal CameraConfig entity for testing."""
    return CameraConfig(
        id="test_tenant/TestCam",
        tenant_id="test_tenant",
        map_id="test_tenant/map01",
        name="TestCam",
        spec=CameraSpec.HIKVISION_DS_2DF8425IX,
        credential=Credential(username="admin", password="secret"),
        ip_address="192.168.1.100",
    )


@pytest.fixture
def calibration() -> CameraCalibration:
    """A CameraCalibration with non-trivial values."""
    return CameraCalibration(
        id="test_tenant/TestCam",
        position=PixelPoint.create(x=500.0, y=300.0),
        height=Meters(12.5),
        base_orientation=Orientation.create(
            yaw=Degrees(45.0),
            pitch=Degrees(-10.0),
        ),
        distortion=LensDistortion(
            k1=Unitless(-0.35),
            k2=Unitless(0.12),
            p1=Unitless(0.001),
            p2=Unitless(-0.002),
        ),
    )


@pytest.fixture
def zero_distortion_calibration() -> CameraCalibration:
    """A CameraCalibration with zero distortion coefficients."""
    return CameraCalibration(
        id="test_tenant/TestCam",
        position=PixelPoint.create(x=100.0, y=200.0),
        height=Meters(8.0),
        base_orientation=Orientation.create(
            yaw=Degrees(90.0),
            pitch=Degrees(0.0),
        ),
        distortion=LensDistortion(),  # all defaults are 0.0
    )


# =============================================================================
# Keys expected in every result dict
# =============================================================================

BASE_KEYS = {"id", "name", "ip", "tenant_id", "model", "sensor_width_mm", "base_focal_length_mm"}
CALIBRATION_KEYS = {"height_m", "pan_offset_deg", "tilt_offset_deg", "k1", "k2", "p1", "p2"}
ALL_KEYS = BASE_KEYS | CALIBRATION_KEYS


# =============================================================================
# Tests — with calibration present
# =============================================================================


class TestWithCalibration:
    """build_legacy_camera_dict when a CameraCalibration is provided."""

    def test_all_keys_present(
        self, camera: CameraConfig, calibration: CameraCalibration
    ) -> None:
        result = build_legacy_camera_dict(camera, calibration)
        assert set(result.keys()) == ALL_KEYS, "Result dict must contain exactly the expected keys"

    def test_base_fields(
        self, camera: CameraConfig, calibration: CameraCalibration
    ) -> None:
        result = build_legacy_camera_dict(camera, calibration)

        assert result["id"] == "test_tenant/TestCam", "id must match camera.id"
        assert result["name"] == "TestCam", "name must match camera.name"
        assert result["ip"] == "192.168.1.100", "ip must match camera.ip_address"
        assert result["tenant_id"] == "test_tenant", "tenant_id must match camera.tenant_id"
        assert result["model"] == "DS-2DF8425IX-AELW", "model must match camera.spec.model_name"

    def test_sensor_fields_are_float(
        self, camera: CameraConfig, calibration: CameraCalibration
    ) -> None:
        result = build_legacy_camera_dict(camera, calibration)

        assert isinstance(result["sensor_width_mm"], float), "sensor_width_mm must be float"
        assert isinstance(result["base_focal_length_mm"], float), "base_focal_length_mm must be float"
        assert result["sensor_width_mm"] == pytest.approx(6.78), "sensor_width_mm value"
        assert result["base_focal_length_mm"] == pytest.approx(5.9), "base_focal_length_mm value"

    def test_calibration_fields(
        self, camera: CameraConfig, calibration: CameraCalibration
    ) -> None:
        result = build_legacy_camera_dict(camera, calibration)

        assert result["height_m"] == pytest.approx(12.5), "height_m must come from calibration.height"
        assert result["pan_offset_deg"] == pytest.approx(45.0), (
            "pan_offset_deg must come from calibration.base_orientation.yaw"
        )
        assert result["tilt_offset_deg"] == pytest.approx(-10.0), (
            "tilt_offset_deg must come from calibration.base_orientation.pitch"
        )
        assert result["k1"] == pytest.approx(-0.35), "k1 must come from calibration.distortion.k1"
        assert result["k2"] == pytest.approx(0.12), "k2 must come from calibration.distortion.k2"
        assert result["p1"] == pytest.approx(0.001), "p1 must come from calibration.distortion.p1"
        assert result["p2"] == pytest.approx(-0.002), "p2 must come from calibration.distortion.p2"

    def test_calibration_values_are_float(
        self, camera: CameraConfig, calibration: CameraCalibration
    ) -> None:
        """All calibration values must be plain float, not Decimal or NewType wrappers."""
        result = build_legacy_camera_dict(camera, calibration)

        for key in CALIBRATION_KEYS:
            assert type(result[key]) is float, f"{key} must be exactly float, got {type(result[key])}"


# =============================================================================
# Tests — without calibration (None)
# =============================================================================


class TestWithoutCalibration:
    """build_legacy_camera_dict when calibration is None."""

    def test_all_keys_still_present(self, camera: CameraConfig) -> None:
        result = build_legacy_camera_dict(camera, None)
        assert set(result.keys()) == ALL_KEYS, (
            "Result dict must contain all keys even when calibration is None"
        )

    def test_base_fields_unchanged(self, camera: CameraConfig) -> None:
        result = build_legacy_camera_dict(camera, None)

        assert result["id"] == "test_tenant/TestCam"
        assert result["name"] == "TestCam"
        assert result["ip"] == "192.168.1.100"
        assert result["tenant_id"] == "test_tenant"
        assert result["model"] == "DS-2DF8425IX-AELW"

    def test_default_calibration_values(self, camera: CameraConfig) -> None:
        result = build_legacy_camera_dict(camera, None)

        assert result["height_m"] == 5.0, "Default height_m should be 5.0"
        assert result["pan_offset_deg"] == 0.0, "Default pan_offset_deg should be 0.0"
        assert result["tilt_offset_deg"] == 0.0, "Default tilt_offset_deg should be 0.0"
        assert result["k1"] == 0.0, "Default k1 should be 0.0"
        assert result["k2"] == 0.0, "Default k2 should be 0.0"
        assert result["p1"] == 0.0, "Default p1 should be 0.0"
        assert result["p2"] == 0.0, "Default p2 should be 0.0"

    def test_default_values_are_float(self, camera: CameraConfig) -> None:
        result = build_legacy_camera_dict(camera, None)

        for key in CALIBRATION_KEYS:
            assert type(result[key]) is float, f"{key} must be exactly float, got {type(result[key])}"


# =============================================================================
# Tests — with calibration but zero distortion
# =============================================================================


class TestWithZeroDistortion:
    """build_legacy_camera_dict when calibration has zero distortion coefficients."""

    def test_zero_distortion_values(
        self,
        camera: CameraConfig,
        zero_distortion_calibration: CameraCalibration,
    ) -> None:
        result = build_legacy_camera_dict(camera, zero_distortion_calibration)

        assert result["k1"] == 0.0, "k1 should be 0.0 for zero distortion"
        assert result["k2"] == 0.0, "k2 should be 0.0 for zero distortion"
        assert result["p1"] == 0.0, "p1 should be 0.0 for zero distortion"
        assert result["p2"] == 0.0, "p2 should be 0.0 for zero distortion"

    def test_zero_distortion_values_are_float(
        self,
        camera: CameraConfig,
        zero_distortion_calibration: CameraCalibration,
    ) -> None:
        result = build_legacy_camera_dict(camera, zero_distortion_calibration)

        for key in ("k1", "k2", "p1", "p2"):
            assert type(result[key]) is float, f"{key} must be exactly float, got {type(result[key])}"

    def test_non_distortion_calibration_fields_populated(
        self,
        camera: CameraConfig,
        zero_distortion_calibration: CameraCalibration,
    ) -> None:
        """Even with zero distortion, height and orientation come from calibration, not defaults."""
        result = build_legacy_camera_dict(camera, zero_distortion_calibration)

        assert result["height_m"] == pytest.approx(8.0), "height_m should use calibration value, not default"
        assert result["pan_offset_deg"] == pytest.approx(90.0), (
            "pan_offset_deg should use calibration value, not default"
        )
        assert result["tilt_offset_deg"] == pytest.approx(0.0), (
            "tilt_offset_deg should use calibration value, not default"
        )
