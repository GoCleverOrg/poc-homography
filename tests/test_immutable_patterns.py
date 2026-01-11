#!/usr/bin/env python3
"""
Comprehensive tests for immutable patterns in the camera geometry pipeline.

This module tests the frozen dataclass behavior, factory methods, pure function
behavior of the immutable state management patterns.

Classes tested:
    - CameraParameters (frozen dataclass)
    - CameraGeometryResult (frozen dataclass)
    - CameraGeometry.compute() (pure function)
    - IntrinsicExtrinsicConfig (frozen dataclass)
    - IntrinsicExtrinsicResult (frozen dataclass)
    - IntrinsicExtrinsicHomography.compute_from_config() (pure function)
    - UTMConverter factory methods
    - GCPCoordinateConverter factory methods

Run with: python -m pytest tests/test_immutable_patterns.py -v
"""

import os
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.camera_parameters import (
    CameraGeometryResult,
    CameraParameters,
    DistortionCoefficients,
    HeightUncertainty,
)
from poc_homography.coordinate_converter import (
    PYPROJ_AVAILABLE,
    GCPCoordinateConverter,
    UTMConverter,
)
from poc_homography.homography_parameters import (
    IntrinsicExtrinsicConfig,
    IntrinsicExtrinsicResult,
)
from poc_homography.intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography
from poc_homography.types import Degrees, Meters, Millimeters, Pixels, Unitless

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_intrinsic_matrix() -> np.ndarray:
    """Return a typical 3x3 camera intrinsic matrix K."""
    return np.array([[1670.0, 0.0, 960.0], [0.0, 1670.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)


@pytest.fixture
def sample_camera_position() -> np.ndarray:
    """Return a typical camera position [X, Y, Z] in meters."""
    return np.array([0.0, 0.0, 10.0], dtype=np.float64)


@pytest.fixture
def sample_camera_params(sample_intrinsic_matrix, sample_camera_position) -> CameraParameters:
    """Create a sample CameraParameters instance for testing."""
    return CameraParameters.create(
        image_width=Pixels(1920),
        image_height=Pixels(1080),
        intrinsic_matrix=sample_intrinsic_matrix,
        camera_position=sample_camera_position,
        pan_deg=Degrees(0.0),
        tilt_deg=Degrees(45.0),
        roll_deg=Degrees(0.0),
        map_width=Pixels(640),
        map_height=Pixels(640),
        pixels_per_meter=Unitless(100.0),
    )


@pytest.fixture
def sample_ie_config(sample_intrinsic_matrix, sample_camera_position) -> IntrinsicExtrinsicConfig:
    """Create a sample IntrinsicExtrinsicConfig instance for testing."""
    return IntrinsicExtrinsicConfig.create(
        camera_matrix=sample_intrinsic_matrix,
        camera_position=sample_camera_position,
        pan_deg=Degrees(0.0),
        tilt_deg=Degrees(45.0),
        roll_deg=Degrees(0.0),
        map_width=Pixels(640),
        map_height=Pixels(640),
        pixels_per_meter=Unitless(100.0),
        sensor_width_mm=Millimeters(7.18),
        base_focal_length_mm=Millimeters(5.9),
        map_id="test_map",
    )


# ============================================================================
# CameraParameters Tests
# ============================================================================


class TestCameraParametersFrozen:
    """Test that CameraParameters is truly frozen (immutable)."""

    def test_frozen_attribute_raises_error(self, sample_camera_params):
        """Attempting to modify a frozen attribute raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_camera_params.pan_deg = Degrees(90.0)

    def test_frozen_image_width_raises_error(self, sample_camera_params):
        """Attempting to modify image_width raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_camera_params.image_width = Pixels(3840)

    def test_frozen_image_height_raises_error(self, sample_camera_params):
        """Attempting to modify image_height raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_camera_params.image_height = Pixels(2160)

    def test_frozen_tilt_deg_raises_error(self, sample_camera_params):
        """Attempting to modify tilt_deg raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_camera_params.tilt_deg = Degrees(60.0)

    def test_frozen_roll_deg_raises_error(self, sample_camera_params):
        """Attempting to modify roll_deg raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_camera_params.roll_deg = Degrees(5.0)

    def test_frozen_map_width_raises_error(self, sample_camera_params):
        """Attempting to modify map_width raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_camera_params.map_width = Pixels(1280)

    def test_frozen_map_height_raises_error(self, sample_camera_params):
        """Attempting to modify map_height raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_camera_params.map_height = Pixels(1280)

    def test_frozen_pixels_per_meter_raises_error(self, sample_camera_params):
        """Attempting to modify pixels_per_meter raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_camera_params.pixels_per_meter = Unitless(200.0)

    def test_frozen_internal_bytes_raises_error(self, sample_camera_params):
        """Attempting to modify internal bytes fields raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_camera_params._intrinsic_matrix_data = b"fake"

    def test_intrinsic_matrix_is_immutable(self, sample_camera_params):
        """The intrinsic_matrix property returns a read-only array."""
        K = sample_camera_params.intrinsic_matrix
        assert not K.flags.writeable, "intrinsic_matrix should be read-only"
        with pytest.raises((ValueError, TypeError)):
            K[0, 0] = 9999.0

    def test_camera_position_is_immutable(self, sample_camera_params):
        """The camera_position property returns a read-only array."""
        pos = sample_camera_params.camera_position
        assert not pos.flags.writeable, "camera_position should be read-only"
        with pytest.raises((ValueError, TypeError)):
            pos[2] = 99.0


class TestCameraParametersCreateFactory:
    """Test the CameraParameters.create() factory method."""

    def test_create_basic(self, sample_intrinsic_matrix, sample_camera_position):
        """Create returns a valid CameraParameters instance."""
        params = CameraParameters.create(
            image_width=Pixels(1920),
            image_height=Pixels(1080),
            intrinsic_matrix=sample_intrinsic_matrix,
            camera_position=sample_camera_position,
            pan_deg=Degrees(45.0),
            tilt_deg=Degrees(30.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
        )

        assert params.image_width == 1920
        assert params.image_height == 1080
        assert params.pan_deg == 45.0
        assert params.tilt_deg == 30.0
        assert params.roll_deg == 0.0
        assert params.map_width == 640
        assert params.map_height == 640
        assert params.pixels_per_meter == 100.0

    def test_create_with_distortion(self, sample_intrinsic_matrix, sample_camera_position):
        """Create with distortion coefficients."""
        distortion = DistortionCoefficients(
            k1=Unitless(-0.1),
            k2=Unitless(0.01),
            p1=Unitless(0.0),
            p2=Unitless(0.0),
            k3=Unitless(0.0),
        )
        params = CameraParameters.create(
            image_width=Pixels(1920),
            image_height=Pixels(1080),
            intrinsic_matrix=sample_intrinsic_matrix,
            camera_position=sample_camera_position,
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
            distortion=distortion,
        )

        assert params.distortion is not None
        assert params.distortion.k1 == -0.1
        assert params.distortion.k2 == 0.01

    def test_create_with_height_uncertainty(self, sample_intrinsic_matrix, sample_camera_position):
        """Create with height uncertainty bounds."""
        height_uncertainty = HeightUncertainty(lower=Meters(9.5), upper=Meters(10.5))
        params = CameraParameters.create(
            image_width=Pixels(1920),
            image_height=Pixels(1080),
            intrinsic_matrix=sample_intrinsic_matrix,
            camera_position=sample_camera_position,
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
            height_uncertainty=height_uncertainty,
        )

        assert params.height_uncertainty is not None
        assert params.height_uncertainty.lower == 9.5
        assert params.height_uncertainty.upper == 10.5

    def test_create_with_affine_matrix(self, sample_intrinsic_matrix, sample_camera_position):
        """Create with optional affine matrix."""
        affine = np.array(
            [
                [0.15, 0.0, 100.0],
                [0.0, -0.15, 200.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        params = CameraParameters.create(
            image_width=Pixels(1920),
            image_height=Pixels(1080),
            intrinsic_matrix=sample_intrinsic_matrix,
            camera_position=sample_camera_position,
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
            affine_matrix=affine,
        )

        assert params.affine_matrix is not None
        assert np.allclose(params.affine_matrix, affine)


class TestCameraParametersValidation:
    """Test __post_init__ validation in CameraParameters."""

    def test_invalid_image_width_raises(self, sample_intrinsic_matrix, sample_camera_position):
        """Creating with non-positive image_width raises ValueError."""
        with pytest.raises(ValueError, match="image_width must be positive"):
            CameraParameters.create(
                image_width=Pixels(0),
                image_height=Pixels(1080),
                intrinsic_matrix=sample_intrinsic_matrix,
                camera_position=sample_camera_position,
                pan_deg=Degrees(0.0),
                tilt_deg=Degrees(45.0),
                roll_deg=Degrees(0.0),
                map_width=Pixels(640),
                map_height=Pixels(640),
                pixels_per_meter=Unitless(100.0),
            )

    def test_invalid_image_height_raises(self, sample_intrinsic_matrix, sample_camera_position):
        """Creating with non-positive image_height raises ValueError."""
        with pytest.raises(ValueError, match="image_height must be positive"):
            CameraParameters.create(
                image_width=Pixels(1920),
                image_height=Pixels(-1),
                intrinsic_matrix=sample_intrinsic_matrix,
                camera_position=sample_camera_position,
                pan_deg=Degrees(0.0),
                tilt_deg=Degrees(45.0),
                roll_deg=Degrees(0.0),
                map_width=Pixels(640),
                map_height=Pixels(640),
                pixels_per_meter=Unitless(100.0),
            )

    def test_invalid_map_width_raises(self, sample_intrinsic_matrix, sample_camera_position):
        """Creating with non-positive map_width raises ValueError."""
        with pytest.raises(ValueError, match="map_width must be positive"):
            CameraParameters.create(
                image_width=Pixels(1920),
                image_height=Pixels(1080),
                intrinsic_matrix=sample_intrinsic_matrix,
                camera_position=sample_camera_position,
                pan_deg=Degrees(0.0),
                tilt_deg=Degrees(45.0),
                roll_deg=Degrees(0.0),
                map_width=Pixels(0),
                map_height=Pixels(640),
                pixels_per_meter=Unitless(100.0),
            )

    def test_invalid_pixels_per_meter_raises(self, sample_intrinsic_matrix, sample_camera_position):
        """Creating with non-positive pixels_per_meter raises ValueError."""
        with pytest.raises(ValueError, match="pixels_per_meter must be positive"):
            CameraParameters.create(
                image_width=Pixels(1920),
                image_height=Pixels(1080),
                intrinsic_matrix=sample_intrinsic_matrix,
                camera_position=sample_camera_position,
                pan_deg=Degrees(0.0),
                tilt_deg=Degrees(45.0),
                roll_deg=Degrees(0.0),
                map_width=Pixels(640),
                map_height=Pixels(640),
                pixels_per_meter=Unitless(-100.0),
            )

    def test_invalid_intrinsic_matrix_shape_raises(self, sample_camera_position):
        """Creating with wrong intrinsic matrix shape raises ValueError."""
        bad_K = np.eye(4)  # 4x4 instead of 3x3
        # The error happens during post_init when trying to reshape bytes to 3x3
        with pytest.raises(ValueError, match="cannot reshape|intrinsic_matrix.*must be.*3, 3"):
            CameraParameters.create(
                image_width=Pixels(1920),
                image_height=Pixels(1080),
                intrinsic_matrix=bad_K,
                camera_position=sample_camera_position,
                pan_deg=Degrees(0.0),
                tilt_deg=Degrees(45.0),
                roll_deg=Degrees(0.0),
                map_width=Pixels(640),
                map_height=Pixels(640),
                pixels_per_meter=Unitless(100.0),
            )


class TestCameraParametersHashability:
    """Test that CameraParameters can be used as dict key."""

    def test_hashable_as_dict_key(self, sample_camera_params):
        """CameraParameters can be used as a dictionary key."""
        cache = {}
        cache[sample_camera_params] = "result_1"
        assert cache[sample_camera_params] == "result_1"

    def test_equal_params_have_same_hash(self, sample_intrinsic_matrix, sample_camera_position):
        """Two CameraParameters with same values have same hash."""
        params1 = CameraParameters.create(
            image_width=Pixels(1920),
            image_height=Pixels(1080),
            intrinsic_matrix=sample_intrinsic_matrix,
            camera_position=sample_camera_position,
            pan_deg=Degrees(45.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
        )
        params2 = CameraParameters.create(
            image_width=Pixels(1920),
            image_height=Pixels(1080),
            intrinsic_matrix=sample_intrinsic_matrix.copy(),
            camera_position=sample_camera_position.copy(),
            pan_deg=Degrees(45.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
        )

        assert hash(params1) == hash(params2)
        assert params1 == params2

    def test_different_params_have_different_hash(
        self, sample_camera_params, sample_intrinsic_matrix, sample_camera_position
    ):
        """CameraParameters with different values have different hashes."""
        params2 = CameraParameters.create(
            image_width=Pixels(1920),
            image_height=Pixels(1080),
            intrinsic_matrix=sample_intrinsic_matrix,
            camera_position=sample_camera_position,
            pan_deg=Degrees(90.0),  # Different pan
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
        )

        assert hash(sample_camera_params) != hash(params2)
        assert sample_camera_params != params2

    def test_usable_in_set(self, sample_camera_params):
        """CameraParameters can be added to a set."""
        param_set = {sample_camera_params}
        assert sample_camera_params in param_set


# ============================================================================
# CameraGeometryResult Tests
# ============================================================================


class TestCameraGeometryResultFrozen:
    """Test that CameraGeometryResult is truly frozen (immutable)."""

    @pytest.fixture
    def sample_result(self) -> CameraGeometryResult:
        """Create a sample CameraGeometryResult instance."""
        H = np.array(
            [
                [100.0, 0.0, 960.0],
                [0.0, 100.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        H_inv = np.linalg.inv(H)
        return CameraGeometryResult.create(
            homography_matrix=H,
            inverse_homography_matrix=H_inv,
            condition_number=1.0,
            determinant=10000.0,
            is_valid=True,
            validation_messages=(),
        )

    def test_frozen_condition_number_raises(self, sample_result):
        """Attempting to modify condition_number raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_result.condition_number = 99.0

    def test_frozen_determinant_raises(self, sample_result):
        """Attempting to modify determinant raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_result.determinant = 99.0

    def test_frozen_is_valid_raises(self, sample_result):
        """Attempting to modify is_valid raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_result.is_valid = False

    def test_homography_matrix_is_immutable(self, sample_result):
        """The homography_matrix property returns a read-only array."""
        H = sample_result.homography_matrix
        assert not H.flags.writeable, "homography_matrix should be read-only"
        with pytest.raises((ValueError, TypeError)):
            H[0, 0] = 9999.0

    def test_inverse_homography_matrix_is_immutable(self, sample_result):
        """The inverse_homography_matrix property returns a read-only array."""
        H_inv = sample_result.inverse_homography_matrix
        assert not H_inv.flags.writeable, "inverse_homography_matrix should be read-only"
        with pytest.raises((ValueError, TypeError)):
            H_inv[0, 0] = 9999.0


class TestCameraGeometryResultCreateFactory:
    """Test the CameraGeometryResult.create() factory method."""

    def test_create_valid_result(self):
        """Create returns a valid CameraGeometryResult instance."""
        H = np.eye(3) * 100.0
        H[2, 2] = 1.0
        H_inv = np.linalg.inv(H)

        result = CameraGeometryResult.create(
            homography_matrix=H,
            inverse_homography_matrix=H_inv,
            condition_number=1.5,
            determinant=10000.0,
            is_valid=True,
            validation_messages=["test message"],
            center_projection_distance=Meters(15.0),
        )

        assert result.is_valid is True
        assert result.condition_number == 1.5
        assert result.determinant == 10000.0
        assert len(result.validation_messages) == 1
        assert result.center_projection_distance == 15.0

    def test_create_with_list_messages(self):
        """Create converts list messages to tuple."""
        H = np.eye(3)
        H_inv = np.eye(3)

        result = CameraGeometryResult.create(
            homography_matrix=H,
            inverse_homography_matrix=H_inv,
            condition_number=1.0,
            determinant=1.0,
            is_valid=True,
            validation_messages=["msg1", "msg2"],  # list, not tuple
        )

        assert isinstance(result.validation_messages, tuple)
        assert result.validation_messages == ("msg1", "msg2")


# ============================================================================
# CameraGeometry.compute() Tests (Pure Function)
# ============================================================================


class TestCameraGeometryComputePureFunction:
    """Test that CameraGeometry.compute() is a pure function."""

    def test_compute_returns_consistent_results(self, sample_camera_params):
        """compute() returns identical results for identical inputs."""
        result1 = CameraGeometry.compute(sample_camera_params)
        result2 = CameraGeometry.compute(sample_camera_params)

        assert np.allclose(result1.homography_matrix, result2.homography_matrix)
        assert np.allclose(result1.inverse_homography_matrix, result2.inverse_homography_matrix)
        assert result1.condition_number == result2.condition_number
        assert result1.determinant == result2.determinant
        assert result1.is_valid == result2.is_valid

    def test_compute_does_not_modify_params(self, sample_camera_params):
        """compute() does not modify the input CameraParameters."""
        # CameraParameters is frozen, so this test ensures no mutation attempts
        original_pan = sample_camera_params.pan_deg
        original_tilt = sample_camera_params.tilt_deg

        _ = CameraGeometry.compute(sample_camera_params)

        assert sample_camera_params.pan_deg == original_pan
        assert sample_camera_params.tilt_deg == original_tilt

    def test_compute_does_not_require_instance(self, sample_camera_params):
        """compute() is a classmethod that works without an instance."""
        # Should work as a classmethod, not requiring any instance state
        result = CameraGeometry.compute(sample_camera_params)
        assert result.is_valid is True

    def test_compute_result_is_immutable(self, sample_camera_params):
        """The result from compute() is immutable."""
        result = CameraGeometry.compute(sample_camera_params)

        with pytest.raises(FrozenInstanceError):
            result.is_valid = False  # type: ignore[misc]  # Intentional for test


# ============================================================================
# IntrinsicExtrinsicConfig Tests
# ============================================================================


class TestIntrinsicExtrinsicConfigFrozen:
    """Test that IntrinsicExtrinsicConfig is truly frozen (immutable)."""

    def test_frozen_pan_deg_raises(self, sample_ie_config):
        """Attempting to modify pan_deg raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_ie_config.pan_deg = Degrees(90.0)

    def test_frozen_tilt_deg_raises(self, sample_ie_config):
        """Attempting to modify tilt_deg raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_ie_config.tilt_deg = Degrees(60.0)

    def test_frozen_map_id_raises(self, sample_ie_config):
        """Attempting to modify map_id raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_ie_config.map_id = "new_map"

    def test_camera_matrix_is_immutable(self, sample_ie_config):
        """The camera_matrix property returns a read-only array."""
        K = sample_ie_config.camera_matrix
        assert not K.flags.writeable, "camera_matrix should be read-only"

    def test_camera_position_is_immutable(self, sample_ie_config):
        """The camera_position property returns a read-only array."""
        pos = sample_ie_config.camera_position
        assert not pos.flags.writeable, "camera_position should be read-only"


class TestIntrinsicExtrinsicConfigFactories:
    """Test IntrinsicExtrinsicConfig factory methods."""

    def test_create_basic(self, sample_intrinsic_matrix, sample_camera_position):
        """Create returns a valid IntrinsicExtrinsicConfig instance."""
        config = IntrinsicExtrinsicConfig.create(
            camera_matrix=sample_intrinsic_matrix,
            camera_position=sample_camera_position,
            pan_deg=Degrees(45.0),
            tilt_deg=Degrees(30.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
            sensor_width_mm=Millimeters(7.18),
            base_focal_length_mm=Millimeters(5.9),
            map_id="test_map",
        )

        assert config.pan_deg == 45.0
        assert config.tilt_deg == 30.0
        assert config.map_id == "test_map"

    def test_from_reference_dict(self, sample_intrinsic_matrix, sample_camera_position):
        """from_reference_dict creates config from dict format."""
        reference = {
            "camera_matrix": sample_intrinsic_matrix,
            "camera_position": sample_camera_position,
            "pan_deg": 45.0,
            "tilt_deg": 30.0,
            "roll_deg": 2.0,
            "map_width": 640,
            "map_height": 640,
        }

        config = IntrinsicExtrinsicConfig.from_reference_dict(
            reference=reference,
            pixels_per_meter=Unitless(100.0),
            sensor_width_mm=Millimeters(7.18),
            base_focal_length_mm=Millimeters(5.9),
            map_id="test_map",
        )

        assert config.pan_deg == 45.0
        assert config.tilt_deg == 30.0
        assert config.roll_deg == 2.0
        assert config.map_id == "test_map"

    def test_from_reference_dict_missing_key_raises(
        self, sample_intrinsic_matrix, sample_camera_position
    ):
        """from_reference_dict raises ValueError for missing required keys."""
        reference = {
            "camera_matrix": sample_intrinsic_matrix,
            # Missing camera_position
            "pan_deg": 45.0,
            "tilt_deg": 30.0,
            "map_width": 640,
            "map_height": 640,
        }

        with pytest.raises(ValueError, match="Missing required reference key"):
            IntrinsicExtrinsicConfig.from_reference_dict(
                reference=reference,
                pixels_per_meter=Unitless(100.0),
                sensor_width_mm=Millimeters(7.18),
                base_focal_length_mm=Millimeters(5.9),
                map_id="test_map",
            )


class TestIntrinsicExtrinsicConfigValidation:
    """Test __post_init__ validation in IntrinsicExtrinsicConfig."""

    def test_invalid_roll_raises(self, sample_intrinsic_matrix, sample_camera_position):
        """Creating with excessive roll angle raises ValueError."""
        with pytest.raises(ValueError, match="roll_deg.*outside valid range"):
            IntrinsicExtrinsicConfig.create(
                camera_matrix=sample_intrinsic_matrix,
                camera_position=sample_camera_position,
                pan_deg=Degrees(0.0),
                tilt_deg=Degrees(45.0),
                roll_deg=Degrees(20.0),  # Exceeds 15 degree threshold
                map_width=Pixels(640),
                map_height=Pixels(640),
                pixels_per_meter=Unitless(100.0),
                sensor_width_mm=Millimeters(7.18),
                base_focal_length_mm=Millimeters(5.9),
                map_id="test_map",
            )

    def test_empty_map_id_raises(self, sample_intrinsic_matrix, sample_camera_position):
        """Creating with empty map_id raises ValueError."""
        with pytest.raises(ValueError, match="map_id must be a non-empty string"):
            IntrinsicExtrinsicConfig.create(
                camera_matrix=sample_intrinsic_matrix,
                camera_position=sample_camera_position,
                pan_deg=Degrees(0.0),
                tilt_deg=Degrees(45.0),
                roll_deg=Degrees(0.0),
                map_width=Pixels(640),
                map_height=Pixels(640),
                pixels_per_meter=Unitless(100.0),
                sensor_width_mm=Millimeters(7.18),
                base_focal_length_mm=Millimeters(5.9),
                map_id="",  # Empty
            )


# ============================================================================
# IntrinsicExtrinsicResult Tests
# ============================================================================


class TestIntrinsicExtrinsicResultFrozen:
    """Test that IntrinsicExtrinsicResult is truly frozen (immutable)."""

    @pytest.fixture
    def sample_ie_result(self, sample_camera_position) -> IntrinsicExtrinsicResult:
        """Create a sample IntrinsicExtrinsicResult instance."""
        H = np.eye(3) * 100.0
        H[2, 2] = 1.0
        H_inv = np.linalg.inv(H)

        return IntrinsicExtrinsicResult.create(
            homography_matrix=H,
            inverse_homography_matrix=H_inv,
            confidence=0.9,
            camera_position=sample_camera_position,
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
        )

    def test_frozen_confidence_raises(self, sample_ie_result):
        """Attempting to modify confidence raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_ie_result.confidence = 0.5

    def test_frozen_is_valid_raises(self, sample_ie_result):
        """Attempting to modify is_valid raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_ie_result.is_valid = False

    def test_homography_matrix_is_immutable(self, sample_ie_result):
        """The homography_matrix property returns a read-only array."""
        H = sample_ie_result.homography_matrix
        assert not H.flags.writeable, "homography_matrix should be read-only"


class TestIntrinsicExtrinsicResultCreateFactory:
    """Test IntrinsicExtrinsicResult.create() factory method."""

    def test_create_computes_validity(self, sample_camera_position):
        """Create computes is_valid based on confidence and condition number."""
        H = np.eye(3) * 100.0
        H[2, 2] = 1.0
        H_inv = np.linalg.inv(H)

        result = IntrinsicExtrinsicResult.create(
            homography_matrix=H,
            inverse_homography_matrix=H_inv,
            confidence=0.9,
            camera_position=sample_camera_position,
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
        )

        assert result.is_valid is True

    def test_create_low_confidence_invalid(self, sample_camera_position):
        """Create marks result invalid when confidence is too low."""
        H = np.eye(3)
        H_inv = np.eye(3)

        result = IntrinsicExtrinsicResult.create(
            homography_matrix=H,
            inverse_homography_matrix=H_inv,
            confidence=0.1,  # Below 0.3 threshold
            camera_position=sample_camera_position,
            pan_deg=Degrees(0.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
        )

        assert result.is_valid is False
        assert any("Confidence" in msg for msg in result.validation_messages)


# ============================================================================
# IntrinsicExtrinsicHomography.compute_from_config() Tests (Pure Function)
# ============================================================================


class TestIntrinsicExtrinsicHomographyComputeFromConfig:
    """Test that compute_from_config() is a pure function."""

    def test_compute_from_config_returns_consistent_results(self, sample_ie_config):
        """compute_from_config() returns identical results for identical inputs."""
        result1 = IntrinsicExtrinsicHomography.compute_from_config(sample_ie_config)
        result2 = IntrinsicExtrinsicHomography.compute_from_config(sample_ie_config)

        assert np.allclose(result1.homography_matrix, result2.homography_matrix)
        assert np.allclose(result1.inverse_homography_matrix, result2.inverse_homography_matrix)
        assert result1.confidence == result2.confidence
        assert result1.is_valid == result2.is_valid

    def test_compute_from_config_does_not_require_instance(self, sample_ie_config):
        """compute_from_config() works as a classmethod without instance."""
        result = IntrinsicExtrinsicHomography.compute_from_config(sample_ie_config)
        assert isinstance(result, IntrinsicExtrinsicResult)
        assert result.is_valid is True

    def test_compute_from_config_result_is_immutable(self, sample_ie_config):
        """The result from compute_from_config() is immutable."""
        result = IntrinsicExtrinsicHomography.compute_from_config(sample_ie_config)

        with pytest.raises(FrozenInstanceError):
            result.confidence = 0.5  # type: ignore[misc]  # Intentional for test


# ============================================================================
# UTMConverter Factory Tests
# ============================================================================


@pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not installed")
class TestUTMConverterFactoryMethods:
    """Test UTMConverter factory methods for immutable pattern."""

    def test_with_reference_creates_new_instance(self):
        """UTMConverter.with_reference() creates a new configured instance."""
        converter = UTMConverter.with_reference(
            lat=Degrees(39.5),
            lon=Degrees(-0.5),
        )

        assert converter._ref_lat == 39.5
        assert converter._ref_lon == -0.5
        assert converter._ref_easting is not None
        assert converter._ref_northing is not None

    def test_with_reference_utm_creates_new_instance(self):
        """UTMConverter.with_reference_utm() creates a new configured instance."""
        converter = UTMConverter.with_reference_utm(
            easting=Meters(737575.0),
            northing=Meters(4391595.0),
        )

        assert converter._ref_easting == 737575.0
        assert converter._ref_northing == 4391595.0
        assert converter._ref_lat is not None
        assert converter._ref_lon is not None

    def test_with_reference_returns_different_instances(self):
        """Each call to with_reference() returns a new instance."""
        converter1 = UTMConverter.with_reference(lat=Degrees(39.5), lon=Degrees(-0.5))
        converter2 = UTMConverter.with_reference(lat=Degrees(40.0), lon=Degrees(-0.4))

        assert converter1 is not converter2
        assert converter1._ref_lat != converter2._ref_lat

    def test_converter_without_reference_raises_on_conversion(self):
        """UTMConverter without reference raises ValueError on conversion."""
        converter = UTMConverter()

        with pytest.raises(ValueError, match="Reference point not set"):
            converter.gps_to_local_xy(39.5, -0.5)


# ============================================================================
# GCPCoordinateConverter Factory Tests
# ============================================================================


@pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not installed")
class TestGCPCoordinateConverterFactoryMethods:
    """Test GCPCoordinateConverter factory methods for immutable pattern."""

    def test_with_reference_gps_creates_new_instance(self):
        """GCPCoordinateConverter.with_reference_gps() creates a configured instance."""
        converter = GCPCoordinateConverter.with_reference_gps(
            lat=Degrees(39.640472),
            lon=Degrees(-0.230194),
        )

        assert converter._ref_lat == 39.640472
        assert converter._ref_lon == -0.230194

    def test_with_reference_utm_creates_new_instance(self):
        """GCPCoordinateConverter.with_reference_utm() creates a configured instance."""
        converter = GCPCoordinateConverter.with_reference_utm(
            easting=Meters(737575.0),
            northing=Meters(4391595.0),
        )

        assert converter._ref_easting == 737575.0
        assert converter._ref_northing == 4391595.0

    def test_with_reference_gps_returns_different_instances(self):
        """Each call to with_reference_gps() returns a new instance."""
        converter1 = GCPCoordinateConverter.with_reference_gps(lat=Degrees(39.5), lon=Degrees(-0.5))
        converter2 = GCPCoordinateConverter.with_reference_gps(lat=Degrees(40.0), lon=Degrees(-0.4))

        assert converter1 is not converter2
        assert converter1._ref_lat != converter2._ref_lat

    def test_converter_without_reference_raises_on_conversion(self):
        """GCPCoordinateConverter without reference raises ValueError on conversion."""
        converter = GCPCoordinateConverter()

        with pytest.raises(ValueError, match="Reference point not set"):
            converter.gps_to_local(39.5, -0.5)


# ============================================================================
# Integration Tests
# ============================================================================


class TestImmutablePatternIntegration:
    """Integration tests verifying the immutable patterns work together."""

    def test_camera_geometry_full_pipeline(self, sample_intrinsic_matrix, sample_camera_position):
        """Full pipeline: CameraParameters -> compute() -> result."""
        # Create immutable params
        params = CameraParameters.create(
            image_width=Pixels(1920),
            image_height=Pixels(1080),
            intrinsic_matrix=sample_intrinsic_matrix,
            camera_position=sample_camera_position,
            pan_deg=Degrees(30.0),
            tilt_deg=Degrees(45.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
        )

        # Compute using pure function
        result = CameraGeometry.compute(params)

        # Results should be valid
        assert result.is_valid is True
        assert not np.allclose(result.homography_matrix, np.eye(3))

    def test_intrinsic_extrinsic_full_pipeline(
        self, sample_intrinsic_matrix, sample_camera_position
    ):
        """Full pipeline: IntrinsicExtrinsicConfig -> compute_from_config() -> result."""
        # Create config
        config = IntrinsicExtrinsicConfig.create(
            camera_matrix=sample_intrinsic_matrix,
            camera_position=sample_camera_position,
            pan_deg=Degrees(45.0),
            tilt_deg=Degrees(30.0),
            roll_deg=Degrees(0.0),
            map_width=Pixels(640),
            map_height=Pixels(640),
            pixels_per_meter=Unitless(100.0),
            sensor_width_mm=Millimeters(7.18),
            base_focal_length_mm=Millimeters(5.9),
            map_id="test_map",
        )

        # Compute using pure function
        result = IntrinsicExtrinsicHomography.compute_from_config(config)

        # Verify result
        assert result.is_valid is True
        assert result.confidence > 0.5
        assert result.pan_deg == 45.0
        assert result.tilt_deg == 30.0

    def test_caching_works_with_hashable_params(self, sample_camera_params):
        """Demonstrate caching pattern with hashable CameraParameters."""
        cache = {}

        def cached_compute(params: CameraParameters) -> CameraGeometryResult:
            if params in cache:
                return cache[params]
            result = CameraGeometry.compute(params)
            cache[params] = result
            return result

        # First call computes
        result1 = cached_compute(sample_camera_params)
        assert len(cache) == 1

        # Second call retrieves from cache
        result2 = cached_compute(sample_camera_params)
        assert len(cache) == 1
        assert result1 is result2  # Same object from cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
