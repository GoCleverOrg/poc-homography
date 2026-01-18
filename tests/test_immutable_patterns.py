"""
Comprehensive tests for immutable patterns in the camera geometry pipeline.

This module tests the frozen dataclass behavior, factory methods, pure function
behavior of the immutable state management patterns.

Classes tested:
    - CameraIntrinsics (frozen dataclass VO)
    - Homography (frozen dataclass VO)
    - Orientation (frozen dataclass VO)
    - Vector3 (frozen dataclass VO)
    - CameraGeometry.compute_from_vo() (pure function)
    - IntrinsicExtrinsicConfig (frozen dataclass)
    - IntrinsicExtrinsicResult (frozen dataclass)
    - IntrinsicExtrinsicHomography.compute_from_config() (pure function)

Run with: python -m pytest tests/test_immutable_patterns.py -v

UPDATED: Refactored for VO-based API
- Removed CameraParameters and CameraGeometryResult tests (classes deleted)
- Added tests for domain VOs (CameraIntrinsics, Homography, Orientation, Vector3)
- Updated integration tests to use compute_from_vo()
"""

import os
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.domain.vo import CameraIntrinsics, Homography, Orientation, Vector3
from poc_homography.homography import (
    IntrinsicExtrinsicConfig,
    IntrinsicExtrinsicHomography,
    IntrinsicExtrinsicResult,
)
from poc_homography.types import Degrees, Millimeters, Pixels, Unitless

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
def sample_intrinsics(sample_intrinsic_matrix) -> CameraIntrinsics:
    """Create a sample CameraIntrinsics VO for testing."""
    return CameraIntrinsics.from_K_matrix(
        sample_intrinsic_matrix,
        Pixels(1920),
        Pixels(1080),
        sensor_width=Millimeters(7.18),
    )


@pytest.fixture
def sample_position_vo() -> Vector3:
    """Create a sample camera position as Vector3 VO."""
    return Vector3.create(0.0, 0.0, 10.0)


@pytest.fixture
def sample_orientation() -> Orientation:
    """Create a sample orientation VO."""
    return Orientation.create(yaw=Degrees(0.0), pitch=Degrees(45.0), roll=Degrees(0.0))


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
# CameraIntrinsics VO Tests
# ============================================================================


class TestCameraIntrinsicsFrozen:
    """Test that CameraIntrinsics is truly frozen (immutable)."""

    def test_frozen_sensor_width_raises(self, sample_intrinsics):
        """Attempting to modify sensor_width raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_intrinsics.sensor_width = Millimeters(10.0)

    def test_frozen_focal_length_raises(self, sample_intrinsics):
        """Attempting to modify focal_length raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_intrinsics.focal_length = Millimeters(50.0)

    def test_k_matrix_is_immutable(self, sample_intrinsics):
        """The K matrix is immutable."""
        K = sample_intrinsics.to_K()
        # Matrix3x3 VO wraps the array - attempting to modify should fail
        with pytest.raises((ValueError, TypeError, AttributeError)):
            K._array[0, 0] = 9999.0


class TestCameraIntrinsicsCreateFactory:
    """Test the CameraIntrinsics factory methods."""

    def test_create_basic(self):
        """Create returns a valid CameraIntrinsics instance."""
        intrinsics = CameraIntrinsics.create(
            sensor_width=Millimeters(7.18),
            base_focal_length=Millimeters(5.9),
            image_width=Pixels(1920),
            image_height=Pixels(1080),
            focal_length=Millimeters(5.9),
        )

        assert intrinsics.sensor_width == 7.18
        assert intrinsics.base_focal_length == 5.9
        assert intrinsics.image_width == 1920
        assert intrinsics.image_height == 1080

    def test_from_k_matrix(self, sample_intrinsic_matrix):
        """from_K_matrix creates valid CameraIntrinsics from K matrix."""
        intrinsics = CameraIntrinsics.from_K_matrix(
            sample_intrinsic_matrix,
            Pixels(1920),
            Pixels(1080),
            sensor_width=Millimeters(7.18),
        )

        # Should have valid dimensions
        assert intrinsics.image_width == 1920
        assert intrinsics.image_height == 1080

        # K matrix should be reconstructed
        K = intrinsics.to_K()._to_array()
        assert K.shape == (3, 3)

    def test_create_rejects_invalid_params(self):
        """Create raises ValueError for invalid parameters."""
        with pytest.raises(ValueError, match="sensor_width must be positive"):
            CameraIntrinsics.create(
                sensor_width=Millimeters(-1.0),
                base_focal_length=Millimeters(5.9),
                image_width=Pixels(1920),
                image_height=Pixels(1080),
                focal_length=Millimeters(5.9),
            )


# ============================================================================
# Homography VO Tests
# ============================================================================


class TestHomographyFrozen:
    """Test that Homography is truly frozen (immutable)."""

    @pytest.fixture
    def sample_homography(self) -> Homography:
        """Create a sample Homography VO."""
        H = np.array(
            [
                [100.0, 0.0, 960.0],
                [0.0, 100.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return Homography.create(H)

    def test_frozen_matrix_raises(self, sample_homography):
        """Attempting to modify _matrix raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_homography._matrix = None

    def test_matrix_is_immutable(self, sample_homography):
        """The homography matrix is immutable."""
        H = sample_homography._matrix
        with pytest.raises((ValueError, TypeError, AttributeError)):
            H._array[0, 0] = 9999.0

    def test_inverse_returns_homography(self, sample_homography):
        """inverse property returns a Homography instance."""
        inv = sample_homography.inverse
        assert isinstance(inv, Homography)

    def test_double_inverse_matches_original(self, sample_homography):
        """H.inverse.inverse should match original H."""
        H_original = sample_homography._matrix.to_array()
        H_double_inverse = sample_homography.inverse.inverse._matrix.to_array()
        assert np.allclose(H_original, H_double_inverse)


class TestHomographyCreateFactory:
    """Test Homography.create() factory method."""

    def test_create_valid_homography(self):
        """Create returns a valid Homography instance."""
        H = np.array(
            [
                [100.0, 0.0, 960.0],
                [0.0, 100.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        homography = Homography.create(H)

        assert homography.determinant != 0
        assert homography.condition_number > 0

    def test_create_rejects_singular_matrix(self):
        """Create raises ValueError for singular matrix."""
        H_singular = np.zeros((3, 3))
        with pytest.raises(ValueError, match="singular"):
            Homography.create(H_singular)


# ============================================================================
# Orientation VO Tests
# ============================================================================


class TestOrientationFrozen:
    """Test that Orientation is truly frozen (immutable)."""

    def test_frozen_yaw_raises(self, sample_orientation):
        """Attempting to modify yaw raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_orientation.yaw = Degrees(90.0)

    def test_frozen_pitch_raises(self, sample_orientation):
        """Attempting to modify pitch raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_orientation.pitch = Degrees(60.0)

    def test_frozen_roll_raises(self, sample_orientation):
        """Attempting to modify roll raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_orientation.roll = Degrees(5.0)


class TestOrientationCreateFactory:
    """Test Orientation.create() factory method."""

    def test_create_basic(self):
        """Create returns a valid Orientation instance."""
        orientation = Orientation.create(
            yaw=Degrees(45.0),
            pitch=Degrees(30.0),
            roll=Degrees(2.0),
        )

        assert orientation.yaw == 45.0
        assert orientation.pitch == 30.0
        assert orientation.roll == 2.0


# ============================================================================
# Vector3 VO Tests
# ============================================================================


class TestVector3Frozen:
    """Test that Vector3 is truly frozen (immutable)."""

    def test_frozen_x_raises(self, sample_position_vo):
        """Attempting to modify x raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_position_vo.x = 100.0

    def test_frozen_y_raises(self, sample_position_vo):
        """Attempting to modify y raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_position_vo.y = 100.0

    def test_frozen_z_raises(self, sample_position_vo):
        """Attempting to modify z raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            sample_position_vo.z = 100.0


class TestVector3CreateFactory:
    """Test Vector3.create() factory method."""

    def test_create_basic(self):
        """Create returns a valid Vector3 instance."""
        vec = Vector3.create(1.0, 2.0, 3.0)

        assert vec.x == 1.0
        assert vec.y == 2.0
        assert vec.z == 3.0

    def test_to_array(self):
        """to_array returns numpy array."""
        vec = Vector3.create(1.0, 2.0, 3.0)
        arr = vec._to_array()

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        assert np.allclose(arr, [1.0, 2.0, 3.0])


# ============================================================================
# CameraGeometry.compute_from_vo() Tests (Pure Function)
# ============================================================================


class TestCameraGeometryComputeFromVoPureFunction:
    """Test that CameraGeometry.compute_from_vo() is a pure function."""

    def test_compute_from_vo_returns_consistent_results(
        self, sample_intrinsics, sample_position_vo, sample_orientation
    ):
        """compute_from_vo() returns identical results for identical inputs."""
        result1 = CameraGeometry.compute_from_vo(
            intrinsics=sample_intrinsics,
            camera_position=sample_position_vo,
            orientation=sample_orientation,
        )
        result2 = CameraGeometry.compute_from_vo(
            intrinsics=sample_intrinsics,
            camera_position=sample_position_vo,
            orientation=sample_orientation,
        )

        H1 = result1._matrix._to_array()
        H2 = result2._matrix._to_array()

        assert np.allclose(H1, H2)
        assert result1.condition_number == result2.condition_number
        assert result1.determinant == result2.determinant

    def test_compute_from_vo_returns_homography_vo(
        self, sample_intrinsics, sample_position_vo, sample_orientation
    ):
        """compute_from_vo() returns a Homography VO."""
        result = CameraGeometry.compute_from_vo(
            intrinsics=sample_intrinsics,
            camera_position=sample_position_vo,
            orientation=sample_orientation,
        )

        assert isinstance(result, Homography)

    def test_compute_from_vo_result_is_immutable(
        self, sample_intrinsics, sample_position_vo, sample_orientation
    ):
        """The result from compute_from_vo() is immutable."""
        result = CameraGeometry.compute_from_vo(
            intrinsics=sample_intrinsics,
            camera_position=sample_position_vo,
            orientation=sample_orientation,
        )

        with pytest.raises(FrozenInstanceError):
            result._matrix = None  # type: ignore[assignment]


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
# Integration Tests
# ============================================================================


class TestImmutablePatternIntegration:
    """Integration tests verifying the immutable patterns work together."""

    def test_camera_geometry_full_pipeline_vo(
        self, sample_intrinsics, sample_position_vo, sample_orientation
    ):
        """Full pipeline: VOs -> compute_from_vo() -> Homography VO."""
        # Compute using pure function with VOs
        result = CameraGeometry.compute_from_vo(
            intrinsics=sample_intrinsics,
            camera_position=sample_position_vo,
            orientation=sample_orientation,
        )

        # Result should be a valid Homography VO
        assert isinstance(result, Homography)
        assert result.determinant != 0

        # Should be able to project and inverse-project
        H = result._matrix._to_array()
        assert not np.allclose(H, np.eye(3))

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

    def test_homography_vo_hashable(self):
        """Homography VO can be used in sets and as dict keys."""
        H = np.array(
            [
                [100.0, 0.0, 960.0],
                [0.0, 100.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        homography = Homography.create(H)

        # Should be hashable
        cache = {homography: "result"}
        assert cache[homography] == "result"

        # Should work in sets
        homography_set = {homography}
        assert homography in homography_set


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
