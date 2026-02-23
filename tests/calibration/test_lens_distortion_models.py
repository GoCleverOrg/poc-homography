"""Tests for lens distortion calibration data models.

Tests the public API and behavior of data models without knowledge
of internal implementation details.
"""

import numpy as np
import pytest

from poc_homography.calibration.lens_distortion.models import (
    CameraLine,
    GroundTruthLine,
    LineCorrespondence,
    MapPixelLine,
    PTZPosition,
)


class TestPTZPosition:
    """Tests for PTZPosition data model."""

    def test_create_valid_position(self):
        """Should create position with valid parameters."""
        pos = PTZPosition(pan_deg=45.0, tilt_deg=30.0, zoom_factor=2.0)

        assert pos.pan_deg == 45.0
        assert pos.tilt_deg == 30.0
        assert pos.zoom_factor == 2.0

    def test_reject_zero_zoom(self):
        """Should reject zoom_factor of zero."""
        with pytest.raises(ValueError, match="zoom_factor must be positive"):
            PTZPosition(pan_deg=0.0, tilt_deg=30.0, zoom_factor=0.0)

    def test_reject_negative_zoom(self):
        """Should reject negative zoom_factor."""
        with pytest.raises(ValueError, match="zoom_factor must be positive"):
            PTZPosition(pan_deg=0.0, tilt_deg=30.0, zoom_factor=-1.0)

    def test_allows_negative_pan_and_tilt(self):
        """Should allow negative pan and tilt angles."""
        pos = PTZPosition(pan_deg=-90.0, tilt_deg=-15.0, zoom_factor=1.0)

        assert pos.pan_deg == -90.0
        assert pos.tilt_deg == -15.0

    def test_is_immutable(self):
        """Should be immutable (frozen dataclass)."""
        pos = PTZPosition(pan_deg=0.0, tilt_deg=30.0, zoom_factor=1.0)

        with pytest.raises(AttributeError):
            pos.pan_deg = 45.0  # type: ignore[misc]


class TestCameraLine:
    """Tests for CameraLine data model."""

    @pytest.fixture
    def ptz_position(self):
        """Standard PTZ position for tests."""
        return PTZPosition(pan_deg=0.0, tilt_deg=30.0, zoom_factor=1.0)

    def test_create_valid_line(self, ptz_position):
        """Should create line with valid parameters."""
        line = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=(100.0, 200.0),
            end_pixel=(300.0, 400.0),
            ptz_position=ptz_position,
        )

        assert line.line_id == "line_1"
        assert line.start_pixel == (100.0, 200.0)
        assert line.end_pixel == (300.0, 400.0)

    def test_reject_invalid_start_pixel(self, ptz_position):
        """Should reject start_pixel with wrong number of elements."""
        with pytest.raises(ValueError, match="start_pixel must have 2 elements"):
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 200.0, 300.0),  # type: ignore[arg-type]
                end_pixel=(300.0, 400.0),
                ptz_position=ptz_position,
            )

    def test_reject_invalid_end_pixel(self, ptz_position):
        """Should reject end_pixel with wrong number of elements."""
        with pytest.raises(ValueError, match="end_pixel must have 2 elements"):
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 200.0),
                end_pixel=(300.0,),  # type: ignore[arg-type]
                ptz_position=ptz_position,
            )

    def test_reject_confidence_below_zero(self, ptz_position):
        """Should reject confidence below 0."""
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 200.0),
                end_pixel=(300.0, 400.0),
                ptz_position=ptz_position,
                confidence=-0.1,
            )

    def test_reject_confidence_above_one(self, ptz_position):
        """Should reject confidence above 1."""
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 200.0),
                end_pixel=(300.0, 400.0),
                ptz_position=ptz_position,
                confidence=1.5,
            )

    def test_length_pixels_horizontal_line(self, ptz_position):
        """Should calculate correct length for horizontal line."""
        line = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=(0.0, 0.0),
            end_pixel=(100.0, 0.0),
            ptz_position=ptz_position,
        )

        assert line.length_pixels == 100.0

    def test_length_pixels_vertical_line(self, ptz_position):
        """Should calculate correct length for vertical line."""
        line = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=(0.0, 0.0),
            end_pixel=(0.0, 50.0),
            ptz_position=ptz_position,
        )

        assert line.length_pixels == 50.0

    def test_length_pixels_diagonal_line(self, ptz_position):
        """Should calculate correct length for diagonal line (3-4-5 triangle)."""
        line = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=(0.0, 0.0),
            end_pixel=(3.0, 4.0),
            ptz_position=ptz_position,
        )

        assert line.length_pixels == 5.0

    def test_angle_degrees_horizontal(self, ptz_position):
        """Should return 0 degrees for horizontal line pointing right."""
        line = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=(0.0, 0.0),
            end_pixel=(100.0, 0.0),
            ptz_position=ptz_position,
        )

        assert line.angle_degrees == 0.0

    def test_angle_degrees_vertical(self, ptz_position):
        """Should return 90 degrees for vertical line pointing down."""
        line = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=(0.0, 0.0),
            end_pixel=(0.0, 100.0),
            ptz_position=ptz_position,
        )

        assert line.angle_degrees == 90.0

    def test_angle_degrees_45_degree(self, ptz_position):
        """Should return 45 degrees for diagonal line."""
        line = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=(0.0, 0.0),
            end_pixel=(100.0, 100.0),
            ptz_position=ptz_position,
        )

        assert line.angle_degrees == 45.0

    def test_sample_points_returns_correct_count(self, ptz_position):
        """Should return requested number of sample points."""
        line = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=(0.0, 0.0),
            end_pixel=(100.0, 100.0),
            ptz_position=ptz_position,
        )

        samples = line.sample_points(num_samples=10)

        assert samples.shape == (10, 2)

    def test_sample_points_includes_endpoints(self, ptz_position):
        """Should include start and end points in samples."""
        line = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=(0.0, 0.0),
            end_pixel=(100.0, 50.0),
            ptz_position=ptz_position,
        )

        samples = line.sample_points(num_samples=5)

        np.testing.assert_array_almost_equal(samples[0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(samples[-1], [100.0, 50.0])

    def test_sample_points_uses_edge_pixels_when_available(self, ptz_position):
        """Should return edge_pixels data when available, not interpolated points."""
        edge_pixels = (
            (10.0, 20.0),
            (15.0, 22.0),
            (20.0, 25.0),
            (25.0, 27.0),
            (30.0, 30.0),
            (35.0, 32.0),
        )
        line = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=(10.0, 20.0),
            end_pixel=(35.0, 32.0),
            ptz_position=ptz_position,
            edge_pixels=edge_pixels,
        )

        samples = line.sample_points(num_samples=100)

        # Should return all 6 edge_pixels since 6 < 100
        assert samples.shape == (6, 2)
        np.testing.assert_array_almost_equal(samples[0], [10.0, 20.0])

    def test_sample_points_subsamples_edge_pixels(self, ptz_position):
        """Should subsample edge_pixels when more than num_samples available."""
        edge_pixels = tuple((float(i), float(i * 2)) for i in range(50))
        line = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=(0.0, 0.0),
            end_pixel=(49.0, 98.0),
            ptz_position=ptz_position,
            edge_pixels=edge_pixels,
        )

        samples = line.sample_points(num_samples=10)

        assert samples.shape == (10, 2)
        # First and last should match edge_pixels endpoints
        np.testing.assert_array_almost_equal(samples[0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(samples[-1], [49.0, 98.0])

    def test_sample_points_edge_pixels_with_one_point(self, ptz_position):
        """Should fall back to linear interpolation when edge_pixels has < 2 points."""
        line = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=(0.0, 0.0),
            end_pixel=(100.0, 0.0),
            ptz_position=ptz_position,
            edge_pixels=((50.0, 0.0),),
        )

        # Only 1 edge pixel (< 2), should fall back to interpolation
        samples = line.sample_points(num_samples=5)
        assert samples.shape == (5, 2)

    def test_sample_points_empty_edge_pixels(self, ptz_position):
        """Should fall back to linear interpolation when edge_pixels is empty."""
        line = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=(0.0, 0.0),
            end_pixel=(100.0, 0.0),
            ptz_position=ptz_position,
            edge_pixels=(),
        )

        samples = line.sample_points(num_samples=5)
        assert samples.shape == (5, 2)

    def test_has_edge_curvature_true_for_curved_edges(self, ptz_position):
        """Should return True when edge_pixels deviate from the chord."""
        line = CameraLine(
            line_id="curved",
            image_path="img.jpg",
            start_pixel=(0.0, 0.0),
            end_pixel=(100.0, 0.0),
            ptz_position=ptz_position,
            edge_pixels=((0.0, 0.0), (50.0, 5.0), (100.0, 0.0)),
        )
        assert line.has_edge_curvature() is True

    def test_has_edge_curvature_false_for_collinear_edges(self, ptz_position):
        """Should return False when edge_pixels lie on a straight line."""
        line = CameraLine(
            line_id="straight",
            image_path="img.jpg",
            start_pixel=(0.0, 0.0),
            end_pixel=(100.0, 0.0),
            ptz_position=ptz_position,
            edge_pixels=((0.0, 0.0), (50.0, 0.0), (100.0, 0.0)),
        )
        assert line.has_edge_curvature() is False

    def test_has_edge_curvature_false_without_edge_pixels(self, ptz_position):
        """Should return False when no edge_pixels are provided."""
        line = CameraLine(
            line_id="no_edges",
            image_path="img.jpg",
            start_pixel=(0.0, 0.0),
            end_pixel=(100.0, 0.0),
            ptz_position=ptz_position,
        )
        assert line.has_edge_curvature() is False

    def test_has_edge_curvature_false_for_two_points(self, ptz_position):
        """Should return False when edge_pixels has only 2 points."""
        line = CameraLine(
            line_id="two_pts",
            image_path="img.jpg",
            start_pixel=(0.0, 0.0),
            end_pixel=(100.0, 0.0),
            ptz_position=ptz_position,
            edge_pixels=((0.0, 0.0), (100.0, 0.0)),
        )
        assert line.has_edge_curvature() is False

    def test_to_dict_and_from_dict_roundtrip(self, ptz_position):
        """Should serialize and deserialize without data loss."""
        original = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=(100.5, 200.5),
            end_pixel=(300.5, 400.5),
            ptz_position=ptz_position,
            line_type="boundary",
            confidence=0.95,
        )

        data = original.to_dict()
        restored = CameraLine.from_dict(data)

        assert restored.line_id == original.line_id
        assert restored.image_path == original.image_path
        assert restored.start_pixel == original.start_pixel
        assert restored.end_pixel == original.end_pixel
        assert restored.ptz_position.pan_deg == original.ptz_position.pan_deg
        assert restored.ptz_position.tilt_deg == original.ptz_position.tilt_deg
        assert restored.ptz_position.zoom_factor == original.ptz_position.zoom_factor
        assert restored.line_type == original.line_type
        assert restored.confidence == original.confidence


class TestGroundTruthLine:
    """Tests for GroundTruthLine data model."""

    def test_create_valid_line(self):
        """Should create line with valid parameters."""
        line = GroundTruthLine(
            line_id="gt_1",
            start_world=(10.0, 20.0),
            end_world=(15.0, 25.0),
            map_id="parking_lot_a",
        )

        assert line.line_id == "gt_1"
        assert line.start_world == (10.0, 20.0)
        assert line.end_world == (15.0, 25.0)

    def test_reject_invalid_start_world(self):
        """Should reject start_world with wrong number of elements."""
        with pytest.raises(ValueError, match="start_world must have 2 elements"):
            GroundTruthLine(
                line_id="gt_1",
                start_world=(10.0,),  # type: ignore[arg-type]
                end_world=(15.0, 25.0),
            )

    def test_length_meters(self):
        """Should calculate correct length in meters."""
        line = GroundTruthLine(
            line_id="gt_1",
            start_world=(0.0, 0.0),
            end_world=(3.0, 4.0),  # 3-4-5 triangle
        )

        assert line.length_meters == 5.0

    def test_to_dict_and_from_dict_roundtrip(self):
        """Should serialize and deserialize without data loss."""
        original = GroundTruthLine(
            line_id="gt_1",
            start_world=(10.5, 20.5),
            end_world=(15.5, 25.5),
            map_id="parking_lot_a",
        )

        data = original.to_dict()
        restored = GroundTruthLine.from_dict(data)

        assert restored.line_id == original.line_id
        assert restored.start_world == original.start_world
        assert restored.end_world == original.end_world
        assert restored.map_id == original.map_id


class TestLineCorrespondence:
    """Tests for LineCorrespondence data model."""

    def test_create_correspondence(self):
        """Should create correspondence between camera and ground truth lines."""
        ptz = PTZPosition(pan_deg=0.0, tilt_deg=30.0, zoom_factor=1.0)
        camera_line = CameraLine(
            line_id="cam_1",
            image_path="/path/to/image.jpg",
            start_pixel=(100.0, 200.0),
            end_pixel=(300.0, 400.0),
            ptz_position=ptz,
        )
        ground_truth = GroundTruthLine(
            line_id="gt_1",
            start_world=(10.0, 20.0),
            end_world=(15.0, 25.0),
        )

        correspondence = LineCorrespondence(
            camera_line=camera_line,
            ground_truth_line=ground_truth,
        )

        assert correspondence.camera_line == camera_line
        assert correspondence.ground_truth_line == ground_truth


class TestMapPixelLine:
    """Tests for MapPixelLine data model."""

    def test_create_valid_line(self):
        """Should create line with valid parameters."""
        line = MapPixelLine(
            line_id="map_1",
            start_pixel=(100, 200),
            end_pixel=(300, 400),
            map_id="aerial_map",
        )

        assert line.line_id == "map_1"
        assert line.start_pixel == (100, 200)
        assert line.end_pixel == (300, 400)

    def test_to_world_coordinates_simple_transform(self):
        """Should convert pixel coordinates to world coordinates."""
        line = MapPixelLine(
            line_id="map_1",
            start_pixel=(0, 0),
            end_pixel=(10, 10),
            map_id="test_map",
        )

        # Simple geotransform: origin at (100, 200), 1 meter per pixel
        geotransform = [100.0, 1.0, 0.0, 200.0, 0.0, 1.0]

        world_line = line.to_world_coordinates(geotransform)

        assert world_line.line_id == "map_1"
        assert world_line.start_world == (100.0, 200.0)
        assert world_line.end_world == (110.0, 210.0)

    def test_to_world_coordinates_with_scale(self):
        """Should handle geotransform with scale factor."""
        line = MapPixelLine(
            line_id="map_1",
            start_pixel=(10, 20),
            end_pixel=(30, 40),
            map_id="test_map",
        )

        # 0.5 meters per pixel
        geotransform = [0.0, 0.5, 0.0, 0.0, 0.0, 0.5]

        world_line = line.to_world_coordinates(geotransform)

        assert world_line.start_world == (5.0, 10.0)
        assert world_line.end_world == (15.0, 20.0)

    def test_to_dict_and_from_dict_roundtrip(self):
        """Should serialize and deserialize without data loss."""
        original = MapPixelLine(
            line_id="map_1",
            start_pixel=(100, 200),
            end_pixel=(300, 400),
            map_id="aerial_map",
        )

        data = original.to_dict()
        restored = MapPixelLine.from_dict(data)

        assert restored.line_id == original.line_id
        assert restored.start_pixel == original.start_pixel
        assert restored.end_pixel == original.end_pixel
        assert restored.map_id == original.map_id
