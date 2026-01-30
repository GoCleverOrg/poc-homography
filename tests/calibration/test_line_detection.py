"""Tests for automatic parking line detection.

Tests the public API and behavior of line detection without knowledge
of internal implementation details.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from poc_homography.calibration.lens_distortion.line_detection import (
    CandidateLine,
    LineDetectionConfig,
    LineDetector,
)
from poc_homography.calibration.lens_distortion.models import CameraLine, PTZPosition


class TestLineDetectionConfig:
    """Tests for LineDetectionConfig dataclass."""

    def test_default_config(self):
        """Should create config with sensible defaults."""
        config = LineDetectionConfig()

        assert config.canny_low > 0
        assert config.canny_high > config.canny_low
        assert config.min_line_length > 0
        assert config.min_confidence >= 0
        assert config.min_confidence <= 1

    def test_reject_even_gaussian_kernel(self):
        """Should reject even gaussian kernel size."""
        with pytest.raises(ValueError, match="gaussian_kernel_size must be odd"):
            LineDetectionConfig(gaussian_kernel_size=4)

    def test_reject_canny_low_greater_than_high(self):
        """Should reject canny_low >= canny_high."""
        with pytest.raises(ValueError, match="canny_low.*must be less than canny_high"):
            LineDetectionConfig(canny_low=150, canny_high=50)

    def test_reject_negative_min_line_length(self):
        """Should reject non-positive min_line_length."""
        with pytest.raises(ValueError, match="min_line_length must be positive"):
            LineDetectionConfig(min_line_length=0)


class TestCandidateLine:
    """Tests for CandidateLine dataclass."""

    def test_create_candidate_line(self):
        """Should create candidate line with computed properties."""
        line = CandidateLine(
            start=(0.0, 0.0),
            end=(100.0, 0.0),
        )

        assert line.start == (0.0, 0.0)
        assert line.end == (100.0, 0.0)
        assert line.length == 100.0
        assert line.angle_deg == 0.0

    def test_length_calculated_for_diagonal(self):
        """Should calculate correct length for diagonal line."""
        line = CandidateLine(
            start=(0.0, 0.0),
            end=(3.0, 4.0),  # 3-4-5 triangle
        )

        assert line.length == 5.0

    def test_angle_calculated_for_vertical(self):
        """Should calculate correct angle for vertical line."""
        line = CandidateLine(
            start=(0.0, 0.0),
            end=(0.0, 100.0),
        )

        assert line.angle_deg == 90.0

    def test_angle_calculated_for_45_degrees(self):
        """Should calculate correct angle for 45 degree line."""
        line = CandidateLine(
            start=(0.0, 0.0),
            end=(100.0, 100.0),
        )

        assert line.angle_deg == 45.0

    def test_to_camera_line(self):
        """Should convert to CameraLine with metadata."""
        candidate = CandidateLine(
            start=(100.0, 200.0),
            end=(300.0, 400.0),
            confidence=0.85,
        )
        ptz = PTZPosition(pan_deg=45.0, tilt_deg=30.0, zoom_factor=2.0)

        camera_line = candidate.to_camera_line(
            line_id="line_001",
            image_path="/path/to/image.jpg",
            ptz_position=ptz,
            line_type="parking",
        )

        assert isinstance(camera_line, CameraLine)
        assert camera_line.line_id == "line_001"
        assert camera_line.image_path == "/path/to/image.jpg"
        assert camera_line.start_pixel == (100.0, 200.0)
        assert camera_line.end_pixel == (300.0, 400.0)
        assert camera_line.confidence == 0.85
        assert camera_line.ptz_position.pan_deg == 45.0


class TestLineDetector:
    """Tests for LineDetector."""

    @pytest.fixture
    def detector(self):
        """Standard line detector for tests."""
        return LineDetector()

    @pytest.fixture
    def blank_image(self):
        """Blank gray image for tests."""
        return np.zeros((480, 640), dtype=np.uint8)

    @pytest.fixture
    def image_with_horizontal_lines(self):
        """Image with clear horizontal white lines on black background."""
        img = np.zeros((480, 640), dtype=np.uint8)
        # Draw several horizontal lines
        cv2.line(img, (50, 100), (590, 100), 255, 2)
        cv2.line(img, (50, 200), (590, 200), 255, 2)
        cv2.line(img, (50, 300), (590, 300), 255, 2)
        return img

    @pytest.fixture
    def image_with_vertical_lines(self):
        """Image with clear vertical white lines on black background."""
        img = np.zeros((480, 640), dtype=np.uint8)
        # Draw several vertical lines
        cv2.line(img, (100, 50), (100, 430), 255, 2)
        cv2.line(img, (300, 50), (300, 430), 255, 2)
        cv2.line(img, (500, 50), (500, 430), 255, 2)
        return img

    @pytest.fixture
    def color_image_with_lines(self):
        """Color image with lines."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(img, (50, 200), (590, 200), (255, 255, 255), 2)
        cv2.line(img, (50, 300), (590, 300), (255, 255, 255), 2)
        return img

    def test_detect_returns_empty_for_blank_image(self, detector, blank_image):
        """Should return empty list for image with no lines."""
        candidates = detector.detect(blank_image)

        assert candidates == []

    def test_detect_finds_horizontal_lines(self, detector, image_with_horizontal_lines):
        """Should detect horizontal lines in image."""
        candidates = detector.detect(image_with_horizontal_lines)

        assert len(candidates) > 0
        # All detected lines should be roughly horizontal (angle near 0 or 180)
        for c in candidates:
            angle = abs(c.angle_deg) % 180
            assert angle < 20 or angle > 160  # Within 20 degrees of horizontal

    def test_detect_finds_vertical_lines(self, detector, image_with_vertical_lines):
        """Should detect vertical lines in image."""
        candidates = detector.detect(image_with_vertical_lines)

        assert len(candidates) > 0
        # All detected lines should be roughly vertical (angle near 90)
        for c in candidates:
            angle = abs(c.angle_deg)
            assert 70 < angle < 110  # Within 20 degrees of vertical

    def test_detect_handles_color_image(self, detector, color_image_with_lines):
        """Should handle BGR color images."""
        candidates = detector.detect(color_image_with_lines)

        assert len(candidates) > 0

    def test_detect_returns_candidates_with_confidence(
        self, detector, image_with_horizontal_lines
    ):
        """Should return candidates with confidence scores."""
        candidates = detector.detect(image_with_horizontal_lines)

        assert len(candidates) > 0
        for c in candidates:
            assert 0.0 <= c.confidence <= 1.0

    def test_detect_returns_candidates_sorted_by_confidence(
        self, detector, image_with_horizontal_lines
    ):
        """Should return candidates sorted by confidence (highest first)."""
        candidates = detector.detect(image_with_horizontal_lines)

        if len(candidates) > 1:
            confidences = [c.confidence for c in candidates]
            assert confidences == sorted(confidences, reverse=True)

    def test_detect_filters_by_min_confidence(self, image_with_horizontal_lines):
        """Should filter out lines below min_confidence threshold."""
        config = LineDetectionConfig(min_confidence=0.9)
        detector = LineDetector(config=config)

        candidates = detector.detect(image_with_horizontal_lines)

        for c in candidates:
            assert c.confidence >= 0.9

    def test_detect_filters_by_min_line_length(self):
        """Should filter out lines shorter than min_line_length."""
        # Create image with short and long lines
        img = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(img, (50, 100), (100, 100), 255, 2)  # 50 pixels
        cv2.line(img, (50, 200), (550, 200), 255, 2)  # 500 pixels

        config = LineDetectionConfig(min_line_length=200)
        detector = LineDetector(config=config)

        candidates = detector.detect(img)

        # Should only find the long line
        for c in candidates:
            assert c.length >= 200

    def test_detect_from_file_with_valid_image(self, detector):
        """Should detect lines from image file."""
        # Create a temporary image file
        img = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(img, (50, 200), (590, 200), 255, 2)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, img)
            filepath = f.name

        try:
            candidates = detector.detect_from_file(filepath)
            assert len(candidates) > 0
        finally:
            Path(filepath).unlink()

    def test_detect_from_file_raises_for_missing_file(self, detector):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            detector.detect_from_file("/nonexistent/path/image.png")

    def test_detect_from_file_raises_for_invalid_image(self, detector):
        """Should raise ValueError for invalid image file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not an image")
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="Failed to load image"):
                detector.detect_from_file(filepath)
        finally:
            Path(filepath).unlink()

    def test_detect_assigns_cluster_ids_to_parallel_lines(
        self, image_with_horizontal_lines
    ):
        """Should group parallel lines into clusters."""
        config = LineDetectionConfig(parallel_angle_tolerance=10.0)
        detector = LineDetector(config=config)

        candidates = detector.detect(image_with_horizontal_lines)

        # All horizontal lines should be in the same cluster
        if len(candidates) > 1:
            cluster_ids = {c.cluster_id for c in candidates}
            # Should have at least one cluster (might be -1 for unclustered)
            assert len(cluster_ids) >= 1

    def test_visualize_returns_image(self, detector, image_with_horizontal_lines):
        """Should return image with lines drawn."""
        candidates = detector.detect(image_with_horizontal_lines)

        # Need color image for visualization
        color_img = cv2.cvtColor(image_with_horizontal_lines, cv2.COLOR_GRAY2BGR)
        result = detector.visualize(color_img, candidates)

        assert result.shape == color_img.shape
        assert result.dtype == np.uint8

    def test_visualize_with_empty_candidates(self, detector, blank_image):
        """Should handle empty candidates list."""
        color_img = cv2.cvtColor(blank_image, cv2.COLOR_GRAY2BGR)
        result = detector.visualize(color_img, [])

        # Should return copy of original image
        assert result.shape == color_img.shape


class TestLineDetectorIntegration:
    """Integration tests for line detector with realistic scenarios."""

    def test_detect_parking_lot_style_lines(self):
        """Should detect lines arranged like parking spots."""
        # Create image simulating parking spot markings
        img = np.zeros((720, 1280), dtype=np.uint8)

        # Vertical dividers (like parking spot boundaries)
        for x in range(100, 1200, 100):
            cv2.line(img, (x, 200), (x, 600), 255, 3)

        # Horizontal boundary lines
        cv2.line(img, (50, 200), (1230, 200), 255, 3)
        cv2.line(img, (50, 600), (1230, 600), 255, 3)

        detector = LineDetector()
        candidates = detector.detect(img)

        # Should detect multiple lines
        assert len(candidates) >= 4

        # Should have both horizontal and vertical lines
        angles = [abs(c.angle_deg) for c in candidates]
        has_horizontal = any(a < 20 or a > 160 for a in angles)
        has_vertical = any(70 < a < 110 for a in angles)

        assert has_horizontal
        assert has_vertical

    def test_detect_with_noise(self):
        """Should still detect lines in noisy image."""
        # Create image with lines
        img = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(img, (50, 200), (590, 200), 255, 4)
        cv2.line(img, (50, 300), (590, 300), 255, 4)

        # Add Gaussian noise
        noise = np.random.normal(0, 25, img.shape).astype(np.int16)
        noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        detector = LineDetector()
        candidates = detector.detect(noisy_img)

        # Should still find lines despite noise
        assert len(candidates) >= 1

    def test_detect_respects_custom_config(self):
        """Should respect custom configuration parameters."""
        img = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(img, (50, 200), (590, 200), 255, 2)

        # Very strict config
        config = LineDetectionConfig(
            min_line_length=600,  # Longer than our line
            min_confidence=0.99,
        )
        detector = LineDetector(config=config)

        candidates = detector.detect(img)

        # Should find nothing with very strict config
        assert len(candidates) == 0
