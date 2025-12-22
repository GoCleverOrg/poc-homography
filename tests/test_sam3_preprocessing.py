#!/usr/bin/env python3
"""
Unit tests for SAM3 image preprocessing functions.

Tests verify preprocessing presets: None, CLAHE, and Black/White Threshold.
"""

import unittest
import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import cv2 for testing
import cv2

# Import will be added after we add the function to unified_gcp_tool
# For now, we'll define it locally for testing
def apply_preprocessing(frame, preprocessing_type):
    """Apply preprocessing to frame for SAM3 detection."""
    if preprocessing_type == 'none' or preprocessing_type is None:
        return frame
    elif preprocessing_type == 'clahe':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    elif preprocessing_type == 'threshold':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    else:
        return frame


class TestApplyPreprocessingNone(unittest.TestCase):
    """Test 'none' preprocessing option."""

    def test_none_returns_same_frame(self):
        """Test that 'none' preprocessing returns the input frame unchanged."""
        # Create a test frame with varying intensities
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = apply_preprocessing(frame, 'none')

        # Should return exactly the same frame
        np.testing.assert_array_equal(result, frame)

    def test_none_value_returns_same_frame(self):
        """Test that None preprocessing type returns the input frame unchanged."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = apply_preprocessing(frame, None)

        np.testing.assert_array_equal(result, frame)

    def test_none_preserves_frame_shape(self):
        """Test that 'none' preprocessing preserves frame shape."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = apply_preprocessing(frame, 'none')

        self.assertEqual(result.shape, (480, 640, 3))

    def test_none_preserves_frame_dtype(self):
        """Test that 'none' preprocessing preserves frame dtype."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = apply_preprocessing(frame, 'none')

        self.assertEqual(result.dtype, np.uint8)


class TestApplyPreprocessingCLAHE(unittest.TestCase):
    """Test CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing."""

    def test_clahe_returns_bgr_frame(self):
        """Test that CLAHE preprocessing returns a BGR frame."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = apply_preprocessing(frame, 'clahe')

        # Should return BGR frame with 3 channels
        self.assertEqual(result.shape, (100, 100, 3))
        self.assertEqual(len(result.shape), 3)

    def test_clahe_preserves_frame_dimensions(self):
        """Test that CLAHE preprocessing preserves frame dimensions."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = apply_preprocessing(frame, 'clahe')

        self.assertEqual(result.shape, (480, 640, 3))

    def test_clahe_preserves_dtype(self):
        """Test that CLAHE preprocessing preserves uint8 dtype."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = apply_preprocessing(frame, 'clahe')

        self.assertEqual(result.dtype, np.uint8)

    def test_clahe_enhances_low_contrast_image(self):
        """Test that CLAHE increases contrast in low-contrast images."""
        # Create a low-contrast image (all values between 100-150)
        frame = np.random.randint(100, 150, (256, 256, 3), dtype=np.uint8)
        original_std = np.std(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        result = apply_preprocessing(frame, 'clahe')
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        enhanced_std = np.std(result_gray)

        # CLAHE should increase the standard deviation (spread of pixel values)
        self.assertGreater(enhanced_std, original_std)

    def test_clahe_output_in_valid_range(self):
        """Test that CLAHE output values are in valid range [0, 255]."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = apply_preprocessing(frame, 'clahe')

        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 255)

    def test_clahe_produces_different_output_than_input(self):
        """Test that CLAHE produces different output than input for non-uniform images."""
        # Create a non-uniform image
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[:128, :] = 100  # Top half darker
        frame[128:, :] = 200  # Bottom half brighter

        result = apply_preprocessing(frame, 'clahe')

        # Should produce different output
        self.assertFalse(np.array_equal(result, frame))


class TestApplyPreprocessingThreshold(unittest.TestCase):
    """Test Black/White threshold preprocessing."""

    def test_threshold_returns_bgr_frame(self):
        """Test that threshold preprocessing returns a BGR frame."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = apply_preprocessing(frame, 'threshold')

        # Should return BGR frame with 3 channels
        self.assertEqual(result.shape, (100, 100, 3))
        self.assertEqual(len(result.shape), 3)

    def test_threshold_preserves_frame_dimensions(self):
        """Test that threshold preprocessing preserves frame dimensions."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = apply_preprocessing(frame, 'threshold')

        self.assertEqual(result.shape, (480, 640, 3))

    def test_threshold_preserves_dtype(self):
        """Test that threshold preprocessing preserves uint8 dtype."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = apply_preprocessing(frame, 'threshold')

        self.assertEqual(result.dtype, np.uint8)

    def test_threshold_produces_binary_values(self):
        """Test that threshold preprocessing produces only black (0) or white (255) values."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = apply_preprocessing(frame, 'threshold')
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        unique_values = np.unique(result_gray)

        # Should only contain 0 and/or 255
        self.assertTrue(all(val in [0, 255] for val in unique_values))

    def test_threshold_on_half_black_half_white_image(self):
        """Test threshold preprocessing on a half-black, half-white image."""
        # Create image that's half black (0), half white (255)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, 50:] = 255

        result = apply_preprocessing(frame, 'threshold')
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Should still be binary (0 and 255)
        unique_values = np.unique(result_gray)
        self.assertTrue(all(val in [0, 255] for val in unique_values))

    def test_threshold_uses_otsu_method(self):
        """Test that threshold uses Otsu's method for automatic threshold selection."""
        # Create a bimodal image (two peaks in histogram)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[:100, :] = 80  # Dark region
        frame[100:, :] = 180  # Bright region

        result = apply_preprocessing(frame, 'threshold')
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Otsu's method should separate the two regions
        # Count black and white pixels
        black_pixels = np.sum(result_gray == 0)
        white_pixels = np.sum(result_gray == 255)
        total_pixels = result_gray.size

        # Should have both black and white pixels
        self.assertGreater(black_pixels, 0)
        self.assertGreater(white_pixels, 0)
        self.assertEqual(black_pixels + white_pixels, total_pixels)


class TestApplyPreprocessingInvalidType(unittest.TestCase):
    """Test handling of invalid preprocessing types."""

    def test_invalid_type_returns_original_frame(self):
        """Test that invalid preprocessing type returns the original frame."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = apply_preprocessing(frame, 'invalid_type')

        # Should return the original frame unchanged
        np.testing.assert_array_equal(result, frame)

    def test_empty_string_returns_original_frame(self):
        """Test that empty string preprocessing type returns the original frame."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = apply_preprocessing(frame, '')

        np.testing.assert_array_equal(result, frame)


class TestApplyPreprocessingOriginalFramePreservation(unittest.TestCase):
    """Test that preprocessing does not modify the original frame."""

    def test_clahe_does_not_modify_original_frame(self):
        """Test that CLAHE preprocessing does not modify the original frame."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        frame_copy = frame.copy()

        apply_preprocessing(frame, 'clahe')

        # Original frame should be unchanged
        np.testing.assert_array_equal(frame, frame_copy)

    def test_threshold_does_not_modify_original_frame(self):
        """Test that threshold preprocessing does not modify the original frame."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        frame_copy = frame.copy()

        apply_preprocessing(frame, 'threshold')

        # Original frame should be unchanged
        np.testing.assert_array_equal(frame, frame_copy)


if __name__ == '__main__':
    unittest.main()
