#!/usr/bin/env python3
"""
Unit tests for MaskMatcher interface and ECCMaskMatcher implementation.

Tests verify the strategy pattern interface, ECC correlation computation,
mask preprocessing, error handling, and edge cases.
"""

import unittest
import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poc_homography.mask_matcher import MaskMatcher, ECCMaskMatcher


class TestMaskMatcherInterface(unittest.TestCase):
    """Test the MaskMatcher abstract interface."""

    def test_ecc_mask_matcher_inherits_from_mask_matcher(self):
        """Test that ECCMaskMatcher inherits from MaskMatcher."""
        self.assertTrue(issubclass(ECCMaskMatcher, MaskMatcher))

    def test_ecc_mask_matcher_has_compute_correlation_method(self):
        """Test that ECCMaskMatcher implements compute_correlation method."""
        matcher = ECCMaskMatcher()
        self.assertTrue(hasattr(matcher, 'compute_correlation'))
        self.assertTrue(callable(getattr(matcher, 'compute_correlation')))


class TestECCMaskMatcherBasicFunctionality(unittest.TestCase):
    """Test basic functionality of ECCMaskMatcher."""

    def setUp(self):
        """Set up test fixtures."""
        self.matcher = ECCMaskMatcher()

    def test_identical_masks_high_correlation(self):
        """Test that identical masks produce correlation close to 1.0."""
        # Create a checkerboard pattern
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[::16, ::16] = 255

        score = self.matcher.compute_correlation(mask, mask.copy())

        # Identical masks should have very high correlation (>0.99)
        self.assertGreater(score, 0.99)
        self.assertLessEqual(score, 1.0)

    def test_different_masks_lower_correlation(self):
        """Test that different masks produce lower correlation."""
        # Create two different patterns
        mask1 = np.zeros((256, 256), dtype=np.uint8)
        mask1[:128, :] = 255  # Top half white

        mask2 = np.zeros((256, 256), dtype=np.uint8)
        mask2[:, :128] = 255  # Left half white

        score = self.matcher.compute_correlation(mask1, mask2)

        # Different masks should have lower correlation than identical ones
        self.assertLess(score, 0.9)

    def test_completely_different_masks_very_low_correlation(self):
        """Test that completely different masks produce very low correlation."""
        # Create two very different patterns
        mask1 = np.zeros((256, 256), dtype=np.uint8)
        mask1[0, 0] = 255  # Single pixel in corner

        mask2 = np.zeros((256, 256), dtype=np.uint8)
        mask2[255, 255] = 255  # Single pixel in opposite corner

        score = self.matcher.compute_correlation(mask1, mask2)

        # Should have very low or zero correlation
        self.assertLessEqual(score, 0.5)

    def test_empty_masks_returns_zero(self):
        """Test that empty (all-black) masks return 0.0 correlation."""
        mask1 = np.zeros((256, 256), dtype=np.uint8)
        mask2 = np.zeros((256, 256), dtype=np.uint8)

        score = self.matcher.compute_correlation(mask1, mask2)

        # Empty masks should return 0.0 (ECC fails to converge)
        self.assertEqual(score, 0.0)

    def test_correlation_in_valid_range(self):
        """Test that correlation is always in [0.0, 1.0] range."""
        # Create random masks
        np.random.seed(42)
        mask1 = (np.random.rand(256, 256) > 0.5).astype(np.uint8) * 255
        mask2 = (np.random.rand(256, 256) > 0.5).astype(np.uint8) * 255

        score = self.matcher.compute_correlation(mask1, mask2)

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestECCMaskMatcherMaskPreprocessing(unittest.TestCase):
    """Test mask preprocessing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.matcher = ECCMaskMatcher()

    def test_different_sized_masks_handled_correctly(self):
        """Test that masks of different sizes are preprocessed correctly."""
        # Create masks of different sizes with similar patterns
        mask1 = np.zeros((128, 256), dtype=np.uint8)
        mask1[::8, ::8] = 255

        mask2 = np.zeros((512, 512), dtype=np.uint8)
        mask2[::32, ::32] = 255

        # Should not raise exception
        score = self.matcher.compute_correlation(mask1, mask2)

        # Score should be valid (both resized to same size internally)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_3d_masks_converted_to_2d(self):
        """Test that 3-channel masks are converted to 2D correctly."""
        # Create identical 3-channel masks (simulating BGR format)
        mask1 = np.zeros((256, 256, 3), dtype=np.uint8)
        mask1[::16, ::16, :] = 255

        mask2 = mask1.copy()

        score = self.matcher.compute_correlation(mask1, mask2)

        # Should have high correlation (identical after conversion to 2D)
        self.assertGreater(score, 0.99)

    def test_single_channel_3d_mask(self):
        """Test 3D mask with single channel."""
        mask1 = np.zeros((256, 256, 1), dtype=np.uint8)
        mask1[::16, ::16, 0] = 255

        mask2 = mask1.copy()

        score = self.matcher.compute_correlation(mask1, mask2)

        self.assertGreater(score, 0.99)


class TestECCMaskMatcherErrorHandling(unittest.TestCase):
    """Test error handling and validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.matcher = ECCMaskMatcher()

    def test_non_numpy_input_raises_type_error(self):
        """Test that non-numpy array inputs raise TypeError."""
        with self.assertRaises(TypeError) as context:
            self.matcher.compute_correlation([1, 2, 3], [4, 5, 6])

        self.assertIn("numpy arrays", str(context.exception))

    def test_empty_array_raises_value_error(self):
        """Test that empty arrays raise ValueError."""
        empty = np.array([], dtype=np.uint8)
        valid = np.zeros((256, 256), dtype=np.uint8)

        with self.assertRaises(ValueError) as context:
            self.matcher.compute_correlation(empty, valid)

        self.assertIn("empty", str(context.exception))

    def test_invalid_dimensions_raise_value_error(self):
        """Test that arrays with invalid dimensions raise ValueError."""
        # 1D array should fail
        mask1d = np.zeros(256, dtype=np.uint8)
        valid = np.zeros((256, 256), dtype=np.uint8)

        with self.assertRaises(ValueError) as context:
            self.matcher.compute_correlation(mask1d, valid)

        self.assertIn("2D or 3D", str(context.exception))

    def test_4d_array_raises_value_error(self):
        """Test that 4D arrays raise ValueError."""
        mask4d = np.zeros((10, 256, 256, 3), dtype=np.uint8)
        valid = np.zeros((256, 256), dtype=np.uint8)

        with self.assertRaises(ValueError) as context:
            self.matcher.compute_correlation(mask4d, valid)

        self.assertIn("2D or 3D", str(context.exception))


class TestECCMaskMatcherCustomParameters(unittest.TestCase):
    """Test ECCMaskMatcher with custom parameters."""

    def test_custom_target_size(self):
        """Test initialization with custom target size."""
        matcher = ECCMaskMatcher(target_size=(256, 256))

        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[::8, ::8] = 255

        score = matcher.compute_correlation(mask, mask.copy())

        self.assertGreater(score, 0.99)

    def test_custom_max_iterations(self):
        """Test initialization with custom max iterations."""
        matcher = ECCMaskMatcher(max_iterations=100)

        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[::16, ::16] = 255

        score = matcher.compute_correlation(mask, mask.copy())

        self.assertGreater(score, 0.99)

    def test_custom_epsilon(self):
        """Test initialization with custom epsilon."""
        matcher = ECCMaskMatcher(epsilon=1e-4)

        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[::16, ::16] = 255

        score = matcher.compute_correlation(mask, mask.copy())

        self.assertGreater(score, 0.99)

    def test_all_custom_parameters(self):
        """Test initialization with all custom parameters."""
        matcher = ECCMaskMatcher(
            target_size=(384, 384),
            max_iterations=75,
            epsilon=5e-4
        )

        self.assertEqual(matcher.target_size, (384, 384))
        self.assertEqual(matcher.max_iterations, 75)
        self.assertEqual(matcher.epsilon, 5e-4)


class TestECCMaskMatcherEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.matcher = ECCMaskMatcher()

    def test_all_white_masks(self):
        """Test correlation between all-white masks."""
        mask1 = np.full((256, 256), 255, dtype=np.uint8)
        mask2 = np.full((256, 256), 255, dtype=np.uint8)

        score = self.matcher.compute_correlation(mask1, mask2)

        # All-white masks may fail to converge (no features)
        # Should return 0.0 gracefully
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_single_pixel_difference(self):
        """Test masks with single pixel difference."""
        mask1 = np.zeros((256, 256), dtype=np.uint8)
        mask1[::16, ::16] = 255

        mask2 = mask1.copy()
        mask2[128, 128] = 0  # Change one pixel

        score = self.matcher.compute_correlation(mask1, mask2)

        # Should still have very high correlation
        self.assertGreater(score, 0.95)

    def test_rotated_mask(self):
        """Test correlation with rotated mask."""
        # Create asymmetric pattern
        mask1 = np.zeros((256, 256), dtype=np.uint8)
        mask1[64:192, 64:128] = 255  # Rectangle

        # Rotate 90 degrees
        mask2 = np.rot90(mask1)

        score = self.matcher.compute_correlation(mask1, mask2)

        # ECC should handle rotation (MOTION_EUCLIDEAN supports rotation)
        # But correlation may be lower than identical
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_small_masks(self):
        """Test with very small masks."""
        mask1 = np.zeros((16, 16), dtype=np.uint8)
        mask1[::4, ::4] = 255

        mask2 = mask1.copy()

        score = self.matcher.compute_correlation(mask1, mask2)

        # Should still work with small masks
        self.assertGreater(score, 0.9)

    def test_large_masks(self):
        """Test with large masks."""
        mask1 = np.zeros((2048, 2048), dtype=np.uint8)
        # Use solid regions that survive downsampling to 512x512
        mask1[512:1536, 512:1536] = 255  # Large central square

        mask2 = mask1.copy()

        score = self.matcher.compute_correlation(mask1, mask2)

        # Should handle large masks (will be resized to 512x512)
        self.assertGreater(score, 0.99)


class TestECCMaskMatcherRealWorldScenarios(unittest.TestCase):
    """Test scenarios that mimic real-world usage."""

    def setUp(self):
        """Set up test fixtures."""
        self.matcher = ECCMaskMatcher()

    def test_water_mask_simulation(self):
        """Test with simulated water segmentation masks."""
        # Simulate water body in center
        mask1 = np.zeros((512, 512), dtype=np.uint8)
        cv2_available = True
        try:
            import cv2
            cv2.circle(mask1, (256, 256), 100, 255, -1)
        except ImportError:
            # Fallback to simple pattern if cv2 not available
            mask1[156:356, 156:356] = 255
            cv2_available = False

        # Same water body, slightly shifted
        mask2 = np.zeros((512, 512), dtype=np.uint8)
        if cv2_available:
            import cv2
            cv2.circle(mask2, (266, 266), 100, 255, -1)  # Shifted by 10 pixels
        else:
            mask2[166:366, 166:366] = 255

        score = self.matcher.compute_correlation(mask1, mask2)

        # Should have good correlation despite shift (ECC handles translation)
        self.assertGreater(score, 0.7)

    def test_sparse_feature_masks(self):
        """Test with sparse feature masks (few features)."""
        mask1 = np.zeros((512, 512), dtype=np.uint8)
        mask1[100, 100] = 255
        mask1[200, 200] = 255
        mask1[300, 300] = 255

        mask2 = mask1.copy()

        score = self.matcher.compute_correlation(mask1, mask2)

        # Sparse features may not converge well
        # Should return valid score or 0.0
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == '__main__':
    unittest.main()
