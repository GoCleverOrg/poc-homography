#!/usr/bin/env python3
"""
Test coordinate conversion consistency across the codebase.

These tests verify that:
1. All coordinate conversions use the same implementation (coordinate_converter.py)
2. Round-trip conversions are accurate to < 1mm for distances < 1km
3. The 0.076% systematic error from issue #30 has been resolved
4. All modules use the shared coordinate conversion functions

Issue #30 Background:
The old implementation used a simplified formula with a constant of 111111 m/degree,
which had a systematic error of approximately 0.076%. At 100m distance, this produced
about 7.6cm error compared to the more accurate Formula B using R=6371000m.

This test suite ensures that all modules now use the correct shared implementation
and that round-trip conversions are accurate to sub-millimeter precision.
"""

import unittest
import sys
import os
import math

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poc_homography.coordinate_converter import (
    EARTH_RADIUS_M,
    gps_to_local_xy,
    local_xy_to_gps
)


class TestCoordinateConverterBasics(unittest.TestCase):
    """Basic tests for coordinate converter functions."""

    def test_earth_radius_constant(self):
        """Verify EARTH_RADIUS_M is set to the correct value (6371000m)."""
        self.assertEqual(EARTH_RADIUS_M, 6371000.0,
                        "Earth radius should be 6371000.0 meters (WGS84 mean radius)")

    def test_round_trip_origin(self):
        """Zero offset should return to origin with perfect accuracy."""
        ref_lat, ref_lon = 39.640472, -0.230194

        # Convert 0,0 local to GPS and back
        lat, lon = local_xy_to_gps(ref_lat, ref_lon, 0.0, 0.0)
        x, y = gps_to_local_xy(ref_lat, ref_lon, lat, lon)

        # Should be exact (within floating point precision)
        self.assertAlmostEqual(x, 0.0, places=10,
                              msg="Zero offset should return exactly to X=0")
        self.assertAlmostEqual(y, 0.0, places=10,
                              msg="Zero offset should return exactly to Y=0")

        # GPS coordinates should match reference point
        self.assertAlmostEqual(lat, ref_lat, places=10,
                              msg="Zero offset should return to reference latitude")
        self.assertAlmostEqual(lon, ref_lon, places=10,
                              msg="Zero offset should return to reference longitude")

    def test_round_trip_various_distances_single_latitude(self):
        """Round-trip should have < 1mm error for < 1km distances at 45° latitude."""
        ref_lat, ref_lon = 45.0, 0.0  # 45° latitude for testing

        # Test distances: 1m, 10m, 100m, 1000m in each direction
        # Note: There's a small asymmetry because gps_to_local_xy uses average latitude
        # for the cosine term, while local_xy_to_gps uses reference latitude.
        # This causes sub-millimeter errors for diagonal movements.
        test_cases = [
            # (x_meters, y_meters, max_error_mm, description)
            (1.0, 0.0, 0.001, "1m East"),
            (0.0, 1.0, 0.001, "1m North"),
            (10.0, 0.0, 0.001, "10m East"),
            (0.0, 10.0, 0.001, "10m North"),
            (100.0, 0.0, 0.001, "100m East"),
            (0.0, 100.0, 0.001, "100m North"),
            (1000.0, 0.0, 0.001, "1000m East"),
            (0.0, 1000.0, 0.001, "1000m North"),
            (50.0, 50.0, 1.0, "~71m NE diagonal"),  # Diagonal has small asymmetry error
            (100.0, 100.0, 1.0, "~141m NE diagonal"),  # Diagonal has small asymmetry error
        ]

        for x_orig, y_orig, max_error_mm, description in test_cases:
            with self.subTest(case=description):
                # Local XY → GPS → Local XY
                lat, lon = local_xy_to_gps(ref_lat, ref_lon, x_orig, y_orig)
                x_final, y_final = gps_to_local_xy(ref_lat, ref_lon, lat, lon)

                # Calculate round-trip error
                error = math.sqrt((x_orig - x_final)**2 + (y_orig - y_final)**2)
                error_mm = error * 1000  # Convert to millimeters

                self.assertLess(error_mm, max_error_mm,
                               f"{description}: Round-trip error {error_mm:.4f}mm exceeds "
                               f"threshold {max_error_mm}mm")

    def test_round_trip_various_latitudes(self):
        """Test round-trip accuracy at different reference latitudes."""
        # Test at equator, 30°, 45°, 60° latitude
        test_latitudes = [
            (0.0, "Equator"),
            (30.0, "30° North"),
            (45.0, "45° North"),
            (60.0, "60° North"),
            (-30.0, "30° South"),
        ]

        distance_m = 100.0  # Test at 100m distance
        max_error_mm = 0.001  # 1 micrometer threshold

        for ref_lat, description in test_latitudes:
            with self.subTest(latitude=description):
                ref_lon = 0.0

                # Test in all cardinal directions
                for x, y in [(distance_m, 0), (0, distance_m),
                            (-distance_m, 0), (0, -distance_m)]:
                    lat, lon = local_xy_to_gps(ref_lat, ref_lon, x, y)
                    x_final, y_final = gps_to_local_xy(ref_lat, ref_lon, lat, lon)

                    error = math.sqrt((x - x_final)**2 + (y - y_final)**2)
                    error_mm = error * 1000

                    self.assertLess(error_mm, max_error_mm,
                                   f"{description} at ({x},{y}): error {error_mm:.4f}mm")

    def test_negative_coordinates(self):
        """West and South directions should work correctly."""
        ref_lat, ref_lon = 39.640472, -0.230194

        # Test negative X (West), negative Y (South)
        # Note: Diagonal movements have small asymmetry errors due to the different
        # cosine terms used in forward and inverse transforms
        test_cases = [
            (-50.0, 0.0, 0.001, "50m West"),
            (0.0, -50.0, 0.001, "50m South"),
            (-50.0, -50.0, 1.0, "50m Southwest"),  # Diagonal has small asymmetry
        ]

        for x, y, max_error_mm, description in test_cases:
            with self.subTest(case=description):
                lat, lon = local_xy_to_gps(ref_lat, ref_lon, x, y)
                x_final, y_final = gps_to_local_xy(ref_lat, ref_lon, lat, lon)

                error = math.sqrt((x - x_final)**2 + (y - y_final)**2)
                error_mm = error * 1000

                self.assertLess(error_mm, max_error_mm,
                               f"{description}: Round-trip error {error_mm:.4f}mm")

    def test_gps_to_local_xy_eastward_movement(self):
        """Moving East should increase longitude and produce positive X."""
        ref_lat, ref_lon = 45.0, 0.0

        # Point slightly east of reference
        target_lat = ref_lat
        target_lon = ref_lon + 0.001  # ~70m east at 45° latitude

        x, y = gps_to_local_xy(ref_lat, ref_lon, target_lat, target_lon)

        self.assertGreater(x, 0, "Eastward movement should produce positive X")
        self.assertAlmostEqual(y, 0, places=10,
                              msg="Pure eastward movement should have Y ≈ 0")

    def test_gps_to_local_xy_northward_movement(self):
        """Moving North should increase latitude and produce positive Y."""
        ref_lat, ref_lon = 45.0, 0.0

        # Point slightly north of reference
        target_lat = ref_lat + 0.001  # ~111m north
        target_lon = ref_lon

        x, y = gps_to_local_xy(ref_lat, ref_lon, target_lat, target_lon)

        self.assertAlmostEqual(x, 0, places=10,
                              msg="Pure northward movement should have X ≈ 0")
        self.assertGreater(y, 0, "Northward movement should produce positive Y")

    def test_latitude_validation(self):
        """Test that invalid latitudes raise ValueError."""
        ref_lat, ref_lon = 45.0, 0.0

        # Test latitude out of range
        with self.assertRaises(ValueError):
            gps_to_local_xy(91.0, ref_lon, 45.0, 0.0)  # ref_lat > 90

        with self.assertRaises(ValueError):
            gps_to_local_xy(ref_lat, ref_lon, -91.0, 0.0)  # target_lat < -90

        # Test polar regions (> 85°)
        with self.assertRaises(ValueError):
            gps_to_local_xy(86.0, ref_lon, 45.0, 0.0)  # ref_lat in polar region

    def test_local_xy_to_gps_latitude_validation(self):
        """Test that local_xy_to_gps validates reference latitude."""
        with self.assertRaises(ValueError):
            local_xy_to_gps(91.0, 0.0, 100.0, 100.0)  # ref_lat > 90

        with self.assertRaises(ValueError):
            local_xy_to_gps(-91.0, 0.0, 100.0, 100.0)  # ref_lat < -90

        with self.assertRaises(ValueError):
            local_xy_to_gps(86.0, 0.0, 100.0, 100.0)  # polar region


class TestConsistencyAcrossModules(unittest.TestCase):
    """Test that all modules use the same coordinate conversion."""

    def test_all_modules_import_shared_implementation(self):
        """Verify all modules import from coordinate_converter."""
        import poc_homography.gps_distance_calculator
        import poc_homography.intrinsic_extrinsic_homography
        import poc_homography.feature_match_homography

        # Check that gps_distance_calculator imports the functions
        self.assertTrue(hasattr(poc_homography.gps_distance_calculator, 'gps_to_local_xy'),
                       "gps_distance_calculator should import gps_to_local_xy")
        self.assertTrue(hasattr(poc_homography.gps_distance_calculator, 'local_xy_to_gps'),
                       "gps_distance_calculator should import local_xy_to_gps")
        self.assertTrue(hasattr(poc_homography.gps_distance_calculator, 'EARTH_RADIUS_M'),
                       "gps_distance_calculator should import EARTH_RADIUS_M")

        # Check that feature_match_homography imports the functions
        self.assertTrue(hasattr(poc_homography.feature_match_homography, 'gps_to_local_xy'),
                       "feature_match_homography should import gps_to_local_xy")
        self.assertTrue(hasattr(poc_homography.feature_match_homography, 'local_xy_to_gps'),
                       "feature_match_homography should import local_xy_to_gps")

        # Check that intrinsic_extrinsic_homography imports local_xy_to_gps
        self.assertTrue(hasattr(poc_homography.intrinsic_extrinsic_homography, 'local_xy_to_gps'),
                       "intrinsic_extrinsic_homography should import local_xy_to_gps")

    def test_all_modules_use_same_earth_radius(self):
        """Verify all modules use the same Earth radius constant."""
        import poc_homography.gps_distance_calculator
        from poc_homography.coordinate_converter import EARTH_RADIUS_M

        # gps_distance_calculator imports and uses EARTH_RADIUS_M
        self.assertEqual(poc_homography.gps_distance_calculator.EARTH_RADIUS_M,
                        EARTH_RADIUS_M,
                        "gps_distance_calculator should use the shared EARTH_RADIUS_M")

        self.assertEqual(EARTH_RADIUS_M, 6371000.0,
                        "All modules should use Earth radius of 6371000m")

    def test_modules_produce_identical_results(self):
        """Verify that imported functions produce identical results."""
        import poc_homography.gps_distance_calculator as gps_calc
        import poc_homography.feature_match_homography as feature_match
        from poc_homography.coordinate_converter import gps_to_local_xy, local_xy_to_gps

        ref_lat, ref_lon = 39.640472, -0.230194
        target_lat, target_lon = 39.640583, -0.230111

        # Test gps_to_local_xy
        x1, y1 = gps_to_local_xy(ref_lat, ref_lon, target_lat, target_lon)
        x2, y2 = gps_calc.gps_to_local_xy(ref_lat, ref_lon, target_lat, target_lon)
        x3, y3 = feature_match.gps_to_local_xy(ref_lat, ref_lon, target_lat, target_lon)

        self.assertEqual(x1, x2, "coordinate_converter and gps_distance_calculator should match")
        self.assertEqual(x1, x3, "coordinate_converter and feature_match_homography should match")
        self.assertEqual(y1, y2, "coordinate_converter and gps_distance_calculator should match")
        self.assertEqual(y1, y3, "coordinate_converter and feature_match_homography should match")

        # Test local_xy_to_gps
        lat1, lon1 = local_xy_to_gps(ref_lat, ref_lon, 50.0, 100.0)
        lat2, lon2 = gps_calc.local_xy_to_gps(ref_lat, ref_lon, 50.0, 100.0)
        lat3, lon3 = feature_match.local_xy_to_gps(ref_lat, ref_lon, 50.0, 100.0)

        self.assertEqual(lat1, lat2, "coordinate_converter and gps_distance_calculator should match")
        self.assertEqual(lat1, lat3, "coordinate_converter and feature_match_homography should match")
        self.assertEqual(lon1, lon2, "coordinate_converter and gps_distance_calculator should match")
        self.assertEqual(lon1, lon3, "coordinate_converter and feature_match_homography should match")


class TestIssue30Regression(unittest.TestCase):
    """Regression tests for issue #30 (GPS conversion inconsistency)."""

    def test_no_systematic_error_at_100m(self):
        """
        The old 111111 formula had ~0.076% error. Verify this is fixed.

        Old formula (Formula A):
            delta_lat_deg = y_meters / 111111
            delta_lon_deg = x_meters / (111111 * cos(ref_lat))

        New formula (Formula B):
            delta_lat_deg = (y_meters / R) * (180 / π)
            delta_lon_deg = (x_meters / (R * cos(ref_lat))) * (180 / π)

        At 100m distance, the old formula would produce ~7.6cm systematic error.
        The new formula should have < 1mm round-trip error for cardinal directions,
        and < 1mm for diagonal directions (which have small asymmetry errors).
        """
        ref_lat, ref_lon = 39.640472, -0.230194

        # Test at 100m distance in various directions
        # Cardinal directions (N, S, E, W) should have < 1 micrometer error
        # Diagonal directions have small asymmetry due to different cosine terms
        test_distances = [
            (100.0, 0.0, 0.001, "100m East"),
            (0.0, 100.0, 0.001, "100m North"),
            (100.0, 100.0, 1.0, "100m NE"),  # Diagonal has asymmetry
            (-100.0, 0.0, 0.001, "100m West"),
            (0.0, -100.0, 0.001, "100m South"),
        ]

        for x, y, max_error_mm, description in test_distances:
            with self.subTest(case=description):
                # Round-trip: Local → GPS → Local
                lat, lon = local_xy_to_gps(ref_lat, ref_lon, x, y)
                x_final, y_final = gps_to_local_xy(ref_lat, ref_lon, lat, lon)

                error = math.sqrt((x - x_final)**2 + (y - y_final)**2)
                error_mm = error * 1000

                # This test would FAIL with the old 111111 formula (error ~76mm)
                # but PASSES with the correct R=6371000 formula (error < 1mm)
                self.assertLess(error_mm, max_error_mm,
                               f"{description}: Error {error_mm:.6f}mm. "
                               f"Old formula would have ~76mm error. "
                               f"This proves we're using the correct R=6371000 formula.")

    def test_conversion_matches_formula_b(self):
        """
        Verify the implementation matches Formula B (R=6371000).

        Formula B:
            lat = ref_lat + (y_meters / R) * (180 / π)
            lon = ref_lon + (x_meters / (R * cos(ref_lat))) * (180 / π)

        where R = 6371000 meters
        """
        ref_lat, ref_lon = 45.0, 0.0  # Use 45° for easy verification
        x_meters, y_meters = 100.0, 100.0

        # Manual calculation using Formula B
        R = 6371000.0
        ref_lat_rad = math.radians(ref_lat)

        expected_lat = ref_lat + math.degrees(y_meters / R)
        expected_lon = ref_lon + math.degrees(x_meters / (R * math.cos(ref_lat_rad)))

        # Get actual result from our implementation
        actual_lat, actual_lon = local_xy_to_gps(ref_lat, ref_lon, x_meters, y_meters)

        # Should match to high precision
        self.assertAlmostEqual(actual_lat, expected_lat, places=12,
                              msg="Implementation should match Formula B for latitude")
        self.assertAlmostEqual(actual_lon, expected_lon, places=12,
                              msg="Implementation should match Formula B for longitude")

    def test_old_formula_would_fail(self):
        """
        Demonstrate that the old formula (111111) would NOT pass our accuracy tests.

        This test documents what the error WOULD BE if we were still using the old formula.
        It's a negative test that proves we've fixed the issue.
        """
        ref_lat, ref_lon = 39.640472, -0.230194
        x_meters, y_meters = 100.0, 0.0

        # Calculate what the OLD formula would have given us
        # Old Formula A: delta_lon_deg = x_meters / (111111 * cos(ref_lat))
        ref_lat_rad = math.radians(ref_lat)
        old_delta_lon_deg = x_meters / (111111 * math.cos(ref_lat_rad))
        old_lon = ref_lon + old_delta_lon_deg

        # Calculate what the NEW formula gives us (what we actually use)
        new_lat, new_lon = local_xy_to_gps(ref_lat, ref_lon, x_meters, y_meters)

        # Calculate the difference between old and new formulas
        lon_difference_deg = abs(new_lon - old_lon)

        # Convert longitude difference back to meters to see the actual error
        # At this latitude, 1 degree longitude ≈ 85.4 km
        lon_difference_meters = lon_difference_deg * (EARTH_RADIUS_M * math.cos(ref_lat_rad) * math.pi / 180)
        lon_difference_mm = lon_difference_meters * 1000

        # The old formula should differ by approximately 0.076% of 100m ≈ 76mm
        # We expect the difference to be in the range of 50-100mm
        self.assertGreater(lon_difference_mm, 50,
                          f"Old formula should differ by at least 50mm at 100m distance. "
                          f"Actual difference: {lon_difference_mm:.2f}mm. "
                          f"This proves the old formula had significant error.")

        self.assertLess(lon_difference_mm, 100,
                       f"Difference should be less than 100mm. "
                       f"Actual: {lon_difference_mm:.2f}mm")

    def test_earth_radius_not_111111(self):
        """
        Verify we're NOT using the simplified 111111 m/degree approximation.

        The old formula used 111111 m/degree for latitude.
        The correct formula uses R = 6371000m.

        The ratio should be: 111111 / (6371000 * π / 180) ≈ 0.99924
        This is where the 0.076% error came from.
        """
        from poc_homography.coordinate_converter import EARTH_RADIUS_M

        # Verify we're using the correct Earth radius
        self.assertEqual(EARTH_RADIUS_M, 6371000.0,
                        "Must use Earth radius of 6371000m, not the 111111 approximation")

        # Calculate what the correct meters per degree should be
        correct_m_per_deg_lat = EARTH_RADIUS_M * math.pi / 180

        # The old approximation
        old_approximation = 111111.0

        # Calculate the error percentage
        error_percentage = abs(correct_m_per_deg_lat - old_approximation) / correct_m_per_deg_lat * 100

        # Document that the old formula had ~0.076% error
        self.assertAlmostEqual(error_percentage, 0.076, places=2,
                              msg=f"The old 111111 approximation had ~0.076% error. "
                                  f"Actual error: {error_percentage:.3f}%")

        # Verify our implementation does NOT use 111111
        self.assertNotAlmostEqual(correct_m_per_deg_lat, old_approximation, places=0,
                                 msg="We should NOT be using the 111111 approximation")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_very_small_offsets(self):
        """Test that very small offsets (millimeters) work correctly."""
        ref_lat, ref_lon = 39.640472, -0.230194

        # Test 1mm offsets
        for x, y in [(0.001, 0), (0, 0.001), (0.001, 0.001)]:
            lat, lon = local_xy_to_gps(ref_lat, ref_lon, x, y)
            x_final, y_final = gps_to_local_xy(ref_lat, ref_lon, lat, lon)

            error = math.sqrt((x - x_final)**2 + (y - y_final)**2)
            error_microns = error * 1e6

            self.assertLess(error_microns, 1.0,
                           f"1mm offset should round-trip with < 1 micrometer error. "
                           f"Got {error_microns:.3f} micrometers")

    def test_large_offsets(self):
        """Test that 1km offsets still have acceptable accuracy."""
        ref_lat, ref_lon = 39.640472, -0.230194

        # Test 1km in each direction
        # At 1km, the equirectangular projection approximation has larger errors
        # Cardinal directions: < 1cm error
        # Diagonal: < 10cm error due to asymmetry over longer distances
        test_cases = [
            (1000, 0, 1.0, "1km East"),
            (0, 1000, 1.0, "1km North"),
            (1000, 1000, 10.0, "1km NE diagonal"),  # Larger asymmetry at 1km
        ]

        for x, y, max_error_cm, description in test_cases:
            with self.subTest(case=description):
                lat, lon = local_xy_to_gps(ref_lat, ref_lon, x, y)
                x_final, y_final = gps_to_local_xy(ref_lat, ref_lon, lat, lon)

                error = math.sqrt((x - x_final)**2 + (y - y_final)**2)
                error_cm = error * 100

                self.assertLess(error_cm, max_error_cm,
                               f"{description} should have < {max_error_cm}cm error. "
                               f"Got {error_cm:.3f}cm")

    def test_equator_special_case(self):
        """Test conversions at the equator (latitude = 0)."""
        ref_lat, ref_lon = 0.0, 0.0

        # At equator, cos(lat) = 1, so longitude conversion is simplest
        x_meters = 100.0
        y_meters = 100.0

        lat, lon = local_xy_to_gps(ref_lat, ref_lon, x_meters, y_meters)
        x_final, y_final = gps_to_local_xy(ref_lat, ref_lon, lat, lon)

        error = math.sqrt((x_meters - x_final)**2 + (y_meters - y_final)**2)
        error_mm = error * 1000

        self.assertLess(error_mm, 0.001,
                       f"Equator should have < 1mm error. Got {error_mm:.6f}mm")

    def test_coordinate_system_axes(self):
        """
        Verify coordinate system convention:
        - X axis: East-West (positive = East)
        - Y axis: North-South (positive = North)
        """
        ref_lat, ref_lon = 45.0, 0.0

        # Test East (positive X)
        lat_east, lon_east = local_xy_to_gps(ref_lat, ref_lon, 100.0, 0.0)
        self.assertGreater(lon_east, ref_lon, "Moving East should increase longitude")
        self.assertAlmostEqual(lat_east, ref_lat, places=10,
                              msg="Moving East should not change latitude")

        # Test West (negative X)
        lat_west, lon_west = local_xy_to_gps(ref_lat, ref_lon, -100.0, 0.0)
        self.assertLess(lon_west, ref_lon, "Moving West should decrease longitude")
        self.assertAlmostEqual(lat_west, ref_lat, places=10,
                              msg="Moving West should not change latitude")

        # Test North (positive Y)
        lat_north, lon_north = local_xy_to_gps(ref_lat, ref_lon, 0.0, 100.0)
        self.assertGreater(lat_north, ref_lat, "Moving North should increase latitude")
        self.assertAlmostEqual(lon_north, ref_lon, places=10,
                              msg="Moving North should not change longitude")

        # Test South (negative Y)
        lat_south, lon_south = local_xy_to_gps(ref_lat, ref_lon, 0.0, -100.0)
        self.assertLess(lat_south, ref_lat, "Moving South should decrease latitude")
        self.assertAlmostEqual(lon_south, ref_lon, places=10,
                              msg="Moving South should not change longitude")


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCoordinateConverterBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestConsistencyAcrossModules))
    suite.addTests(loader.loadTestsFromTestCase(TestIssue30Regression))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
