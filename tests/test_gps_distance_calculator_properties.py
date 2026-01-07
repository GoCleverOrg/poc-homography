#!/usr/bin/env python3
"""
Property-based tests for GPS distance calculator using Hypothesis.

These tests verify mathematical properties that MUST hold for the Haversine
distance formula and bearing calculations on a sphere. Property-based testing
generates hundreds of random GPS coordinate pairs to ensure these invariants
hold universally, not just for hand-picked test cases.

Mathematical Background:
-----------------------
The Haversine formula computes great circle distance on a sphere:
    a = sin²(Δφ/2) + cos(φ1)×cos(φ2)×sin²(Δλ/2)
    distance = 2 × R × atan2(√a, √(1-a))

where φ is latitude, λ is longitude, and R is Earth's radius (6,371,000 m).

The bearing formula computes the initial bearing from point A to point B:
    bearing = atan2(sin(Δλ)×cos(φ2), cos(φ1)×sin(φ2) - sin(φ1)×cos(φ2)×cos(Δλ))

Properties Verified:
-------------------
1. Distance Symmetry: distance(A, B) == distance(B, A)
   - Great circle distance is symmetric (distance is the same both directions)

2. Triangle Inequality: distance(A, C) <= distance(A, B) + distance(B, C)
   - The direct route is always shorter than or equal to any detour

3. Zero Distance: distance(A, A) == 0
   - Distance from any point to itself is zero

4. Distance Positive Definite: distance(A, B) >= 0 for all A, B
   - Distance is always non-negative

5. Meridian Distance: Distance along meridian (same longitude) should match
   latitude difference × Earth radius × π/180
   - Special case check for North-South movement

6. Equator Distance: Distance along equator (latitude=0) should match
   longitude difference × Earth radius × π/180
   - Special case check for East-West movement at equator

7. Bearing North Property: Bearing due north should be 0° (or 360°)
   - Cardinal direction verification

8. Bearing East Property: Bearing due east should be 90°
   - Cardinal direction verification

Reference:
---------
- EARTH_RADIUS_M = 6,371,000 meters (from coordinate_converter.py)
- Haversine formula: https://en.wikipedia.org/wiki/Haversine_formula
- Great circle distance: https://en.wikipedia.org/wiki/Great-circle_distance

Note on Bearing Consistency:
----------------------------
A bearing consistency test (verifying that bearing(A,B) ≈ bearing(B,A) + 180°)
was considered but not included due to complex edge cases in spherical geometry.
While this property holds approximately for most coordinate pairs, convergence of
meridians and great circle curvature cause significant deviations (>10°) at certain
latitude/distance combinations, making it unsuitable for property-based testing
across the full coordinate space.
"""

import unittest
import sys
import os
import math
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hypothesis import given, strategies as st, settings, assume
from poc_homography.coordinate_converter import EARTH_RADIUS_M
from poc_homography.gps_distance_calculator import (
    haversine_distance,
    bearing_between_points
)


# Hypothesis strategies for generating GPS coordinates
# Latitude: -85 to 85 degrees (avoid polar regions where formulas break down)
# Longitude: -180 to 180 degrees (full range)
latitude_strategy = st.floats(min_value=-85.0, max_value=85.0, allow_nan=False, allow_infinity=False)
longitude_strategy = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)


def gps_coordinate_strategy():
    """Generate a GPS coordinate (lat, lon) pair."""
    return st.tuples(latitude_strategy, longitude_strategy)


def gps_coordinate_pair_strategy():
    """Generate two GPS coordinates for testing pairwise properties."""
    return st.tuples(gps_coordinate_strategy(), gps_coordinate_strategy())


def gps_coordinate_triple_strategy():
    """Generate three GPS coordinates for testing triangle inequality."""
    return st.tuples(
        gps_coordinate_strategy(),
        gps_coordinate_strategy(),
        gps_coordinate_strategy()
    )


def normalize_bearing(bearing: float) -> float:
    """Normalize bearing to [0, 360) range."""
    return bearing % 360.0


class TestGPSDistanceCalculatorProperties(unittest.TestCase):
    """Property-based tests for GPS distance and bearing calculations."""

    @given(gps_coordinate_pair_strategy())
    @settings(max_examples=100)
    def test_distance_symmetry(self, coords):
        """
        Property: Distance is symmetric - distance(A, B) == distance(B, A).

        WHY this must hold:
        Great circle distance on a sphere is inherently symmetric. The shortest
        path between two points has the same length regardless of which point
        you start from. This is a fundamental property of metric spaces.

        Mathematical basis:
        The Haversine formula computes the great circle distance using the
        central angle between two points. Since sin²(x) and cos(x) are symmetric
        operations, and the formula treats both points equivalently in the
        calculation of Δφ and Δλ, the distance must be the same in both directions.
        """
        (lat1, lon1), (lat2, lon2) = coords

        dist_forward = haversine_distance(lat1, lon1, lat2, lon2)
        dist_reverse = haversine_distance(lat2, lon2, lat1, lon1)

        # Distances should be exactly equal (within floating point precision)
        np.testing.assert_allclose(
            dist_forward,
            dist_reverse,
            rtol=1e-10,
            atol=1e-6,  # 1 micrometer absolute tolerance
            err_msg=f"Distance symmetry violated: "
                    f"d({lat1:.6f},{lon1:.6f} → {lat2:.6f},{lon2:.6f}) = {dist_forward:.6f}m, "
                    f"d({lat2:.6f},{lon2:.6f} → {lat1:.6f},{lon1:.6f}) = {dist_reverse:.6f}m"
        )

    @given(gps_coordinate_triple_strategy())
    @settings(max_examples=100)
    def test_triangle_inequality(self, coords):
        """
        Property: Triangle inequality - distance(A, C) <= distance(A, B) + distance(B, C).

        WHY this must hold:
        The triangle inequality is a fundamental property of metric spaces and
        geometry. On a sphere, the great circle path between two points is the
        shortest possible path. Any detour through a third point must be at least
        as long as the direct path (and usually longer).

        Mathematical basis:
        For any three points A, B, C on a sphere, the direct great circle distance
        from A to C cannot exceed the sum of the great circle distances A→B and B→C.
        This is because the great circle is the geodesic (shortest path) on the
        sphere's surface. The only case where equality holds is when B lies on the
        great circle path from A to C.

        Numerical tolerance:
        We use a small tolerance (1mm + 1e-6 relative) to account for:
        1. Floating point rounding errors in trigonometric calculations
        2. Numerical precision limitations in the atan2 function
        3. Round-off errors when summing distances
        """
        (lat_a, lon_a), (lat_b, lon_b), (lat_c, lon_c) = coords

        dist_ac = haversine_distance(lat_a, lon_a, lat_c, lon_c)
        dist_ab = haversine_distance(lat_a, lon_a, lat_b, lon_b)
        dist_bc = haversine_distance(lat_b, lon_b, lat_c, lon_c)

        detour_distance = dist_ab + dist_bc

        # Triangle inequality: direct distance <= sum of two sides
        # Allow small numerical tolerance (1mm + relative error)
        tolerance = 1e-3 + 1e-6 * max(dist_ac, detour_distance)

        self.assertLessEqual(
            dist_ac,
            detour_distance + tolerance,
            msg=f"Triangle inequality violated:\n"
                f"  A=({lat_a:.6f}, {lon_a:.6f})\n"
                f"  B=({lat_b:.6f}, {lon_b:.6f})\n"
                f"  C=({lat_c:.6f}, {lon_c:.6f})\n"
                f"  Direct A→C: {dist_ac:.6f}m\n"
                f"  Detour A→B→C: {detour_distance:.6f}m\n"
                f"  Violation: {dist_ac - detour_distance:.6f}m > tolerance {tolerance:.6f}m"
        )

    @given(gps_coordinate_strategy())
    @settings(max_examples=100)
    def test_zero_distance_property(self, coord):
        """
        Property: Zero distance - distance from any point to itself is zero.

        WHY this must hold:
        This is the identity property of metric spaces. The distance from any
        point to itself must be exactly zero. This is one of the fundamental
        axioms that defines what a "distance" means mathematically.

        Mathematical basis:
        In the Haversine formula, when (lat1, lon1) == (lat2, lon2):
        - Δφ = 0 and Δλ = 0
        - sin²(0/2) = 0
        - a = 0
        - distance = 2 × R × atan2(0, 1) = 2 × R × 0 = 0

        The calculation should produce exactly 0.0, not just a very small number.
        """
        lat, lon = coord

        distance = haversine_distance(lat, lon, lat, lon)

        # Distance to self should be exactly zero
        np.testing.assert_allclose(
            distance,
            0.0,
            rtol=0,
            atol=1e-10,  # Essentially zero, allowing only for floating point noise
            err_msg=f"Distance from ({lat:.6f}, {lon:.6f}) to itself is {distance:.10e}, expected 0.0"
        )

    @given(gps_coordinate_pair_strategy())
    @settings(max_examples=100)
    def test_distance_positive_definite(self, coords):
        """
        Property: Distance is positive definite - distance(A, B) >= 0 for all A, B.

        WHY this must hold:
        Distance is a measure of separation and can never be negative. This is
        another fundamental axiom of metric spaces. The only way to have zero
        distance is if the two points are the same (tested separately).

        Mathematical basis:
        The Haversine formula computes:
            distance = 2 × R × atan2(√a, √(1-a))

        Since R > 0 (Earth's radius), and atan2(√a, √(1-a)) returns a value in
        [0, π] for the great circle calculation, the result is always >= 0.

        Non-negative outputs from atan2:
        - √a >= 0 (square root is non-negative)
        - √(1-a) >= 0 (when a <= 1, which is always true for valid coordinates)
        - atan2(positive, positive) returns angle in [0, π/2]
        - 2 × R × [0, π/2] >= 0
        """
        (lat1, lon1), (lat2, lon2) = coords

        distance = haversine_distance(lat1, lon1, lat2, lon2)

        self.assertGreaterEqual(
            distance,
            0.0,
            msg=f"Distance is negative: d({lat1:.6f},{lon1:.6f} → {lat2:.6f},{lon2:.6f}) = {distance:.6f}m"
        )

    @given(latitude_strategy, st.floats(min_value=-180, max_value=180),
           st.floats(min_value=0.001, max_value=10.0))
    @settings(max_examples=100)
    def test_meridian_distance_property(self, lat1, lon, lat_offset_deg):
        """
        Property: Distance along a meridian (same longitude) should equal
        the latitude difference × Earth radius × π/180.

        WHY this must hold:
        A meridian is a great circle passing through both poles. When traveling
        along a meridian (constant longitude), you're moving along a circle of
        radius R (Earth's radius). The arc length formula for a circle is:
            arc_length = radius × angle_in_radians

        Mathematical basis:
        For movement along a meridian (Δλ = 0):
        - The great circle distance simplifies considerably
        - Distance = R × Δφ (where Δφ is in radians)
        - Distance = R × (Δφ_degrees × π/180)

        This is a special case where the Haversine formula should reduce to
        simple arc length calculation.

        Tolerance: 1mm + relative error to account for floating point arithmetic.
        """
        lat2 = lat1 + lat_offset_deg

        # Skip if lat2 goes out of valid range
        assume(-85.0 <= lat2 <= 85.0)

        distance = haversine_distance(lat1, lon, lat2, lon)
        expected_distance = EARTH_RADIUS_M * abs(lat_offset_deg) * (math.pi / 180.0)

        # Allow small numerical tolerance
        np.testing.assert_allclose(
            distance,
            expected_distance,
            rtol=1e-6,
            atol=1e-3,  # 1mm tolerance
            err_msg=f"Meridian distance incorrect:\n"
                    f"  From: ({lat1:.6f}, {lon:.6f})\n"
                    f"  To:   ({lat2:.6f}, {lon:.6f})\n"
                    f"  Calculated: {distance:.6f}m\n"
                    f"  Expected:   {expected_distance:.6f}m\n"
                    f"  Difference: {abs(distance - expected_distance):.6f}m"
        )

    @given(st.floats(min_value=-180, max_value=180),
           st.floats(min_value=0.001, max_value=10.0))
    @settings(max_examples=100)
    def test_equator_distance_property(self, lon1, lon_offset_deg):
        """
        Property: Distance along the equator (latitude=0) should equal
        the longitude difference × Earth radius × π/180.

        WHY this must hold:
        The equator is a great circle with radius R (Earth's radius). When
        traveling along the equator (latitude = 0), you're moving along a circle
        of radius R. The arc length formula applies directly:
            arc_length = radius × angle_in_radians

        Mathematical basis:
        For movement along the equator (φ = 0):
        - cos(0) = 1
        - The Haversine formula simplifies to pure longitudinal distance
        - Distance = R × Δλ (where Δλ is in radians)
        - Distance = R × (Δλ_degrees × π/180)

        This is another special case where the Haversine formula should reduce
        to simple arc length calculation.

        Tolerance: 1mm + relative error to account for floating point arithmetic.
        """
        lat = 0.0  # Equator
        lon2 = lon1 + lon_offset_deg

        # Normalize longitude to [-180, 180]
        lon2 = ((lon2 + 180) % 360) - 180

        distance = haversine_distance(lat, lon1, lat, lon2)

        # Calculate expected distance (shortest path around sphere)
        lon_diff = abs(lon2 - lon1)
        if lon_diff > 180:
            lon_diff = 360 - lon_diff  # Take shorter path around sphere

        expected_distance = EARTH_RADIUS_M * lon_diff * (math.pi / 180.0)

        # Allow small numerical tolerance
        np.testing.assert_allclose(
            distance,
            expected_distance,
            rtol=1e-6,
            atol=1e-3,  # 1mm tolerance
            err_msg=f"Equator distance incorrect:\n"
                    f"  From: ({lat:.6f}, {lon1:.6f})\n"
                    f"  To:   ({lat:.6f}, {lon2:.6f})\n"
                    f"  Longitude difference: {lon_diff:.6f}°\n"
                    f"  Calculated: {distance:.6f}m\n"
                    f"  Expected:   {expected_distance:.6f}m\n"
                    f"  Difference: {abs(distance - expected_distance):.6f}m"
        )

    @given(gps_coordinate_strategy())
    @settings(max_examples=100)
    def test_bearing_north_property(self, coord):
        """
        Property: Bearing due north from any point should be 0° (or 360°).

        WHY this must hold:
        Moving due north means increasing latitude while keeping longitude constant.
        By navigation convention, North is defined as 0° (or equivalently 360°).

        Mathematical basis:
        When moving due north (Δλ = 0, Δφ > 0):
        - sin(Δλ) = 0
        - The bearing formula should yield 0°

        This tests that the bearing calculation correctly handles cardinal directions.
        """
        lat, lon = coord

        # Move 0.1 degrees north (small enough to avoid polar regions)
        lat_north = lat + 0.1
        assume(lat_north <= 85.0)  # Stay away from pole

        bearing = bearing_between_points(lat, lon, lat_north, lon)

        # Bearing should be 0° (North) or 360° (equivalent)
        bearing_normalized = normalize_bearing(bearing)

        # Check if bearing is close to 0° or 360°
        is_north = bearing_normalized < 0.01 or bearing_normalized > 359.99

        self.assertTrue(
            is_north,
            msg=f"Bearing due north incorrect:\n"
                f"  From: ({lat:.6f}, {lon:.6f})\n"
                f"  To:   ({lat_north:.6f}, {lon:.6f})\n"
                f"  Bearing: {bearing:.4f}° (expected ~0° or ~360°)"
        )

    @given(gps_coordinate_strategy())
    @settings(max_examples=100)
    def test_bearing_east_property(self, coord):
        """
        Property: Bearing due east from any point should be 90°.

        WHY this must hold:
        Moving due east means increasing longitude while keeping latitude constant.
        By navigation convention, East is defined as 90°.

        Mathematical basis:
        When moving due east (Δφ = 0, Δλ > 0):
        - The bearing formula should yield 90°

        This tests that the bearing calculation correctly handles cardinal directions.

        Note: Due to spherical geometry, "due east" at different latitudes travels
        along different circles (not great circles except at equator), but the
        initial bearing should still be 90°.
        """
        lat, lon = coord

        # Move 0.1 degrees east (small enough for initial bearing to be meaningful)
        lon_east = lon + 0.1
        assume(lon_east <= 180.0)  # Stay within valid range

        # Correct parameter order: (lat1, lon1, lat2, lon2)
        bearing = bearing_between_points(lat, lon, lat, lon_east)

        # Bearing should be 90° (East)
        # Allow slightly larger tolerance due to spherical geometry effects at higher latitudes
        np.testing.assert_allclose(
            bearing,
            90.0,
            rtol=0,
            atol=0.05,  # 0.05 degree tolerance (spherical geometry effects)
            err_msg=f"Bearing due east incorrect:\n"
                    f"  From: ({lat:.6f}, {lon:.6f})\n"
                    f"  To:   ({lat:.6f}, {lon_east:.6f})\n"
                    f"  Bearing: {bearing:.4f}° (expected 90.0°)"
        )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
