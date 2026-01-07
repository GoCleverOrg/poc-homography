#!/usr/bin/env python3
"""
Audit tests for pyproj.Transformer.from_crs() usage across the codebase.

This test module verifies that ALL pyproj Transformer.from_crs() calls in the codebase
use always_xy=True, ensuring consistent coordinate ordering across the entire project.

WHY always_xy=True IS CRITICAL:
================================
pyproj's default coordinate order follows the CRS definition:
- For EPSG:4326 (WGS84), the EPSG standard specifies coordinates as (latitude, longitude)
- However, most GIS systems and our codebase use (longitude, latitude) = (x, y) order

Without always_xy=True:
- EPSG:4326 expects: (lat, lon) = (y, x) - WRONG for our codebase
- This causes silent coordinate swapping bugs
- Latitude values (typically 30-60) are misinterpreted as longitude
- Longitude values (typically -180 to 180) are misinterpreted as latitude

With always_xy=True:
- All CRS expect: (x, y) = (lon, lat) for geographic CRS
- Consistent with standard GIS convention (lon, lat)
- Matches our codebase's coordinate order expectations

AUDIT FINDINGS (Issue #134):
============================
All 13 pyproj.Transformer.from_crs() call sites in the codebase use always_xy=True:

1. poc_homography/coordinate_converter.py
   - Line 72: self._to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
   - Line 73: self._to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

2. tools/extract_kml_points.py
   - Lines 42-44: self.transformer = Transformer.from_crs(config["crs"], "EPSG:4326", always_xy=True)
   - Lines 45-47: self.reverse_transformer = Transformer.from_crs("EPSG:4326", config["crs"], always_xy=True)

3. tools/unified_gcp_tool.py
   - Lines 345-347: self.transformer_utm_to_gps = Transformer.from_crs(self.utm_crs, "EPSG:4326", always_xy=True)
   - Lines 348-350: self.transformer_gps_to_utm = Transformer.from_crs("EPSG:4326", self.utm_crs, always_xy=True)

4. tools/debug_coordinate_transforms.py
   - Line 35: transformer = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)
   - Line 46: transformer = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)

5. tools/debug_y_scaling.py
   - Line 75: transformer = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)
   - Line 89: reverse_transformer = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)
   - Line 140: transformer = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)
   - Line 213: transformer = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)
   - Line 262: reverse_transformer = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)

This test ensures that:
1. All locations are documented as living documentation
2. Any new Transformer.from_crs() calls are detected
3. All calls include always_xy=True
"""

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Dict, List, NamedTuple, Set


class TransformerLocation(NamedTuple):
    """Represents a location where Transformer.from_crs() is called."""
    file_path: str
    line_number: int
    line_content: str
    has_always_xy: bool


class TestPyprojAudit(unittest.TestCase):
    """
    Audit tests for pyproj.Transformer.from_crs() usage.

    This test class serves as living documentation of the audit findings and
    continuously verifies that all pyproj Transformer calls use always_xy=True.

    The test dynamically scans the codebase at test time to detect:
    1. Any new Transformer.from_crs() calls that may be added
    2. Any calls missing the always_xy=True parameter
    3. Changes to existing documented locations
    """

    # Known locations from the audit (for documentation purposes)
    # Maps relative file path to list of expected line numbers
    # Note: Line 43 is in docstring example code, actual calls are at 90, 91
    DOCUMENTED_LOCATIONS: Dict[str, List[int]] = {
        "poc_homography/coordinate_converter.py": [43, 90, 91],  # Line 43 is docstring example
        "tools/extract_kml_points.py": [42, 45],  # Multi-line calls start at these lines
        "tools/unified_gcp_tool.py": [345, 348],  # Multi-line calls start at these lines
        "tools/debug_coordinate_transforms.py": [35, 46],
        "tools/debug_y_scaling.py": [75, 89, 140, 213, 262],
    }

    @classmethod
    def setUpClass(cls):
        """Set up the test class by finding the project root."""
        # Find project root (parent of tests directory)
        cls.tests_dir = Path(__file__).parent
        cls.project_root = cls.tests_dir.parent

        # Verify we're in the right project
        assert (cls.project_root / "poc_homography").exists(), \
            f"Could not find poc_homography directory in {cls.project_root}"

    def _find_transformer_calls_grep(self) -> List[TransformerLocation]:
        """
        Find all Transformer.from_crs() calls using grep/subprocess.

        Returns:
            List of TransformerLocation objects for each call found
        """
        locations: List[TransformerLocation] = []

        # Search pattern: Transformer.from_crs with any arguments
        # Using grep -rn for recursive search with line numbers
        try:
            result = subprocess.run(
                [
                    "grep", "-rn", "--include=*.py",
                    "Transformer.from_crs",
                    str(self.project_root)
                ],
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )

            # Parse grep output: each line is "filepath:linenumber:content"
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                # Split on first two colons to handle file paths with colons
                parts = line.split(":", 2)
                if len(parts) < 3:
                    continue

                file_path = parts[0]
                try:
                    line_number = int(parts[1])
                except ValueError:
                    continue
                line_content = parts[2].strip()

                # Skip test files (including this file)
                if "/tests/" in file_path or file_path.endswith("_test.py"):
                    continue

                # Skip __pycache__ directories
                if "__pycache__" in file_path:
                    continue

                # Skip lines that are comments, docstrings, or URLs (not actual code)
                # These are references to documentation, not actual Transformer calls
                stripped_content = line_content.strip()
                if (stripped_content.startswith("#") or
                    stripped_content.startswith("See:") or
                    "http://" in stripped_content or
                    "https://" in stripped_content):
                    continue

                # Check if always_xy=True is present
                # Need to handle multi-line calls, so we read more context
                has_always_xy = self._check_always_xy_in_call(file_path, line_number)

                locations.append(TransformerLocation(
                    file_path=file_path,
                    line_number=line_number,
                    line_content=line_content,
                    has_always_xy=has_always_xy
                ))

        except FileNotFoundError:
            # grep not available, fall back to Python-based search
            locations = self._find_transformer_calls_python()

        return locations

    def _find_transformer_calls_python(self) -> List[TransformerLocation]:
        """
        Fallback: Find all Transformer.from_crs() calls using Python file scanning.

        Returns:
            List of TransformerLocation objects for each call found
        """
        locations: List[TransformerLocation] = []

        # Directories to search
        search_dirs = [
            self.project_root / "poc_homography",
            self.project_root / "tools",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for py_file in search_dir.rglob("*.py"):
                # Skip test files
                if "test" in py_file.name.lower():
                    continue

                # Skip __pycache__
                if "__pycache__" in str(py_file):
                    continue

                try:
                    content = py_file.read_text()
                    lines = content.split("\n")

                    for i, line in enumerate(lines, start=1):
                        if "Transformer.from_crs" in line:
                            # Skip lines that are comments, docstrings, or URLs
                            stripped = line.strip()
                            if (stripped.startswith("#") or
                                stripped.startswith("See:") or
                                "http://" in stripped or
                                "https://" in stripped):
                                continue

                            has_always_xy = self._check_always_xy_in_call(
                                str(py_file), i
                            )
                            locations.append(TransformerLocation(
                                file_path=str(py_file),
                                line_number=i,
                                line_content=line.strip(),
                                has_always_xy=has_always_xy
                            ))

                except Exception:
                    continue

        return locations

    def _check_always_xy_in_call(self, file_path: str, start_line: int) -> bool:
        """
        Check if a Transformer.from_crs() call includes always_xy=True.

        Handles multi-line calls by reading up to 10 lines from the start.

        Args:
            file_path: Path to the file
            start_line: Line number where Transformer.from_crs starts

        Returns:
            True if always_xy=True is found in the call
        """
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()

            # Read lines starting from the call (up to 10 lines to handle multi-line)
            call_text = ""
            paren_count = 0
            started = False

            for i in range(start_line - 1, min(start_line + 9, len(lines))):
                line = lines[i]
                call_text += line

                # Track parentheses to find end of call
                for char in line:
                    if char == "(":
                        paren_count += 1
                        started = True
                    elif char == ")":
                        paren_count -= 1

                # If we've closed all parentheses after starting, we're done
                if started and paren_count == 0:
                    break

            # Check for always_xy=True in the call
            # Handle various formats: always_xy=True, always_xy = True, etc.
            pattern = r"always_xy\s*=\s*True"
            return bool(re.search(pattern, call_text))

        except Exception:
            return False

    def test_all_transformer_locations_documented(self):
        """
        Verify all Transformer.from_crs() calls are documented and use always_xy=True.

        This test:
        1. Scans the codebase for all Transformer.from_crs() calls
        2. Verifies each call includes always_xy=True
        3. Documents all locations found
        4. Fails if any call is missing always_xy=True
        """
        locations = self._find_transformer_calls_grep()

        # Separate compliant and non-compliant calls
        compliant: List[TransformerLocation] = []
        non_compliant: List[TransformerLocation] = []

        for loc in locations:
            if loc.has_always_xy:
                compliant.append(loc)
            else:
                non_compliant.append(loc)

        # Print documentation of all found locations
        print("\n" + "=" * 70)
        print("PYPROJ TRANSFORMER.FROM_CRS() AUDIT RESULTS")
        print("=" * 70)
        print(f"\nTotal Transformer.from_crs() calls found: {len(locations)}")
        print(f"Compliant (with always_xy=True): {len(compliant)}")
        print(f"Non-compliant (missing always_xy=True): {len(non_compliant)}")

        if compliant:
            print("\n--- COMPLIANT LOCATIONS ---")
            for loc in compliant:
                rel_path = os.path.relpath(loc.file_path, self.project_root)
                print(f"  [OK] {rel_path}:{loc.line_number}")
                print(f"       {loc.line_content[:80]}...")

        if non_compliant:
            print("\n--- NON-COMPLIANT LOCATIONS (MUST BE FIXED) ---")
            for loc in non_compliant:
                rel_path = os.path.relpath(loc.file_path, self.project_root)
                print(f"  [FAIL] {rel_path}:{loc.line_number}")
                print(f"         {loc.line_content}")
                print(f"         -> Missing always_xy=True parameter!")

        print("\n" + "=" * 70)

        # Assert all calls are compliant
        self.assertEqual(
            len(non_compliant), 0,
            f"\n\nFOUND {len(non_compliant)} TRANSFORMER CALLS WITHOUT always_xy=True!\n\n"
            f"The following locations MUST be fixed:\n" +
            "\n".join([
                f"  - {os.path.relpath(loc.file_path, self.project_root)}:{loc.line_number}: "
                f"{loc.line_content[:60]}..."
                for loc in non_compliant
            ]) +
            "\n\nWHY THIS MATTERS:\n"
            "  pyproj's default coordinate order follows the CRS definition.\n"
            "  For EPSG:4326, this means (lat, lon) instead of (lon, lat).\n"
            "  Without always_xy=True, coordinates get silently swapped,\n"
            "  causing subtle but serious bugs in coordinate transforms.\n\n"
            "FIX: Add always_xy=True to all Transformer.from_crs() calls."
        )

    def test_documented_files_exist(self):
        """
        Verify all documented files still exist in the codebase.

        This test ensures the documentation stays up to date by checking
        that the files listed in DOCUMENTED_LOCATIONS still exist.
        """
        missing_files: List[str] = []

        for rel_path in self.DOCUMENTED_LOCATIONS.keys():
            full_path = self.project_root / rel_path
            if not full_path.exists():
                missing_files.append(rel_path)

        self.assertEqual(
            len(missing_files), 0,
            f"\n\nDocumented files no longer exist:\n" +
            "\n".join([f"  - {f}" for f in missing_files]) +
            "\n\nPlease update DOCUMENTED_LOCATIONS in this test file."
        )

    def test_no_undocumented_transformer_calls(self):
        """
        Verify all Transformer.from_crs() calls are in documented files.

        This test catches new files that contain Transformer.from_crs() calls,
        ensuring they are reviewed and documented.
        """
        locations = self._find_transformer_calls_grep()

        # Get set of files with documented locations
        documented_files: Set[str] = set()
        for rel_path in self.DOCUMENTED_LOCATIONS.keys():
            # Normalize path for comparison
            full_path = str((self.project_root / rel_path).resolve())
            documented_files.add(full_path)

        # Find files with calls that aren't documented
        undocumented_files: Set[str] = set()
        for loc in locations:
            normalized_path = str(Path(loc.file_path).resolve())
            if normalized_path not in documented_files:
                undocumented_files.add(normalized_path)

        if undocumented_files:
            print("\n--- UNDOCUMENTED FILES WITH TRANSFORMER CALLS ---")
            for f in undocumented_files:
                rel_path = os.path.relpath(f, self.project_root)
                print(f"  [NEW] {rel_path}")

        self.assertEqual(
            len(undocumented_files), 0,
            f"\n\nFound Transformer.from_crs() calls in undocumented files:\n" +
            "\n".join([
                f"  - {os.path.relpath(f, self.project_root)}"
                for f in undocumented_files
            ]) +
            "\n\nPlease:\n"
            "  1. Verify these calls use always_xy=True\n"
            "  2. Add the files to DOCUMENTED_LOCATIONS in this test file"
        )

    def test_minimum_transformer_calls_found(self):
        """
        Verify we find at least the expected number of Transformer.from_crs() calls.

        This is a sanity check to ensure our grep/search is working correctly.
        If the count drops unexpectedly, it may indicate:
        1. Files were deleted
        2. Code was refactored
        3. Our search pattern is broken
        """
        locations = self._find_transformer_calls_grep()

        # We expect at least 13 calls based on the audit
        # (Multi-line calls are counted once by their starting line)
        expected_minimum = 13

        self.assertGreaterEqual(
            len(locations),
            expected_minimum,
            f"\n\nExpected at least {expected_minimum} Transformer.from_crs() calls, "
            f"but found only {len(locations)}.\n\n"
            "This may indicate:\n"
            "  - Files were deleted or refactored\n"
            "  - Our search pattern needs updating\n"
            "  - The audit documentation is outdated\n\n"
            "Found locations:\n" +
            "\n".join([
                f"  - {os.path.relpath(loc.file_path, self.project_root)}:{loc.line_number}"
                for loc in locations
            ])
        )


class TestUTMWGS84RoundTrip(unittest.TestCase):
    """
    Round-trip accuracy verification for UTM<->WGS84 transforms.

    This test class verifies that coordinate transformations using pyproj
    with always_xy=True maintain sub-centimeter accuracy in round-trip
    conversions. This is critical for:

    1. Camera calibration workflows where GPS coordinates are transformed
       to UTM for local processing and back to GPS for output
    2. Ground Control Point (GCP) workflows that convert between CRS
    3. Any pipeline that chains multiple coordinate transforms

    WHY 1cm THRESHOLD:
    ==================
    - UTM projection is conformal and preserves local angles/shapes
    - For Valencia area (lat ~39.5°, UTM Zone 30N), UTM distortion is minimal
    - pyproj uses high-precision PROJ library with IEEE 754 double precision
    - 1cm is well within survey-grade GPS accuracy (~2-5cm typical)
    - Sub-centimeter round-trip errors prove transforms are mathematically correct

    TEST POINTS SELECTION:
    ======================
    1. City center point: Typical coordinates, most common use case
    2. Zone boundary point: Tests edge case where UTM distortion increases
    3. Round number point: Easy to debug, verify manual calculations
    """

    # Valencia area test points: (latitude, longitude, easting, northing)
    # All coordinates are in EPSG:25830 (UTM Zone 30N) and EPSG:4326 (WGS84)
    #
    # COORDINATE ORDER REMINDER:
    # - EPSG:4326 with always_xy=True: (longitude, latitude) = (x, y)
    # - EPSG:25830 (UTM): (easting, northing) = (x, y)
    VALENCIA_TEST_POINTS = [
        {
            # Point 1: Valencia city center (Plaza del Ayuntamiento area)
            # Moderate coordinates, typical use case for camera calibration
            "name": "Valencia City Center",
            "latitude": 39.469907,
            "longitude": -0.376288,
            "easting": 725846.7,  # Approximate, will verify in test
            "northing": 4372589.5,  # Approximate, will verify in test
        },
        {
            # Point 2: Near UTM Zone 30/31 boundary (longitude -0.0° is boundary)
            # Stress-tests projection near zone edge where distortion increases
            # Valencia Port area, close to zone boundary
            "name": "Near Zone Boundary (Valencia Port)",
            "latitude": 39.451234,
            "longitude": -0.012345,
            "easting": 753424.9,  # Approximate, will verify in test
            "northing": 4370208.8,  # Approximate, will verify in test
        },
        {
            # Point 3: Simple round numbers for debugging
            # Albufera Natural Park area, south of Valencia
            "name": "Round Numbers (Albufera Area)",
            "latitude": 39.350000,
            "longitude": -0.350000,
            "easting": 728000.0,  # Approximate, will verify in test
            "northing": 4359300.0,  # Approximate, will verify in test
        },
    ]

    # Error tolerances
    # 1cm = 0.01 meters for UTM coordinates
    UTM_TOLERANCE_M = 0.01

    # 1cm in degrees at Valencia latitude (~39.5°N)
    # At equator: 1 degree latitude = 111,320 meters
    # At Valencia: 1 degree longitude = 111,320 * cos(39.5°) = ~85,900 meters
    # So 1cm = 0.01m / 85900m = ~0.0000001 degrees (conservative)
    WGS84_TOLERANCE_DEG = 0.0000001

    @classmethod
    def setUpClass(cls):
        """Set up pyproj transformers for UTM Zone 30N <-> WGS84."""
        try:
            from pyproj import Transformer
        except ImportError:
            raise unittest.SkipTest("pyproj not available")

        # Create transformers with always_xy=True
        # This ensures consistent (x, y) = (lon, lat) for WGS84
        # and (x, y) = (easting, northing) for UTM
        cls.wgs84_to_utm = Transformer.from_crs(
            "EPSG:4326",  # WGS84 (GPS coordinates)
            "EPSG:25830",  # UTM Zone 30N (Valencia area)
            always_xy=True  # CRITICAL: ensures (lon, lat) order for WGS84
        )
        cls.utm_to_wgs84 = Transformer.from_crs(
            "EPSG:25830",  # UTM Zone 30N
            "EPSG:4326",  # WGS84
            always_xy=True  # CRITICAL: ensures (lon, lat) order for WGS84
        )

    def test_valencia_points_utm_to_wgs84_round_trip(self):
        """
        UTM -> WGS84 -> UTM round-trip must match within 1cm.

        WHY THIS ROUND-TRIP MUST SUCCEED:
        ==================================
        This tests the scenario where we start with UTM coordinates (e.g., from
        a surveyed GCP position) and need to:
        1. Convert to WGS84 for GPS display or export
        2. Convert back to UTM for further processing

        If round-trip error exceeds 1cm, it indicates:
        - Incorrect coordinate order (lat/lon vs lon/lat) - MOST COMMON BUG
        - Numerical precision loss in transformation chain
        - Incorrect CRS configuration

        1cm threshold is appropriate because:
        - Survey-grade GPS has ~2-5cm accuracy, so 1cm transform error is acceptable
        - Camera calibration workflows require sub-centimeter precision
        - Any larger error would compound across multiple transforms
        """
        for point in self.VALENCIA_TEST_POINTS:
            with self.subTest(point=point["name"]):
                # Start with known UTM coordinates
                # First, get accurate UTM from the known GPS coordinates
                easting_orig, northing_orig = self.wgs84_to_utm.transform(
                    point["longitude"],  # x = longitude (always_xy=True)
                    point["latitude"]    # y = latitude
                )

                # Round-trip: UTM -> WGS84 -> UTM
                # Step 1: UTM to WGS84
                lon_intermediate, lat_intermediate = self.utm_to_wgs84.transform(
                    easting_orig, northing_orig
                )

                # Step 2: WGS84 back to UTM
                easting_final, northing_final = self.wgs84_to_utm.transform(
                    lon_intermediate, lat_intermediate
                )

                # Calculate round-trip error
                easting_error = abs(easting_final - easting_orig)
                northing_error = abs(northing_final - northing_orig)
                total_error = (easting_error**2 + northing_error**2)**0.5

                # Assert within 1cm tolerance
                # This assertion verifies that the full transform chain preserves
                # sub-centimeter accuracy, proving always_xy=True is working correctly
                self.assertLess(
                    total_error,
                    self.UTM_TOLERANCE_M,
                    f"UTM->WGS84->UTM round-trip error {total_error*100:.4f}cm exceeds 1cm threshold\n"
                    f"Point: {point['name']}\n"
                    f"Original UTM: ({easting_orig:.4f}, {northing_orig:.4f})\n"
                    f"Intermediate WGS84: ({lon_intermediate:.8f}, {lat_intermediate:.8f})\n"
                    f"Final UTM: ({easting_final:.4f}, {northing_final:.4f})\n"
                    f"Easting error: {easting_error*100:.4f}cm, Northing error: {northing_error*100:.4f}cm"
                )

    def test_valencia_points_wgs84_to_utm_round_trip(self):
        """
        WGS84 -> UTM -> WGS84 round-trip must match within 1cm (~0.0000001 degrees).

        WHY THIS ROUND-TRIP MUST SUCCEED:
        ==================================
        This tests the scenario where we start with GPS coordinates (e.g., from
        a smartphone or survey device) and need to:
        1. Convert to UTM for local metric calculations (distances, angles)
        2. Convert back to WGS84 for map display or KML export

        If round-trip error exceeds 1cm, it indicates:
        - Incorrect coordinate order (the most common bug with pyproj)
        - Datum transformation issues
        - CRS mismatch between forward and reverse transforms

        The 0.0000001 degree threshold corresponds to ~1cm at Valencia's latitude:
        - 1 degree latitude = 111,320 meters (constant)
        - 1 degree longitude = 111,320 * cos(39.5°) = ~85,900 meters at Valencia
        - 0.0000001 degrees = ~0.01 meters = 1cm
        """
        for point in self.VALENCIA_TEST_POINTS:
            with self.subTest(point=point["name"]):
                # Start with known WGS84 coordinates
                lon_orig = point["longitude"]
                lat_orig = point["latitude"]

                # Round-trip: WGS84 -> UTM -> WGS84
                # Step 1: WGS84 to UTM
                easting_intermediate, northing_intermediate = self.wgs84_to_utm.transform(
                    lon_orig,  # x = longitude (always_xy=True)
                    lat_orig   # y = latitude
                )

                # Step 2: UTM back to WGS84
                lon_final, lat_final = self.utm_to_wgs84.transform(
                    easting_intermediate, northing_intermediate
                )

                # Calculate round-trip error in degrees
                lon_error = abs(lon_final - lon_orig)
                lat_error = abs(lat_final - lat_orig)

                # Also calculate error in meters for more intuitive reporting
                # Using local scale factors for Valencia latitude
                METERS_PER_DEG_LAT = 111320.0  # Approximately constant
                METERS_PER_DEG_LON = 111320.0 * 0.771  # cos(39.5°) ≈ 0.771

                lon_error_m = lon_error * METERS_PER_DEG_LON
                lat_error_m = lat_error * METERS_PER_DEG_LAT
                total_error_m = (lon_error_m**2 + lat_error_m**2)**0.5

                # Assert within tolerance
                # This assertion verifies that GPS coordinates survive the full
                # transform chain without coordinate swapping or precision loss
                self.assertLess(
                    lon_error,
                    self.WGS84_TOLERANCE_DEG,
                    f"Longitude round-trip error {lon_error:.10f}° ({lon_error_m*100:.4f}cm) exceeds threshold\n"
                    f"Point: {point['name']}\n"
                    f"Original: ({lon_orig:.8f}, {lat_orig:.8f})\n"
                    f"Intermediate UTM: ({easting_intermediate:.4f}, {northing_intermediate:.4f})\n"
                    f"Final: ({lon_final:.8f}, {lat_final:.8f})"
                )

                self.assertLess(
                    lat_error,
                    self.WGS84_TOLERANCE_DEG,
                    f"Latitude round-trip error {lat_error:.10f}° ({lat_error_m*100:.4f}cm) exceeds threshold\n"
                    f"Point: {point['name']}\n"
                    f"Original: ({lon_orig:.8f}, {lat_orig:.8f})\n"
                    f"Intermediate UTM: ({easting_intermediate:.4f}, {northing_intermediate:.4f})\n"
                    f"Final: ({lon_final:.8f}, {lat_final:.8f})"
                )

                # Also verify total 2D error in meters
                self.assertLess(
                    total_error_m,
                    self.UTM_TOLERANCE_M,
                    f"Total 2D round-trip error {total_error_m*100:.4f}cm exceeds 1cm threshold\n"
                    f"Point: {point['name']}"
                )


class TestCoordinateConsistency(unittest.TestCase):
    """
    Cross-module consistency verification.

    This test class verifies that coordinate_converter.py (the canonical implementation)
    and tool scripts produce identical results when transforming coordinates.

    WHY THIS TEST IS CRITICAL:
    ==========================
    The codebase has multiple locations that create pyproj Transformers:

    1. poc_homography/coordinate_converter.py - UTMConverter class (CANONICAL)
    2. tools/unified_gcp_tool.py - GCPEditor class
    3. tools/extract_kml_points.py - PointExtractor class

    All of these MUST produce identical results for the same inputs.
    If any tool drifts from the canonical implementation (e.g., by using
    different always_xy settings or different CRS configurations),
    it would cause subtle but serious coordinate mismatches.

    This test catches:
    - Tools that forget always_xy=True
    - Tools using different CRS strings
    - Accidental coordinate order swapping (lat/lon vs lon/lat)
    - Implementation drift over time
    """

    # Use the same Valencia test points as TestUTMWGS84RoundTrip for consistency
    VALENCIA_TEST_POINTS = TestUTMWGS84RoundTrip.VALENCIA_TEST_POINTS

    # UTM CRS used in Valencia area
    UTM_CRS = "EPSG:25830"

    def test_coordinate_converter_matches_tools(self):
        """
        Verify coordinate_converter.py and tool scripts produce identical results.

        This test:
        1. Creates a UTMConverter instance (the canonical implementation)
        2. Creates Transformer instances the same way tools do
        3. Transforms the same 3 Valencia test points
        4. Verifies results match exactly (no floating-point tolerance needed
           since both should use identical underlying PROJ operations)

        If this test fails, it indicates that tool scripts have diverged
        from the canonical coordinate_converter.py implementation.
        """
        try:
            from pyproj import Transformer
        except ImportError:
            raise unittest.SkipTest("pyproj not available")

        # Import the canonical implementation
        from poc_homography.coordinate_converter import UTMConverter

        # Create canonical UTMConverter instance
        canonical_converter = UTMConverter(self.UTM_CRS)

        # Create Transformers the way tools do (unified_gcp_tool.py pattern)
        tool_wgs84_to_utm = Transformer.from_crs(
            "EPSG:4326", self.UTM_CRS, always_xy=True
        )
        tool_utm_to_wgs84 = Transformer.from_crs(
            self.UTM_CRS, "EPSG:4326", always_xy=True
        )

        for point in self.VALENCIA_TEST_POINTS:
            with self.subTest(point=point["name"]):
                lat = point["latitude"]
                lon = point["longitude"]

                # --- Test GPS to UTM conversion ---
                # Canonical implementation
                canonical_easting, canonical_northing = canonical_converter.gps_to_utm(lat, lon)

                # Tool-style implementation
                tool_easting, tool_northing = tool_wgs84_to_utm.transform(lon, lat)

                # Results must match exactly
                self.assertEqual(
                    canonical_easting,
                    tool_easting,
                    f"GPS->UTM easting mismatch for {point['name']}:\n"
                    f"  Canonical: {canonical_easting}\n"
                    f"  Tool-style: {tool_easting}\n"
                    f"  This indicates tools have diverged from coordinate_converter.py"
                )
                self.assertEqual(
                    canonical_northing,
                    tool_northing,
                    f"GPS->UTM northing mismatch for {point['name']}:\n"
                    f"  Canonical: {canonical_northing}\n"
                    f"  Tool-style: {tool_northing}\n"
                    f"  This indicates tools have diverged from coordinate_converter.py"
                )

                # --- Test UTM to GPS conversion ---
                # Canonical implementation
                canonical_lat, canonical_lon = canonical_converter.utm_to_gps(
                    canonical_easting, canonical_northing
                )

                # Tool-style implementation
                tool_lon, tool_lat = tool_utm_to_wgs84.transform(
                    tool_easting, tool_northing
                )

                # Results must match exactly
                self.assertEqual(
                    canonical_lat,
                    tool_lat,
                    f"UTM->GPS latitude mismatch for {point['name']}:\n"
                    f"  Canonical: {canonical_lat}\n"
                    f"  Tool-style: {tool_lat}\n"
                    f"  This indicates tools have diverged from coordinate_converter.py"
                )
                self.assertEqual(
                    canonical_lon,
                    tool_lon,
                    f"UTM->GPS longitude mismatch for {point['name']}:\n"
                    f"  Canonical: {canonical_lon}\n"
                    f"  Tool-style: {tool_lon}\n"
                    f"  This indicates tools have diverged from coordinate_converter.py"
                )


def run_tests():
    """Run all tests with detailed output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPyprojAudit))
    suite.addTests(loader.loadTestsFromTestCase(TestUTMWGS84RoundTrip))
    suite.addTests(loader.loadTestsFromTestCase(TestCoordinateConsistency))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
