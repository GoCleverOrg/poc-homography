#!/usr/bin/env python3
"""
Unit tests for KML generation module.

Tests the KML generator's ability to create valid KML 2.2 compliant XML files
with correct structure, styles, placemarks, and coordinate formatting.
"""

import pytest
import tempfile
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from poc_homography.kml_generator import (
    generate_kml,
    create_output_directory,
    haversine_distance
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_gcps():
    """Sample GCP data for testing."""
    return [
        {
            'gps': {'latitude': 39.640600, 'longitude': -0.230200},
            'image': {'u': 400.0, 'v': 300.0},
            'metadata': {'description': 'P#01 - Test Point 1', 'accuracy': 'high'}
        },
        {
            'gps': {'latitude': 39.640620, 'longitude': -0.229800},
            'image': {'u': 2100.0, 'v': 320.0},
            'metadata': {'description': 'P#02 - Test Point 2', 'accuracy': 'medium'}
        },
    ]


@pytest.fixture
def sample_validation_results():
    """Sample validation results for testing."""
    return {
        'details': [
            {'projected_gps': (39.640605, -0.230195), 'error_meters': 0.56},
            {'projected_gps': (39.640625, -0.229805), 'error_meters': 0.71},
        ]
    }


@pytest.fixture
def sample_camera_gps():
    """Sample camera GPS position for testing."""
    return {'latitude': 39.640500, 'longitude': -0.230000}


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# Haversine Distance Tests
# ============================================================================

def test_haversine_distance_known_values():
    """Test haversine distance calculation with known GPS coordinates."""
    # Test case 1: Two nearby points (approximately 35-40 meters apart)
    lat1, lon1 = 39.640600, -0.230200
    lat2, lon2 = 39.640620, -0.229800

    distance = haversine_distance(lat1, lon1, lat2, lon2)

    # Expected distance is approximately 35-40 meters
    assert 30.0 < distance < 45.0, f"Expected ~35-40m, got {distance:.2f}m"

    # Test case 2: Same point should return 0
    distance_same = haversine_distance(lat1, lon1, lat1, lon1)
    assert distance_same == 0.0, f"Same point should have 0 distance, got {distance_same}"

    # Test case 3: Another known distance (approximately 100m)
    lat3, lon3 = 39.640600, -0.230200
    lat4, lon4 = 39.641500, -0.230200  # ~100m north

    distance_100m = haversine_distance(lat3, lon3, lat4, lon4)
    assert 95.0 < distance_100m < 105.0, f"Expected ~100m, got {distance_100m:.2f}m"


def test_haversine_distance_large_distance():
    """Test haversine distance with points far apart (cross-city distances)."""
    # Valencia, Spain - approximate coordinates for two distant points
    lat1, lon1 = 39.469907, -0.376288  # Valencia city center
    lat2, lon2 = 39.640600, -0.230200  # Test location (Port area)

    distance = haversine_distance(lat1, lon1, lat2, lon2)

    # Expected distance is approximately 20-25 km
    assert 18000 < distance < 27000, f"Expected ~20-25km, got {distance:.2f}m"

    # Verify symmetry: distance(A,B) == distance(B,A)
    distance_reverse = haversine_distance(lat2, lon2, lat1, lon1)
    assert abs(distance - distance_reverse) < 0.01, "Distance calculation should be symmetric"


# ============================================================================
# Output Directory Tests
# ============================================================================

def test_create_output_directory():
    """Test that output directory is created correctly."""
    output_dir = create_output_directory()

    # Verify directory exists
    assert os.path.isdir(output_dir), f"Output directory should exist: {output_dir}"

    # Verify it's an absolute path
    assert os.path.isabs(output_dir), "Output directory should be absolute path"

    # Verify it ends with 'output'
    assert output_dir.endswith('output'), "Directory should be named 'output'"


def test_create_output_directory_idempotent():
    """Test that calling create_output_directory twice doesn't fail."""
    # First call
    output_dir1 = create_output_directory()
    assert os.path.isdir(output_dir1)

    # Second call should succeed and return same path
    output_dir2 = create_output_directory()
    assert os.path.isdir(output_dir2)
    assert output_dir1 == output_dir2, "Both calls should return same path"


# ============================================================================
# KML Structure and Validation Tests
# ============================================================================

def test_generate_kml_creates_valid_xml(sample_gcps, sample_validation_results,
                                        sample_camera_gps, temp_output_dir):
    """Test that generated KML can be parsed as valid XML."""
    output_path = os.path.join(temp_output_dir, 'test.kml')

    # Generate KML
    generate_kml(sample_gcps, sample_validation_results, sample_camera_gps, output_path)

    # Parse the generated XML
    try:
        tree = ET.parse(output_path)
        root = tree.getroot()
        assert root.tag.endswith('kml'), "Root element should be 'kml'"
    except ET.ParseError as e:
        pytest.fail(f"Generated KML is not valid XML: {e}")


def test_generate_kml_has_correct_structure(sample_gcps, sample_validation_results,
                                            sample_camera_gps, temp_output_dir):
    """Test that generated KML has correct document structure."""
    output_path = os.path.join(temp_output_dir, 'test_structure.kml')

    # Generate KML
    generate_kml(sample_gcps, sample_validation_results, sample_camera_gps, output_path)

    # Parse and verify structure
    tree = ET.parse(output_path)
    root = tree.getroot()

    # Find Document element
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
    document = root.find('kml:Document', namespaces)
    assert document is not None, "Document element should exist"

    # Verify all 4 folders exist
    folders = document.findall('kml:Folder', namespaces)
    folder_names = [f.find('kml:name', namespaces).text for f in folders]

    expected_folders = ['Camera Position', 'Original GCPs', 'Projected GCPs', 'Error Lines']
    for expected in expected_folders:
        assert expected in folder_names, f"Folder '{expected}' should exist"

    # Verify all 4 styles exist
    styles = document.findall('kml:Style', namespaces)
    style_ids = [s.get('id') for s in styles]

    expected_styles = ['camera-position', 'original-gcp', 'projected-gcp', 'error-line']
    for expected in expected_styles:
        assert expected in style_ids, f"Style '{expected}' should exist"


def test_generate_kml_placemark_count(sample_gcps, sample_validation_results,
                                     sample_camera_gps, temp_output_dir):
    """Test that KML contains correct number of placemarks."""
    output_path = os.path.join(temp_output_dir, 'test_count.kml')

    num_gcps = len(sample_gcps)

    # Generate KML
    generate_kml(sample_gcps, sample_validation_results, sample_camera_gps, output_path)

    # Parse and count placemarks
    tree = ET.parse(output_path)
    root = tree.getroot()
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}

    document = root.find('kml:Document', namespaces)
    folders = document.findall('kml:Folder', namespaces)

    # Organize folders by name
    folder_dict = {}
    for folder in folders:
        name = folder.find('kml:name', namespaces).text
        folder_dict[name] = folder

    # Test Camera Position folder (should have 1 placemark)
    camera_folder = folder_dict['Camera Position']
    camera_placemarks = camera_folder.findall('kml:Placemark', namespaces)
    assert len(camera_placemarks) == 1, "Camera Position folder should have 1 placemark"

    # Test Original GCPs folder (should have N placemarks)
    original_folder = folder_dict['Original GCPs']
    original_placemarks = original_folder.findall('kml:Placemark', namespaces)
    assert len(original_placemarks) == num_gcps, \
        f"Original GCPs folder should have {num_gcps} placemarks"

    # Test Projected GCPs folder (should have N placemarks)
    projected_folder = folder_dict['Projected GCPs']
    projected_placemarks = projected_folder.findall('kml:Placemark', namespaces)
    assert len(projected_placemarks) == num_gcps, \
        f"Projected GCPs folder should have {num_gcps} placemarks"

    # Test Error Lines folder (should have N LineString elements)
    error_folder = folder_dict['Error Lines']
    error_placemarks = error_folder.findall('kml:Placemark', namespaces)
    assert len(error_placemarks) == num_gcps, \
        f"Error Lines folder should have {num_gcps} placemarks"

    # Verify each error placemark has a LineString
    for placemark in error_placemarks:
        line_string = placemark.find('kml:LineString', namespaces)
        assert line_string is not None, "Error placemark should contain LineString"


def test_generate_kml_coordinates_format(sample_gcps, sample_validation_results,
                                        sample_camera_gps, temp_output_dir):
    """Test that coordinates use correct KML format: longitude,latitude,0."""
    output_path = os.path.join(temp_output_dir, 'test_coords.kml')

    # Generate KML
    generate_kml(sample_gcps, sample_validation_results, sample_camera_gps, output_path)

    # Parse and check coordinate format
    tree = ET.parse(output_path)
    root = tree.getroot()
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}

    # Get all coordinate elements
    coordinates = root.findall('.//kml:coordinates', namespaces)
    assert len(coordinates) > 0, "Should have coordinate elements"

    for coord_elem in coordinates:
        coord_text = coord_elem.text.strip()

        # Split by space for LineStrings, or use as-is for Points
        coord_parts = coord_text.split()

        for coord_part in coord_parts:
            parts = coord_part.split(',')

            # Should have 3 parts: longitude, latitude, altitude
            assert len(parts) == 3, f"Coordinate should have 3 parts: {coord_part}"

            lon, lat, alt = parts

            # Verify format (should be numeric)
            try:
                lon_val = float(lon)
                lat_val = float(lat)
                alt_val = float(alt)
            except ValueError:
                pytest.fail(f"Coordinate parts should be numeric: {coord_part}")

            # Verify longitude is in valid range (-180 to 180)
            assert -180 <= lon_val <= 180, f"Longitude out of range: {lon_val}"

            # Verify latitude is in valid range (-90 to 90)
            assert -90 <= lat_val <= 90, f"Latitude out of range: {lat_val}"

            # Verify altitude is 0 (we use ground-level coordinates)
            assert alt_val == 0, f"Altitude should be 0: {alt_val}"

    # Verify specific camera coordinate format
    camera_lat = sample_camera_gps['latitude']
    camera_lon = sample_camera_gps['longitude']
    expected_camera_coords = f"{camera_lon},{camera_lat},0"

    # Find camera placemark
    document = root.find('kml:Document', namespaces)
    folders = document.findall('kml:Folder', namespaces)

    for folder in folders:
        name = folder.find('kml:name', namespaces).text
        if name == 'Camera Position':
            placemark = folder.find('kml:Placemark', namespaces)
            point = placemark.find('kml:Point', namespaces)
            coords = point.find('kml:coordinates', namespaces).text.strip()
            assert coords == expected_camera_coords, \
                f"Camera coordinates format incorrect. Expected: {expected_camera_coords}, Got: {coords}"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_generate_kml_with_empty_gcps(sample_validation_results, sample_camera_gps,
                                     temp_output_dir):
    """Test that empty GCP list raises ValueError."""
    output_path = os.path.join(temp_output_dir, 'test_empty.kml')

    empty_gcps = []

    # Should raise ValueError for empty GCPs
    with pytest.raises(ValueError, match="GCPs list cannot be empty"):
        generate_kml(empty_gcps, sample_validation_results, sample_camera_gps, output_path)


def test_generate_kml_missing_validation_details(sample_gcps, sample_camera_gps,
                                                 temp_output_dir):
    """Test that missing validation details raises ValueError."""
    output_path = os.path.join(temp_output_dir, 'test_missing.kml')

    # Missing 'details' key
    invalid_results = {'foo': 'bar'}

    with pytest.raises(ValueError, match="validation_results must contain 'details' key"):
        generate_kml(sample_gcps, invalid_results, sample_camera_gps, output_path)


def test_generate_kml_invalid_camera_gps(sample_gcps, sample_validation_results,
                                        temp_output_dir):
    """Test that invalid camera GPS raises ValueError."""
    output_path = os.path.join(temp_output_dir, 'test_invalid_camera.kml')

    # Missing longitude
    invalid_camera = {'latitude': 39.640500}

    with pytest.raises(ValueError, match="camera_gps must contain 'latitude' and 'longitude' keys"):
        generate_kml(sample_gcps, sample_validation_results, invalid_camera, output_path)


# ============================================================================
# File Output Tests
# ============================================================================

def test_generate_kml_file_saved(sample_gcps, sample_validation_results,
                                sample_camera_gps, temp_output_dir):
    """Test that KML file is properly saved to disk."""
    output_path = os.path.join(temp_output_dir, 'test_save.kml')

    # Generate KML
    generate_kml(sample_gcps, sample_validation_results, sample_camera_gps, output_path)

    # Verify file exists
    assert os.path.isfile(output_path), f"KML file should exist at {output_path}"

    # Verify file is non-empty
    file_size = os.path.getsize(output_path)
    assert file_size > 0, "KML file should not be empty"

    # Verify file starts with XML declaration
    with open(output_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        assert first_line.startswith('<?xml'), \
            f"KML file should start with XML declaration, got: {first_line}"


def test_generate_kml_file_content_readable(sample_gcps, sample_validation_results,
                                           sample_camera_gps, temp_output_dir):
    """Test that generated KML file is readable and contains expected content."""
    output_path = os.path.join(temp_output_dir, 'test_content.kml')

    # Generate KML
    generate_kml(sample_gcps, sample_validation_results, sample_camera_gps, output_path)

    # Read file content
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Verify key elements are present
    assert '<?xml version=' in content, "Should have XML declaration"
    assert '<kml xmlns="http://www.opengis.net/kml/2.2">' in content, "Should have KML root"
    assert '<Document>' in content, "Should have Document element"
    assert 'Camera Position' in content, "Should have Camera Position folder"
    assert 'Original GCPs' in content, "Should have Original GCPs folder"
    assert 'Projected GCPs' in content, "Should have Projected GCPs folder"
    assert 'Error Lines' in content, "Should have Error Lines folder"


# ============================================================================
# Accuracy Tests
# ============================================================================

def test_kml_error_distance_accuracy(sample_gcps, temp_output_dir):
    """Test that error distance in KML matches haversine calculation."""
    # Create validation results with known coordinates
    orig_lat, orig_lon = 39.640600, -0.230200
    proj_lat, proj_lon = 39.640605, -0.230195

    # Calculate expected error distance
    expected_error = haversine_distance(orig_lat, orig_lon, proj_lat, proj_lon)

    validation_results = {
        'details': [
            {'projected_gps': (proj_lat, proj_lon), 'error_meters': expected_error}
        ]
    }

    camera_gps = {'latitude': 39.640500, 'longitude': -0.230000}
    output_path = os.path.join(temp_output_dir, 'test_error.kml')

    # Generate KML
    generate_kml([sample_gcps[0]], validation_results, camera_gps, output_path)

    # Parse and verify error distance in description
    tree = ET.parse(output_path)
    root = tree.getroot()
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}

    # Find Error Lines folder
    document = root.find('kml:Document', namespaces)
    folders = document.findall('kml:Folder', namespaces)

    for folder in folders:
        name = folder.find('kml:name', namespaces).text
        if name == 'Error Lines':
            placemark = folder.find('kml:Placemark', namespaces)
            description = placemark.find('kml:description', namespaces).text

            # Extract error value from description (format: "Error distance: X.XXm")
            import re
            match = re.search(r'Error distance: ([\d.]+)m', description)
            assert match, f"Error distance not found in description: {description}"

            error_in_kml = float(match.group(1))

            # Verify it matches our expected error (within 0.01m tolerance)
            assert abs(error_in_kml - expected_error) < 0.01, \
                f"Error distance mismatch. Expected: {expected_error:.2f}m, Got: {error_in_kml:.2f}m"


def test_kml_line_string_coordinates_match_points(sample_gcps, sample_validation_results,
                                                  sample_camera_gps, temp_output_dir):
    """Test that LineString coordinates match original and projected point coordinates."""
    output_path = os.path.join(temp_output_dir, 'test_line_coords.kml')

    # Generate KML
    generate_kml(sample_gcps, sample_validation_results, sample_camera_gps, output_path)

    # Parse and verify
    tree = ET.parse(output_path)
    root = tree.getroot()
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}

    document = root.find('kml:Document', namespaces)
    folders = document.findall('kml:Folder', namespaces)

    # Organize folders
    folder_dict = {}
    for folder in folders:
        name = folder.find('kml:name', namespaces).text
        folder_dict[name] = folder

    # Get coordinates from Original GCPs
    original_folder = folder_dict['Original GCPs']
    original_placemarks = original_folder.findall('kml:Placemark', namespaces)

    # Get coordinates from Projected GCPs
    projected_folder = folder_dict['Projected GCPs']
    projected_placemarks = projected_folder.findall('kml:Placemark', namespaces)

    # Get coordinates from Error Lines
    error_folder = folder_dict['Error Lines']
    error_placemarks = error_folder.findall('kml:Placemark', namespaces)

    # For each error line, verify it connects correct original and projected points
    for i, error_pm in enumerate(error_placemarks):
        # Get LineString coordinates
        line_string = error_pm.find('kml:LineString', namespaces)
        line_coords = line_string.find('kml:coordinates', namespaces).text.strip()
        line_points = line_coords.split()

        assert len(line_points) == 2, "LineString should have exactly 2 points"

        # Get original point coordinates
        orig_point = original_placemarks[i].find('kml:Point', namespaces)
        orig_coords = orig_point.find('kml:coordinates', namespaces).text.strip()

        # Get projected point coordinates
        proj_point = projected_placemarks[i].find('kml:Point', namespaces)
        proj_coords = proj_point.find('kml:coordinates', namespaces).text.strip()

        # Verify line connects these points
        assert line_points[0] == orig_coords, \
            f"LineString first point should match original GCP coordinates"
        assert line_points[1] == proj_coords, \
            f"LineString second point should match projected GCP coordinates"


# ============================================================================
# Multi-GCP Tests
# ============================================================================

def test_generate_kml_with_many_gcps(sample_camera_gps, temp_output_dir):
    """Test KML generation with larger number of GCPs."""
    # Create 10 GCPs
    num_gcps = 10
    gcps = []
    validation_results = {'details': []}

    base_lat, base_lon = 39.640600, -0.230200

    for i in range(num_gcps):
        # Create GCP with slight offset
        lat = base_lat + i * 0.0001
        lon = base_lon + i * 0.0001

        gcp = {
            'gps': {'latitude': lat, 'longitude': lon},
            'image': {'u': 100.0 + i * 200, 'v': 100.0 + i * 50},
            'metadata': {'description': f'GCP_{i+1}', 'accuracy': 'high'}
        }
        gcps.append(gcp)

        # Create corresponding validation result
        proj_lat = lat + 0.000005
        proj_lon = lon + 0.000005
        error = haversine_distance(lat, lon, proj_lat, proj_lon)

        validation_results['details'].append({
            'projected_gps': (proj_lat, proj_lon),
            'error_meters': error
        })

    output_path = os.path.join(temp_output_dir, 'test_many_gcps.kml')

    # Generate KML
    generate_kml(gcps, validation_results, sample_camera_gps, output_path)

    # Verify file was created
    assert os.path.isfile(output_path)

    # Parse and verify counts
    tree = ET.parse(output_path)
    root = tree.getroot()
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}

    # Count all placemarks
    all_placemarks = root.findall('.//kml:Placemark', namespaces)

    # Expected: 1 camera + num_gcps original + num_gcps projected + num_gcps error lines
    expected_count = 1 + num_gcps + num_gcps + num_gcps
    assert len(all_placemarks) == expected_count, \
        f"Expected {expected_count} placemarks, got {len(all_placemarks)}"


if __name__ == '__main__':
    """Run tests with pytest."""
    pytest.main([__file__, '-v'])
