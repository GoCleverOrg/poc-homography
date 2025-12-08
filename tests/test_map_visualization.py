#!/usr/bin/env python3
"""
Integration tests for the web visualization module.

Tests the map debug server's ability to generate HTML visualizations,
find available ports, export annotated frames, and serve debug content.
"""

import pytest
import tempfile
import os
import socket
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock

from poc_homography.map_debug_server import (
    find_available_port,
    generate_html,
    start_server
)
from tests.test_gcp_round_trip_validation import export_frame_with_markers


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
            'metadata': {'description': 'P#01 - Test Point 1'}
        },
        {
            'gps': {'latitude': 39.640620, 'longitude': -0.229800},
            'image': {'u': 2100.0, 'v': 320.0},
            'metadata': {'description': 'P#02 - Test Point 2'}
        },
        {
            'gps': {'latitude': 39.640400, 'longitude': -0.230000},
            'image': {'u': 1280.0, 'v': 720.0},
            'metadata': {'description': 'P#03 - Center Point'}
        },
    ]


@pytest.fixture
def sample_validation_results():
    """Sample validation results for testing."""
    return {
        'details': [
            {
                'projected_gps': (39.640605, -0.230205),
                'projected_pixel': (402.3, 301.5),
                'error_meters': 0.56,
                'image_point': (400.0, 300.0)
            },
            {
                'projected_gps': (39.640618, -0.229805),
                'projected_pixel': (2098.7, 318.2),
                'error_meters': 0.45,
                'image_point': (2100.0, 320.0)
            },
            {
                'projected_gps': (39.640398, -0.230002),
                'projected_pixel': (1281.1, 719.8),
                'error_meters': 0.23,
                'image_point': (1280.0, 720.0)
            },
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


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    # Create a simple test image (1920x1080)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, :] = (50, 50, 50)  # Dark gray background

    # Draw grid for visual reference
    for i in range(0, 1920, 100):
        cv2.line(frame, (i, 0), (i, 1080), (100, 100, 100), 1)
    for i in range(0, 1080, 100):
        cv2.line(frame, (0, i), (1920, i), (100, 100, 100), 1)

    return frame


# ============================================================================
# Port Finding Tests
# ============================================================================

def test_find_available_port_default():
    """Test that find_available_port returns a port in expected range."""
    port = find_available_port()

    # Verify port is an integer
    assert isinstance(port, int), "Port should be an integer"

    # Verify port is in reasonable range
    assert 8080 <= port < 8090, f"Port should be in range 8080-8089, got {port}"

    # Verify port is actually available by binding to it
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', port))


def test_find_available_port_fallback():
    """Test that find_available_port falls back when default port is occupied."""
    # Bind to port 8080 to force fallback
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(('', 8080))
        blocker.listen(1)

        # Find available port should skip 8080
        port = find_available_port(start_port=8080, max_attempts=10)

        # Should get 8081 or higher
        assert port >= 8081, f"Port should be 8081 or higher, got {port}"
        assert port < 8090, f"Port should be less than 8090, got {port}"


def test_find_available_port_all_occupied():
    """Test that find_available_port raises error when all ports occupied."""
    # Create sockets to occupy multiple consecutive ports
    sockets = []
    start_port = 9000  # Use higher range to avoid conflicts

    try:
        # Occupy 5 consecutive ports
        for i in range(5):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', start_port + i))
            s.listen(1)
            sockets.append(s)

        # Try to find port with max_attempts=5, should fail
        with pytest.raises(RuntimeError, match="Could not find an available port"):
            find_available_port(start_port=start_port, max_attempts=5)

    finally:
        # Clean up sockets
        for s in sockets:
            s.close()


# ============================================================================
# HTML Generation Tests
# ============================================================================

def test_generate_html_contains_required_elements(sample_gcps, sample_validation_results,
                                                  sample_camera_gps):
    """Test that generated HTML contains all required elements."""
    html = generate_html(
        camera_frame_path='frame.jpg',
        kml_path='validation.kml',
        camera_gps=sample_camera_gps,
        gcps=sample_gcps,
        validation_results=sample_validation_results
    )

    # Verify Leaflet.js CDN link
    assert 'unpkg.com/leaflet@1.9.4/dist/leaflet.css' in html, \
        "HTML should contain Leaflet CSS CDN link"
    assert 'unpkg.com/leaflet@1.9.4/dist/leaflet.js' in html, \
        "HTML should contain Leaflet JS CDN link"

    # Verify ESRI tile URL
    assert 'server.arcgisonline.com/ArcGIS/rest/services/World_Imagery' in html, \
        "HTML should contain ESRI World Imagery tile URL"

    # Verify side-by-side layout structure
    assert '<div class="container">' in html, "HTML should have container div"
    assert '<div class="content">' in html, "HTML should have content div"
    assert '<div class="panel">' in html, "HTML should have panel divs"

    # Verify camera frame img element
    assert 'id="cameraFrame"' in html, "HTML should have camera frame img element"
    assert 'src="frame.jpg"' in html, "HTML should reference camera frame path"

    # Verify map div element
    assert 'id="map"' in html, "HTML should have map div element"

    # Verify canvas for GCP overlay
    assert 'id="gcpCanvas"' in html, "HTML should have GCP canvas element"


def test_generate_html_gcps_embedded(sample_gcps, sample_validation_results,
                                     sample_camera_gps):
    """Test that GCP data is properly embedded in HTML JavaScript."""
    html = generate_html(
        camera_frame_path='frame.jpg',
        kml_path='validation.kml',
        camera_gps=sample_camera_gps,
        gcps=sample_gcps,
        validation_results=sample_validation_results
    )

    # Verify GCP coordinates appear in JavaScript section
    for gcp in sample_gcps:
        lat = gcp['gps']['latitude']
        lon = gcp['gps']['longitude']

        # Coordinates should appear in the embedded JSON data
        assert str(lat) in html, f"GCP latitude {lat} should be in HTML"
        assert str(lon) in html, f"GCP longitude {lon} should be in HTML"

    # Verify GCP descriptions are included
    for gcp in sample_gcps:
        desc = gcp['metadata']['description']
        assert desc in html, f"GCP description '{desc}' should be in HTML"


def test_generate_html_camera_gps_centered(sample_gcps, sample_validation_results,
                                           sample_camera_gps):
    """Test that camera GPS coordinates are used for map initialization."""
    html = generate_html(
        camera_frame_path='frame.jpg',
        kml_path='validation.kml',
        camera_gps=sample_camera_gps,
        gcps=sample_gcps,
        validation_results=sample_validation_results
    )

    # Verify camera GPS appears in JavaScript
    camera_lat = sample_camera_gps['latitude']
    camera_lon = sample_camera_gps['longitude']

    assert str(camera_lat) in html, "Camera latitude should be in HTML"
    assert str(camera_lon) in html, "Camera longitude should be in HTML"

    # Verify map.setView call with camera coordinates
    assert 'map.setView' in html, "HTML should contain map.setView call"


def test_generate_html_google_maps_without_api_key(sample_gcps, sample_validation_results,
                                                    sample_camera_gps):
    """Test that Google Maps layer is not enabled without API key."""
    # Ensure no GOOGLE_MAPS_API_KEY in environment
    with patch.dict(os.environ, {}, clear=True):
        html = generate_html(
            camera_frame_path='frame.jpg',
            kml_path='validation.kml',
            camera_gps=sample_camera_gps,
            gcps=sample_gcps,
            validation_results=sample_validation_results
        )

        # Google Maps API key variable should be empty
        assert "const googleMapsApiKey = '';" in html or \
               'const googleMapsApiKey = "";' in html, \
               "Google Maps API key should be empty string"

        # Conditional check should prevent Google Maps layer
        assert 'if (googleMapsApiKey)' in html, \
            "HTML should have conditional check for Google Maps API key"


def test_generate_html_google_maps_with_api_key(sample_gcps, sample_validation_results,
                                                 sample_camera_gps):
    """Test that Google Maps layer is enabled with API key."""
    test_api_key = 'TEST_API_KEY_12345'

    # Mock GOOGLE_MAPS_API_KEY environment variable
    with patch.dict(os.environ, {'GOOGLE_MAPS_API_KEY': test_api_key}):
        html = generate_html(
            camera_frame_path='frame.jpg',
            kml_path='validation.kml',
            camera_gps=sample_camera_gps,
            gcps=sample_gcps,
            validation_results=sample_validation_results
        )

        # Verify API key is embedded in HTML
        assert test_api_key in html, "Google Maps API key should be in HTML"

        # Verify Google Maps tile URL template is present
        assert 'mt1.google.com/vt/lyrs=s' in html, \
            "HTML should contain Google Maps tile URL when API key is present"


def test_generate_html_valid_structure(sample_gcps, sample_validation_results,
                                       sample_camera_gps):
    """Test that generated HTML has valid structure."""
    html = generate_html(
        camera_frame_path='frame.jpg',
        kml_path='validation.kml',
        camera_gps=sample_camera_gps,
        gcps=sample_gcps,
        validation_results=sample_validation_results
    )

    # Verify HTML starts with DOCTYPE
    assert html.strip().startswith('<!DOCTYPE html>'), \
        "HTML should start with DOCTYPE declaration"

    # Verify essential HTML tags
    assert '<html' in html, "HTML should contain <html> tag"
    assert '</html>' in html, "HTML should contain </html> closing tag"
    assert '<head>' in html, "HTML should contain <head> tag"
    assert '</head>' in html, "HTML should contain </head> closing tag"
    assert '<body>' in html, "HTML should contain <body> tag"
    assert '</body>' in html, "HTML should contain </body> closing tag"

    # Verify meta charset
    assert 'charset="UTF-8"' in html or "charset='UTF-8'" in html, \
        "HTML should specify UTF-8 charset"


# ============================================================================
# Server and File Output Tests
# ============================================================================

def test_server_output_directory_created(sample_gcps, sample_validation_results,
                                         sample_camera_gps, sample_frame,
                                         temp_output_dir):
    """Test that server creates output files in specified directory."""
    # Create mock assets
    frame_path = os.path.join(temp_output_dir, 'frame.jpg')
    kml_path = os.path.join(temp_output_dir, 'test.kml')

    # Save sample frame
    cv2.imwrite(frame_path, sample_frame)

    # Create minimal KML file
    with open(kml_path, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
        f.write('  <Document><name>Test</name></Document>\n')
        f.write('</kml>\n')

    # Mock webbrowser.open to prevent actual browser launch
    with patch('webbrowser.open'):
        # Mock server to prevent it from running indefinitely
        with patch('poc_homography.map_debug_server.socketserver.TCPServer') as mock_server:
            mock_server_instance = MagicMock()
            mock_server.return_value.__enter__ = MagicMock(return_value=mock_server_instance)
            mock_server.return_value.__exit__ = MagicMock(return_value=False)

            # Mock serve_forever to return immediately
            mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()

            # Start server (will exit immediately due to mock)
            try:
                start_server(
                    output_dir=temp_output_dir,
                    camera_frame_path=frame_path,
                    kml_path=kml_path,
                    camera_gps=sample_camera_gps,
                    gcps=sample_gcps,
                    validation_results=sample_validation_results,
                    auto_open=False
                )
            except KeyboardInterrupt:
                pass  # Expected due to mock

    # Verify index.html was created
    index_path = os.path.join(temp_output_dir, 'index.html')
    assert os.path.isfile(index_path), "index.html should be created"

    # Verify index.html is non-empty
    assert os.path.getsize(index_path) > 0, "index.html should not be empty"

    # Verify assets are present
    assert os.path.isfile(frame_path), "Frame should be in output directory"
    assert os.path.isfile(kml_path), "KML should be in output directory"


# ============================================================================
# Frame Export Tests
# ============================================================================

def test_export_frame_with_markers_creates_file(sample_frame, sample_gcps,
                                                sample_validation_results,
                                                temp_output_dir):
    """Test that export_frame_with_markers creates a file."""
    output_path = os.path.join(temp_output_dir, 'annotated_frame.jpg')

    # Export frame with markers
    result_path = export_frame_with_markers(
        frame=sample_frame,
        gcps=sample_gcps,
        validation_results=sample_validation_results,
        output_path=output_path
    )

    # Verify file was created
    assert os.path.isfile(output_path), "Annotated frame should be created"

    # Verify returned path is absolute
    assert os.path.isabs(result_path), "Returned path should be absolute"

    # Verify file is non-empty
    file_size = os.path.getsize(output_path)
    assert file_size > 0, "Annotated frame should not be empty"

    # Verify it's a valid image by loading it
    loaded_frame = cv2.imread(output_path)
    assert loaded_frame is not None, "Exported file should be a valid image"
    assert loaded_frame.shape == sample_frame.shape, \
        "Exported frame should have same dimensions as input"


def test_export_frame_with_markers_draws_elements(sample_gcps, sample_validation_results,
                                                  temp_output_dir):
    """Test that export_frame_with_markers actually draws markers."""
    # Create a white frame for easy marker detection
    white_frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 255

    output_path = os.path.join(temp_output_dir, 'marked_frame.jpg')

    # Export frame with markers
    export_frame_with_markers(
        frame=white_frame,
        gcps=sample_gcps,
        validation_results=sample_validation_results,
        output_path=output_path
    )

    # Load the exported frame
    marked_frame = cv2.imread(output_path)

    # Verify frame has changed from all white
    # If markers were drawn, some pixels should not be white (255, 255, 255)
    non_white_pixels = np.sum(np.any(marked_frame != 255, axis=2))

    assert non_white_pixels > 0, \
        "Exported frame should have non-white pixels (markers drawn)"

    # Check that markers were drawn near expected GCP locations
    for gcp in sample_gcps:
        u = int(gcp['image']['u'])
        v = int(gcp['image']['v'])

        # Check a small region around the GCP location
        if 0 <= v < 1080 and 0 <= u < 1920:
            region = marked_frame[max(0, v-10):min(1080, v+10),
                                 max(0, u-10):min(1920, u+10)]

            # Region should have some non-white pixels
            region_non_white = np.sum(np.any(region != 255, axis=2))
            assert region_non_white > 0, \
                f"Region around GCP at ({u}, {v}) should have markers"


# ============================================================================
# Integration Tests
# ============================================================================

def test_integration_end_to_end(sample_frame, sample_gcps, sample_validation_results,
                                sample_camera_gps, temp_output_dir):
    """Test complete integration: generate all outputs."""
    # Import KML generator
    from poc_homography.kml_generator import generate_kml

    # Step 1: Export annotated frame
    frame_path = os.path.join(temp_output_dir, 'annotated_frame.jpg')
    export_frame_with_markers(
        frame=sample_frame,
        gcps=sample_gcps,
        validation_results=sample_validation_results,
        output_path=frame_path
    )

    # Step 2: Generate KML
    kml_path = os.path.join(temp_output_dir, 'validation.kml')
    generate_kml(
        gcps=sample_gcps,
        validation_results=sample_validation_results,
        camera_gps=sample_camera_gps,
        output_path=kml_path
    )

    # Step 3: Generate HTML
    html = generate_html(
        camera_frame_path='annotated_frame.jpg',
        kml_path='validation.kml',
        camera_gps=sample_camera_gps,
        gcps=sample_gcps,
        validation_results=sample_validation_results
    )

    # Write HTML to output directory
    html_path = os.path.join(temp_output_dir, 'index.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    # Verify all three outputs exist
    assert os.path.isfile(frame_path), "Annotated frame should exist"
    assert os.path.isfile(kml_path), "KML file should exist"
    assert os.path.isfile(html_path), "HTML file should exist"

    # Verify all outputs are non-empty
    assert os.path.getsize(frame_path) > 0, "Annotated frame should not be empty"
    assert os.path.getsize(kml_path) > 0, "KML file should not be empty"
    assert os.path.getsize(html_path) > 0, "HTML file should not be empty"

    # Verify KML is valid XML
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()
        assert root.tag.endswith('kml'), "KML root should be 'kml' tag"
    except ET.ParseError:
        pytest.fail("Generated KML is not valid XML")

    # Verify HTML contains expected content
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    assert '<!DOCTYPE html>' in html_content, "HTML should have DOCTYPE"
    assert 'annotated_frame.jpg' in html_content, "HTML should reference frame"
    assert 'gcpData' in html_content, "HTML should contain embedded GCP data"
    assert 'leaflet' in html_content.lower(), "HTML should include Leaflet"


def test_generate_html_with_empty_validation_results(sample_gcps, sample_camera_gps):
    """Test HTML generation with empty validation results."""
    # Empty validation results
    empty_results = {'details': []}

    html = generate_html(
        camera_frame_path='frame.jpg',
        kml_path='validation.kml',
        camera_gps=sample_camera_gps,
        gcps=sample_gcps,
        validation_results=empty_results
    )

    # Should still generate valid HTML
    assert '<!DOCTYPE html>' in html, "Should generate valid HTML"
    assert 'id="map"' in html, "Should have map div"

    # GCPs should still be present
    for gcp in sample_gcps:
        lat = gcp['gps']['latitude']
        assert str(lat) in html, f"GCP data should be in HTML even without validation"


def test_generate_html_with_no_projected_gps(sample_gcps, sample_camera_gps):
    """Test HTML generation when validation results lack projected GPS."""
    # Validation results without projected_gps
    incomplete_results = {
        'details': [
            {'error_meters': 0.0}  # No projected_gps field
        ]
    }

    html = generate_html(
        camera_frame_path='frame.jpg',
        kml_path='validation.kml',
        camera_gps=sample_camera_gps,
        gcps=sample_gcps,
        validation_results=incomplete_results
    )

    # Should still generate valid HTML
    assert '<!DOCTYPE html>' in html, "Should generate valid HTML"
    assert 'leaflet' in html.lower(), "Should include Leaflet"


if __name__ == '__main__':
    """Run tests with pytest."""
    pytest.main([__file__, '-v'])
