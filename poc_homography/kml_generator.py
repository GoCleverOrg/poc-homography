#!/usr/bin/env python3
"""
KML Generation Module for POC Homography Project.

This module generates KML 2.2 compliant XML files from GCP data for visualization
in Google Earth and other KML-compatible mapping applications.

Example Usage:
    from poc_homography.kml_generator import generate_kml, create_output_directory

    # Prepare data
    gcps = [
        {
            'gps': {'latitude': 39.640600, 'longitude': -0.230200},
            'image': {'u': 400.0, 'v': 300.0},
            'metadata': {'description': 'P#01 - Corner', 'accuracy': 'high'}
        },
        # ... more GCPs
    ]

    validation_results = {
        'details': [
            {
                'projected_gps': (39.640605, -0.230205),
                'error_meters': 0.5
            },
            # ... more results
        ]
    }

    camera_gps = {'latitude': 39.640500, 'longitude': -0.230000}

    # Generate KML
    output_dir = create_output_directory()
    output_path = f"{output_dir}/gcp_validation.kml"
    generate_kml(gcps, validation_results, camera_gps, output_path)
"""

import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance in meters between two GPS points using Haversine formula.

    This function computes the great-circle distance between two points on Earth
    specified by their latitude and longitude coordinates.

    Args:
        lat1: Latitude of first point in decimal degrees
        lon1: Longitude of first point in decimal degrees
        lat2: Latitude of second point in decimal degrees
        lon2: Longitude of second point in decimal degrees

    Returns:
        Distance between the two points in meters

    Example:
        >>> distance = haversine_distance(39.640600, -0.230200, 39.640620, -0.229800)
        >>> print(f"Distance: {distance:.2f}m")
        Distance: 35.42m
    """
    R = 6371000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def create_output_directory() -> str:
    """
    Ensure the output directory exists.

    Creates the 'output/' directory if it doesn't exist. This directory is used
    to store generated KML files and other output artifacts.

    Returns:
        Absolute path to the output directory

    Example:
        >>> output_dir = create_output_directory()
        >>> print(output_dir)
        /path/to/poc-homography/output
    """
    # Get the project root (parent of poc_homography module)
    module_dir = Path(__file__).parent
    project_root = module_dir.parent
    output_dir = project_root / 'output'

    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    return str(output_dir)


def _create_style(style_id: str, color: str, icon_scale: float = 1.0,
                  line_width: float = 2.0) -> Element:
    """
    Create a KML Style element with specified color and properties.

    Args:
        style_id: Unique identifier for the style
        color: KML color in format 'aabbggrr' (alpha, blue, green, red)
        icon_scale: Scale factor for icon size (default: 1.0)
        line_width: Width of line in pixels (default: 2.0)

    Returns:
        XML Element representing the Style
    """
    style = Element('Style', {'id': style_id})

    # Icon style
    icon_style = SubElement(style, 'IconStyle')
    SubElement(icon_style, 'color').text = color
    SubElement(icon_style, 'scale').text = str(icon_scale)
    icon = SubElement(icon_style, 'Icon')
    SubElement(icon, 'href').text = 'http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png'

    # Line style
    line_style = SubElement(style, 'LineStyle')
    SubElement(line_style, 'color').text = color
    SubElement(line_style, 'width').text = str(line_width)

    return style


def _create_placemark(name: str, description: str, coordinates: str,
                      style_url: str) -> Element:
    """
    Create a KML Placemark element for a point.

    Args:
        name: Name/title of the placemark
        description: Detailed description text
        coordinates: KML coordinate string (longitude,latitude,altitude)
        style_url: Reference to style ID (e.g., '#original-gcp')

    Returns:
        XML Element representing the Placemark
    """
    placemark = Element('Placemark')
    SubElement(placemark, 'name').text = name
    SubElement(placemark, 'description').text = description
    SubElement(placemark, 'styleUrl').text = style_url

    point = SubElement(placemark, 'Point')
    SubElement(point, 'coordinates').text = coordinates

    return placemark


def _create_line_placemark(name: str, description: str,
                          coord1: Tuple[float, float],
                          coord2: Tuple[float, float],
                          style_url: str) -> Element:
    """
    Create a KML Placemark element for a line connecting two points.

    Args:
        name: Name/title of the line placemark
        description: Detailed description text
        coord1: First point as (latitude, longitude) tuple
        coord2: Second point as (latitude, longitude) tuple
        style_url: Reference to style ID (e.g., '#error-line')

    Returns:
        XML Element representing the Placemark with LineString geometry
    """
    placemark = Element('Placemark')
    SubElement(placemark, 'name').text = name
    SubElement(placemark, 'description').text = description
    SubElement(placemark, 'styleUrl').text = style_url

    line_string = SubElement(placemark, 'LineString')
    SubElement(line_string, 'tessellate').text = '1'

    # KML format: longitude,latitude,altitude
    coordinates_text = (
        f"{coord1[1]},{coord1[0]},0 "
        f"{coord2[1]},{coord2[0]},0"
    )
    SubElement(line_string, 'coordinates').text = coordinates_text

    return placemark


def generate_kml(gcps: List[Dict],
                 validation_results: Dict,
                 camera_gps: Dict[str, float],
                 output_path: str) -> None:
    """
    Generate a KML 2.2 compliant XML file from GCP data and validation results.

    This function creates a comprehensive KML file that visualizes:
    - Camera position
    - Original GCP positions
    - Projected GCP positions (from round-trip validation)
    - Error lines connecting original and projected positions

    Args:
        gcps: List of GCP dictionaries with structure:
            {
                'gps': {'latitude': float, 'longitude': float},
                'image': {'u': float, 'v': float},
                'metadata': {'description': str, 'accuracy': str}
            }
        validation_results: Dictionary containing:
            {
                'details': [
                    {
                        'projected_gps': (lat, lon),
                        'error_meters': float,
                        'image_point': (u, v)  # optional
                    },
                    ...
                ]
            }
        camera_gps: Dictionary with camera position:
            {'latitude': float, 'longitude': float}
        output_path: Path string where KML file will be saved

    Returns:
        None (writes KML file to output_path)

    Raises:
        ValueError: If input data is malformed or missing required fields
        IOError: If unable to write to output_path

    Example:
        >>> gcps = [
        ...     {
        ...         'gps': {'latitude': 39.640600, 'longitude': -0.230200},
        ...         'image': {'u': 400.0, 'v': 300.0},
        ...         'metadata': {'description': 'P#01', 'accuracy': 'high'}
        ...     }
        ... ]
        >>> results = {
        ...     'details': [
        ...         {
        ...             'projected_gps': (39.640605, -0.230205),
        ...             'error_meters': 0.5
        ...         }
        ...     ]
        ... }
        >>> camera = {'latitude': 39.640500, 'longitude': -0.230000}
        >>> generate_kml(gcps, results, camera, 'output/validation.kml')
    """
    # Validate inputs
    if not gcps:
        raise ValueError("GCPs list cannot be empty")
    if not validation_results or 'details' not in validation_results:
        raise ValueError("validation_results must contain 'details' key")
    if not camera_gps or 'latitude' not in camera_gps or 'longitude' not in camera_gps:
        raise ValueError("camera_gps must contain 'latitude' and 'longitude' keys")

    # Create root KML element
    kml = Element('kml', {'xmlns': 'http://www.opengis.net/kml/2.2'})
    document = SubElement(kml, 'Document')
    SubElement(document, 'name').text = 'GCP Round-Trip Validation'

    # Create styles
    styles = {
        'camera-position': ('ff00ff00', 1.2, 3.0),  # Green, larger icon
        'original-gcp': ('ff00ff00', 1.0, 2.0),     # Green
        'projected-gcp': ('ff0000ff', 1.0, 2.0),    # Blue
        'error-line': ('ff00ffff', 1.0, 2.0),       # Yellow
    }

    for style_id, (color, icon_scale, line_width) in styles.items():
        style = _create_style(style_id, color, icon_scale, line_width)
        document.append(style)

    # Create Camera Position folder
    camera_folder = Element('Folder')
    SubElement(camera_folder, 'name').text = 'Camera Position'

    camera_lat = camera_gps['latitude']
    camera_lon = camera_gps['longitude']
    camera_coords = f"{camera_lon},{camera_lat},0"
    camera_desc = f"Camera GPS position\nLatitude: {camera_lat:.6f}\nLongitude: {camera_lon:.6f}"

    camera_placemark = _create_placemark(
        'Camera',
        camera_desc,
        camera_coords,
        '#camera-position'
    )
    camera_folder.append(camera_placemark)

    # Create Original GCPs folder
    original_folder = Element('Folder')
    SubElement(original_folder, 'name').text = 'Original GCPs'

    for i, gcp in enumerate(gcps):
        gps = gcp['gps']
        image = gcp['image']
        metadata = gcp.get('metadata', {})

        lat = gps['latitude']
        lon = gps['longitude']
        u = image['u']
        v = image['v']

        name = metadata.get('description', f'GCP {i+1}')
        accuracy = metadata.get('accuracy', 'unknown')

        coords = f"{lon},{lat},0"
        desc = (
            f"Original GCP Position\n"
            f"Latitude: {lat:.6f}\n"
            f"Longitude: {lon:.6f}\n"
            f"Image pixel: ({u:.1f}, {v:.1f})\n"
            f"Accuracy: {accuracy}"
        )

        placemark = _create_placemark(name, desc, coords, '#original-gcp')
        original_folder.append(placemark)

    # Create Projected GCPs folder
    projected_folder = Element('Folder')
    SubElement(projected_folder, 'name').text = 'Projected GCPs'

    details = validation_results['details']

    for i, (gcp, detail) in enumerate(zip(gcps, details)):
        if 'projected_gps' not in detail:
            continue

        proj_lat, proj_lon = detail['projected_gps']
        image = gcp['image']
        u = image['u']
        v = image['v']

        metadata = gcp.get('metadata', {})
        original_name = metadata.get('description', f'GCP {i+1}')
        name = f"{original_name} (projected)"

        coords = f"{proj_lon},{proj_lat},0"
        desc = (
            f"Round-trip projection from pixel ({u:.1f}, {v:.1f})\n"
            f"Projected Latitude: {proj_lat:.6f}\n"
            f"Projected Longitude: {proj_lon:.6f}"
        )

        placemark = _create_placemark(name, desc, coords, '#projected-gcp')
        projected_folder.append(placemark)

    # Create Error Lines folder
    error_folder = Element('Folder')
    SubElement(error_folder, 'name').text = 'Error Lines'

    for i, (gcp, detail) in enumerate(zip(gcps, details)):
        if 'projected_gps' not in detail or 'error_meters' not in detail:
            continue

        gps = gcp['gps']
        orig_lat = gps['latitude']
        orig_lon = gps['longitude']

        proj_lat, proj_lon = detail['projected_gps']
        error_m = detail['error_meters']

        metadata = gcp.get('metadata', {})
        original_name = metadata.get('description', f'GCP {i+1}')
        name = f"{original_name} Error"

        desc = f"Error distance: {error_m:.2f}m"

        line_placemark = _create_line_placemark(
            name,
            desc,
            (orig_lat, orig_lon),
            (proj_lat, proj_lon),
            '#error-line'
        )
        error_folder.append(line_placemark)

    # Add folders to document
    document.append(camera_folder)
    document.append(original_folder)
    document.append(projected_folder)
    document.append(error_folder)

    # Create tree and write to file
    tree = ElementTree(kml)

    # Pretty print the XML
    indent(tree, space='  ', level=0)

    # Write to file with XML declaration
    with open(output_path, 'wb') as f:
        tree.write(f, encoding='UTF-8', xml_declaration=True)

    print(f"KML file generated successfully: {output_path}")
    print(f"  Camera position: ({camera_lat:.6f}, {camera_lon:.6f})")
    print(f"  Original GCPs: {len(gcps)}")
    print(f"  Projected GCPs: {len([d for d in details if 'projected_gps' in d])}")
    print(f"  Error lines: {len([d for d in details if 'error_meters' in d])}")


if __name__ == '__main__':
    """
    Example usage and test of the KML generator.
    """
    # Create sample data
    sample_gcps = [
        {
            'gps': {'latitude': 39.640600, 'longitude': -0.230200},
            'image': {'u': 400.0, 'v': 300.0},
            'metadata': {'description': 'P#01 - Top Left', 'accuracy': 'high'}
        },
        {
            'gps': {'latitude': 39.640620, 'longitude': -0.229800},
            'image': {'u': 2100.0, 'v': 320.0},
            'metadata': {'description': 'P#02 - Top Right', 'accuracy': 'high'}
        },
        {
            'gps': {'latitude': 39.640400, 'longitude': -0.230000},
            'image': {'u': 1280.0, 'v': 720.0},
            'metadata': {'description': 'P#03 - Center', 'accuracy': 'medium'}
        },
    ]

    sample_results = {
        'details': [
            {
                'projected_gps': (39.640605, -0.230205),
                'error_meters': 0.56,
                'image_point': (400.0, 300.0)
            },
            {
                'projected_gps': (39.640618, -0.229805),
                'error_meters': 0.45,
                'image_point': (2100.0, 320.0)
            },
            {
                'projected_gps': (39.640398, -0.230002),
                'error_meters': 0.23,
                'image_point': (1280.0, 720.0)
            },
        ]
    }

    sample_camera = {
        'latitude': 39.640500,
        'longitude': -0.230000
    }

    # Generate KML
    output_dir = create_output_directory()
    output_file = os.path.join(output_dir, 'example_gcp_validation.kml')

    print("Generating example KML file...")
    generate_kml(sample_gcps, sample_results, sample_camera, output_file)
    print(f"\nExample KML file created at: {output_file}")
    print("You can open this file in Google Earth or any KML viewer.")
