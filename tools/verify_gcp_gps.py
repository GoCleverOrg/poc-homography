#!/usr/bin/env python3
"""
GCP GPS Verification Tool

Generates an interactive map to verify GCP GPS coordinates against satellite imagery.
Opens in browser with GCPs plotted on Google/OSM satellite view for visual inspection.

Usage:
    python verify_gcp_gps.py --gcps gcps.yaml --output map.html
    python verify_gcp_gps.py --gcps gcps.yaml --camera Valte --show-fov
"""

import argparse
import math
import os
import sys
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from poc_homography.camera_config import get_camera_by_name_safe
from poc_homography.gps_distance_calculator import dms_to_dd
from poc_homography.satellite_layers import generate_satellite_layers_js


def get_camera_config_decimal(camera_name: str) -> dict:
    """
    Get camera config with GPS coordinates converted to decimal degrees.

    Args:
        camera_name: Name of the camera (e.g., "Valte", "Setram")

    Returns:
        Camera config dict with lat/lon in decimal degrees, or None if not found
    """
    cam = get_camera_by_name_safe(camera_name)
    if not cam:
        return None

    # Convert DMS coordinates to decimal degrees
    return {
        "lat": dms_to_dd(cam["lat"]),
        "lon": dms_to_dd(cam["lon"]),
        "height_m": cam["height_m"],
        "pan_offset_deg": cam["pan_offset_deg"],
    }


def load_gcps_from_yaml(yaml_path: str, image_height: int = 1080) -> tuple:
    """
    Load GCPs from YAML file.

    Returns (gcps, ptz_info, metadata)
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML required")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    gcps = []
    ptz_info = None
    metadata = {}

    if "gcps" in data:
        # Simple format
        for gcp in data["gcps"]:
            gcps.append(
                {
                    "lat": gcp["lat"],
                    "lon": gcp["lon"],
                    "name": gcp.get("name", "GCP"),
                    "pixel_u": gcp.get("pixel_u", 0),
                    "pixel_v": gcp.get("pixel_v", 0),
                }
            )
    elif "homography" in data:
        # Complex format from capture tool
        ctx = data["homography"]["feature_match"]["camera_capture_context"]
        ptz_info = ctx.get("ptz_position", {})
        metadata = {
            "camera_name": ctx.get("camera_name"),
            "capture_timestamp": ctx.get("capture_timestamp"),
            "notes": ctx.get("notes"),
            "coordinate_system": ctx.get("coordinate_system"),
        }

        coordinate_system = ctx.get("coordinate_system")

        for gcp in data["homography"]["feature_match"]["ground_control_points"]:
            v = gcp["image"]["v"]
            if coordinate_system is None:
                v = image_height - v

            gcps.append(
                {
                    "lat": gcp["gps"]["latitude"],
                    "lon": gcp["gps"]["longitude"],
                    "name": gcp.get("metadata", {}).get("description", "GCP"),
                    "pixel_u": gcp["image"]["u"],
                    "pixel_v": v,
                }
            )

    return gcps, ptz_info, metadata


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point 1 to point 2."""
    dlat = (lat2 - lat1) * 111320
    dlon = (lon2 - lon1) * 111320 * math.cos(math.radians(lat1))
    return math.degrees(math.atan2(dlon, dlat)) % 360


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two GPS points."""
    dlat = (lat2 - lat1) * 111320
    dlon = (lon2 - lon1) * 111320 * math.cos(math.radians(lat1))
    return math.sqrt(dlat**2 + dlon**2)


def generate_map_html(
    gcps, camera_config=None, ptz_info=None, metadata=None, title="GCP Verification Map"
):
    """Generate interactive HTML map with GCPs plotted."""

    # Check for Google Maps API key and generate satellite layers
    google_maps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    satellite_layers_js = generate_satellite_layers_js(
        google_api_key=google_maps_api_key if google_maps_api_key else None, default_layer="google"
    )

    # Calculate center point
    if gcps:
        center_lat = sum(g["lat"] for g in gcps) / len(gcps)
        center_lon = sum(g["lon"] for g in gcps) / len(gcps)
    elif camera_config:
        center_lat = camera_config["lat"]
        center_lon = camera_config["lon"]
    else:
        center_lat, center_lon = 39.64, -0.23

    # Calculate bounds for zoom
    if gcps:
        min_lat = min(g["lat"] for g in gcps)
        max_lat = max(g["lat"] for g in gcps)
        min_lon = min(g["lon"] for g in gcps)
        max_lon = max(g["lon"] for g in gcps)
    else:
        min_lat, max_lat = center_lat - 0.001, center_lat + 0.001
        min_lon, max_lon = center_lon - 0.001, center_lon + 0.001

    # Build GCP markers JavaScript
    gcp_markers_js = ""
    for i, gcp in enumerate(gcps):
        # Calculate distance and bearing from camera if available
        extra_info = ""
        if camera_config:
            dist = calculate_distance(
                camera_config["lat"], camera_config["lon"], gcp["lat"], gcp["lon"]
            )
            bearing = calculate_bearing(
                camera_config["lat"], camera_config["lon"], gcp["lat"], gcp["lon"]
            )
            extra_info = f"<br>Distance: {dist:.1f}m<br>Bearing: {bearing:.1f}&deg;"

        popup_content = (
            f"<b>{gcp['name']}</b><br>"
            f"GPS: {gcp['lat']:.6f}, {gcp['lon']:.6f}<br>"
            f"Pixel: ({gcp['pixel_u']:.1f}, {gcp['pixel_v']:.1f})"
            f"{extra_info}"
        )

        gcp_markers_js += f"""
        L.circleMarker([{gcp["lat"]}, {gcp["lon"]}], {{
            radius: 6,
            fillColor: '#ff4444',
            color: '#aa0000',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        }}).addTo(map).bindPopup('{popup_content}');
        """

    # Camera marker and FOV cone
    camera_js = ""
    if camera_config:
        camera_js = f"""
        // Camera position marker
        L.marker([{camera_config["lat"]}, {camera_config["lon"]}], {{
            icon: L.divIcon({{
                className: 'camera-icon',
                html: '<div style="background:#0066ff;width:12px;height:12px;border-radius:50%;border:2px solid white;"></div>',
                iconSize: [16, 16],
                iconAnchor: [8, 8]
            }})
        }}).addTo(map).bindPopup('<b>Camera</b><br>Height: {camera_config["height_m"]:.2f}m<br>Pan offset: {camera_config["pan_offset_deg"]:.1f}&deg;');
        """

        # Add FOV cone if PTZ info available
        if ptz_info and "pan" in ptz_info:
            pan = ptz_info["pan"] + camera_config.get("pan_offset_deg", 0)
            # Approximate FOV based on zoom (60° at zoom=1)
            zoom = ptz_info.get("zoom", 1.0)
            fov = 60.0 / zoom

            # Calculate FOV cone points (50m distance for visualization)
            cone_distance = 50
            left_bearing = pan - fov / 2
            right_bearing = pan + fov / 2

            def point_at_bearing(lat, lon, bearing, distance):
                dlat = distance * math.cos(math.radians(bearing)) / 111320
                dlon = (
                    distance
                    * math.sin(math.radians(bearing))
                    / (111320 * math.cos(math.radians(lat)))
                )
                return lat + dlat, lon + dlon

            left_lat, left_lon = point_at_bearing(
                camera_config["lat"], camera_config["lon"], left_bearing, cone_distance
            )
            right_lat, right_lon = point_at_bearing(
                camera_config["lat"], camera_config["lon"], right_bearing, cone_distance
            )
            center_lat_fov, center_lon_fov = point_at_bearing(
                camera_config["lat"], camera_config["lon"], pan, cone_distance
            )

            camera_js += f"""
            // FOV cone
            L.polygon([
                [{camera_config["lat"]}, {camera_config["lon"]}],
                [{left_lat}, {left_lon}],
                [{right_lat}, {right_lon}]
            ], {{
                color: '#0066ff',
                fillColor: '#0066ff',
                fillOpacity: 0.15,
                weight: 1
            }}).addTo(map).bindPopup('FOV: {fov:.1f}&deg;<br>Pan: {pan:.1f}&deg;');

            // Center line
            L.polyline([
                [{camera_config["lat"]}, {camera_config["lon"]}],
                [{center_lat_fov}, {center_lon_fov}]
            ], {{
                color: '#0066ff',
                weight: 2,
                dashArray: '5,5'
            }}).addTo(map);
            """

    # Metadata info
    meta_info = ""
    if metadata:
        meta_parts = []
        if metadata.get("camera_name"):
            meta_parts.append(f"Camera: {metadata['camera_name']}")
        if metadata.get("capture_timestamp"):
            meta_parts.append(f"Captured: {metadata['capture_timestamp']}")
        if metadata.get("notes"):
            meta_parts.append(f"Notes: {metadata['notes']}")
        if metadata.get("coordinate_system") is None:
            meta_parts.append("Format: Legacy (leaflet_y, converted)")
        meta_info = "<br>".join(meta_parts)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
        .info-panel {{
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 1000;
            background: white;
            padding: 10px 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            max-width: 300px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            cursor: move;
            user-select: none;
        }}
        .info-panel h3 {{ margin: 0 0 10px 0; font-size: 14px; cursor: move; }}
        .legend {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd; }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-panel" id="infoPanel">
        <h3>GCP Verification Map <span style="font-size:10px;color:#999;">drag to move</span></h3>
        <div>GCPs: {len(gcps)}</div>
        <div>{meta_info}</div>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-dot" style="background:#ff4444;border:2px solid #aa0000;"></div>
                <span>GCP Location</span>
            </div>
            {"<div class='legend-item'><div class='legend-dot' style='background:#0066ff;border:2px solid white;'></div><span>Camera</span></div>" if camera_config else ""}
        </div>
        <div style="margin-top:10px;font-size:11px;color:#666;">
            Click markers for details.<br>
            Use satellite view to verify GPS accuracy.
        </div>
    </div>
    <script>
        var map = L.map('map').fitBounds([
            [{min_lat - 0.0002}, {min_lon - 0.0003}],
            [{max_lat + 0.0002}, {max_lon + 0.0003}]
        ]);

        // Satellite layer configuration (from shared module)
        {satellite_layers_js}

        // Add GCP markers
        {gcp_markers_js}

        // Add camera marker and FOV
        {camera_js}

        // Make info panel draggable
        (function() {{
            var panel = document.getElementById('infoPanel');
            var isDragging = false;
            var offsetX, offsetY;

            panel.addEventListener('mousedown', function(e) {{
                isDragging = true;
                offsetX = e.clientX - panel.offsetLeft;
                offsetY = e.clientY - panel.offsetTop;
                panel.style.opacity = '0.8';
            }});

            document.addEventListener('mousemove', function(e) {{
                if (isDragging) {{
                    panel.style.left = (e.clientX - offsetX) + 'px';
                    panel.style.top = (e.clientY - offsetY) + 'px';
                    panel.style.right = 'auto';
                }}
            }});

            document.addEventListener('mouseup', function() {{
                isDragging = false;
                panel.style.opacity = '1';
            }});
        }})();
    </script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Verify GCP GPS coordinates on an interactive map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_gcp_gps.py --gcps my_gcps.yaml
  python verify_gcp_gps.py --gcps gcps.yaml --camera Valte --output map.html
  python verify_gcp_gps.py --gcps gcps.yaml --camera Valte --no-browser
        """,
    )
    parser.add_argument("--gcps", "-g", required=True, help="Path to YAML file with GCPs")
    parser.add_argument("--camera", "-c", help="Camera name to show position and FOV")
    parser.add_argument("--output", "-o", help="Output HTML file (default: gcp_map.html)")
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open browser automatically"
    )

    args = parser.parse_args()

    # Load GCPs
    print(f"Loading GCPs from: {args.gcps}")
    gcps, ptz_info, metadata = load_gcps_from_yaml(args.gcps)
    print(f"  Loaded {len(gcps)} GCPs")

    if metadata.get("coordinate_system") is None:
        print("  Note: Converted from legacy leaflet_y format")

    # Get camera config if specified
    camera_config = None
    if args.camera:
        camera_config = get_camera_config_decimal(args.camera)
        if camera_config:
            print(
                f"  Camera: {args.camera} at ({camera_config['lat']:.6f}, {camera_config['lon']:.6f})"
            )
        else:
            print(f"  Warning: Unknown camera '{args.camera}'")

    # Generate map
    title = f"GCP Verification - {os.path.basename(args.gcps)}"
    html = generate_map_html(gcps, camera_config, ptz_info, metadata, title)

    # Write output
    output_path = args.output or "gcp_map.html"
    with open(output_path, "w") as f:
        f.write(html)
    print(f"\nMap saved to: {output_path}")

    # Calculate statistics
    if camera_config and gcps:
        distances = [
            calculate_distance(camera_config["lat"], camera_config["lon"], g["lat"], g["lon"])
            for g in gcps
        ]
        bearings = [
            calculate_bearing(camera_config["lat"], camera_config["lon"], g["lat"], g["lon"])
            for g in gcps
        ]

        print("\nGCP Statistics:")
        print(f"  Distance range: {min(distances):.1f}m - {max(distances):.1f}m")
        print(f"  Bearing range: {min(bearings):.1f}° - {max(bearings):.1f}°")

        if ptz_info and "pan" in ptz_info:
            expected_bearing = ptz_info["pan"] + camera_config.get("pan_offset_deg", 0)
            print(f"  Expected center bearing: {expected_bearing:.1f}°")

    # Open in browser
    if not args.no_browser:
        abs_path = os.path.abspath(output_path)
        print("\nOpening in browser...")
        webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()
