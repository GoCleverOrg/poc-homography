"""
GCP verification utilities for validating GPS coordinates against satellite imagery.

Generates interactive HTML maps with GCPs plotted on satellite imagery for visual inspection.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from pathlib import Path

from poc_homography.types import Degrees, Meters, PixelsFloat


@dataclass
class GCPMarker:
    """A ground control point marker for map visualization."""

    lat: Degrees
    lon: Degrees
    name: str
    pixel_u: PixelsFloat
    pixel_v: PixelsFloat


@dataclass
class CameraLocation:
    """Camera location and configuration for map visualization."""

    lat: Degrees
    lon: Degrees
    height_m: Meters
    pan_offset_deg: Degrees


@dataclass
class PTZInfo:
    """PTZ (Pan-Tilt-Zoom) information for FOV visualization."""

    pan: Degrees
    tilt: Degrees
    zoom: float


@dataclass
class GCPMetadata:
    """Metadata from GCP YAML file."""

    camera_name: str | None = None
    capture_timestamp: str | None = None
    notes: str | None = None
    coordinate_system: str | None = None


def load_gcps_from_yaml(
    yaml_path: Path, image_height: int = 1080
) -> tuple[list[GCPMarker], PTZInfo | None, GCPMetadata]:
    """
    Load GCPs from YAML file.

    Args:
        yaml_path: Path to YAML file containing GCP data
        image_height: Image height in pixels (for legacy coordinate conversion)

    Returns:
        Tuple of (gcps, ptz_info, metadata)
    """
    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    gcps: list[GCPMarker] = []
    ptz_info: PTZInfo | None = None
    metadata = GCPMetadata()

    if "gcps" in data:
        # Simple format
        for gcp in data["gcps"]:
            gcps.append(
                GCPMarker(
                    lat=Degrees(gcp["lat"]),
                    lon=Degrees(gcp["lon"]),
                    name=gcp.get("name", "GCP"),
                    pixel_u=PixelsFloat(gcp.get("pixel_u", 0)),
                    pixel_v=PixelsFloat(gcp.get("pixel_v", 0)),
                )
            )
    elif "homography" in data:
        # Complex format from capture tool
        ctx = data["homography"]["feature_match"]["camera_capture_context"]
        ptz_data = ctx.get("ptz_position", {})
        if ptz_data:
            ptz_info = PTZInfo(
                pan=Degrees(ptz_data.get("pan", 0.0)),
                tilt=Degrees(ptz_data.get("tilt", 0.0)),
                zoom=ptz_data.get("zoom", 1.0),
            )

        metadata = GCPMetadata(
            camera_name=ctx.get("camera_name"),
            capture_timestamp=ctx.get("capture_timestamp"),
            notes=ctx.get("notes"),
            coordinate_system=ctx.get("coordinate_system"),
        )

        coordinate_system = ctx.get("coordinate_system")

        for gcp in data["homography"]["feature_match"]["ground_control_points"]:
            v = gcp["image"]["v"]
            if coordinate_system is None:
                v = image_height - v

            gcps.append(
                GCPMarker(
                    lat=Degrees(gcp["gps"]["latitude"]),
                    lon=Degrees(gcp["gps"]["longitude"]),
                    name=gcp.get("metadata", {}).get("description", "GCP"),
                    pixel_u=PixelsFloat(gcp["image"]["u"]),
                    pixel_v=PixelsFloat(v),
                )
            )

    return gcps, ptz_info, metadata


def calculate_bearing(lat1: Degrees, lon1: Degrees, lat2: Degrees, lon2: Degrees) -> Degrees:
    """
    Calculate bearing from point 1 to point 2.

    Args:
        lat1: Latitude of point 1
        lon1: Longitude of point 1
        lat2: Latitude of point 2
        lon2: Longitude of point 2

    Returns:
        Bearing in degrees (0-360)
    """
    dlat = (lat2 - lat1) * 111320
    dlon = (lon2 - lon1) * 111320 * math.cos(math.radians(lat1))
    return Degrees(math.degrees(math.atan2(dlon, dlat)) % 360)


def calculate_distance(lat1: Degrees, lon1: Degrees, lat2: Degrees, lon2: Degrees) -> Meters:
    """
    Calculate distance in meters between two GPS points.

    Uses simple Euclidean approximation suitable for small distances.

    Args:
        lat1: Latitude of point 1
        lon1: Longitude of point 1
        lat2: Latitude of point 2
        lon2: Longitude of point 2

    Returns:
        Distance in meters
    """
    dlat = (lat2 - lat1) * 111320
    dlon = (lon2 - lon1) * 111320 * math.cos(math.radians(lat1))
    return Meters(math.sqrt(dlat**2 + dlon**2))


def _generate_satellite_layers_js(
    google_api_key: str | None = None,
    default_layer: str = "google",
    max_zoom: int = 23,
    max_native_zoom: int = 19,
) -> str:
    """
    Generate JavaScript code for Leaflet.js satellite layer definitions.

    Creates layer definitions for 5 satellite providers:
    - OSM Street Map
    - ESRI Satellite
    - PNOA Spain (high-resolution Spanish orthophotos)
    - Google Satellite (conditional on API key)
    - Hybrid (ESRI + OSM overlay)

    Args:
        google_api_key: Optional Google Maps API key
        default_layer: Which layer to activate by default
        max_zoom: Maximum zoom level for over-zooming support
        max_native_zoom: Native tile resolution zoom level

    Returns:
        JavaScript code string defining baseLayers object and adding default to map
    """
    layer_var_names: dict[str, str] = {
        "osm": "osm",
        "esri": "satellite",
        "pnoa": "pnoa",
        "google": "google",
        "hybrid": "hybrid",
    }

    default_js_var = layer_var_names.get(default_layer, layer_var_names["google"])

    js_parts: list[str] = []

    # OSM Street Map
    js_parts.append("""
        // OSM Street Map
        var osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors',
            maxZoom: 19
        });""")

    # ESRI Satellite with over-zoom support
    js_parts.append(f"""
        // ESRI Satellite with over-zoom support
        var satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles &copy; Esri',
            maxNativeZoom: {max_native_zoom},
            maxZoom: {max_zoom}
        }});""")

    # PNOA Spain (high-resolution orthophotos)
    js_parts.append("""
        // PNOA - Spanish high-resolution orthophotos (25-50cm resolution)
        var pnoa = L.tileLayer.wms('https://www.ign.es/wms-inspire/pnoa-ma', {
            layers: 'OI.OrthoimageCoverage',
            format: 'image/png',
            transparent: true,
            attribution: 'PNOA &copy; IGN',
            maxZoom: 22
        });""")

    # Google Satellite (conditional)
    if google_api_key:
        js_parts.append(f"""
        // Google Satellite
        var google = L.tileLayer('https://mt1.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}&key={google_api_key}', {{
            attribution: '&copy; Google',
            maxZoom: 21
        }});""")
    else:
        js_parts.append("""
        // Google Satellite (no API key)
        var google = L.tileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
            attribution: '&copy; Google',
            maxZoom: 21
        });""")

    # Hybrid layer (ESRI + OSM overlay)
    js_parts.append(f"""
        // Hybrid (ESRI satellite + OSM streets overlay)
        var hybrid = L.layerGroup([
            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
                maxNativeZoom: {max_native_zoom},
                maxZoom: {max_zoom}
            }}),
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                opacity: 0.3
            }})
        ]);""")

    # Add default layer to map
    js_parts.append(f"""
        // Add default layer to map
        {default_js_var}.addTo(map);""")

    # Build baseLayers object for layer control
    js_parts.append("""
        // Layer control
        var baseLayers = {
            'Street Map': osm,
            'ESRI Satellite': satellite,
            'PNOA Spain (Best)': pnoa,
            'Google Satellite': google,
            'Hybrid': hybrid
        };

        L.control.layers(baseLayers).addTo(map);""")

    return "\n".join(js_parts)


def generate_verification_map(
    gcps: list[GCPMarker],
    camera_config: CameraLocation | None = None,
    ptz_info: PTZInfo | None = None,
    metadata: GCPMetadata | None = None,
    title: str = "GCP Verification Map",
) -> str:
    """
    Generate interactive HTML map with GCPs plotted.

    Args:
        gcps: List of GCP markers to plot
        camera_config: Optional camera location for distance/bearing calculations
        ptz_info: Optional PTZ info for FOV cone visualization
        metadata: Optional metadata to display
        title: Map title

    Returns:
        Complete HTML string for the verification map
    """
    # Check for Google Maps API key and generate satellite layers
    google_maps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    satellite_layers_js = _generate_satellite_layers_js(
        google_api_key=google_maps_api_key if google_maps_api_key else None,
        default_layer="google",
    )

    # Calculate center point
    if gcps:
        center_lat = sum(g.lat for g in gcps) / len(gcps)
        center_lon = sum(g.lon for g in gcps) / len(gcps)
    elif camera_config:
        center_lat = camera_config.lat
        center_lon = camera_config.lon
    else:
        center_lat, center_lon = Degrees(39.64), Degrees(-0.23)

    # Calculate bounds for zoom
    if gcps:
        min_lat = min(g.lat for g in gcps)
        max_lat = max(g.lat for g in gcps)
        min_lon = min(g.lon for g in gcps)
        max_lon = max(g.lon for g in gcps)
    else:
        min_lat, max_lat = center_lat - 0.001, center_lat + 0.001
        min_lon, max_lon = center_lon - 0.001, center_lon + 0.001

    # Build GCP markers JavaScript
    gcp_markers_js = ""
    for gcp in gcps:
        # Calculate distance and bearing from camera if available
        extra_info = ""
        if camera_config:
            dist = calculate_distance(camera_config.lat, camera_config.lon, gcp.lat, gcp.lon)
            bearing = calculate_bearing(camera_config.lat, camera_config.lon, gcp.lat, gcp.lon)
            extra_info = f"<br>Distance: {dist:.1f}m<br>Bearing: {bearing:.1f}&deg;"

        popup_content = (
            f"<b>{gcp.name}</b><br>"
            f"GPS: {gcp.lat:.6f}, {gcp.lon:.6f}<br>"
            f"Pixel: ({gcp.pixel_u:.1f}, {gcp.pixel_v:.1f})"
            f"{extra_info}"
        )

        gcp_markers_js += f"""
        L.circleMarker([{gcp.lat}, {gcp.lon}], {{
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
        L.marker([{camera_config.lat}, {camera_config.lon}], {{
            icon: L.divIcon({{
                className: 'camera-icon',
                html: '<div style="background:#0066ff;width:12px;height:12px;border-radius:50%;border:2px solid white;"></div>',
                iconSize: [16, 16],
                iconAnchor: [8, 8]
            }})
        }}).addTo(map).bindPopup('<b>Camera</b><br>Height: {camera_config.height_m:.2f}m<br>Pan offset: {camera_config.pan_offset_deg:.1f}&deg;');
        """

        # Add FOV cone if PTZ info available
        if ptz_info:
            pan = ptz_info.pan + camera_config.pan_offset_deg
            # Approximate FOV based on zoom (60° at zoom=1)
            fov = 60.0 / ptz_info.zoom

            # Calculate FOV cone points (50m distance for visualization)
            cone_distance = Meters(50)
            left_bearing = Degrees(pan - fov / 2)
            right_bearing = Degrees(pan + fov / 2)

            def point_at_bearing(
                lat: Degrees, lon: Degrees, bearing: Degrees, distance: Meters
            ) -> tuple[Degrees, Degrees]:
                dlat = distance * math.cos(math.radians(bearing)) / 111320
                dlon = (
                    distance
                    * math.sin(math.radians(bearing))
                    / (111320 * math.cos(math.radians(lat)))
                )
                return Degrees(lat + dlat), Degrees(lon + dlon)

            left_lat, left_lon = point_at_bearing(
                camera_config.lat, camera_config.lon, left_bearing, cone_distance
            )
            right_lat, right_lon = point_at_bearing(
                camera_config.lat, camera_config.lon, right_bearing, cone_distance
            )
            center_lat_fov, center_lon_fov = point_at_bearing(
                camera_config.lat, camera_config.lon, Degrees(pan), cone_distance
            )

            camera_js += f"""
            // FOV cone
            L.polygon([
                [{camera_config.lat}, {camera_config.lon}],
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
                [{camera_config.lat}, {camera_config.lon}],
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
        if metadata.camera_name:
            meta_parts.append(f"Camera: {metadata.camera_name}")
        if metadata.capture_timestamp:
            meta_parts.append(f"Captured: {metadata.capture_timestamp}")
        if metadata.notes:
            meta_parts.append(f"Notes: {metadata.notes}")
        if metadata.coordinate_system is None:
            meta_parts.append("Format: Legacy (leaflet_y, converted)")
        meta_info = "<br>".join(meta_parts)

    camera_legend = (
        "<div class='legend-item'><div class='legend-dot' style='background:#0066ff;border:2px solid white;'></div><span>Camera</span></div>"
        if camera_config
        else ""
    )

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
            {camera_legend}
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
