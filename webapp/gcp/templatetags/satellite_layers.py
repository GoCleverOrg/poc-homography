"""
Django Template Tag for Satellite Layer Configuration.

Provides reusable Leaflet.js satellite layer configuration for map visualization.
Converted from poc_homography.satellite_layers for Django template usage.

Usage in templates:
    {% load satellite_layers %}
    <script>
        {% satellite_layers_js google_api_key=api_key default_layer='google' %}
    </script>
"""

from __future__ import annotations

from django import template
from django.utils.safestring import SafeString, mark_safe

register = template.Library()

# Configuration constants
DEFAULT_MAX_ZOOM = 23
DEFAULT_MAX_NATIVE_ZOOM = 19
DEFAULT_LAYER = "google"

# Valid layer names
VALID_LAYERS = frozenset({"osm", "esri", "pnoa", "google", "hybrid"})

# Map layer names to JavaScript variable names
LAYER_VAR_NAMES: dict[str, str] = {
    "osm": "osm",
    "esri": "satellite",
    "pnoa": "pnoa",
    "google": "google",
    "hybrid": "hybrid",
}


@register.simple_tag
def satellite_layers_js(
    google_api_key: str | None = None,
    default_layer: str = DEFAULT_LAYER,
    max_zoom: int = DEFAULT_MAX_ZOOM,
    max_native_zoom: int = DEFAULT_MAX_NATIVE_ZOOM,
) -> SafeString:
    """
    Generate JavaScript code for Leaflet.js satellite layer definitions.

    Creates layer definitions for 5 satellite providers:
    - OSM Street Map
    - ESRI Satellite
    - PNOA Spain (high-resolution Spanish orthophotos)
    - Google Satellite (conditional on API key)
    - Hybrid (ESRI + OSM overlay)

    Args:
        google_api_key: Optional Google Maps API key. If None or empty,
            Google Satellite layer uses unauthenticated endpoint.
        default_layer: Which layer to activate by default. Options:
            'google', 'esri', 'pnoa', 'osm', 'hybrid'.
        max_zoom: Maximum zoom level for over-zooming support.
        max_native_zoom: Native tile resolution zoom level.

    Returns:
        JavaScript code string defining baseLayers object and adding default to map.
        The code assumes a Leaflet map object named 'map' already exists.
    """
    # Get JavaScript variable name for default layer
    default_js_var = LAYER_VAR_NAMES.get(default_layer, LAYER_VAR_NAMES[DEFAULT_LAYER])

    # Build JavaScript code
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

    return mark_safe("\n".join(js_parts))
