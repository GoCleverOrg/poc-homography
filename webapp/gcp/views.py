"""
Views for Map Points capture and visualization.

Django view wrappers that use MapPointRegistry for persistence with per-site storage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from poc_homography.map_points import MapPoint, MapPointRegistry
from poc_homography.sites_config import SITES, get_all_site_names, get_site_by_name

# Project root and data directories
# Path: views.py -> gcp/ -> webapp/ -> project_root/
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MAP_POINTS_DIR = DATA_DIR / "map_points"


def _ensure_map_points_dir() -> None:
    """Ensure map_points data directory exists."""
    MAP_POINTS_DIR.mkdir(parents=True, exist_ok=True)


def _get_site_map_points_file(site_name: str) -> Path:
    """Get the map points file path for a site.

    Args:
        site_name: Name of the site (e.g., "Valte", "Setram")

    Returns:
        Path to the site's YAML file (e.g., data/map_points/valte.yaml)
    """
    return MAP_POINTS_DIR / f"{site_name.lower()}.yaml"


def _resolve_safe_file_path(filename: str) -> Path | None:
    """
    Safely resolve a filename to a path within the allowed gcps directory.

    Prevents path traversal attacks by:
    1. Extracting only the filename (stripping any directory components)
    2. Constructing the path within DATA_DIR/gcps
    3. Validating the resolved path is still within DATA_DIR

    Args:
        filename: User-provided filename (may contain malicious path components)

    Returns:
        Safe resolved Path within gcps directory, or None if invalid
    """
    # Extract just the filename, stripping any directory components
    safe_name = Path(filename).name

    # Reject empty filenames or hidden files
    if not safe_name or safe_name.startswith("."):
        return None

    # Construct path within allowed directory
    gcps_dir = DATA_DIR / "gcps"
    file_path = (gcps_dir / safe_name).resolve()

    # Verify the resolved path is still within DATA_DIR
    try:
        file_path.relative_to(DATA_DIR.resolve())
    except ValueError:
        return None

    return file_path


def _load_registry_for_site(site_name: str) -> MapPointRegistry:
    """Load MapPointRegistry from disk for a specific site.

    Args:
        site_name: Name of the site to load map points for

    Returns:
        MapPointRegistry loaded from file, or empty registry if file doesn't exist
    """
    file_path = _get_site_map_points_file(site_name)
    if file_path.exists():
        return MapPointRegistry.load(file_path)
    return MapPointRegistry(map_id=site_name.lower(), points={})


def _save_registry_for_site(registry: MapPointRegistry, site_name: str) -> None:
    """Save MapPointRegistry to disk for a specific site.

    Args:
        registry: The MapPointRegistry to save
        site_name: Name of the site to save map points for
    """
    _ensure_map_points_dir()
    file_path = _get_site_map_points_file(site_name)
    registry.save(file_path)


def _point_to_dict(point_id: str, point: MapPoint) -> dict[str, Any]:
    """Convert a MapPoint to dict with its id included."""
    return {"id": point_id, **point.to_dict()}


def _site_to_dict(site) -> dict[str, Any]:
    """Convert a Site to dict for JSON serialization."""
    return {
        "name": site.name,
        "image_path": site.image_path,
        "geotransform": site.geotransform,
        "utm_crs": site.utm_crs,
        "is_configured": site.is_configured,
    }


def index(request: HttpRequest) -> HttpResponse:
    """Landing page with links to tools."""
    return render(request, "gcp/index.html", {"title": "Homography Map Points Tools"})


def gcp_capture(request: HttpRequest) -> HttpResponse:
    """Map Points capture interface for marking points on site reference images."""
    # Get all sites for the selector
    sites_data = [_site_to_dict(site) for site in SITES]

    # Default to first configured site, or first site if none configured
    default_site = next((s for s in SITES if s.is_configured), SITES[0] if SITES else None)
    default_site_name = default_site.name if default_site else ""

    # Load existing points for default site
    if default_site_name:
        registry = _load_registry_for_site(default_site_name)
        points_data = [_point_to_dict(pid, p) for pid, p in registry.points.items()]
        map_id = registry.map_id
    else:
        points_data = []
        map_id = "default"

    context: dict[str, Any] = {
        "title": "Map Points Capture Tool",
        "map_id": map_id,
        "points": points_data,
        "sites": sites_data,
        "default_site": default_site_name,
    }
    return render(request, "gcp/gcp_capture.html", context)


def debug_map(request: HttpRequest) -> HttpResponse:
    """
    Debug visualization for MapPoint data.

    Displays MapPoints from storage for verification.
    Since MapPoints use pixel coordinates (not GPS), this shows a
    list view rather than a geographic map.
    """
    # Load registry from site parameter or file parameter
    site_param = request.GET.get("site")
    file_param = request.GET.get("file")

    if site_param:
        # Load from site-specific file
        site = get_site_by_name(site_param)
        if site:
            registry = _load_registry_for_site(site_param)
        else:
            registry = MapPointRegistry(map_id="default", points={})
    elif file_param:
        # Legacy: load from file parameter
        file_path = _resolve_safe_file_path(file_param)
        if file_path and file_path.exists():
            registry = MapPointRegistry.load(file_path)
        else:
            registry = MapPointRegistry(map_id="default", points={})
    else:
        # Default to first configured site
        default_site = next((s for s in SITES if s.is_configured), None)
        if default_site:
            registry = _load_registry_for_site(default_site.name)
        else:
            registry = MapPointRegistry(map_id="default", points={})

    points_data = [_point_to_dict(pid, p) for pid, p in registry.points.items()]

    context: dict[str, Any] = {
        "map_id": registry.map_id,
        "points": points_data,
        "point_count": len(points_data),
    }

    return render(request, "gcp/debug_map.html", context)


@require_GET
def api_get_gcps(request: HttpRequest) -> JsonResponse:
    """
    API endpoint to get MapPoints from storage for a site.

    Query params:
        - site: Name of the site to load points for (required)

    Returns:
        JSON with success status and list of MapPoint data.
    """
    site_name = request.GET.get("site")

    if not site_name:
        return JsonResponse(
            {"success": False, "error": "Missing 'site' parameter"},
            status=400,
        )

    site = get_site_by_name(site_name)
    if not site:
        return JsonResponse(
            {"success": False, "error": f"Site '{site_name}' not found"},
            status=404,
        )

    registry = _load_registry_for_site(site_name)
    points = [_point_to_dict(pid, p) for pid, p in registry.points.items()]

    return JsonResponse(
        {
            "success": True,
            "site": site_name,
            "map_id": registry.map_id,
            "points": points,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def api_save_gcps(request: HttpRequest) -> JsonResponse:
    """
    API endpoint to save MapPoints to storage for a site.

    Expects JSON body with:
        - site: str - Site name (required)
        - map_id: str - Map identifier (optional, defaults to site name lowercase)
        - points: list - List of point objects with id, pixel_x, pixel_y
    """
    data = json.loads(request.body)
    site_name = data.get("site")

    if not site_name:
        return JsonResponse(
            {"success": False, "error": "Missing 'site' field"},
            status=400,
        )

    site = get_site_by_name(site_name)
    if not site:
        return JsonResponse(
            {"success": False, "error": f"Site '{site_name}' not found"},
            status=404,
        )

    map_id = data.get("map_id", site_name.lower())
    points_data = data.get("points", [])

    # Build MapPoints from data
    points: dict[str, MapPoint] = {}
    for p in points_data:
        point_id = str(p["id"])  # Extract id
        point = MapPoint.from_dict(p)  # MapPoint ignores the id field
        points[point_id] = point  # Use id as key

    # Create and save registry
    registry = MapPointRegistry(map_id=map_id, points=points)
    _save_registry_for_site(registry, site_name)

    return JsonResponse(
        {
            "success": True,
            "message": f"Saved {len(points)} MapPoints for site '{site_name}'",
            "site": site_name,
            "map_id": map_id,
        }
    )


@csrf_exempt
@require_http_methods(["DELETE"])
def api_delete_gcp(request: HttpRequest, gcp_id: int) -> JsonResponse:
    """
    API endpoint to delete a specific MapPoint from a site.

    Query params:
        - site: Name of the site (required)

    Args:
        gcp_id: ID of the MapPoint to delete (passed as int but used as string key)
    """
    site_name = request.GET.get("site")

    if not site_name:
        return JsonResponse(
            {"success": False, "error": "Missing 'site' parameter"},
            status=400,
        )

    site = get_site_by_name(site_name)
    if not site:
        return JsonResponse(
            {"success": False, "error": f"Site '{site_name}' not found"},
            status=404,
        )

    point_id = str(gcp_id)
    registry = _load_registry_for_site(site_name)

    if point_id not in registry.points:
        return JsonResponse(
            {"success": False, "error": f"MapPoint {point_id} not found"},
            status=404,
        )

    # Create new registry without the deleted point
    new_points = {k: v for k, v in registry.points.items() if k != point_id}
    new_registry = MapPointRegistry(map_id=registry.map_id, points=new_points)
    _save_registry_for_site(new_registry, site_name)

    return JsonResponse(
        {
            "success": True,
            "message": f"Deleted MapPoint {point_id} from site '{site_name}'",
        }
    )


@require_GET
def api_get_sites(request: HttpRequest) -> JsonResponse:
    """
    API endpoint to get all available sites.

    Returns:
        JSON with list of site configurations.
    """
    sites_data = [_site_to_dict(site) for site in SITES]
    return JsonResponse(
        {
            "success": True,
            "sites": sites_data,
        }
    )


@require_GET
def api_get_site(request: HttpRequest, site_name: str) -> JsonResponse:
    """
    API endpoint to get a specific site configuration.

    Args:
        site_name: Name of the site to retrieve

    Returns:
        JSON with site configuration.
    """
    site = get_site_by_name(site_name)
    if not site:
        return JsonResponse(
            {"success": False, "error": f"Site '{site_name}' not found"},
            status=404,
        )

    return JsonResponse(
        {
            "success": True,
            "site": _site_to_dict(site),
        }
    )
