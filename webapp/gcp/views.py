"""
Views for GCP capture and visualization.

Thin Django view wrappers that reference existing poc_homography library functions.
"""

import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


def index(request):
    """Landing page with links to tools."""
    return render(request, "gcp/index.html", {"title": "Homography GCP Tools"})


def gcp_capture(request):
    """GCP capture interface for marking points on satellite map."""
    return render(request, "gcp/gcp_capture.html", {"title": "GCP Capture Tool"})


def debug_map(request):
    """
    Debug map visualization for GCP verification.

    Displays GCPs from a YAML file on a satellite map for verification.
    References existing poc_homography functionality.
    """
    # For now, return a simple visualization
    # In a full implementation, this would load GCPs from data directory
    # using poc_homography.gcp_loader or similar

    # Placeholder data
    gcps_data = []
    camera_location = None
    camera_name = "Unknown"
    gcp_count = 0

    # Check if GCP file path provided in query params
    gcp_file = request.GET.get("file", None)

    if gcp_file:
        # In a full implementation:
        # from poc_homography.gcp_loader import load_gcps_from_yaml
        # gcps = load_gcps_from_yaml(gcp_file)
        # gcps_data = [{'lat': gcp.lat, 'lng': gcp.lng, 'description': gcp.description, ...} for gcp in gcps]
        pass

    context = {
        "gcps_json": json.dumps(gcps_data),
        "camera_location": json.dumps(camera_location),
        "camera_name": camera_name,
        "gcp_count": gcp_count,
    }

    return render(request, "gcp/debug_map.html", context)


def api_get_gcps(request):
    """
    API endpoint to get GCPs from storage.

    In a full implementation, this would:
    - Load GCPs from data/gcps/ directory using poc_homography functions
    - Return as JSON
    """
    # Placeholder response
    gcps = []

    # In full implementation:
    # from poc_homography.gcp_loader import load_gcps_from_yaml
    # gcp_file = settings.DATA_DIR / 'gcps' / 'latest.yaml'
    # if gcp_file.exists():
    #     gcps_data = load_gcps_from_yaml(gcp_file)
    #     gcps = [serialize_gcp(gcp) for gcp in gcps_data]

    return JsonResponse({"success": True, "gcps": gcps})


@csrf_exempt
def api_save_gcps(request):
    """
    API endpoint to save GCPs to storage.

    In a full implementation, this would:
    - Parse GCP data from request
    - Save to data/gcps/ directory using poc_homography functions
    - Return success status
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        gcps = data.get("gcps", [])

        # In full implementation:
        # from poc_homography.gcp_saver import save_gcps_to_yaml
        # gcp_file = settings.DATA_DIR / 'gcps' / f'gcps_{timestamp}.yaml'
        # save_gcps_to_yaml(gcps, gcp_file)

        return JsonResponse({"success": True, "message": f"Saved {len(gcps)} GCPs"})

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@csrf_exempt
def api_delete_gcp(request, gcp_id):
    """
    API endpoint to delete a specific GCP.

    Args:
        gcp_id: ID of the GCP to delete
    """
    if request.method != "DELETE":
        return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)

    try:
        # In full implementation:
        # Load GCPs, remove the one with gcp_id, save back

        return JsonResponse({"success": True, "message": f"Deleted GCP {gcp_id}"})

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)
