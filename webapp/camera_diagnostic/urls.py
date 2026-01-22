"""URL patterns for camera diagnostic app."""

from django.urls import path

from . import views

app_name = "camera_diagnostic"

urlpatterns = [
    # Main page
    path("", views.index, name="index"),
    # API endpoints - Tenants and Cameras
    path("api/tenants/", views.api_tenants, name="api_tenants"),
    path("api/cameras/", views.api_cameras, name="api_cameras"),
    # RTSP streaming endpoints (camera_name is now camera_id like "valte_cam01")
    path("api/video-stream/<str:camera_name>/", views.api_video_stream, name="api_video_stream"),
    path("api/test-rtsp/<str:camera_name>/", views.api_test_rtsp, name="api_test_rtsp"),
    path(
        "api/capture-snapshot/<str:camera_name>/",
        views.api_capture_snapshot,
        name="api_capture_snapshot",
    ),
    # Web UI testing endpoint
    path("api/test-webui/<str:camera_name>/", views.api_test_webui, name="api_test_webui"),
    # PTZ API testing endpoint
    path("api/test-ptz/<str:camera_name>/", views.api_test_ptz, name="api_test_ptz"),
]
