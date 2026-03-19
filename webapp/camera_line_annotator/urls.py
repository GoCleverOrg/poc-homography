"""URL patterns for camera line annotator app."""

from django.urls import path

from . import views

app_name = "camera_line_annotator"

urlpatterns = [
    # Main page
    path("", views.index, name="index"),
    # API endpoints
    path("api/images/", views.api_images, name="api_images"),
    path("api/switch-image/", views.api_switch_image, name="api_switch_image"),
    path("api/line-ids/", views.api_line_ids, name="api_line_ids"),
    path("api/annotations/", views.api_annotations, name="api_annotations"),
    path("api/annotations/create/", views.api_annotations_create, name="api_annotations_create"),
    path(
        "api/annotations/<str:line_id>/",
        views.api_annotation_detail,
        name="api_annotation_detail",
    ),
    path("api/camera-status/", views.api_camera_status, name="api_camera_status"),
    # Image serving
    path("image/", views.serve_image, name="serve_image"),
]
