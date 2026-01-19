"""
URL configuration for GCP app.

Separates page views from API endpoints for cleaner routing.
"""

from django.urls import path

from . import views

app_name = "gcp"

# Page views - render HTML templates
page_patterns = [
    path("", views.index, name="index"),
    path("debug/", views.debug_map, name="debug_map"),
]

urlpatterns = page_patterns
