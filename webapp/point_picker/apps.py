"""Django app configuration for point_picker."""

from django.apps import AppConfig


class PointPickerConfig(AppConfig):
    """Configuration for the Point Picker Django app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "point_picker"
    verbose_name = "GeoTIFF Point Picker"
