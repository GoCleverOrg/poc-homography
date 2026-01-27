"""Django app configuration for lens_calibration."""

from django.apps import AppConfig


class LensCalibrationConfig(AppConfig):
    """Configuration for the Lens Calibration Django app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "lens_calibration"
    verbose_name = "Lens Distortion Calibration Tool"
