"""Django app configuration for point_picker."""

from pathlib import Path

from django.apps import AppConfig


class PointPickerConfig(AppConfig):
    """Configuration for the Point Picker Django app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "point_picker"
    verbose_name = "GeoTIFF Point Picker"

    def ready(self) -> None:
        """Initialize point picker state when app loads."""
        from .state import get_state, initialize_state

        # Check if already initialized (avoid double initialization in runserver)
        try:
            get_state()
            return  # Already initialized
        except RuntimeError:
            pass

        # Paths relative to webapp directory
        webapp_dir = Path(__file__).resolve().parent.parent
        project_root = webapp_dir.parent

        # Use the Cartografia valencia map and GCPs
        map_file = project_root / "Cartografia_valencia.tif"
        gcp_file = (
            project_root / "tests" / "homography" / "test_data" / "Cartografia_valencia_gcps.yaml"
        )

        if map_file.exists():
            initialize_state(map_file)

            # Load existing GCPs if available
            if gcp_file.exists():
                state = get_state()
                state.load_registry(gcp_file)
