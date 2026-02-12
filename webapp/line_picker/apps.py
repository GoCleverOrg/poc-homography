"""Django app configuration for line_picker."""

from pathlib import Path

from django.apps import AppConfig


class LinePickerConfig(AppConfig):
    """Configuration for the Line Picker Django app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "line_picker"
    verbose_name = "Line Picker"

    def ready(self) -> None:
        """Initialize line picker state when app loads."""
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

        # Use the Cartografia valencia map
        map_file = project_root / "Cartografia_valencia.tif"
        gcps_dir = project_root / "data" / "gcps"

        if map_file.exists() and gcps_dir.exists():
            from poc_homography.map_points.gcp_registry import list_map_ids

            available = list_map_ids(gcps_dir)
            if available:
                initialize_state(map_file, gcps_dir, available[0])
