"""Django app configuration for point_picker."""

from pathlib import Path

from django.apps import AppConfig
from homography_web.frame_utils import DEFAULT_MAP_ID


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

        # Use the default map
        map_file = project_root / f"{DEFAULT_MAP_ID}.tif"
        gcps_dir = project_root / "data" / "gcps"

        if map_file.exists():
            initialize_state(map_file)

            # Load existing GCPs from repository if available
            if gcps_dir.exists():
                from poc_homography.map_points.gcp_registry import from_gcp_repo, list_map_ids

                available = list_map_ids(gcps_dir)
                if available:
                    state = get_state()
                    state.load_from_repo(gcps_dir, available[0])
