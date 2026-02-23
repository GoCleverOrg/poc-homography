"""Django app configuration for point_picker."""

from django.apps import AppConfig
from homography_web.frame_utils import (
    GCPS_DIR,
    get_default_map_id,
    get_map_entity,
    get_map_image_path,
)


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

        # Use the default map
        default_map = get_default_map_id()
        if default_map is None:
            return  # No map configured for default tenant

        try:
            map_file = get_map_image_path(default_map)
            map_entity = get_map_entity(default_map)
        except FileNotFoundError:
            return

        initialize_state(
            map_file,
            width=int(map_entity.photo.width),
            height=int(map_entity.photo.height),
        )

        # Load existing GCPs from repository if available
        if GCPS_DIR.exists():
            from poc_homography.map_points.gcp_registry import list_map_ids

            available = list_map_ids(GCPS_DIR)
            if available:
                state = get_state()
                state.load_from_repo(GCPS_DIR, available[0])
