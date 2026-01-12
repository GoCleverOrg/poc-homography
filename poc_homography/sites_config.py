"""Site configuration file.

Central location for all site definitions. Sites represent physical locations
that cameras observe. Multiple cameras can be associated with the same site.
"""

from __future__ import annotations

from poc_homography.sites import Site

# =============================================================================
# SITE CONFIGURATIONS
# =============================================================================
# Sites define physical locations with reference imagery and coordinate systems.
# Each site includes:
# - name: Unique identifier for the site
# - image_path: Path to reference PNG/GeoTIFF (relative to project root)
# - geotransform: GDAL 6-parameter transform [origin_e, px_w, rot_r, origin_n, rot_c, px_h]
# - utm_crs: Coordinate reference system identifier
#
# Sites with empty image_path are placeholders awaiting configuration.
# =============================================================================

SITES: list[Site] = [
    Site(
        name="Valte",
        image_path="Cartografia_valencia.png",
        geotransform=[737575.05, 0.15, 0, 4391595.45, 0, -0.15],
        utm_crs="EPSG:25830",
    ),
    Site(
        name="Setram",
        image_path="",
        geotransform=[],
        utm_crs="",
    ),
    Site(
        name="AutoTerminal",
        image_path="",
        geotransform=[],
        utm_crs="",
    ),
    Site(
        name="ICO",
        image_path="",
        geotransform=[],
        utm_crs="",
    ),
]


def get_site_by_name(site_name: str) -> Site | None:
    """Find site configuration by name.

    Args:
        site_name: Name of the site (e.g., "Valte", "Setram")

    Returns:
        Site instance or None if not found
    """
    return next((site for site in SITES if site.name == site_name), None)


def get_all_site_names() -> list[str]:
    """Get list of all site names.

    Returns:
        List of site name strings
    """
    return [site.name for site in SITES]
