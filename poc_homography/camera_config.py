"""
Camera configuration file.
Central location for all camera settings and credentials.

Tenant data is loaded from DDD YAML repository (data/tenants/).
Camera data remains hardcoded here for now (contains calibration
parameters not yet in the DDD model).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from poc_homography.infrastructure.repositories import RepoPostgresTenant

# Legacy global credentials - kept for backwards compatibility
# Prefer tenant-specific credentials defined in per-tenant env vars
USERNAME = os.getenv("CAMERA_USERNAME")
PASSWORD = os.getenv("CAMERA_PASSWORD")


def _project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


@lru_cache(maxsize=1)
def _get_tenant_repo():
    """Lazily create the tenant repository (singleton)."""
    from poc_homography.infrastructure.repositories import RepoYamlTenant

    return RepoYamlTenant(_project_root() / "data" / "tenants")


def _get_tenant_repo_pg(session: Session) -> RepoPostgresTenant:
    """Create a PostgreSQL-backed tenant repository for the given session.

    Unlike the YAML variant this is *not* cached because the repository is
    bound to a specific SQLAlchemy session whose lifetime is managed by the
    caller.
    """
    from poc_homography.infrastructure.repositories import RepoPostgresTenant as _Cls

    return _Cls(session)


# =============================================================================
# CAMERA LENS SPECIFICATIONS (Hikvision DS-2DF8425IX-AELW series)
# =============================================================================
# Source: Hikvision official datasheet
# - Sensor: 1/1.8" progressive scan CMOS (diagonal ~8.86mm)
# - Resolution: 2560×1440 (4MP, 16:9 aspect ratio)
# - Focal Length: 5.9mm (wide) to 147.5mm (tele)
# - Optical Zoom: 25× (zoom_factor = focal_length / 5.9)
# - Horizontal FOV: 59.8° (wide) to 3.3° (tele)
# - Aperture: F1.5 (max)
#
# Note: The geometric sensor width (6.78mm) differs from the physical sensor
# because the stated FOV accounts for some lens distortion effects.
# =============================================================================

# Default camera sensor parameters for Hikvision DS-2DF8425IX series
DEFAULT_SENSOR_WIDTH_MM = 6.78  # Calculated from 59.8° FOV at 5.9mm focal length
DEFAULT_BASE_FOCAL_LENGTH_MM = 5.9  # Minimum focal length at 1x zoom

# =============================================================================
# CALIBRATION TABLE FORMAT (Optional)
# =============================================================================
# The calibration_table field allows defining zoom-dependent intrinsic parameters
# to replace the linear focal length approximation. Real lenses exhibit non-linear
# zoom-to-focal-length relationships and zoom-dependent distortion coefficients.
#
# Format: Dictionary mapping zoom_factor (float) to intrinsic parameters (dict)
#
# Example:
# "calibration_table": {
#     1.0: {
#         "fx": 1825.3,      # Focal length in pixels (horizontal)
#         "fy": 1823.1,      # Focal length in pixels (vertical)
#         "cx": 1280.0,      # Principal point x-coordinate (pixels)
#         "cy": 720.0,       # Principal point y-coordinate (pixels)
#         "k1": -0.341,      # Radial distortion coefficient 1
#         "k2": 0.788,       # Radial distortion coefficient 2
#         "p1": 0.0,         # Tangential distortion coefficient 1
#         "p2": 0.0,         # Tangential distortion coefficient 2
#         "k3": 0.0          # Radial distortion coefficient 3
#     },
#     5.0: {
#         "fx": 9120.5, "fy": 9115.2, "cx": 1282.1, "cy": 721.3,
#         "k1": -0.298, "k2": 0.654, "p1": 0.001, "p2": 0.0, "k3": 0.0
#     },
#     # ... additional zoom levels
# }
#
# Calibration Procedure:
# 1. Capture checkerboard images at multiple zoom levels (e.g., 1.0, 5.0, 10.0, 15.0, 20.0, 25.0)
# 2. Use OpenCV calibrateCamera() for each zoom level to obtain intrinsic matrix and distortion coefficients
# 3. Populate calibration_table with results
# 4. IntrinsicExtrinsicHomography will linearly interpolate between discrete zoom levels
#
# Interpolation Behavior:
# - Zoom values between calibrated points: linear interpolation of fx, fy, cx, cy, k1-k3, p1-p2
# - Zoom values below minimum: uses lowest calibrated zoom level (no extrapolation)
# - Zoom values above maximum: uses highest calibrated zoom level (no extrapolation)
# - If calibration_table is None: falls back to linear focal length approximation
# =============================================================================

# =============================================================================
# TENANT AND CAMERA CONFIGURATION
# =============================================================================
# A Tenant represents a deployment site (e.g., "Valte", "Setram").
# Each Tenant can have multiple cameras, named CamXX (e.g., Cam01, Cam02).
# Each camera belongs to exactly one tenant via tenant_id.
# =============================================================================

# Tenant definitions are now loaded from DDD YAML repository (data/tenants/).
# Credentials are loaded from environment variables: {TENANT_ID}_CAMERA_USERNAME, {TENANT_ID}_CAMERA_PASSWORD
# Falls back to global CAMERA_USERNAME/CAMERA_PASSWORD if tenant-specific not set

# Camera configurations - each camera belongs to a tenant
CAMERAS = [
    {
        "id": "valte_cam01",
        "tenant_id": "valte",
        "name": "Cam01",
        "ip": "10.207.99.178",
        "model": "DS-2DF8425IX-AELW(T5)",
        "lat": "39°38'25.72\"N",
        "lon": "0°13'48.63\"W",
        "height_m": 4.71,  # Calibrated 2025-12-11 with comprehensive_calibration.py
        # Pan offset: angle from North when camera reports pan=0
        # True bearing = reported_pan + pan_offset_deg
        # Calibration: Point camera at known landmark, calculate true bearing,
        # then pan_offset = true_bearing - reported_pan
        "pan_offset_deg": 51.7,  # Calibrated 2025-12-11 (was 65°, optimized to 51.7°)
        # Tilt offset: correction for reported tilt angle
        # Effective tilt = reported_tilt + tilt_offset_deg
        # Calibrated by minimizing GCP projection error
        "tilt_offset_deg": -0.25,  # Calibrated 2025-12-11 (camera reports ~0.25° higher than actual)
        # Lens distortion coefficients (OpenCV model)
        # Calibrated using checkerboard or GCP-based optimization
        "k1": -0.341052,  # Radial distortion (negative = barrel distortion)
        "k2": 0.787571,  # Secondary radial distortion
        "p1": 0.0,  # Tangential distortion (not calibrated)
        "p2": 0.0,  # Tangential distortion (not calibrated)
        # Sensor/lens parameters (use defaults if not specified)
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        # Zoom-dependent intrinsic calibration table (optional)
        # If None, uses linear focal length approximation
        # See CALIBRATION TABLE FORMAT documentation above for details
        "calibration_table": None,
        # GeoTIFF reference parameters for georeferencing
        # Updated to use GDAL 6-parameter GeoTransform format (Issue #133)
        # GeoTransform: [origin_easting, pixel_width, row_rotation, origin_northing, col_rotation, pixel_height]
        # For north-up rasters, row_rotation=0 and col_rotation=0
        "geotiff_params": {
            "geotransform": [737575.05, 0.15, 0, 4391595.45, 0, -0.15],
            "utm_crs": "EPSG:25830",
        },
        "description": "Valte Cam01 - primary camera",
    },
    {
        "id": "setram_cam01",
        "tenant_id": "setram",
        "name": "Cam01",
        "ip": "10.237.100.15",
        "model": "DS-2DF8425IX-AELW(T5)",  # Assumed same model
        "lat": "41°19'46.8\"N",
        "lon": "2°08'31.3\"E",
        "height_m": 5.0,  # Default height, calibrate with GPS validation
        "pan_offset_deg": 0.0,  # Pan=0 points north (default, needs calibration)
        "tilt_offset_deg": 0.0,  # Default, needs calibration
        # Distortion not calibrated yet
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        # Zoom-dependent intrinsic calibration table (optional)
        "calibration_table": None,
        "description": "Setram Cam01 - primary camera",
    },
    {
        "id": "icozee-camptz-01",
        "tenant_id": "icozee",
        "name": "Cam01",
        "ip": "10.107.50.2",
        "model": "DS-2DF-8425IX-AELW(T5)",
        # Sensor/lens parameters (use defaults if not specified)
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        # Zoom-dependent intrinsic calibration table (optional)
        # If None, uses linear focal length approximation
        # See CALIBRATION TABLE FORMAT documentation above for details
        "calibration_table": None,
        "description": "Icozee Cam01",
    },
    {
        "id": "icozee-camptz-02",
        "tenant_id": "icozee",
        "name": "Cam02",
        "ip": "10.107.50.3",
        "model": "DS-2DF-8425IX-AELW(T5)",
        # Sensor/lens parameters (use defaults if not specified)
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        # Zoom-dependent intrinsic calibration table (optional)
        # If None, uses linear focal length approximation
        # See CALIBRATION TABLE FORMAT documentation above for details
        "calibration_table": None,
        "description": "Icozee Cam02",
    },
    {
        "id": "icozee-camptz-03",
        "tenant_id": "icozee",
        "name": "Cam03",
        "ip": "10.107.50.4",
        "model": "DS-2DF-8425IX-AELW(T5)",
        # Sensor/lens parameters (use defaults if not specified)
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        # Zoom-dependent intrinsic calibration table (optional)
        # If None, uses linear focal length approximation
        # See CALIBRATION TABLE FORMAT documentation above for details
        "calibration_table": None,
        "description": "Icozee Cam03",
    },
    {
        "id": "icozee-camptz-04",
        "tenant_id": "icozee",
        "name": "Cam04",
        "ip": "10.107.50.5",
        "model": "DS-2DF-8425IX-AELW(T5)",
        # Sensor/lens parameters (use defaults if not specified)
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        # Zoom-dependent intrinsic calibration table (optional)
        # If None, uses linear focal length approximation
        # See CALIBRATION TABLE FORMAT documentation above for details
        "calibration_table": None,
        "description": "Icozee Cam04",
    },
    {
        "id": "icozee-camptz-05",
        "tenant_id": "icozee",
        "name": "Cam05",
        "ip": "10.107.50.6",
        "model": "DS-2DF-8425IX-AELW(T5)",
        # Sensor/lens parameters (use defaults if not specified)
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        # Zoom-dependent intrinsic calibration table (optional)
        # If None, uses linear focal length approximation
        # See CALIBRATION TABLE FORMAT documentation above for details
        "calibration_table": None,
        "description": "Icozee Cam05",
    },
    {
        "id": "icozee-camptz-06",
        "tenant_id": "icozee",
        "name": "Cam06",
        "ip": "10.107.50.7",
        "model": "DS-2DF-8425IX-AELW(T5)",
        # Sensor/lens parameters (use defaults if not specified)
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        # Zoom-dependent intrinsic calibration table (optional)
        # If None, uses linear focal length approximation
        # See CALIBRATION TABLE FORMAT documentation above for details
        "calibration_table": None,
        "description": "Icozee Cam06",
    },
    {
        "id": "icozee-camptz-07",
        "tenant_id": "icozee",
        "name": "Cam07",
        "ip": "10.107.50.8",
        "model": "DS-2DF-8425IX-AELW(T5)",
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        "calibration_table": None,
        "description": "Icozee Cam07",
    },
    {
        "id": "icozee-camptz-08",
        "tenant_id": "icozee",
        "name": "Cam08",
        "ip": "10.107.50.9",
        "model": "DS-2DF-8425IX-AELW(T5)",
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        "calibration_table": None,
        "description": "Icozee Cam08",
    },
    {
        "id": "icozee-camptz-09",
        "tenant_id": "icozee",
        "name": "Cam09",
        "ip": "10.107.50.10",
        "model": "DS-2DF-8425IX-AELW(T5)",
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        "calibration_table": None,
        "description": "Icozee Cam09",
    },
    {
        "id": "icozee-camptz-10",
        "tenant_id": "icozee",
        "name": "Cam10",
        "ip": "10.107.50.11",
        "model": "DS-2DF-8425IX-AELW(T5)",
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        "calibration_table": None,
        "description": "Icozee Cam10",
    },
    {
        "id": "icozee-camptz-11",
        "tenant_id": "icozee",
        "name": "Cam11",
        "ip": "10.107.50.12",
        "model": "DS-2DF-8425IX-AELW(T5)",
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        "calibration_table": None,
        "description": "Icozee Cam11",
    },
    {
        "id": "icozee-camptz-12",
        "tenant_id": "icozee",
        "name": "Cam12",
        "ip": "10.107.50.13",
        "model": "DS-2DF-8425IX-AELW(T5)",
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        "calibration_table": None,
        "description": "Icozee Cam12",
    },
    {
        "id": "icozee-camptz-13",
        "tenant_id": "icozee",
        "name": "Cam13",
        "ip": "10.107.50.14",
        "model": "DS-2DF-8425IX-AELW(T5)",
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        "calibration_table": None,
        "description": "Icozee Cam13",
    },
    {
        "id": "icozee-camptz-14",
        "tenant_id": "icozee",
        "name": "Cam14",
        "ip": "10.107.50.15",
        "model": "DS-2DF-8425IX-AELW(T5)",
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        "calibration_table": None,
        "description": "Icozee Cam14",
    },
    {
        "id": "icozee-camptz-15",
        "tenant_id": "icozee",
        "name": "Cam15",
        "ip": "10.107.50.16",
        "model": "DS-2DF-8425IX-AELW(T5)",
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        "calibration_table": None,
        "description": "Icozee Cam15",
    },
    {
        "id": "icozee-camptz-16",
        "tenant_id": "icozee",
        "name": "Cam16",
        "ip": "10.107.50.17",
        "model": "DS-2DF-8425IX-AELW(T5)",
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        "calibration_table": None,
        "description": "Icozee Cam16",
    },
]


# =============================================================================
# TENANT FUNCTIONS
# =============================================================================


def _tenant_to_dict(tenant) -> dict:
    """Convert a Tenant entity to the legacy dict format."""
    d = tenant.to_dict()
    # Inject credentials from environment variables
    tid = tenant.id.upper()
    d["username"] = os.getenv(f"{tid}_CAMERA_USERNAME")
    d["password"] = os.getenv(f"{tid}_CAMERA_PASSWORD")
    return d


def get_tenants() -> list:
    """
    Get list of all tenant configurations.

    Returns:
        List of tenant configuration dicts (loaded from DDD YAML repo).
    """
    return [_tenant_to_dict(t) for t in _get_tenant_repo().get_all()]


def get_tenant_by_id(tenant_id: str) -> dict | None:
    """
    Find tenant configuration by ID.

    Args:
        tenant_id: ID of the tenant (e.g., "valte", "setram")

    Returns:
        Tenant configuration dict or None if not found
    """
    tenant = _get_tenant_repo().get(tenant_id)
    return _tenant_to_dict(tenant) if tenant else None


def get_tenant_by_name(tenant_name: str) -> dict | None:
    """
    Find tenant configuration by name.

    Args:
        tenant_name: Name of the tenant (e.g., "Valte", "Setram")

    Returns:
        Tenant configuration dict or None if not found
    """
    for tenant in _get_tenant_repo().get_all():
        if tenant.name == tenant_name:
            return _tenant_to_dict(tenant)
    return None


def get_tenants_pg(session: Session) -> list:
    """Get list of all tenant configurations (PostgreSQL-backed).

    Args:
        session: SQLAlchemy session.

    Returns:
        List of tenant configuration dicts.
    """
    return [_tenant_to_dict(t) for t in _get_tenant_repo_pg(session).get_all()]


def get_tenant_by_id_pg(session: Session, tenant_id: str) -> dict | None:
    """Find tenant configuration by ID (PostgreSQL-backed).

    Args:
        session: SQLAlchemy session.
        tenant_id: ID of the tenant (e.g., "valte", "setram").

    Returns:
        Tenant configuration dict or None if not found.
    """
    tenant = _get_tenant_repo_pg(session).get(tenant_id)
    return _tenant_to_dict(tenant) if tenant else None


def get_tenant_by_name_pg(session: Session, tenant_name: str) -> dict | None:
    """Find tenant configuration by name (PostgreSQL-backed).

    Args:
        session: SQLAlchemy session.
        tenant_name: Name of the tenant (e.g., "Valte", "Setram").

    Returns:
        Tenant configuration dict or None if not found.
    """
    for tenant in _get_tenant_repo_pg(session).get_all():
        if tenant.name == tenant_name:
            return _tenant_to_dict(tenant)
    return None


def get_cameras_for_tenant(tenant_id: str) -> list:
    """
    Get all cameras belonging to a tenant.

    Reads from the database-backed :class:`CameraRegistry` cache, falling back
    to the hardcoded ``CAMERAS`` list when the database is unavailable.

    Args:
        tenant_id: ID of the tenant

    Returns:
        List of camera configuration dicts for the tenant
    """
    from poc_homography.camera_registry import get_registry

    return get_registry().get_cameras_for_tenant(tenant_id)


def get_tenant_credentials(tenant_id: str) -> tuple[str | None, str | None]:
    """
    Get credentials for a tenant.

    Loads tenant-specific credentials from environment variables:
    {TENANT_ID}_CAMERA_USERNAME / {TENANT_ID}_CAMERA_PASSWORD.
    Falls back to global CAMERA_USERNAME/CAMERA_PASSWORD if not set.

    Args:
        tenant_id: ID of the tenant

    Returns:
        Tuple of (username, password), either from tenant env vars or global fallback
    """
    tid = tenant_id.upper()
    username = os.getenv(f"{tid}_CAMERA_USERNAME") or USERNAME
    password = os.getenv(f"{tid}_CAMERA_PASSWORD") or PASSWORD
    return username, password


# =============================================================================
# CAMERA FUNCTIONS
# =============================================================================


def get_camera_configs() -> list:
    """
    Get list of all camera configurations.

    Returns:
        List of camera configuration dicts containing camera parameters,
        GPS coordinates, and calibration data. Does not require credentials
        and does not generate RTSP URLs.

    Reads from the database-backed :class:`CameraRegistry` cache, falling back
    to the hardcoded ``CAMERAS`` list when the database is unavailable.
    """
    from poc_homography.camera_registry import get_registry

    return get_registry().get_all_cameras()


def get_camera_by_id(camera_id: str) -> dict | None:
    """
    Find camera configuration by ID.

    Reads from the database-backed :class:`CameraRegistry` cache, falling back
    to the hardcoded ``CAMERAS`` list when the database is unavailable or the
    camera is absent from the database.

    Args:
        camera_id: Full camera ID (e.g., "valte_cam01", "setram_cam01")

    Returns:
        Camera configuration dict or None if not found
    """
    from poc_homography.camera_registry import get_registry

    return get_registry().get_camera_by_id(camera_id)


def get_camera_by_name(camera_name: str) -> dict | None:
    """
    Find camera configuration by name.

    For backwards compatibility, this searches by:
    1. Full camera ID (e.g., "valte_cam01")
    2. Legacy tenant name (e.g., "Valte" -> finds first camera for that tenant)

    Args:
        camera_name: Name/ID of the camera

    Returns:
        Camera configuration dict or None if not found
    """
    # First try exact ID match
    cam = get_camera_by_id(camera_name)
    if cam:
        return cam

    # Then try legacy tenant name match (backwards compatibility)
    # Find tenant by name and return first camera
    tenant = get_tenant_by_name(camera_name)
    if tenant:
        cameras = get_cameras_for_tenant(tenant["id"])
        if cameras:
            return cameras[0]

    return None


def get_rtsp_url(camera_name: str, stream_type: str = "main") -> str | None:
    """
    Get RTSP URL for a camera.

    Prefers per-camera credentials stored in the database (surfaced as
    ``username`` / ``password`` on the camera dict via :class:`CameraRegistry`).
    When the camera has no per-camera credentials (fallback mode), uses
    tenant-specific credentials, otherwise falls back to global
    CAMERA_USERNAME/CAMERA_PASSWORD environment variables.

    Args:
        camera_name: Name or ID of the camera
        stream_type: "main" (101) or "sub" (102)

    Returns:
        Full RTSP URL or None if camera not found

    Raises:
        ValueError: If camera credentials are not set
    """
    cam = get_camera_by_name(camera_name)
    if not cam:
        return None

    tenant_id = cam.get("tenant_id") or ""

    # Prefer per-camera credentials from the database; fall back to
    # tenant-level credentials (which themselves fall back to the globals).
    username = cam.get("username")
    password = cam.get("password")
    if not username or not password:
        username, password = get_tenant_credentials(tenant_id)

    # Validate that credentials are set
    if not username or not password:
        tenant = get_tenant_by_id(tenant_id)
        tenant_name = tenant["name"] if tenant else tenant_id
        raise ValueError(
            f"Camera credentials not set for tenant '{tenant_name}'. "
            f"Please set {tenant_id.upper()}_CAMERA_USERNAME and {tenant_id.upper()}_CAMERA_PASSWORD "
            "environment variables, or set global CAMERA_USERNAME/CAMERA_PASSWORD as fallback."
        )

    channel = "101" if stream_type == "main" else "102"
    return f"rtsp://{username}:{password}@{cam['ip']}:554/Streaming/Channels/{channel}"


# Validation
if __name__ == "__main__":
    print("Camera Configuration")
    print("=" * 70)
    print(f"\nGlobal Credentials (fallback): {USERNAME} / {'*' * len(PASSWORD or '')}")

    tenants = get_tenants()
    print(f"\nConfigured Tenants: {len(tenants)}")
    for tenant in tenants:
        cameras = get_cameras_for_tenant(tenant["id"])
        username, password = get_tenant_credentials(tenant["id"])
        cred_source = "tenant" if tenant.get("username") else "global"
        print(f"\n{tenant['name']} ({tenant['id']}):")
        location = tenant.get("location", {})
        print(f"  Location: {location.get('lat', '')}, {location.get('lon', '')}")
        print(f"  Credentials: {username} / {'*' * len(password or '')} ({cred_source})")
        print(f"  Cameras: {len(cameras)}")
        for cam in cameras:
            print(f"    - {cam['name']} ({cam['id']}): {cam['ip']}")
