"""
Camera configuration file.
Central location for all camera settings and credentials.

Tenant data is loaded from DDD YAML repository (data/tenants/).
Camera data is loaded from DDD YAML repository (data/cameras/).
Calibration data is loaded from DDD YAML repository (data/calibrations/).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poc_homography.domain.entities.camera_calibration import CameraCalibration
    from poc_homography.domain.entities.camera_config import CameraConfig

# Legacy global credentials - kept for backwards compatibility
# Prefer tenant-specific credentials defined in per-tenant env vars
USERNAME = os.getenv("CAMERA_USERNAME")
PASSWORD = os.getenv("CAMERA_PASSWORD")


def _project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


# =============================================================================
# CAMERA LENS SPECIFICATIONS (Hikvision DS-2DF8425IX-AELW series)
# =============================================================================
# Source: Hikvision official datasheet
# - Sensor: 1/1.8" progressive scan CMOS (diagonal ~8.86mm)
# - Resolution: 2560x1440 (4MP, 16:9 aspect ratio)
# - Focal Length: 5.9mm (wide) to 147.5mm (tele)
# - Optical Zoom: 25x (zoom_factor = focal_length / 5.9)
# - Horizontal FOV: 59.8 deg (wide) to 3.3 deg (tele)
# - Aperture: F1.5 (max)
#
# Note: The geometric sensor width (6.78mm) differs from the physical sensor
# because the stated FOV accounts for some lens distortion effects.
# =============================================================================

# Default camera sensor parameters for Hikvision DS-2DF8425IX series
DEFAULT_SENSOR_WIDTH_MM = 6.78  # Calculated from 59.8 deg FOV at 5.9mm focal length
DEFAULT_BASE_FOCAL_LENGTH_MM = 5.9  # Minimum focal length at 1x zoom


# =============================================================================
# LEGACY ID MAPPING
# =============================================================================
# Maps old hardcoded camera IDs to DDD entity IDs ({tenant_id}/{name}).
# This provides backward compatibility for callers using the old naming
# convention. New code should use the DDD IDs directly.
_LEGACY_ID_MAP: dict[str, str] = {
    "valte_cam01": "valte/Valte",
    "setram_cam01": "setram/Cam01",
    "icozee-camptz-01": "icozee/Cam01",
    "icozee-camptz-02": "icozee/Cam02",
    "icozee-camptz-03": "icozee/Cam03",
    "icozee-camptz-04": "icozee/Cam04",
    "icozee-camptz-05": "icozee/Cam05",
    "icozee-camptz-06": "icozee/Cam06",
    "icozee-camptz-07": "icozee/Cam07",
    "icozee-camptz-08": "icozee/Cam08",
    "icozee-camptz-09": "icozee/Cam09",
    "icozee-camptz-10": "icozee/Cam10",
    "icozee-camptz-11": "icozee/Cam11",
    "icozee-camptz-12": "icozee/Cam12",
    "icozee-camptz-13": "icozee/Cam13",
    "icozee-camptz-14": "icozee/Cam14",
    "icozee-camptz-15": "icozee/Cam15",
    "icozee-camptz-16": "icozee/Cam16",
}


# =============================================================================
# REPOSITORY SINGLETONS
# =============================================================================


@lru_cache(maxsize=1)
def _get_tenant_repo():
    """Lazily create the tenant repository (singleton)."""
    from poc_homography.infrastructure.repositories import RepoYamlTenant

    return RepoYamlTenant(_project_root() / "data" / "tenants")


@lru_cache(maxsize=1)
def _get_camera_config_repo():
    """Lazily create the camera config repository (singleton)."""
    from poc_homography.infrastructure.repositories import RepoYamlCameraConfig

    return RepoYamlCameraConfig(_project_root() / "data" / "cameras")


@lru_cache(maxsize=1)
def _get_camera_calibration_repo():
    """Lazily create the camera calibration repository (singleton)."""
    from poc_homography.infrastructure.repositories import RepoYamlCameraCalibration

    return RepoYamlCameraCalibration(_project_root() / "data" / "calibrations")


def _resolve_camera_id(camera_id: str) -> str:
    """Resolve a camera ID, mapping legacy IDs to DDD entity IDs.

    Args:
        camera_id: Either a legacy ID (e.g., "valte_cam01") or a
            DDD entity ID (e.g., "valte/Valte").

    Returns:
        The DDD entity ID.
    """
    return _LEGACY_ID_MAP.get(camera_id, camera_id)


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


def get_cameras_for_tenant(tenant_id: str) -> list[CameraConfig]:
    """
    Get all cameras belonging to a tenant.

    Args:
        tenant_id: ID of the tenant

    Returns:
        List of CameraConfig entities for the tenant
    """
    return list(_get_camera_config_repo().get_by_tenant(tenant_id).values())  # type: ignore[return-value]


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


def get_camera_configs() -> list[CameraConfig]:
    """
    Get list of all camera configurations.

    Returns:
        List of CameraConfig entities loaded from the DDD YAML repo.
    """
    return _get_camera_config_repo().get_all()


def get_camera_by_id(camera_id: str) -> CameraConfig | None:
    """
    Find camera configuration by ID.

    Supports both legacy IDs (e.g., "valte_cam01") and DDD entity IDs
    (e.g., "valte/Valte").

    Args:
        camera_id: Full camera ID

    Returns:
        CameraConfig entity or None if not found
    """
    resolved_id = _resolve_camera_id(camera_id)
    return _get_camera_config_repo().get(resolved_id)


def get_camera_by_name(camera_name: str) -> CameraConfig | None:
    """
    Find camera configuration by name.

    For backwards compatibility, this searches by:
    1. Full camera ID (including legacy IDs)
    2. Legacy tenant name (e.g., "Valte" -> finds first camera for that tenant)

    Args:
        camera_name: Name/ID of the camera

    Returns:
        CameraConfig entity or None if not found
    """
    # First try exact ID match (including legacy ID resolution)
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


def get_calibration_by_camera_id(camera_id: str) -> CameraCalibration | None:
    """
    Get calibration data for a camera.

    Args:
        camera_id: Camera ID (legacy or DDD format)

    Returns:
        CameraCalibration entity or None if not found
    """
    resolved_id = _resolve_camera_id(camera_id)
    return _get_camera_calibration_repo().get(resolved_id)


def build_legacy_camera_dict(
    camera: CameraConfig,
    calibration: CameraCalibration | None,
) -> dict:
    """Build a legacy camera dict from DDD entities for backward compatibility.

    Used by CLI code that still expects the old dict format with fields
    like 'height_m', 'pan_offset_deg', 'k1', 'k2', etc.
    """
    from typing import Any

    result: dict[str, Any] = {
        "id": camera.id,
        "name": camera.name,
        "ip": camera.ip_address,
        "tenant_id": camera.tenant_id,
        "model": camera.spec.model_name,
        "sensor_width_mm": float(camera.spec.sensor_width),
        "base_focal_length_mm": float(camera.spec.base_focal_length),
    }
    if calibration:
        result["height_m"] = float(calibration.height)
        result["pan_offset_deg"] = float(calibration.base_orientation.yaw)
        result["tilt_offset_deg"] = float(calibration.base_orientation.pitch)
        result["k1"] = float(calibration.distortion.k1)
        result["k2"] = float(calibration.distortion.k2)
        result["p1"] = float(calibration.distortion.p1)
        result["p2"] = float(calibration.distortion.p2)
    else:
        result["height_m"] = 5.0
        result["pan_offset_deg"] = 0.0
        result["tilt_offset_deg"] = 0.0
        result["k1"] = 0.0
        result["k2"] = 0.0
        result["p1"] = 0.0
        result["p2"] = 0.0
    return result


def get_rtsp_url(camera_name: str, stream_type: str = "main") -> str | None:
    """
    Get RTSP URL for a camera.

    Uses tenant-specific credentials if available, otherwise falls back to
    global CAMERA_USERNAME/CAMERA_PASSWORD environment variables.

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

    # Get tenant-specific credentials (falls back to global)
    tenant_id = cam.tenant_id or ""
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

    ip = cam.ip_address
    if not ip:
        return None

    channel = "101" if stream_type == "main" else "102"
    return f"rtsp://{username}:{password}@{ip}:554/Streaming/Channels/{channel}"


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
            print(f"    - {cam.name} ({cam.id}): {cam.ip_address}")
