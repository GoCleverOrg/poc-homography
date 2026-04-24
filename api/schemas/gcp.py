"""Pydantic response schemas for GCP endpoints."""

from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Tenant schemas
# ---------------------------------------------------------------------------


class TenantLocation(BaseModel):
    """Optional geographic location of a tenant site."""

    lat: str
    lon: str


class TenantOut(BaseModel):
    """Serialised representation of a :class:`Tenant` entity."""

    id: str
    name: str
    description: str | None = None
    location: TenantLocation | None = None


class TenantListResponse(BaseModel):
    """Envelope for ``GET /api/tenants/``."""

    tenants: list[TenantOut]


# ---------------------------------------------------------------------------
# Map schemas
# ---------------------------------------------------------------------------


class GeoTransformOut(BaseModel):
    """Affine transformation parameters (mirrors ``GeoTransform.to_dict()``)."""

    origin_easting: float
    pixel_width: float
    row_rotation: float
    origin_northing: float
    col_rotation: float
    pixel_height: float


class GeoTiffOut(BaseModel):
    """GeoTIFF metadata (mirrors ``GeoTiff.to_dict()``)."""

    geotransform: GeoTransformOut
    crs: str


class PhotoOut(BaseModel):
    """Image file with dimensions (mirrors ``Photo.to_dict()``)."""

    path: str
    width: int
    height: int


class MapOut(BaseModel):
    """Serialised representation of a :class:`Map` entity."""

    id: str
    tenant_id: str
    photo: PhotoOut
    geotiff: GeoTiffOut


class MapListResponse(BaseModel):
    """Envelope for ``GET /api/tenants/{tenant_id}/maps/``."""

    maps: list[MapOut]


# ---------------------------------------------------------------------------
# Map IDs schema
# ---------------------------------------------------------------------------


class MapIdsResponse(BaseModel):
    """Envelope for ``GET /api/map-ids/``."""

    map_ids: list[str]
