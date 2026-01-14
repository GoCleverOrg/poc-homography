"""FastAPI application for GeoTIFF point picker."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import tifffile
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from poc_homography.geotiff_utils import apply_geotransform
from poc_homography.map_points.map_point import MapPoint
from poc_homography.map_points.map_point_registry import MapPointRegistry

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Tag abbreviation mapping
TAG_ABBREVIATIONS = {
    "parking_spot": "PS",
    "arrows": "AR",
    "crosswalk": "CW",
    "extra": "EX",
}

ABBREVIATION_TO_TAG = {v: k for k, v in TAG_ABBREVIATIONS.items()}

# Valid tag values for validation
VALID_TAGS = Literal["parking_spot", "arrows", "crosswalk", "extra"]


# Pydantic request models for input validation
class AddPointRequest(BaseModel):
    """Request model for adding a new point."""

    tag: VALID_TAGS = Field(default="extra", description="Tag category for the point")
    pixel_x: float = Field(..., description="X pixel coordinate")
    pixel_y: float = Field(..., description="Y pixel coordinate")
    id: str | None = Field(default=None, description="Optional custom point ID")


class UpdatePointRequest(BaseModel):
    """Request model for updating point coordinates."""

    pixel_x: float = Field(..., description="New X pixel coordinate")
    pixel_y: float = Field(..., description="New Y pixel coordinate")


class ExportRequest(BaseModel):
    """Request model for exporting points."""

    path: str = Field(default="", description="Export file path")


class ImportRequest(BaseModel):
    """Request model for importing points."""

    path: str = Field(..., description="Import file path")


# GeoTIFF tag IDs
TIFFTAG_GEOKEYDIRECTORY = 34735
TIFFTAG_GEODOUBLEPARAMS = 34736
TIFFTAG_GEOASCIIPARAMS = 34737
TIFFTAG_MODELPIXELSCALE = 33550
TIFFTAG_MODELTIEPOINT = 33922
TIFFTAG_MODELTRANSFORMATION = 34264


def _extract_geotransform(tif: tifffile.TiffFile) -> tuple[list[float] | None, str | None]:
    """Extract GeoTIFF geotransform and CRS info from TIFF tags.

    Args:
        tif: Open tifffile TiffFile object.

    Returns:
        Tuple of (geotransform, crs_string).
        geotransform is 6-element list [origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height]
        or None if not available.
    """
    page = tif.pages[0]
    # type: ignore needed because tifffile types are incomplete
    tags = {tag.name: tag for tag in page.tags.values()}  # type: ignore[union-attr]

    geotransform = None
    crs = None

    # Try to extract GeoTIFF metadata
    if "ModelPixelScaleTag" in tags and "ModelTiepointTag" in tags:
        # Common GeoTIFF format: pixel scale + tiepoint
        try:
            scale = tags["ModelPixelScaleTag"].value
            tiepoint = tags["ModelTiepointTag"].value

            # GDAL-style geotransform: [originX, pixelWidth, rotationX, originY, rotationY, pixelHeight]
            # tiepoint = [I, J, K, X, Y, Z] where (I,J,K) is pixel coord and (X,Y,Z) is map coord
            origin_x = tiepoint[3] - tiepoint[0] * scale[0]
            origin_y = tiepoint[4] + tiepoint[1] * scale[1]

            geotransform = [
                float(origin_x),  # GT[0]: origin X
                float(scale[0]),  # GT[1]: pixel width
                0.0,  # GT[2]: rotation (typically 0)
                float(origin_y),  # GT[3]: origin Y
                0.0,  # GT[4]: rotation (typically 0)
                -float(scale[1]),  # GT[5]: pixel height (negative for north-up)
            ]
        except (IndexError, TypeError, ValueError):
            pass

    elif "ModelTransformationTag" in tags:
        # Alternative: 4x4 transformation matrix
        try:
            matrix = tags["ModelTransformationTag"].value
            geotransform = [
                float(matrix[3]),  # origin X
                float(matrix[0]),  # pixel width
                float(matrix[1]),  # rotation
                float(matrix[7]),  # origin Y
                float(matrix[4]),  # rotation
                float(matrix[5]),  # pixel height
            ]
        except (IndexError, TypeError, ValueError):
            pass

    # Try to get CRS info from GeoKeyDirectoryTag
    if "GeoKeyDirectoryTag" in tags:
        try:
            geo_keys = tags["GeoKeyDirectoryTag"].value
            # Look for ProjectedCSTypeGeoKey (3072) or GeographicTypeGeoKey (2048)
            for i in range(4, len(geo_keys), 4):
                key_id = geo_keys[i]
                if key_id == 3072:  # ProjectedCSTypeGeoKey
                    epsg = geo_keys[i + 3]
                    crs = f"EPSG:{epsg}"
                    break
                elif key_id == 2048:  # GeographicTypeGeoKey
                    epsg = geo_keys[i + 3]
                    crs = f"EPSG:{epsg}"
        except (IndexError, TypeError, ValueError):
            pass

    return geotransform, crs


class PointPickerState:
    """Mutable state for the point picker application."""

    def __init__(
        self,
        image_path: Path,
        geotransform: list[float] | None = None,
        crs: str | None = None,
    ) -> None:
        """Initialize state with image file.

        Args:
            image_path: Path to the image file (PNG, TIFF, etc.).
            geotransform: Optional 6-parameter geotransform [origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height].
            crs: Optional CRS string (e.g., "EPSG:25830").
        """
        self.geotiff_path = image_path  # Keep name for compatibility
        self.map_id = image_path.stem
        self.registry = MapPointRegistry(map_id=self.map_id, points={})

        # Detect file type and load accordingly
        suffix = image_path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            # Load TIFF metadata using tifffile
            with tifffile.TiffFile(image_path) as tif:
                page = tif.pages[0]
                # type: ignore needed because tifffile types are incomplete
                self.width: int = page.imagewidth  # type: ignore[union-attr]
                self.height: int = page.imagelength  # type: ignore[union-attr]

                # Extract geotransform and CRS from TIFF if not provided
                if geotransform is None or crs is None:
                    tiff_gt, tiff_crs = _extract_geotransform(tif)
                    if geotransform is None:
                        geotransform = tiff_gt
                    if crs is None:
                        crs = tiff_crs
        else:
            # Load other image formats (PNG, JPG, etc.) using PIL
            with Image.open(image_path) as img:
                self.width = img.width
                self.height = img.height

        self.geotransform = geotransform
        self.crs = crs

    def get_next_id(self, tag: str) -> str:
        """Get the next auto-incremented ID for a tag.

        Args:
            tag: Tag name (parking_spot, arrows, crosswalk, extra).

        Returns:
            Next ID string (e.g., "PS5" if PS1-PS4 exist).
        """
        abbrev = TAG_ABBREVIATIONS.get(tag)
        if not abbrev:
            raise ValueError(f"Unknown tag: {tag}")

        # Find max number for this prefix
        max_num = 0
        for point_id in self.registry.points:
            if point_id.startswith(abbrev):
                try:
                    num = int(point_id[len(abbrev) :])
                    max_num = max(max_num, num)
                except ValueError:
                    pass
        return f"{abbrev}{max_num + 1}"

    def add_point(
        self, tag: str, pixel_x: float, pixel_y: float, point_id: str | None = None
    ) -> str:
        """Add a new point with auto-generated or custom ID.

        Args:
            tag: Tag category.
            pixel_x: X pixel coordinate.
            pixel_y: Y pixel coordinate.
            point_id: Optional custom ID. If None, auto-generates based on tag.

        Returns:
            Point ID (generated or provided).
        """
        if point_id is None:
            point_id = self.get_next_id(tag)
        point = MapPoint(pixel_x=pixel_x, pixel_y=pixel_y)

        # Create new registry with added point (immutable pattern)
        new_points = dict(self.registry.points)
        new_points[point_id] = point
        self.registry = MapPointRegistry(map_id=self.map_id, points=new_points)

        return point_id

    def update_point(self, point_id: str, pixel_x: float, pixel_y: float) -> None:
        """Update point coordinates (for drag operations).

        Args:
            point_id: Point ID to update.
            pixel_x: New X pixel coordinate.
            pixel_y: New Y pixel coordinate.
        """
        if point_id not in self.registry.points:
            raise KeyError(f"Point not found: {point_id}")

        new_points = dict(self.registry.points)
        new_points[point_id] = MapPoint(pixel_x=pixel_x, pixel_y=pixel_y)
        self.registry = MapPointRegistry(map_id=self.map_id, points=new_points)

    def delete_point(self, point_id: str) -> None:
        """Delete a point.

        Args:
            point_id: Point ID to delete.
        """
        if point_id not in self.registry.points:
            raise KeyError(f"Point not found: {point_id}")

        new_points = dict(self.registry.points)
        del new_points[point_id]
        self.registry = MapPointRegistry(map_id=self.map_id, points=new_points)

    def load_registry(self, path: Path) -> None:
        """Load points from YAML/JSON file.

        Args:
            path: Path to the file.
        """
        self.registry = MapPointRegistry.load(path)
        self.map_id = self.registry.map_id

    def save_registry(self, path: Path) -> None:
        """Save points to YAML/JSON file.

        Args:
            path: Path to the file.
        """
        self.registry.save(path)

    def get_geo_coords(self, pixel_x: float, pixel_y: float) -> tuple[float, float] | None:
        """Convert pixel coordinates to geographic coordinates.

        Args:
            pixel_x: X pixel coordinate.
            pixel_y: Y pixel coordinate.

        Returns:
            Tuple of (easting, northing) or None if no geotransform.
        """
        if self.geotransform:
            return apply_geotransform(pixel_x, pixel_y, self.geotransform)
        return None


# Module-level state, initialized by create_app
_state: PointPickerState | None = None


def get_state() -> PointPickerState:
    """Get the current application state."""
    if _state is None:
        raise RuntimeError("Application not initialized")
    return _state


def create_app(
    image_path: Path,
    geotransform: list[float] | None = None,
    crs: str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        image_path: Path to the image file (PNG, TIFF, etc.).
        geotransform: Optional 6-parameter geotransform.
        crs: Optional CRS string (e.g., "EPSG:25830").

    Returns:
        Configured FastAPI application.
    """
    global _state
    _state = PointPickerState(image_path, geotransform=geotransform, crs=crs)

    app = FastAPI(
        title="GeoTIFF Point Picker",
        description="Web tool for picking and managing points on GeoTIFF images",
    )

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Register routes
    _register_routes(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        """Serve the main HTML page."""
        html_path = Path(__file__).parent / "static" / "index.html"
        if not html_path.exists():
            raise HTTPException(status_code=500, detail="index.html not found")
        return html_path.read_text()

    @app.get("/api/image/info")
    async def image_info() -> dict:
        """Get image metadata."""
        state = get_state()
        return {
            "width": state.width,
            "height": state.height,
            "geotransform": state.geotransform,
            "crs": state.crs,
            "filename": state.geotiff_path.name,
        }

    @app.get("/api/image/tile")
    async def get_tile(
        x: int = Query(..., description="Tile X coordinate"),
        y: int = Query(..., description="Tile Y coordinate"),
        z: int = Query(..., description="Zoom level"),
        size: int = Query(256, description="Tile size"),
    ) -> Response:
        """Get an image tile at specified coordinates and zoom level.

        OpenSeadragon uses a pyramid where:
        - Level 0 = most zoomed out (fewest tiles)
        - Max level = full resolution (most tiles)

        At level z, the image appears at resolution: original / 2^(max_level - z)
        """
        state = get_state()

        # Calculate max level for the pyramid
        max_level = math.ceil(math.log2(max(state.width, state.height)))

        # At level z, each pixel in the tile grid corresponds to 2^(max_level-z) original pixels
        level_scale = 2 ** (max_level - z)

        # Calculate bounds in original image coordinates
        # Each tile covers (size * level_scale) original pixels
        x0 = x * size * level_scale
        y0 = y * size * level_scale
        x1 = (x + 1) * size * level_scale
        y1 = (y + 1) * size * level_scale

        # Clamp to image bounds
        x0 = max(0, min(x0, state.width))
        y0 = max(0, min(y0, state.height))
        x1 = max(0, min(x1, state.width))
        y1 = max(0, min(y1, state.height))

        if x1 <= x0 or y1 <= y0:
            # Return transparent tile
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return Response(content=buffer.getvalue(), media_type="image/png")

        # Read the tile based on file type
        suffix = state.geotiff_path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            # Read from TIFF using tifffile
            with tifffile.TiffFile(state.geotiff_path) as tif:
                page = tif.pages[0]
                data = page.asarray()

                # Extract the tile region
                if data.ndim == 2:
                    tile_data = data[y0:y1, x0:x1]
                    img = Image.fromarray(normalize_array(tile_data), mode="L")
                    img = img.convert("RGB")
                elif data.ndim == 3:
                    if data.shape[2] >= 3:
                        tile_data = data[y0:y1, x0:x1, :3]
                        img = Image.fromarray(normalize_array(tile_data), mode="RGB")
                    else:
                        tile_data = data[y0:y1, x0:x1, 0]
                        img = Image.fromarray(normalize_array(tile_data), mode="L")
                        img = img.convert("RGB")
                else:
                    if data.shape[0] in (1, 3, 4):
                        if data.shape[0] == 1:
                            tile_data = data[0, y0:y1, x0:x1]
                            img = Image.fromarray(normalize_array(tile_data), mode="L")
                            img = img.convert("RGB")
                        else:
                            tile_data = np.transpose(data[:3, y0:y1, x0:x1], (1, 2, 0))
                            img = Image.fromarray(normalize_array(tile_data), mode="RGB")
                    else:
                        tile_data = data[y0:y1, x0:x1] if data.ndim == 2 else data[y0:y1, x0:x1, 0]
                        img = Image.fromarray(normalize_array(tile_data), mode="L")
                        img = img.convert("RGB")
        else:
            # Read from other formats (PNG, JPG, etc.) using PIL
            with Image.open(state.geotiff_path) as full_img:
                # Crop to tile region
                img = full_img.crop((x0, y0, x1, y1))
                if img.mode != "RGB":
                    img = img.convert("RGB")

        # Resize to tile size
        img = img.resize((size, size), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

    @app.get("/api/image/full")
    async def get_full_image(
        max_size: int = Query(2048, description="Maximum dimension"),
    ) -> Response:
        """Get the full image scaled to max_size."""
        state = get_state()

        suffix = state.geotiff_path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            with tifffile.TiffFile(state.geotiff_path) as tif:
                page = tif.pages[0]
                data = page.asarray()

                # Convert to image
                if data.ndim == 2:
                    img = Image.fromarray(normalize_array(data), mode="L")
                    img = img.convert("RGB")
                elif data.ndim == 3:
                    if data.shape[2] >= 3:
                        img = Image.fromarray(normalize_array(data[:, :, :3]), mode="RGB")
                    else:
                        img = Image.fromarray(normalize_array(data[:, :, 0]), mode="L")
                        img = img.convert("RGB")
                else:
                    if data.shape[0] in (1, 3, 4):
                        if data.shape[0] == 1:
                            img = Image.fromarray(normalize_array(data[0]), mode="L")
                            img = img.convert("RGB")
                        else:
                            img_array = np.transpose(data[:3], (1, 2, 0))
                            img = Image.fromarray(normalize_array(img_array), mode="RGB")
                    else:
                        img = Image.fromarray(normalize_array(data), mode="L")
                        img = img.convert("RGB")
        else:
            # Load other formats with PIL using context manager to prevent resource leak
            with Image.open(state.geotiff_path) as pil_img:
                img = pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img.copy()

        # Scale to max_size while preserving aspect ratio
        ratio = min(max_size / img.width, max_size / img.height)
        if ratio < 1:
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

    @app.get("/api/points")
    async def get_points() -> dict:
        """Get all points."""
        state = get_state()
        return {
            "map_id": state.registry.map_id,
            "points": [
                {
                    "id": pid,
                    "pixel_x": p.pixel_x,
                    "pixel_y": p.pixel_y,
                    "tag": _get_tag_from_id(pid),
                }
                for pid, p in state.registry.points.items()
            ],
        }

    @app.post("/api/points")
    async def add_point(data: AddPointRequest) -> dict:
        """Add a new point."""
        state = get_state()
        point_id = state.add_point(data.tag, data.pixel_x, data.pixel_y, point_id=data.id)
        point = state.registry.points[point_id]

        return {
            "id": point_id,
            "pixel_x": point.pixel_x,
            "pixel_y": point.pixel_y,
            "tag": _get_tag_from_id(point_id),
        }

    @app.put("/api/points/{point_id}")
    async def update_point(point_id: str, data: UpdatePointRequest) -> dict:
        """Update a point's coordinates."""
        state = get_state()
        try:
            state.update_point(point_id, data.pixel_x, data.pixel_y)
            point = state.registry.points[point_id]
            return {
                "id": point_id,
                "pixel_x": point.pixel_x,
                "pixel_y": point.pixel_y,
                "tag": _get_tag_from_id(point_id),
            }
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Point not found: {point_id}")

    @app.delete("/api/points/{point_id}")
    async def delete_point(point_id: str) -> dict:
        """Delete a point."""
        state = get_state()
        try:
            state.delete_point(point_id)
            return {"deleted": point_id}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Point not found: {point_id}")

    @app.get("/api/points/next-id")
    async def get_next_id(tag: str = Query(..., description="Tag category")) -> dict:
        """Get the next ID for a tag category."""
        state = get_state()
        try:
            next_id = state.get_next_id(tag)
            return {"tag": tag, "next_id": next_id}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/geo-coords")
    async def get_geo_coords(
        pixel_x: float = Query(...),
        pixel_y: float = Query(...),
    ) -> dict:
        """Convert pixel coordinates to geographic coordinates."""
        state = get_state()
        coords = state.get_geo_coords(pixel_x, pixel_y)
        if coords:
            return {
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "easting": coords[0],
                "northing": coords[1],
                "crs": state.crs,
            }
        return {
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "easting": None,
            "northing": None,
            "crs": None,
        }

    @app.post("/api/export")
    async def export_points(data: ExportRequest) -> dict:
        """Export points to YAML file."""
        state = get_state()
        path = Path(data.path) if data.path else Path(f"{state.map_id}_points.yaml")
        state.save_registry(path)
        return {"exported": str(path), "count": len(state.registry.points)}

    @app.post("/api/import")
    async def import_points(data: ImportRequest) -> dict:
        """Import points from YAML file."""
        state = get_state()
        path = Path(data.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        state.load_registry(path)
        return {
            "imported": str(path),
            "count": len(state.registry.points),
            "map_id": state.registry.map_id,
        }


def normalize_array(arr: NDArray) -> NDArray[np.uint8]:
    """Normalize array values to 0-255 range for display.

    Args:
        arr: Input array.

    Returns:
        Normalized uint8 array.
    """
    if arr.dtype == np.uint8:
        return arr

    # Handle floating point and other types
    arr = arr.astype(np.float64)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)

    if max_val - min_val > 0:
        arr = (arr - min_val) / (max_val - min_val) * 255
    else:
        arr = np.zeros_like(arr)

    return arr.astype(np.uint8)


def _get_tag_from_id(point_id: str) -> str:
    """Extract tag from point ID prefix.

    Args:
        point_id: Point ID (e.g., "PS1", "AR2").

    Returns:
        Tag name (e.g., "parking_spot", "arrows").
    """
    for abbrev, tag in ABBREVIATION_TO_TAG.items():
        if point_id.startswith(abbrev):
            return tag
    return "extra"
