"""FastAPI application for GeoTIFF point picker."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

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

    def __init__(self, geotiff_path: Path) -> None:
        """Initialize state with GeoTIFF file.

        Args:
            geotiff_path: Path to the GeoTIFF file.
        """
        self.geotiff_path = geotiff_path
        self.map_id = geotiff_path.stem
        self.registry = MapPointRegistry(map_id=self.map_id, points={})

        # Load GeoTIFF metadata using tifffile
        with tifffile.TiffFile(geotiff_path) as tif:
            page = tif.pages[0]
            # type: ignore needed because tifffile types are incomplete
            self.width: int = page.imagewidth  # type: ignore[union-attr]
            self.height: int = page.imagelength  # type: ignore[union-attr]

            # Extract geotransform and CRS
            self.geotransform, self.crs = _extract_geotransform(tif)

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

    def add_point(self, tag: str, pixel_x: float, pixel_y: float) -> str:
        """Add a new point with auto-generated ID.

        Args:
            tag: Tag category.
            pixel_x: X pixel coordinate.
            pixel_y: Y pixel coordinate.

        Returns:
            Generated point ID.
        """
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


def create_app(geotiff_path: Path) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        geotiff_path: Path to the GeoTIFF file.

    Returns:
        Configured FastAPI application.
    """
    global _state
    _state = PointPickerState(geotiff_path)

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
        if html_path.exists():
            return html_path.read_text()
        return _get_embedded_html()

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
        """Get an image tile at specified coordinates and zoom level."""
        state = get_state()

        # Calculate tile bounds in image coordinates
        # At zoom level z, there are 2^z tiles
        scale = 2**z
        tile_width = state.width / scale
        tile_height = state.height / scale

        x0 = int(x * tile_width)
        y0 = int(y * tile_height)
        x1 = int((x + 1) * tile_width)
        y1 = int((y + 1) * tile_height)

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

        # Read the tile from the GeoTIFF using tifffile
        with tifffile.TiffFile(state.geotiff_path) as tif:
            page = tif.pages[0]
            data = page.asarray()

            # Extract the tile region
            if data.ndim == 2:
                # Single band - grayscale
                tile_data = data[y0:y1, x0:x1]
                img = Image.fromarray(normalize_array(tile_data), mode="L")
                img = img.convert("RGB")
            elif data.ndim == 3:
                if data.shape[2] >= 3:
                    # RGB or RGBA (height, width, channels)
                    tile_data = data[y0:y1, x0:x1, :3]
                    img = Image.fromarray(normalize_array(tile_data), mode="RGB")
                else:
                    # Use first channel
                    tile_data = data[y0:y1, x0:x1, 0]
                    img = Image.fromarray(normalize_array(tile_data), mode="L")
                    img = img.convert("RGB")
            else:
                # Fallback for other shapes
                if data.shape[0] in (1, 3, 4):
                    # Channels first format (channels, height, width)
                    if data.shape[0] == 1:
                        tile_data = data[0, y0:y1, x0:x1]
                        img = Image.fromarray(normalize_array(tile_data), mode="L")
                        img = img.convert("RGB")
                    else:
                        tile_data = np.transpose(data[:3, y0:y1, x0:x1], (1, 2, 0))
                        img = Image.fromarray(normalize_array(tile_data), mode="RGB")
                else:
                    # Treat as grayscale
                    tile_data = data[y0:y1, x0:x1] if data.ndim == 2 else data[y0:y1, x0:x1, 0]
                    img = Image.fromarray(normalize_array(tile_data), mode="L")
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
    async def add_point(data: dict) -> dict:
        """Add a new point."""
        state = get_state()
        tag = data.get("tag", "extra")
        pixel_x = float(data["pixel_x"])
        pixel_y = float(data["pixel_y"])

        point_id = state.add_point(tag, pixel_x, pixel_y)
        point = state.registry.points[point_id]

        return {
            "id": point_id,
            "pixel_x": point.pixel_x,
            "pixel_y": point.pixel_y,
            "tag": tag,
        }

    @app.put("/api/points/{point_id}")
    async def update_point(point_id: str, data: dict) -> dict:
        """Update a point's coordinates."""
        state = get_state()
        try:
            pixel_x = float(data["pixel_x"])
            pixel_y = float(data["pixel_y"])
            state.update_point(point_id, pixel_x, pixel_y)
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
    async def export_points(data: dict) -> dict:
        """Export points to YAML file."""
        state = get_state()
        path = Path(data.get("path", f"{state.map_id}_points.yaml"))
        state.save_registry(path)
        return {"exported": str(path), "count": len(state.registry.points)}

    @app.post("/api/import")
    async def import_points(data: dict) -> dict:
        """Import points from YAML file."""
        state = get_state()
        path = Path(data["path"])
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


def _get_embedded_html() -> str:
    """Return embedded HTML for the point picker UI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GeoTIFF Point Picker</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: system-ui, -apple-system, sans-serif; display: flex; height: 100vh; }
        #sidebar {
            width: 280px; background: #f5f5f5; padding: 16px;
            display: flex; flex-direction: column; gap: 16px;
            border-right: 1px solid #ddd;
        }
        #viewer { flex: 1; background: #222; }
        h1 { font-size: 18px; color: #333; }
        .section { background: white; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .section-title { font-size: 14px; font-weight: 600; margin-bottom: 8px; color: #666; }
        .tag-buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .tag-btn {
            padding: 10px; border: 2px solid #ddd; border-radius: 6px;
            background: white; cursor: pointer; font-size: 13px;
            transition: all 0.2s;
        }
        .tag-btn:hover { border-color: #999; }
        .tag-btn.active { border-color: #0066cc; background: #e6f0ff; }
        .tag-btn.parking_spot.active { border-color: #e63946; background: #fde2e4; }
        .tag-btn.arrows.active { border-color: #2a9d8f; background: #d8f3dc; }
        .tag-btn.crosswalk.active { border-color: #f4a261; background: #fff3e0; }
        .tag-btn.extra.active { border-color: #9c27b0; background: #f3e5f5; }
        #next-id { font-size: 16px; font-weight: bold; color: #333; text-align: center; padding: 8px; }
        .btn-group { display: flex; gap: 8px; }
        .btn {
            flex: 1; padding: 10px; border: none; border-radius: 6px;
            cursor: pointer; font-size: 13px; font-weight: 500;
        }
        .btn-primary { background: #0066cc; color: white; }
        .btn-primary:hover { background: #0052a3; }
        .btn-secondary { background: #e0e0e0; color: #333; }
        .btn-secondary:hover { background: #d0d0d0; }
        #coords {
            font-family: monospace; font-size: 12px; color: #666;
            background: #fff; padding: 8px; border-radius: 4px;
        }
        #point-list { max-height: 200px; overflow-y: auto; font-size: 12px; }
        .point-item {
            display: flex; justify-content: space-between; align-items: center;
            padding: 6px 8px; border-bottom: 1px solid #eee; cursor: pointer;
        }
        .point-item:hover { background: #f0f0f0; }
        .point-item.selected { background: #e6f0ff; }
        .point-item .delete-btn {
            background: #ff4444; color: white; border: none;
            padding: 2px 6px; border-radius: 4px; cursor: pointer; font-size: 11px;
        }
        .marker {
            width: 20px; height: 20px; margin-left: -10px; margin-top: -10px;
            border-radius: 50%; border: 3px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            cursor: move; position: absolute;
            display: flex; align-items: center; justify-content: center;
            font-size: 9px; font-weight: bold; color: white;
        }
        .marker.parking_spot { background: #e63946; }
        .marker.arrows { background: #2a9d8f; }
        .marker.crosswalk { background: #f4a261; }
        .marker.extra { background: #9c27b0; }
        .marker.selected { border-color: #ffeb3b; box-shadow: 0 0 8px #ffeb3b; }
        #filename { font-size: 12px; color: #888; word-break: break-all; }
        .modal {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.5); align-items: center; justify-content: center;
            z-index: 1000;
        }
        .modal.active { display: flex; }
        .modal-content {
            background: white; padding: 24px; border-radius: 12px;
            min-width: 300px; max-width: 400px;
        }
        .modal-title { font-size: 18px; margin-bottom: 16px; }
        .modal input {
            width: 100%; padding: 10px; border: 1px solid #ddd;
            border-radius: 6px; margin-bottom: 16px; font-size: 14px;
        }
        .modal-buttons { display: flex; gap: 8px; justify-content: flex-end; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>Point Picker</h1>
        <div id="filename"></div>
        <div class="section">
            <div class="section-title">Tag Category</div>
            <div class="tag-buttons">
                <button class="tag-btn parking_spot" data-tag="parking_spot">Parking Spot</button>
                <button class="tag-btn arrows" data-tag="arrows">Arrows</button>
                <button class="tag-btn crosswalk" data-tag="crosswalk">Crosswalk</button>
                <button class="tag-btn extra" data-tag="extra">Extra</button>
            </div>
            <div id="next-id">Next: PS1</div>
        </div>
        <div class="section">
            <div class="section-title">Coordinates</div>
            <div id="coords">Hover over image...</div>
        </div>
        <div class="section">
            <div class="section-title">Points</div>
            <div id="point-list"></div>
        </div>
        <div class="section">
            <div class="btn-group">
                <button class="btn btn-primary" id="export-btn">Export YAML</button>
                <button class="btn btn-secondary" id="import-btn">Import</button>
            </div>
        </div>
    </div>
    <div id="viewer"></div>

    <div id="export-modal" class="modal">
        <div class="modal-content">
            <div class="modal-title">Export Points</div>
            <input type="text" id="export-path" placeholder="filename.yaml">
            <div class="modal-buttons">
                <button class="btn btn-secondary" onclick="closeModal('export-modal')">Cancel</button>
                <button class="btn btn-primary" onclick="doExport()">Export</button>
            </div>
        </div>
    </div>
    <div id="import-modal" class="modal">
        <div class="modal-content">
            <div class="modal-title">Import Points</div>
            <input type="text" id="import-path" placeholder="path/to/file.yaml">
            <div class="modal-buttons">
                <button class="btn btn-secondary" onclick="closeModal('import-modal')">Cancel</button>
                <button class="btn btn-primary" onclick="doImport()">Import</button>
            </div>
        </div>
    </div>

    <script>
        let viewer;
        let imageInfo;
        let currentTag = 'parking_spot';
        let selectedPointId = null;
        let markers = {};
        let isDragging = false;
        let dragPointId = null;

        async function init() {
            // Get image info
            const resp = await fetch('/api/image/info');
            imageInfo = await resp.json();
            document.getElementById('filename').textContent = imageInfo.filename;

            // Initialize OpenSeadragon
            viewer = OpenSeadragon({
                id: 'viewer',
                prefixUrl: 'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/',
                tileSources: {
                    height: imageInfo.height,
                    width: imageInfo.width,
                    tileSize: 256,
                    getTileUrl: (level, x, y) => {
                        const z = level;
                        return `/api/image/tile?x=${x}&y=${y}&z=${z}`;
                    }
                },
                showNavigator: true,
                navigatorPosition: 'BOTTOM_RIGHT',
                minZoomLevel: 0.5,
                maxZoomLevel: 20,
                visibilityRatio: 0.5,
                constrainDuringPan: true,
            });

            // Add click handler for placing points
            viewer.addHandler('canvas-click', async (e) => {
                if (isDragging) return;
                if (e.quick) {
                    const viewportPoint = viewer.viewport.pointFromPixel(e.position);
                    const imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint);

                    if (imagePoint.x >= 0 && imagePoint.x < imageInfo.width &&
                        imagePoint.y >= 0 && imagePoint.y < imageInfo.height) {
                        await addPoint(imagePoint.x, imagePoint.y);
                    }
                }
            });

            // Add mouse move handler for coordinates
            viewer.addHandler('canvas-drag', (e) => {
                if (dragPointId) {
                    const viewportPoint = viewer.viewport.pointFromPixel(e.position);
                    const imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint);
                    updateMarkerPosition(dragPointId, imagePoint.x, imagePoint.y);
                }
            });

            viewer.addHandler('canvas-drag-end', async (e) => {
                if (dragPointId) {
                    const viewportPoint = viewer.viewport.pointFromPixel(e.position);
                    const imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint);
                    await updatePointCoords(dragPointId, imagePoint.x, imagePoint.y);
                    dragPointId = null;
                    isDragging = false;
                }
            });

            new OpenSeadragon.MouseTracker({
                element: viewer.canvas,
                moveHandler: async (e) => {
                    const viewportPoint = viewer.viewport.pointFromPixel(e.position);
                    const imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint);
                    await updateCoords(imagePoint.x, imagePoint.y);
                }
            });

            // Setup tag buttons
            document.querySelectorAll('.tag-btn').forEach(btn => {
                btn.addEventListener('click', () => selectTag(btn.dataset.tag));
            });

            // Setup export/import
            document.getElementById('export-btn').addEventListener('click', () => {
                document.getElementById('export-path').value = imageInfo.filename.replace(/\\.[^.]+$/, '_points.yaml');
                openModal('export-modal');
            });
            document.getElementById('import-btn').addEventListener('click', () => openModal('import-modal'));

            // Initial setup
            selectTag('parking_spot');
            await loadPoints();
        }

        function selectTag(tag) {
            currentTag = tag;
            document.querySelectorAll('.tag-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.tag === tag);
            });
            updateNextId();
        }

        async function updateNextId() {
            const resp = await fetch(`/api/points/next-id?tag=${currentTag}`);
            const data = await resp.json();
            document.getElementById('next-id').textContent = `Next: ${data.next_id}`;
        }

        async function updateCoords(x, y) {
            if (x < 0 || x >= imageInfo.width || y < 0 || y >= imageInfo.height) {
                document.getElementById('coords').textContent = 'Outside image';
                return;
            }

            let text = `Pixel: ${x.toFixed(1)}, ${y.toFixed(1)}`;

            if (imageInfo.geotransform) {
                const resp = await fetch(`/api/geo-coords?pixel_x=${x}&pixel_y=${y}`);
                const data = await resp.json();
                if (data.easting !== null) {
                    text += `\\nGeo: ${data.easting.toFixed(2)}, ${data.northing.toFixed(2)}`;
                    if (data.crs) text += `\\nCRS: ${data.crs}`;
                }
            }
            document.getElementById('coords').textContent = text;
        }

        async function addPoint(x, y) {
            const resp = await fetch('/api/points', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tag: currentTag, pixel_x: x, pixel_y: y })
            });
            const point = await resp.json();
            addMarker(point);
            updatePointList();
            updateNextId();
        }

        async function loadPoints() {
            const resp = await fetch('/api/points');
            const data = await resp.json();
            // Clear existing markers
            Object.keys(markers).forEach(id => removeMarker(id));
            // Add markers for all points
            data.points.forEach(point => addMarker(point));
            updatePointList();
            updateNextId();
        }

        function addMarker(point) {
            const element = document.createElement('div');
            element.className = `marker ${point.tag}`;
            element.dataset.pointId = point.id;
            element.textContent = point.id.replace(/[A-Z]+/, '');

            // Make marker draggable
            element.addEventListener('mousedown', (e) => {
                e.preventDefault();
                isDragging = true;
                dragPointId = point.id;
                selectPoint(point.id);
            });

            const overlay = viewer.addOverlay({
                element: element,
                location: viewer.viewport.imageToViewportCoordinates(point.pixel_x, point.pixel_y),
                placement: OpenSeadragon.Placement.CENTER
            });

            markers[point.id] = { element, point };
        }

        function removeMarker(pointId) {
            if (markers[pointId]) {
                viewer.removeOverlay(markers[pointId].element);
                delete markers[pointId];
            }
        }

        function updateMarkerPosition(pointId, x, y) {
            if (markers[pointId]) {
                const location = viewer.viewport.imageToViewportCoordinates(x, y);
                viewer.updateOverlay(markers[pointId].element, location);
            }
        }

        async function updatePointCoords(pointId, x, y) {
            await fetch(`/api/points/${pointId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pixel_x: x, pixel_y: y })
            });
            if (markers[pointId]) {
                markers[pointId].point.pixel_x = x;
                markers[pointId].point.pixel_y = y;
            }
            updatePointList();
        }

        function selectPoint(pointId) {
            selectedPointId = pointId;
            Object.values(markers).forEach(m => {
                m.element.classList.toggle('selected', m.point.id === pointId);
            });
            updatePointList();
        }

        function updatePointList() {
            const list = document.getElementById('point-list');
            list.innerHTML = '';
            Object.values(markers).forEach(m => {
                const item = document.createElement('div');
                item.className = `point-item ${m.point.id === selectedPointId ? 'selected' : ''}`;
                item.innerHTML = `
                    <span>${m.point.id}: (${m.point.pixel_x.toFixed(1)}, ${m.point.pixel_y.toFixed(1)})</span>
                    <button class="delete-btn" onclick="deletePoint('${m.point.id}')">X</button>
                `;
                item.addEventListener('click', (e) => {
                    if (!e.target.classList.contains('delete-btn')) {
                        selectPoint(m.point.id);
                        // Pan to point
                        const location = viewer.viewport.imageToViewportCoordinates(m.point.pixel_x, m.point.pixel_y);
                        viewer.viewport.panTo(location);
                    }
                });
                list.appendChild(item);
            });
        }

        async function deletePoint(pointId) {
            await fetch(`/api/points/${pointId}`, { method: 'DELETE' });
            removeMarker(pointId);
            if (selectedPointId === pointId) selectedPointId = null;
            updatePointList();
            updateNextId();
        }

        function openModal(id) {
            document.getElementById(id).classList.add('active');
        }

        function closeModal(id) {
            document.getElementById(id).classList.remove('active');
        }

        async function doExport() {
            const path = document.getElementById('export-path').value;
            if (!path) return;
            const resp = await fetch('/api/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path })
            });
            const result = await resp.json();
            alert(`Exported ${result.count} points to ${result.exported}`);
            closeModal('export-modal');
        }

        async function doImport() {
            const path = document.getElementById('import-path').value;
            if (!path) return;
            try {
                const resp = await fetch('/api/import', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path })
                });
                if (!resp.ok) {
                    const err = await resp.json();
                    alert('Import failed: ' + err.detail);
                    return;
                }
                const result = await resp.json();
                alert(`Imported ${result.count} points from ${result.imported}`);
                closeModal('import-modal');
                await loadPoints();
            } catch (e) {
                alert('Import failed: ' + e.message);
            }
        }

        init();
    </script>
</body>
</html>"""
