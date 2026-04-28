"""Shared tile and full-image rendering for OpenSeadragon endpoints.

Centralises the TIFF ndim branching, ``normalize_array`` handling,
clamp/resize logic that was previously duplicated across point_picker,
line_picker and homography_precision routers.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from api.utils.frame_helpers import normalize_array


def _read_tiff_as_rgb(image_path: Path, region: tuple[int, int, int, int] | None = None) -> Image.Image:
    """Read a TIFF file and return an RGB :class:`PIL.Image`.

    Parameters
    ----------
    image_path:
        Path to the TIFF file.
    region:
        Optional ``(x0, y0, x1, y1)`` crop region in original pixel
        coordinates.  When *None* the full image is returned.
    """
    with tifffile.TiffFile(image_path) as tif:
        page = tif.pages[0]
        data = page.asarray()

        # Apply region crop first (operates on the numpy array).
        if region is not None:
            x0, y0, x1, y1 = region
            if data.ndim <= 3:
                # (H, W) or (H, W, C) -- spatial dims are the first two
                data = data[y0:y1, x0:x1]
            else:
                # Channel-first layout (C, H, W)
                data = data[:, y0:y1, x0:x1]

        if data.ndim == 2:
            img = Image.fromarray(normalize_array(data), mode="L")
            return img.convert("RGB")

        if data.ndim == 3:
            if data.shape[2] >= 3:
                return Image.fromarray(normalize_array(data[:, :, :3]), mode="RGB")
            img = Image.fromarray(normalize_array(data[:, :, 0]), mode="L")
            return img.convert("RGB")

        # ndim > 3 -- channel-first layout
        if data.shape[0] == 1:
            img = Image.fromarray(normalize_array(data[0]), mode="L")
            return img.convert("RGB")
        img_array = np.transpose(data[:3], (1, 2, 0))
        return Image.fromarray(normalize_array(img_array), mode="RGB")


def render_tile(
    image_path: Path,
    width: int,
    height: int,
    x: int,
    y: int,
    z: int,
    size: int,
) -> bytes:
    """Render an OpenSeadragon tile as PNG bytes.

    Parameters
    ----------
    image_path:
        Path to the source image (TIFF or regular image).
    width, height:
        Full image dimensions in pixels.
    x, y, z:
        Tile grid coordinates and zoom level.
    size:
        Tile size in pixels (e.g. 256).

    Returns
    -------
    bytes
        PNG-encoded tile image.
    """
    max_level = math.ceil(math.log2(max(width, height)))
    level_scale = 2 ** (max_level - z)

    # Bounds in original image coordinates
    x0 = x * size * level_scale
    y0 = y * size * level_scale
    x1 = (x + 1) * size * level_scale
    y1 = (y + 1) * size * level_scale

    # Clamp to image bounds
    x0 = max(0, min(x0, width))
    y0 = max(0, min(y0, height))
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))

    if x1 <= x0 or y1 <= y0:
        # Transparent tile
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    suffix = image_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        img = _read_tiff_as_rgb(image_path, region=(x0, y0, x1, y1))
    else:
        with Image.open(image_path) as full_img:
            img = full_img.crop((x0, y0, x1, y1))
            if img.mode != "RGB":
                img = img.convert("RGB")

    img = img.resize((size, size), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_full_image(image_path: Path, max_size: int) -> bytes:
    """Render the full image scaled to *max_size* as PNG bytes.

    Parameters
    ----------
    image_path:
        Path to the source image (TIFF or regular image).
    max_size:
        Maximum dimension (width or height) of the output.

    Returns
    -------
    bytes
        PNG-encoded image.
    """
    suffix = image_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        img = _read_tiff_as_rgb(image_path)
    else:
        with Image.open(image_path) as pil_img:
            img = pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img.copy()

    # Scale preserving aspect ratio
    ratio = min(max_size / img.width, max_size / img.height)
    if ratio < 1:
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
