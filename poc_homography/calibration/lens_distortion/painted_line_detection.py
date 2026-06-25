"""Painted-line detector for floor markings (parking / apron lines).

Generic Canny+Hough (see :class:`LineDetector`) keys off intensity gradients and
fires on every high-contrast edge in the frame -- shadows, kerbs, vehicle
panels. Painted floor lines have a much more specific signature: they are bright
*and* either white or yellow, narrow, and elongated. This detector exploits that
signature directly:

1. Build a *painted response* image = max(white-top-hat, yellow-response) where
   the white top-hat (morphological) isolates bright structures thinner than a
   small fraction of the image width, and the yellow response ``min(G, R) - B``
   lights up yellow paint that a plain intensity top-hat would miss.
2. Threshold at a high percentile (keep only the brightest painted pixels).
3. Morphologically close to bridge dashes / scuffs.
4. Thin to a 1px centerline (``cv2.ximgproc.thinning`` when the contrib module
   is present, otherwise a morphological-skeleton fallback).
5. Connected components -> keep elongated, line-like components.
6. Fit each component and sample ``num_edge_samples`` ``edge_pixels`` *along the
   real painted centerline* (not the chord), so :meth:`CameraLine.has_edge_curvature`
   sees the genuine distortion signal.

The detector matches the output contract of :class:`LineDetector`: ``detect``
returns objects exposing ``to_camera_line(line_id, image_path, ptz_position)``
producing a :class:`CameraLine` with populated ``edge_pixels``. It is therefore a
drop-in alternative the orchestrator can select.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from poc_homography.calibration.lens_distortion.models import CameraLine, PTZPosition

logger = logging.getLogger(__name__)

# cv2.ximgproc (the contrib thinning) is optional; degrade gracefully when absent.
try:  # pragma: no cover - availability depends on the installed opencv build
    _XIMGPROC_THINNING = cv2.ximgproc.thinning  # type: ignore[attr-defined]
    HAS_XIMGPROC = True
except AttributeError:  # pragma: no cover
    _XIMGPROC_THINNING = None
    HAS_XIMGPROC = False


@dataclass(frozen=True)
class PaintedLineConfig:
    """Configuration for painted-line detection.

    Attributes:
        tophat_kernel_frac: White top-hat kernel size as a fraction of image
            width (isolates bright structures thinner than this).
        response_percentile: Percentile of the painted response kept as paint.
        close_kernel: Square structuring-element side for the morphological
            close that bridges dashes and small scuffs (pixels).
        min_component_length: Minimum bounding-box major-axis length (pixels) a
            component must span to be considered a line.
        min_elongation: Minimum bounding-box aspect ratio (major / minor) for a
            component to count as line-like (rejects blobs).
        min_pixels: Minimum centerline pixel count for a component to be kept.
        num_edge_samples: Number of ``edge_pixels`` sampled along each centerline.
        max_lines: Cap on the number of returned lines (longest kept first).
    """

    tophat_kernel_frac: float = 0.02
    response_percentile: float = 99.0
    close_kernel: int = 5
    min_component_length: float = 60.0
    min_elongation: float = 4.0
    min_pixels: int = 30
    num_edge_samples: int = 40
    max_lines: int = 40

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 < self.tophat_kernel_frac < 1.0:
            raise ValueError(f"tophat_kernel_frac must be in (0, 1), got {self.tophat_kernel_frac}")
        if not 0.0 < self.response_percentile < 100.0:
            raise ValueError(
                f"response_percentile must be in (0, 100), got {self.response_percentile}"
            )
        if self.num_edge_samples < 3:
            raise ValueError(f"num_edge_samples must be >= 3, got {self.num_edge_samples}")


@dataclass(frozen=True)
class PaintedLine:
    """A painted line detected by :class:`PaintedLineDetector`.

    Exposes the same ``to_camera_line`` seam as
    :class:`~poc_homography.calibration.lens_distortion.line_detection.CandidateLine`
    so the two detectors are interchangeable.

    Attributes:
        start: (x, y) start point in pixels (centerline extremum).
        end: (x, y) end point in pixels (other centerline extremum).
        edge_pixels: Centerline coordinates sampled along the painted stroke,
            shape (N, 2); carries the distortion signal.
        confidence: Detection confidence (0..1); proportional to elongation.
    """

    start: tuple[float, float]
    end: tuple[float, float]
    edge_pixels: np.ndarray
    confidence: float = 1.0

    @property
    def length(self) -> float:
        """Chord length between the endpoints in pixels."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return float(np.hypot(dx, dy))

    def to_camera_line(
        self,
        line_id: str,
        image_path: str,
        ptz_position: PTZPosition,
        line_type: str = "painted",
    ) -> CameraLine:
        """Convert to a :class:`CameraLine` with populated ``edge_pixels``.

        Args:
            line_id: Unique identifier for the line.
            image_path: Path to the source image.
            ptz_position: Camera PTZ position when the image was captured.
            line_type: Category label stored on the line.

        Returns:
            A :class:`CameraLine` carrying the sampled centerline as
            ``edge_pixels`` (so ``has_edge_curvature`` is meaningful).
        """
        from poc_homography.calibration.lens_distortion.models import CameraLine

        edge_pixels_tuple = (
            tuple((float(p[0]), float(p[1])) for p in self.edge_pixels)
            if self.edge_pixels is not None and len(self.edge_pixels) > 0
            else None
        )
        return CameraLine(
            line_id=line_id,
            image_path=image_path,
            start_pixel=self.start,
            end_pixel=self.end,
            ptz_position=ptz_position,
            line_type=line_type,
            confidence=self.confidence,
            edge_pixels=edge_pixels_tuple,
        )


def _thin(binary: np.ndarray) -> np.ndarray:
    """Thin a binary mask to a 1px skeleton.

    Uses ``cv2.ximgproc.thinning`` when the contrib module is installed,
    otherwise a morphological-skeleton fallback (iterated erode/open).

    Args:
        binary: Non-zero foreground mask (uint8).

    Returns:
        A uint8 skeleton mask (values 0 / 255).
    """
    if HAS_XIMGPROC and _XIMGPROC_THINNING is not None:  # pragma: no cover
        return _XIMGPROC_THINNING(binary)

    out = np.zeros(binary.shape, dtype=np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return out
    # Restrict the (iterative, O(thickness)) morphological skeleton to the
    # foreground bounding box -- thinning a full multi-megapixel frame is
    # dominated by background; the bbox is typically a thin band.
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = (binary[y0:y1, x0:x1] > 0).astype(np.uint8) * 255

    skeleton = np.zeros_like(crop)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    working = crop
    while cv2.countNonZero(working) > 0:
        opened = cv2.morphologyEx(working, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(working, opened)
        working = cv2.erode(working, element)
        skeleton = cv2.bitwise_or(skeleton, temp)

    out[y0:y1, x0:x1] = skeleton
    return out


def _painted_response(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Compute the painted-line response (max of white top-hat and yellow).

    Args:
        image: Input image (BGR or single-channel).
        kernel_size: White top-hat structuring-element side (odd, pixels).

    Returns:
        A single-channel float32 response image (higher = more paint-like).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)
        yellow = np.minimum(g, r) - b
    else:
        gray = image
        yellow = np.zeros_like(gray, dtype=np.float32)

    ksize = max(3, kernel_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    white = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    return np.maximum(white, np.clip(yellow, 0.0, None))


def _order_along_principal_axis(pts: np.ndarray) -> np.ndarray:
    """Sort centerline points along their dominant (PCA) direction.

    Args:
        pts: (N, 2) array of (x, y) centerline coordinates.

    Returns:
        The points ordered monotonically along the principal axis.
    """
    centered = pts - pts.mean(axis=0)
    # Principal direction = eigenvector of the largest covariance eigenvalue.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    t = centered @ direction
    return pts[np.argsort(t)]


class PaintedLineDetector:
    """Detects painted floor lines (parking / apron markings) in an image.

    A drop-in alternative to :class:`LineDetector`: ``detect`` returns
    :class:`PaintedLine` objects exposing ``to_camera_line``.
    """

    def __init__(self, config: PaintedLineConfig | None = None) -> None:
        """Initialize the detector.

        Args:
            config: Detection configuration. Uses defaults when ``None``.
        """
        self.config = config or PaintedLineConfig()

    def detect(self, image: np.ndarray) -> list[PaintedLine]:
        """Detect painted lines in an image.

        Args:
            image: Input image as a numpy array (BGR or grayscale).

        Returns:
            Detected painted lines, longest first, capped at ``max_lines``.
        """
        cfg = self.config
        height, width = image.shape[:2]

        kernel_size = max(3, int(width * cfg.tophat_kernel_frac))
        response = _painted_response(image, kernel_size)

        positive = response[response > 0]
        if positive.size == 0:
            logger.debug("Painted response empty; no paint-like structure")
            return []
        # High-percentile gate, but never above half the peak: saturated paint
        # cores would otherwise collapse the threshold onto the peak and drop the
        # antialiased shoulders, fragmenting each stroke into stubs.
        peak = float(response.max())
        percentile = float(np.percentile(positive, cfg.response_percentile))
        threshold = min(percentile, 0.5 * peak)
        mask = (response >= max(threshold, 1.0)).astype(np.uint8) * 255

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (cfg.close_kernel, cfg.close_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        skeleton = _thin(mask)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton)
        # Group all skeleton pixels by label in a single linear pass, instead of
        # an O(num_labels * image_size) per-label ``np.where`` scan.
        fg_ys, fg_xs = np.where(skeleton > 0)
        fg_labels = labels[fg_ys, fg_xs]
        order = np.argsort(fg_labels, kind="stable")
        fg_ys, fg_xs, fg_labels = fg_ys[order], fg_xs[order], fg_labels[order]

        lines: list[PaintedLine] = []
        for label in range(1, num_labels):
            if int(stats[label, cv2.CC_STAT_AREA]) < cfg.min_pixels:
                continue
            lo = int(np.searchsorted(fg_labels, label, side="left"))
            hi = int(np.searchsorted(fg_labels, label, side="right"))
            pts = np.column_stack([fg_xs[lo:hi], fg_ys[lo:hi]]).astype(np.float64)
            if len(pts) < cfg.min_pixels:
                continue
            line = self._fit_component(pts, height, width)
            if line is not None:
                lines.append(line)

        lines.sort(key=lambda line: line.length, reverse=True)
        kept = lines[: cfg.max_lines]
        logger.debug("Detected %d painted line(s)", len(kept))
        return kept

    def _fit_component(self, pts: np.ndarray, height: int, width: int) -> PaintedLine | None:
        """Fit one connected component into a :class:`PaintedLine`.

        Rejects components that are too short or too round (not line-like),
        orders the centerline along its principal axis, and subsamples it to
        ``num_edge_samples`` ``edge_pixels``.

        Args:
            pts: (N, 2) centerline coordinates of the component.
            height: Image height (unused bound check placeholder).
            width: Image width (unused bound check placeholder).

        Returns:
            A :class:`PaintedLine`, or ``None`` if the component is not a line.
        """
        del height, width  # endpoints already lie within bounds by construction
        cfg = self.config

        ordered = _order_along_principal_axis(pts)
        centered = ordered - ordered.mean(axis=0)
        # Spread along major vs minor axis -> elongation.
        _, sv, _ = np.linalg.svd(centered, full_matrices=False)
        major = float(sv[0])
        minor = float(sv[1]) if len(sv) > 1 else 0.0
        elongation = major / minor if minor > 1e-6 else float("inf")

        start = (float(ordered[0][0]), float(ordered[0][1]))
        end = (float(ordered[-1][0]), float(ordered[-1][1]))
        chord = float(np.hypot(end[0] - start[0], end[1] - start[1]))

        if chord < cfg.min_component_length or elongation < cfg.min_elongation:
            return None

        if len(ordered) <= cfg.num_edge_samples:
            edge_pixels = ordered
        else:
            idx = np.linspace(0, len(ordered) - 1, cfg.num_edge_samples).astype(int)
            edge_pixels = ordered[idx]

        confidence = float(min(1.0, elongation / (cfg.min_elongation * 4.0)))
        return PaintedLine(start=start, end=end, edge_pixels=edge_pixels, confidence=confidence)


__all__ = [
    "HAS_XIMGPROC",
    "PaintedLine",
    "PaintedLineConfig",
    "PaintedLineDetector",
]
