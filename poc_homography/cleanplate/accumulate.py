"""
Per-cell accumulation of floor-labeled color samples across frames.

:class:`CellAccumulator` collects, for every raster cell, up to
``max_samples_per_cell`` color samples drawn ONLY from frames where the cell was
labeled empty floor. It is memory-bounded via a per-cell reservoir cap. After
all frames are added, :meth:`reduce` collapses the samples to a single empty
orthophoto using a robust per-cell statistic (median or quantized mode) and
reports per-cell coverage counts.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MAX_SAMPLES_PER_CELL = 32
DEFAULT_MODE_QUANT = 16


class CellAccumulator:
    """
    Memory-bounded per-cell reservoir of floor color samples.

    The accumulator stores a fixed-size sample buffer per raster cell. Each
    :meth:`add_frame` call appends one color sample to every covered cell, up to
    the reservoir cap (older samples are kept; excess are dropped). A separate
    ``coverage`` counter tracks how many valid samples each cell has seen
    overall (even beyond the cap), so callers can identify never-empty cells.

    Attributes:
        height: Raster height in cells.
        width: Raster width in cells.
        max_samples_per_cell: Reservoir cap per cell.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        max_samples_per_cell: int = DEFAULT_MAX_SAMPLES_PER_CELL,
    ) -> None:
        """
        Initialize an empty accumulator.

        Args:
            shape: Raster shape ``(height, width)`` in cells.
            max_samples_per_cell: Maximum stored samples per cell (> 0).
        """
        if max_samples_per_cell <= 0:
            raise ValueError("max_samples_per_cell must be positive")
        self.height, self.width = shape
        self.max_samples_per_cell = max_samples_per_cell
        # Sample buffer: (max_samples, H, W, 3) float32; fill count per cell.
        self._samples = np.zeros(
            (max_samples_per_cell, self.height, self.width, 3), dtype=np.float32
        )
        self._fill = np.zeros((self.height, self.width), dtype=np.int32)
        self._coverage = np.zeros((self.height, self.width), dtype=np.int32)

    def add_frame(self, color_raster: np.ndarray, valid_raster: np.ndarray) -> None:
        """
        Add one ortho-rectified frame's floor samples to the reservoir.

        Args:
            color_raster: ``(H, W, 3)`` color values in raster space.
            valid_raster: ``(H, W)`` boolean mask; True marks floor samples to
                accumulate for this frame.
        """
        valid = np.asarray(valid_raster, dtype=bool)
        if valid.shape != (self.height, self.width):
            raise ValueError(
                f"valid_raster shape {valid.shape} != raster shape {(self.height, self.width)}"
            )
        color = np.asarray(color_raster, dtype=np.float32)

        self._coverage += valid.astype(np.int32)

        # Only write cells that are valid AND still have reservoir room.
        has_room = self._fill < self.max_samples_per_cell
        writable = valid & has_room
        if not writable.any():
            return

        rows, cols = np.nonzero(writable)
        slots = self._fill[rows, cols]
        self._samples[slots, rows, cols, :] = color[rows, cols, :]
        self._fill[rows, cols] += 1

    @property
    def coverage(self) -> np.ndarray:
        """Per-cell count of floor samples seen, ``(H, W)`` int32."""
        return self._coverage.copy()

    def reduce(self, method: str = "median") -> tuple[np.ndarray, np.ndarray]:
        """
        Collapse per-cell samples into a single empty-floor orthophoto.

        Args:
            method: ``"median"`` for the per-channel median of stored samples,
                or ``"mode"`` for the most frequent coarse-quantized color.

        Returns:
            Tuple ``(orthophoto, coverage)`` where ``orthophoto`` is
            ``(H, W, 3)`` uint8 (zeros where no samples) and ``coverage`` is
            ``(H, W)`` int32.

        Raises:
            ValueError: If ``method`` is not ``"median"`` or ``"mode"``.
        """
        if method == "median":
            ortho = self._reduce_median()
        elif method == "mode":
            ortho = self._reduce_mode()
        else:
            raise ValueError(f"Unknown reduction method: {method!r}")
        return ortho, self._coverage.copy()

    def _reduce_median(self) -> np.ndarray:
        """Per-cell, per-channel median over the stored reservoir samples."""
        ortho = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        max_fill = int(self._fill.max(initial=0))
        for n in range(1, max_fill + 1):
            cells = self._fill == n
            if not cells.any():
                continue
            rows, cols = np.nonzero(cells)
            # (n, k, 3) samples for the k cells with exactly n samples.
            stack = self._samples[:n, rows, cols, :]
            med = np.median(stack, axis=0)
            ortho[rows, cols, :] = np.clip(np.rint(med), 0, 255).astype(np.uint8)
        return ortho

    def _reduce_mode(self, quant: int = DEFAULT_MODE_QUANT) -> np.ndarray:
        """Per-cell mode of coarse-quantized colors over stored samples."""
        ortho = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        rows_all, cols_all = np.nonzero(self._fill > 0)
        for r, c in zip(rows_all.tolist(), cols_all.tolist()):
            n = int(self._fill[r, c])
            samples = self._samples[:n, r, c, :]
            quantized = (samples // quant).astype(np.int64)
            keys = (
                quantized[:, 0] * (256 // quant) ** 2
                + quantized[:, 1] * (256 // quant)
                + quantized[:, 2]
            )
            uniq, counts = np.unique(keys, return_counts=True)
            best_key = uniq[int(np.argmax(counts))]
            # Average the actual samples that fall in the winning bin.
            members = samples[keys == best_key]
            ortho[r, c, :] = np.clip(np.rint(members.mean(axis=0)), 0, 255).astype(np.uint8)
        return ortho
