"""Acceptance and unit tests for the clean-plate reconstruction pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from poc_homography.cleanplate import (
    CellAccumulator,
    CleanPlateFrame,
    GroundRaster,
    ortho_rectify_frame,
    reconstruct_clean_plate,
    write_clean_plate,
)
from poc_homography.cleanplate.synthetic import ALWAYS_COVERED_CELL
from poc_homography.types import Meters, Unitless

# ---------------------------------------------------------------------------
# GroundRaster unit tests
# ---------------------------------------------------------------------------


def test_ground_raster_shape(small_raster: GroundRaster) -> None:
    """Shape derives from extent times resolution."""
    assert small_raster.shape == (96, 96)
    assert small_raster.height == 96
    assert small_raster.width == 96


def test_world_cell_round_trip(small_raster: GroundRaster) -> None:
    """world_to_cell and cell_to_world are mutual inverses."""
    x = np.array([0.5, 3.0, 5.5])
    y = np.array([0.5, 3.0, 5.5])
    col, row = small_raster.world_to_cell(x, y)
    x2, y2 = small_raster.cell_to_world(col, row)
    assert np.allclose(x, x2)
    assert np.allclose(y, y2)


def test_world_to_cell_y_flip(small_raster: GroundRaster) -> None:
    """Row increases as world-Y decreases (top-left origin)."""
    _, row_top = small_raster.world_to_cell(0.0, small_raster.y_max)
    _, row_bottom = small_raster.world_to_cell(0.0, small_raster.y_min)
    assert float(row_top) == pytest.approx(0.0)
    assert float(row_bottom) == pytest.approx(small_raster.height)


def test_world_to_raster_matrix_matches_world_to_cell(small_raster: GroundRaster) -> None:
    """The affine matrix agrees with world_to_cell."""
    matrix = small_raster.world_to_raster_matrix()
    pt = np.array([2.5, 4.0, 1.0])
    out = matrix @ pt
    col, row = small_raster.world_to_cell(2.5, 4.0)
    assert out[0] == pytest.approx(float(col))
    assert out[1] == pytest.approx(float(row))


def test_invalid_extent_raises() -> None:
    """Degenerate extents are rejected."""
    with pytest.raises(ValueError):
        GroundRaster(
            x_min=Meters(1.0),
            x_max=Meters(0.0),
            y_min=Meters(0.0),
            y_max=Meters(1.0),
            pixels_per_meter=Unitless(10.0),
        )


def test_inverted_y_extent_raises() -> None:
    """An inverted Y extent is rejected."""
    with pytest.raises(ValueError):
        GroundRaster(
            x_min=Meters(0.0),
            x_max=Meters(1.0),
            y_min=Meters(1.0),
            y_max=Meters(0.0),
            pixels_per_meter=Unitless(10.0),
        )


def test_non_positive_ppm_raises() -> None:
    """A non-positive resolution is rejected."""
    with pytest.raises(ValueError):
        GroundRaster(
            x_min=Meters(0.0),
            x_max=Meters(1.0),
            y_min=Meters(0.0),
            y_max=Meters(1.0),
            pixels_per_meter=Unitless(0.0),
        )


def test_non_integral_product_raises() -> None:
    """A non-integral extent*ppm product desyncs cell mapping and is rejected."""
    # 1.5m * 16ppm = 24 (integral, fine) but 1.0m * 1.5ppm = 1.5 (non-integral).
    with pytest.raises(ValueError, match="integral"):
        GroundRaster(
            x_min=Meters(0.0),
            x_max=Meters(1.0),
            y_min=Meters(0.0),
            y_max=Meters(1.0),
            pixels_per_meter=Unitless(1.5),
        )


def test_integral_product_constructs() -> None:
    """A valid integral raster constructs and reports the expected shape."""
    raster = GroundRaster(
        x_min=Meters(0.0),
        x_max=Meters(1.5),
        y_min=Meters(0.0),
        y_max=Meters(2.0),
        pixels_per_meter=Unitless(16.0),
    )
    assert raster.shape == (32, 24)


# ---------------------------------------------------------------------------
# CellAccumulator unit tests
# ---------------------------------------------------------------------------


def test_accumulator_median() -> None:
    """Median reduction over per-cell samples; coverage counts valid frames."""
    acc = CellAccumulator((2, 2), max_samples_per_cell=8)
    valid = np.ones((2, 2), dtype=bool)
    for value in (10, 20, 90):
        acc.add_frame(np.full((2, 2, 3), value, dtype=np.float32), valid)
    ortho, coverage = acc.reduce(method="median")
    assert np.all(coverage == 3)
    assert np.all(ortho == 20)  # median(10,20,90)


def test_accumulator_mode() -> None:
    """Mode reduction returns the dominant quantized color."""
    acc = CellAccumulator((1, 1), max_samples_per_cell=8)
    valid = np.ones((1, 1), dtype=bool)
    for value in (100, 100, 100, 200):
        acc.add_frame(np.full((1, 1, 3), value, dtype=np.float32), valid)
    ortho, _ = acc.reduce(method="mode")
    assert np.all(ortho == 100)


def test_accumulator_respects_validity() -> None:
    """Cells never marked valid stay at zero coverage."""
    acc = CellAccumulator((2, 2), max_samples_per_cell=4)
    valid = np.array([[True, False], [False, False]])
    acc.add_frame(np.full((2, 2, 3), 50, dtype=np.float32), valid)
    _, coverage = acc.reduce()
    assert coverage[0, 0] == 1
    assert coverage[0, 1] == 0


def test_accumulator_reservoir_cap() -> None:
    """Coverage keeps counting beyond the reservoir cap."""
    acc = CellAccumulator((1, 1), max_samples_per_cell=2)
    valid = np.ones((1, 1), dtype=bool)
    for value in (10, 20, 30, 40):
        acc.add_frame(np.full((1, 1, 3), value, dtype=np.float32), valid)
    _, coverage = acc.reduce()
    assert coverage[0, 0] == 4  # all four seen
    # Only first two samples stored -> median(10, 20) == 15.
    ortho, _ = acc.reduce(method="median")
    assert np.all(ortho == 15)


# ---------------------------------------------------------------------------
# Ortho unit tests
# ---------------------------------------------------------------------------


def test_ortho_identity_like() -> None:
    """A world->image homography equal to the raster affine warps to itself."""
    raster = GroundRaster(
        x_min=Meters(0.0),
        x_max=Meters(4.0),
        y_min=Meters(0.0),
        y_max=Meters(4.0),
        pixels_per_meter=Unitless(8.0),
    )
    # If H == world_to_raster_matrix, then image_to_raster == identity.
    homography = raster.world_to_raster_matrix()
    image = np.random.default_rng(0).integers(0, 255, size=(*raster.shape, 3), dtype=np.uint8)
    result = ortho_rectify_frame(image, homography, raster)
    assert result is not None
    assert np.array_equal(result.color, image)
    assert result.valid.all()


def test_ortho_all_floor_mask_equals_no_mask_footprint() -> None:
    """An all-floor mask yields identical validity to the no-mask footprint path.

    Equivalence pin for the ortho refactor: when the floor mask is all-True the
    warped mask already encodes the in-image footprint, so passing it must give
    the same ``valid`` raster as passing no mask at all.
    """
    raster = GroundRaster(
        x_min=Meters(0.0),
        x_max=Meters(3.0),
        y_min=Meters(0.0),
        y_max=Meters(3.0),
        pixels_per_meter=Unitless(8.0),
    )
    # A non-identity world->image homography so the footprint is a real subset.
    homography = raster.world_to_raster_matrix() @ np.array(
        [[1.0, 0.2, 1.5], [0.1, 1.0, 1.0], [0.0, 0.0, 1.0]]
    )
    image = np.random.default_rng(1).integers(0, 255, size=(40, 50, 3), dtype=np.uint8)
    all_floor = np.ones(image.shape[:2], dtype=bool)

    with_mask = ortho_rectify_frame(image, homography, raster, floor_mask=all_floor)
    without_mask = ortho_rectify_frame(image, homography, raster)
    assert with_mask is not None
    assert without_mask is not None
    assert np.array_equal(with_mask.valid, without_mask.valid)


def test_ortho_singular_returns_none() -> None:
    """A singular homography is skipped gracefully."""
    raster = GroundRaster(
        x_min=Meters(0.0),
        x_max=Meters(2.0),
        y_min=Meters(0.0),
        y_max=Meters(2.0),
        pixels_per_meter=Unitless(8.0),
    )
    singular = np.zeros((3, 3), dtype=np.float64)
    image = np.zeros((*raster.shape, 3), dtype=np.uint8)
    assert ortho_rectify_frame(image, singular, raster) is None


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------


def test_recovers_clean_background_within_tolerance(
    synthetic_visits: tuple[np.ndarray, list[CleanPlateFrame]],
    small_raster: GroundRaster,
) -> None:
    """Reconstruction recovers the clean background over the covered region."""
    background, frames = synthetic_visits
    result = reconstruct_clean_plate(frames, small_raster, method="median", photometric=False)

    # No silent no-op: some cells must be covered and compared.
    assert result.coverage.max() > 0

    # Compare only on a well-covered, non-inpainted interior region to avoid
    # warp-edge resampling artifacts.
    recoverable = (result.coverage >= 2) & (~result.inpainted_mask)
    recoverable[:6, :] = False
    recoverable[-6:, :] = False
    recoverable[:, :6] = False
    recoverable[:, -6:] = False
    assert recoverable.sum() > 100  # compared region is non-empty

    diff = np.abs(result.orthophoto.astype(np.float32) - background.astype(np.float32))
    mae = diff[recoverable].mean()
    assert mae < 20.0, f"mean abs error too high: {mae}"


def test_inpaints_never_empty_cells(
    synthetic_visits: tuple[np.ndarray, list[CleanPlateFrame]],
    small_raster: GroundRaster,
) -> None:
    """The always-covered cell has zero coverage, is inpainted, and finite."""
    _, frames = synthetic_visits
    result = reconstruct_clean_plate(frames, small_raster, method="median", photometric=False)

    row, col = ALWAYS_COVERED_CELL
    assert result.coverage[row, col] == 0
    assert result.inpainted_mask[row, col]
    assert np.all(np.isfinite(result.orthophoto[row, col]))
    # Inpainting fills from neighbors -> not left as a black hole.
    assert result.orthophoto[row, col].sum() > 0


def test_write_clean_plate_png_and_tiff(
    synthetic_visits: tuple[np.ndarray, list[CleanPlateFrame]],
    small_raster: GroundRaster,
    tmp_path,
) -> None:
    """Write helper emits PNG orthophoto and TIFF coverage."""
    _, frames = synthetic_visits
    result = reconstruct_clean_plate(frames, small_raster, photometric=False)
    ortho_path = tmp_path / "ortho.png"
    cov_path = tmp_path / "coverage.tif"
    write_clean_plate(result, ortho_path, cov_path)
    assert ortho_path.exists()
    assert cov_path.exists()


def test_photometric_consumes_gain(small_raster: GroundRaster) -> None:
    """Per-frame gain is consumed: gain-leveled fusion differs from gain-ignored.

    Two frames see the same floor but at different capture gains (one bright,
    one dark). With gain wired in, both are leveled toward a common reference
    before fusion, so the reconstruction differs from the path that ignores
    gain entirely (``photometric=False``).
    """
    raster = small_raster
    homography = raster.world_to_raster_matrix()
    base = np.random.default_rng(0).integers(40, 200, size=(*raster.shape, 3), dtype=np.uint8)
    # Frame B is the same scene captured at 2x the gain (twice as bright).
    bright = np.clip(base.astype(np.float32) * 2.0, 0, 255).astype(np.uint8)
    full_mask = np.ones(raster.shape, dtype=np.uint8) * 255

    frames = [
        CleanPlateFrame(
            image=base, floor_mask=full_mask, ground_homography=homography, gain=Unitless(1.0)
        ),
        CleanPlateFrame(
            image=bright, floor_mask=full_mask, ground_homography=homography, gain=Unitless(2.0)
        ),
    ]

    with_gain = reconstruct_clean_plate(frames, raster, method="median", photometric=True)
    without = reconstruct_clean_plate(frames, raster, method="median", photometric=False)

    covered = with_gain.coverage >= 2
    assert covered.any()
    # Gain is actually consumed: the leveled result differs from the raw path.
    assert not np.array_equal(with_gain.orthophoto[covered], without.orthophoto[covered])


@pytest.mark.slow
def test_end_to_end_more_visits(small_raster: GroundRaster) -> None:
    """Heavier end-to-end run with more visits still recovers the background."""
    from poc_homography.cleanplate import make_synthetic_visits

    background, frames = make_synthetic_visits(n_visits=8, raster=small_raster, seed=3)
    result = reconstruct_clean_plate(frames, small_raster, method="median", photometric=True)
    recoverable = (result.coverage >= 2) & (~result.inpainted_mask)
    recoverable[:8, :] = False
    recoverable[-8:, :] = False
    recoverable[:, :8] = False
    recoverable[:, -8:] = False
    assert recoverable.sum() > 100
    diff = np.abs(result.orthophoto.astype(np.float32) - background.astype(np.float32))
    assert diff[recoverable].mean() < 30.0
