# Clean-Plate (Empty-Floor Orthophoto) Reconstruction Methods

This page documents the candidate methods for **offline clean-plate
reconstruction** — recovering an empty-floor orthophoto of the surveyed ground
plane from a multi-visit survey dataset — and states the recommended approach.
The input is the C1 survey dataset (see
[survey_dataset_schema.md](./survey_dataset_schema.md)), reprocessed without a
camera (see [survey_offline_reprocessing.md](./survey_offline_reprocessing.md)).

---

## 1\. Problem framing

A survey visits the scene on an **hourly cadence across multiple days**. Every
ground point on the floor is therefore observed many times — under different
poses, lighting, and transient occupancy (people, vehicles, equipment passing
through). Each observed frame carries enough state to be ortho-rectified onto
the common ground plane: the computed intrinsics $K$ and the recovered pose
$(R, t)$ (see [ptz_intrinsics_and_pose.md](./ptz_intrinsics_and_pose.md)) give
the image→floor homography, and a **per-frame floor mask** labels which pixels
are bare floor versus occluded by a mover.

Two observations drive the design:

1. **A pixel only has to be seen empty once.** Because each ground cell is
   sampled many times, recovering the clean plate at that cell reduces to
   selecting from the subset of samples that land on floor-labeled pixels. The
   transient-occupancy problem becomes a per-cell selection over time, not a
   per-frame restoration problem.
2. **Fuse into one common ground-plane raster, not per-pose mosaics.** Every
   pose and every visit is resampled directly into a **single global
   ortho-rectified raster** of the floor (a fixed-resolution grid in floor
   coordinates). Fusing in this shared coordinate frame means there are no
   per-pose image boundaries to stitch, so the classic mosaic **seam** problem
   does not arise — overlapping poses simply contribute more samples to the
   same cells.

The unit of reconstruction is therefore the **ground cell**, and the core
operation is a mask-aware reduction over the floor-labeled samples that fall in
each cell across all poses and all times.

---

## 2\. Method comparison

The four candidate methods below all consume the same input — ortho-rectified,
mask-labeled samples per ground cell — and differ in how they reduce those
samples to a clean value. `cells` denotes the number of ground cells in the
raster; `samples` the (variable) number of floor-labeled samples per cell.

### 2.1 Mask-aware temporal median / mode

**How it works.** For each ground cell, gather only the samples whose source
pixel is floor-labeled, then reduce them with a robust statistic: per-channel
**median** for continuous color, or **mode** (or a small color-cluster vote) for
quantized/posterized floors. Movers, shadows of movers, and specular flashes are
outliers in the floor-labeled subset and are rejected by the robust reduction.

| Aspect | Notes |
|--------|-------|
| Pros | Seam-free single global raster; robust to transient movers and per-sample mask noise; trivially parallel per cell; streams sample-by-sample so the full stack of frames never needs to be resident; deterministic and easy to audit. |
| Cons | A cell that is **never** seen empty (always occluded) yields no value and must be filled by another method; ignores photometric drift unless samples are normalized first; median over a small sample count can be noisy. |
| Cost | Time `O(cells × samples)`; memory `O(cells)` for streaming accumulators (or `O(cells × samples)` only if an exact median buffer is kept). CPU-only. |
| When to use | The default for offline batch reconstruction where most cells are seen empty at least once across the visit set. |

### 2.2 RPCA (Robust Principal Component Analysis)

**How it works.** Stack the co-registered samples into a matrix and decompose it
as **low-rank background + sparse foreground** ($M = L + S$). The low-rank
component $L$ is the static clean plate (it is the same across time up to
lighting), and the sparse component $S$ absorbs the transient movers. Solved by
an iterative convex relaxation (e.g. principal component pursuit /
inexact ALM).

| Aspect | Notes |
|--------|-------|
| Pros | Can recover the background **without** relying on the per-frame masks (it discovers the sparse foreground itself); models global lighting structure in the low-rank term; strong, well-studied baseline for background recovery. |
| Cons | Iterative SVD-based solver is computationally heavy; assumes the foreground is genuinely sparse, which fails when occupancy is persistent or large; needs the co-registered stack resident; harder to make seam-free across heterogeneous poses than a direct ground-cell reduction. |
| Cost | Time dominated by repeated SVDs per iteration — far above a single reduction pass; memory `O(cells × samples)` to hold the stack plus the low-rank factors. CPU-heavy. |
| When to use | As a **comparison baseline** to validate the cheaper median result, or where reliable per-frame masks are unavailable and foreground is provably sparse. |

### 2.3 Inpainting

**How it works.** For ground cells that are **never** seen empty, synthesize a
value from spatial context — neighboring recovered cells (diffusion / PatchMatch
/ exemplar fill) — or borrow from another pose/time that did see the cell. This
is a **fill** step, not a primary reducer: it only runs on the residual holes
left by a per-cell reduction.

| Aspect | Notes |
|--------|-------|
| Pros | Closes the holes a temporal reducer cannot fill; cheap when holes are few; produces a hole-free deliverable. |
| Cons | Synthesized, not observed — texture/geometry under a permanently occluded cell is a guess; quality degrades as hole size grows; not a standalone clean-plate method. |
| Cost | Time proportional to hole area and patch search radius, not to the full raster; memory `O(cells)`. CPU-only. |
| When to use | **Fallback** for never-empty cells after the primary reduction; keep the inpainted mask so synthesized cells are flagged downstream. |

### 2.4 Multi-band (Laplacian pyramid) blending

**How it works.** Where multiple poses overlap a region, build a Laplacian
pyramid per contributing source and blend bands across poses so that
low-frequency exposure differences are merged on coarse levels while
high-frequency detail is preserved on fine levels. This is a **refinement** that
suppresses residual brightness steps between poses, applied on top of the fused
raster.

| Aspect | Notes |
|--------|-------|
| Pros | Hides residual low-frequency intensity differences between overlapping poses; standard, well-understood technique; visually smooth result. |
| Cons | Solves a problem (cross-pose seams) that direct ground-cell fusion largely avoids by construction, so it is mostly cosmetic here; pyramid construction adds memory and passes; can smear genuine floor detail if mis-weighted. |
| Cost | Time `O(cells × pyramid_levels)`; memory for the pyramid (`~1.33 × raster` per band set). CPU-only. |
| When to use | Optional polish when photometric leveling alone leaves visible low-frequency steps in the final orthophoto. |

---

## 3\. Photometric normalization across visits

Samples for one cell come from different times of day and different exposure
settings, so they must be brought to a common photometric reference **before**
reduction, otherwise the median mixes incompatible brightnesses.

Two complementary strategies, both grounded in data the dataset already
carries:

- **Exposure-metadata normalization (preferred).** Each `FrameRecord` carries a
  per-frame `FullOptics` snapshot with `exposure_type`, `shutter`, `gain`, and
  `iris` (see [survey_dataset_schema.md](./survey_dataset_schema.md) and
  `poc_homography/domain/entities/survey/frame_record.py`). Use these to scale
  each sample toward a reference exposure — e.g. compensate for relative
  shutter/gain so a frame shot at higher gain is brought down before it enters
  the reduction. This is physically motivated and per-frame exact where the
  fields are populated; fields are optional, so fall back gracefully when a
  given frame lacks them.
- **Time-of-day bucketing (fallback).** When exposure metadata is missing or
  unreliable, bucket samples by capture time (`capture.timestamp_at_capture`)
  into time-of-day bins and reduce within consistent-lighting buckets, or
  estimate a per-bucket gain offset from cells observed across buckets. This
  needs no optics metadata but is coarser than per-frame compensation.

In both cases a residual global leveling pass (or the multi-band blend in
[2.4](#24-multi-band-laplacian-pyramid-blending)) can remove whatever
low-frequency drift remains.

---

## 4\. Recommendation

**Primary method: mask-aware temporal-median ortho-rectified ground-plane
fusion.** Resample every pose and visit into one global floor raster, and for
each ground cell take the per-channel median (or mode for quantized floors) over
the floor-labeled samples, after photometric normalization from `FullOptics`
exposure metadata.

Rationale against the alternatives:

- **Versus RPCA.** The median reduction is `O(cells × samples)` with streaming
  `O(cells)` memory and no iterative solver, whereas RPCA runs repeated SVDs
  over a resident stack and assumes sparse foreground — a poor fit for a survey
  that may see sustained occupancy. The median also exploits the per-frame masks
  we already have, which RPCA's value is to do without. RPCA is **reserved as a
  comparison baseline** to validate the median output, not as the production
  path, precisely because it is heavier in both CPU and memory for no gain when
  good masks exist.
- **Versus inpainting.** Inpainting synthesizes rather than observes, so it
  cannot be the primary method when most cells *are* seen empty. It is retained
  as the **fallback for never-empty cells** only.
- **Versus multi-band blending.** Direct ground-cell fusion is seam-free by
  construction, so blending is a cosmetic refinement, not a necessity. It is
  kept **optional**, alongside photometric leveling, for residual low-frequency
  steps.

The method is CPU-friendly, fully offline, produces a **single seam-free global
raster**, and is robust to transient movers by construction. The composed
pipeline is therefore: photometric normalization → mask-aware temporal-median
fusion → inpainting fallback for holes → optional multi-band blend / photometric
leveling refinement.

---

## 5\. Pipeline & usage

The implemented module lives at `poc_homography/cleanplate/`. It consumes a C1
survey run, ortho-rectifies each floor-masked frame onto the shared ground-plane
raster using the per-frame intrinsics and pose, performs the photometric
normalization and mask-aware temporal-median reduction described above, applies
the inpainting fallback to residual holes, and writes the clean-plate orthophoto
plus a provenance mask (which cells were observed vs. synthesized).

The high-level entry point is the CLI:

```bash
hom cleanplate reconstruct --run-id <run_id>
```

It targets a survey run identically whether that run is a real local-fixture
dataset or a synthetic one of the same structure.

> **Prototype data caveat — factual accuracy.** The survey run
> `smoke-37dbd4a0` referenced in the source issue is **not present in this
> worktree**. Survey datasets are local fixtures (see
> [test_fixtures.md](./test_fixtures.md)) and are not present here, and that
> run id does not exist locally. The shipped prototype is therefore generated
> from a **synthetic multi-visit dataset that mimics the same C1 structure**
> (multiple poses × multiple time buckets, with per-frame floor masks). The
> exact same `hom cleanplate reconstruct` command targets a real survey run
> identically once its fixture data is present locally.

### 5.1 Shipped prototype artifact

The committed prototype was produced deterministically (fixed `--seed 0`) by the
self-validating `synth` command, which builds a synthetic multi-visit dataset,
runs the full reconstruction, and reports the mean absolute error (MAE) of the
reconstruction against the known clean background over covered cells:

```bash
mkdir -p docs/assets/cleanplate
uv run hom cleanplate synth \
  --output docs/assets/cleanplate/clean_plate.png \
  --coverage-output docs/assets/cleanplate/coverage.tif \
  --truth-output docs/assets/cleanplate/truth.png \
  --n-visits 6 --seed 0 \
  --x-min 0 --x-max 6 --y-min 0 --y-max 6 --pixels-per-meter 16
```

Reported result: `visits=6 seed=0 coverage=99.8% MAE=13.0371`.

Output artifacts (committed under `docs/assets/cleanplate/`):

| File | Contents |
| ---- | -------- |
| `clean_plate.png` | Reconstructed empty-floor orthophoto (96×96, 6 m × 6 m at 16 px/m). |
| `truth.png` | Ground-truth clean background (for visual comparison). |
| `coverage.tif` | Per-cell observed-sample count (int32 raster); `0` = inpainted hole. |

To run against a **real** survey run instead (once a fixture dataset is present
locally), point `reconstruct` at the run directory and id — the pipeline is
identical:

```bash
uv run hom cleanplate reconstruct \
  --run-dir data/survey --run-id <run_id> --camera-id <camera_id> \
  --x-min <m> --x-max <m> --y-min <m> --y-max <m> --pixels-per-meter <ppm> \
  --output clean_plate.png --coverage-output coverage.tif
```
