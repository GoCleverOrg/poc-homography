# Productization Ideas — PTZ Calibration & Floor-Geometry Toolkit

This page catalogues candidate "products" that can be carved out of the PTZ
homography / ortho-floor work, with an honest read of what already exists in the
codebase, the gap to a shippable MVP, the value, and — critically — the
**physical accuracy ceiling** each one inherits from the single-PTZ-camera,
flat-ground model.

> **Accuracy ceiling (applies to everything geometric here).** A purely-rotating
> PTZ camera has (near) zero translation, so it cannot triangulate ground
> relief. Combined with a small pan/tilt **lever-arm** (rotation axes offset from
> the optical centre) and real floor undulation, this leaves an **irreducible
> ~cm-scale floor-registration floor** (≈25 mm median / ≈65 mm p90 line
> disagreement on our terminal scene). This was proven by experiment: pose
> bundle-adjustment cannot reduce it (it sits in the optimiser null space), a
> physical ≤3° ground slope does nothing, and only a degenerate non-physical
> plane "fixes" it in-sample. Beating cm-level needs a richer sensor model
> (lever-arm + floor DEM) or a second viewpoint (a baseline). Every product
> below is honest about whether it lives within, or is limited by, this ceiling.

Maturity is a rough "% to MVP" given current code.

---

## A. Automatic lens-distortion calibration per camera zoom — **~80%, ship first**

Find radial distortion coefficients (k1, k2) for every zoom level of a camera,
fully automatically, and store them so any downstream geometry can undistort
frames first.

**What already exists**
- `poc_homography/calibration/lens_distortion/` — a complete module:
  `distortion_solver.py` (plumb-line / line-straightness solver),
  `line_detection.py`, `scene_self_calibration.py` (group `(image, zoom)` pairs
  by zoom → detect lines → solve distortion+intrinsics per zoom → assemble a
  table, skipping zooms with too few usable lines), `calibration_table.py`
  (`CameraCalibrationTable` with zoom interpolation), `apply_calibration.py`.
- Domain: `domain/entities/lens_calibration_table.py`,
  `domain/vo/zoom_calibration_entry.py`, `domain/vo/lens_distortion.py`.
- Persistence: `infrastructure/repositories/repo_{yaml,postgres}_lens_calibration_table.py`,
  example output `calibration_results/valte_cam01_calibration.yaml`.
- Requirements/quality gates: `docs/lens_calibration_requirements.md`
  (≥15 pts/line, ≥10 lines, ≥2 quadrants, 2–3 orientations, 50/50 hold-out,
  validation-RMSE < baseline).
- Validated offline (maglor PoC): plumb-line k1=−0.183 / k2=0.098,
  held-out raw bow 0.75→0.36 px; per-**model** k1 std ≈ 0.011 (shareable);
  stable on repeat. k3/tangential overfit → keep k1,k2.

**Gap to MVP** (the product wrapper):
1. **Per-model table**, not just per-camera (codebase currently stores
   per-camera — see `CameraSpec` note). Aggregate many cameras-of-a-model into a
   shareable model default + variance.
2. **Multi-run-across-time benchmarking** that recomputes/updates the model
   table automatically; precision improves with more runs.
3. **Parameterised** run-count, zoom tick/range granularity, sensitivity.
4. **Visibility precheck** that *fails with a clear error* when the scene can't
   support calibration (no straight-line structure / insufficient coverage),
   plus a PTZ survey step to *find* a usable view.
5. Live capture loop with mandatory PTZ save/restore.

**Value:** high and foundational — every other geometric product depends on
undistorted frames. **Within the ceiling** (distortion is a lens property,
unaffected by the ground/relief limit). This is the recommended first ship.

---

## B. 2.5D floor calibration: reported (pan, tilt, zoom) → floor homography — **~50%, flagship**

Given a camera's *reported* pan/tilt/zoom, return the homography mapping image
pixels ↔ ground plane (the core "spatial world model": pixel↔world for any PTZ
pose).

**What already exists**
- `poc_homography/camera/intrinsics.py` (`compute_intrinsics`, K from zoom),
  `camera_geometry.py`, `homography/`, `horizon/`.
- The single-camera pose model (Rc2w + flat ground) and the automatic per-frame
  pan/tilt/focal **correction** via bundle-adjustment (the capability mira's
  hand-fit table lacks).

**Gap to MVP**
- Automate the *extrinsic* calibration end-to-end: zoom→focal curve, tilt-frame
  offset, camera height, and **pan/tilt reporting bias** per camera; persist and
  validate over time.
- A clean API: `homography_for(pan, tilt, zoom) -> Matrix3x3`.

**Honest ceiling:** **cm-level**, bounded by the lever-arm + relief limit. Great
for zones, counting, coarse localisation, dewarped overlays; **not** for mm
survey. Optionally extend with a lever-arm + floor-DEM model to push further
(research).

**Value:** highest strategic value — it's the reusable spatial model. Ship
scoped explicitly to cm accuracy.

---

## C. Automatic "perfect lines" for flat painted surfaces — **~60%**

Given a flat painted surface (parking lot, terminal apron), output the
*idealised* line network — straight, continuous, parallel-within-family,
perpendicular-across-families, equal-spacing — i.e. machine annotations, plus the
deviation of reality from the ideal as a **defect signal**.

**What exists** (maglor PoC, `model_loop4.py`): white+yellow line detection,
SAM vehicle exclusion on the analysis layer, **data-driven** orientation families
+ measured equally-spaced comb, a **self-converging nested loop**
(reality→model→feed-back→refit; line positions settle 30.5→5.4→1.1 mm), and a
mismatch = step-from-own-straightedge readout. In-repo line/annotation pieces:
`calibration/lens_distortion/line_detection.py`, `calibration/annotation.py`,
`domain/vo/{line_trace,ortho_line}.py`.

**Gap to MVP:** port the maglor loop into the repo; generalise beyond
parking-lot markings; recover short dashes/ticks; accept rectified or per-frame
input.

**Value:** auto-labelling for ML, pavement-marking QC, as-built-vs-design.
**Within the ceiling** for *detection*; the *mismatch* it reports is bounded by
the same cm floor (it measures it honestly).

---

## D. Stitch / ortho registration-QC report — **~65%, novel**

Automatic "is this mosaic correctly registered?" report for any multi-image
stitch/ortho: **cracks** (no-data seam tears), **line mismatches** (mm, from
tile-overlap disagreement *and* model-straightedge steps), and **bow**, each with
viewable full-res crops and a convergence/quality summary.

**What exists** (maglor PoC): the overlap-disagreement validator (median 0 /
p90 20 mm registration), the owner-map crack detector, the model-fit mismatch
detector — all validated by viewing, with false positives (cars, floor text)
traced and removed.

**Gap to MVP:** package as a CLI/report over an arbitrary tile set + owner map;
stabilise thresholds; HTML report.

**Value:** a quality *gate* for stitching/ortho pipelines. Honest by
construction. **Reports** the ceiling rather than being limited by it.

---

## E. Autonomous PTZ "survey" primitive — **~70%, reusable**

"Move the camera to find the best view of X" — sweep pan/tilt/zoom, score
candidate views (e.g. line-length × radial-spread × low-occlusion), with
**mandatory PTZ save/restore** so production cameras are never left moved.

**What exists:** `webapp/camera_survey/` (multi-phase survey, scoring, manifest),
the maglor survey-then-calibrate step (#47, all 8 cameras restored exactly),
`domain/vo/survey_plan_config.py`, `cli/survey.py`.

**Gap to MVP:** extract a small reusable "find a good view for task T" service +
the save/restore guard as a first-class, test-covered primitive.

**Value:** underpins A and any "go look at this" automation; safety-critical
save/restore is a differentiator.

---

## F. Clean-plate-as-a-service — **~55%, strong infra fit**

Time-average frames at a fixed pose (+ SAM masking) → a transient-free background
plate (no cars/people).

**What exists:** `poc_homography/cleanplate/`, `cli/cleanplate.py`,
`docs/cleanplate_reconstruction_methods.md`, and the MinIO+Neon+gallery infra
already built (epic tasks #18–20). No-car-mask + SAM PoC done; time-averaging
queued.

**Gap to MVP:** wire the time-averaging "clean-plate" step to the gallery infra;
schedule/refresh.

**Value:** clean plates feed calibration (stable line structure), change
detection, and presentation.

---

## G. PTZ accuracy / calibration-drift monitor — **~40%, low effort**

From pose-BA residuals: how accurately does a camera's *reported* pan/tilt match
reality, and does it **drift over time**? A camera-health / recalibration signal.

**What exists:** the pose-BA Δpan/Δtilt residuals (maglor `stage_c.py`); benchmark
task #46 pending. Uses existing captured data — **no camera movement**.

**Gap to MVP:** a periodic job that computes per-camera pointing bias + tracks it
over runs; alert on drift.

**Value:** operational reliability; tells you *when* to recalibrate (feeds A/B).

---

## H. One-camera metric floor-map generator — **~60%**

Turnkey: rotate an existing PTZ camera → get a **cm-accurate** metric floor map
(no drone), shipped *with* the QC report from (D) and the honest accuracy spec.

**What exists:** the full maglor pipeline (undistort → pose-BA → IPM → seam
composite → inpaint), `FINAL_floor_mosaic.png`; in-repo `survey/`, `maps/`,
`map_points/`.

**Gap to MVP:** productionise the pipeline as a repeatable job; bundle the QC
report; document the cm spec.

**Value:** rapid site floor map from hardware already deployed. Explicitly a
cm-level product (see ceiling).

---

## I. Lever-arm + floor-DEM calibration — **research, not MVP**

The principled route to beat the cm ceiling: jointly model the PTZ lever-arm
translation and a floor relief map. May still be ill-conditioned from a single
rotating camera (no baseline). Track as R&D, not a near-term product.

---

## Recommended sequencing

1. **A** (lens distortion per zoom) — nearly done, foundational. Pair with **E**
   (its survey primitive).
2. **B** (2.5D floor homography) — flagship spatial model, scoped to cm accuracy.
3. **C + D** — package together as "flat-surface line extraction + registration QC".
4. **F, G** — supporting operational capabilities.
5. **I** — research track only.
