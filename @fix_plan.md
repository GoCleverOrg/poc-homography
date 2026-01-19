# Adaptive Deadcode Elimination & Test Repair Plan

## Current State
- **Tests Status**: FAILING (from previous deadcode elimination)
- **CRITICAL BLOCKER**: Production code has 12 W0212 violations (protected-access)
- **First Priority**: Fix production code W0212 violations (PHASE 0)
- **Second Priority**: Fix test W0212 violations and failing tests (PHASE 1)
- **Third Priority**: Continue deadcode elimination (PHASE 2)

---

## PHASE 0: FIX PRODUCTION CODE W0212 VIOLATIONS (Active - MANDATORY)

**DECISION**: User selected Option C - Fix production code violations FIRST, then tests.

**Goal**: Eliminate all 12 W0212 (protected-access) violations in `poc_homography/` production code.

### Why This Matters
- Production code violations must be fixed before test violations
- Validates that Value Object encapsulation is respected in production
- Provides pattern for fixing test violations afterward
- Prevents introducing more protected-access in tests

### Production Code Violations (12 total)

#### Group 1: `._to_array()` access (5 violations)
- [ ] `camera_geometry.py:348` - accessing `._to_array` of Matrix/Vector
- [ ] `camera_geometry.py:349` - accessing `._to_array` of Matrix/Vector
- [ ] `camera_geometry.py:350` - accessing `._to_array` of Matrix/Vector
- [ ] `cli/camera.py:135` - accessing `._to_array` of Matrix/Vector
- [ ] `domain/vo/rotation.py:117` - accessing `._to_array` of Matrix

#### Group 2: `._update_display()` access (4 violations)
- [ ] `calibration/interactive.py:465` - accessing `._update_display`
- [ ] `calibration/interactive.py:521` - accessing `._update_display`
- [ ] `cli/frame.py:407` - accessing `._update_display`
- [ ] `cli/frame.py:486` - accessing `._update_display`

#### Group 3: Other protected members (3 violations)
- [ ] `application/services/frame_capture_service.py:94` - accessing `._data_dir`
- [ ] `domain/vo/orientation.py:152` - accessing `._rotation`
- [ ] `domain/vo/rotation.py:108` - accessing `._matrix`

### Strategy

**MANDATORY: Use validate-design skill before EACH fix**
```
Skill(skill="validate-design")
```

For each violation:
1. Read the file and understand the context
2. Invoke validate-design skill with the proposed fix
3. Follow skill's guidance to use public API
4. Fix the violation
5. Run `uv run pylint poc_homography/` to verify fix
6. Commit with message: `fix(validation): eliminate W0212 in <file>`
7. Move to next violation

### Exit Criteria for Phase 0
- [ ] All 12 production code W0212 violations eliminated
- [ ] `uv run pylint poc_homography/` returns 0 W0212 errors
- [ ] `uv run poe validate` passes
- [ ] Tests may still fail (that's Phase 1)
- [ ] Ready to proceed to Phase 1 (test violations)

---

## PHASE 1: TEST REPAIR MODE (Pending Phase 0 Completion)
**Goal**: Get all tests to pass before continuing deadcode elimination

### Pre-Work: Analyze Current Failures
- [ ] Run `uv run pytest tests/ -v --tb=short` to see all failures
- [ ] Categorize each failure as Type A, B, or C (see PROMPT.md)
- [ ] Create execution plan based on failure types

### Type A: Tests for Removed Code (DELETE TESTS)
- [ ] Identify all tests that import or call removed code
- [ ] Delete test functions for removed methods/classes
- [ ] Delete empty test files if all tests removed
- [ ] Run tests after each deletion to verify progress

### Type B: Tests Broken by Refactoring (FIX TESTS)
- [ ] Fix import path errors (moved classes/functions)
- [ ] Update function signatures (changed parameters)
- [ ] Update assertions (changed return values/types)
- [ ] Fix mocking/patching paths after refactoring

### Type C: Tests Revealing Real Bugs (FIX CODE OR REVERT)
- [ ] Analyze each "real bug" failure
- [ ] Determine if removed code should be restored
- [ ] Fix production code if bug is real
- [ ] Revert deadcode removal if code wasn't actually dead

### Exit Criteria for Phase 1
- [ ] All tests passing: `uv run pytest tests/ -v` → 100% pass
- [ ] Validation passing: `uv run poe validate` → clean
- [ ] Ready to proceed to Phase 2

---

## PHASE 2: DEADCODE ELIMINATION MODE (Pending Phase 1 Completion)

### High Priority: 100% Confidence Findings (Safest)
These are definitely unused - remove without hesitation:

- [ ] `feature_match.py:741` - unused variable 'point_id'
- [ ] `interface.py:206` - unused variable 'point_id'
- [ ] `interface.py:220` - unused variable 'point_id_prefix'
- [ ] `map_points.py:218` - unused variable 'point_id'
- [ ] `map_points.py:261` - unused variable 'point_id_prefix'
- [ ] `map_points.py:49` - unused variable 'inverse_matrix'
- [ ] `map_points.py:50` - unused variable 'num_gcps'

### Medium Priority: 60% Confidence Methods (Verify First)

#### rotation.py (Domain Value Objects)
- [ ] Verify usage: `rg "from_matrix" --type py`
- [ ] Remove `from_matrix` method (line 73) if unused
- [ ] Delete tests for `from_matrix`
- [ ] Verify usage: `rg "to_matrix" --type py`
- [ ] Remove `to_matrix` method (line 88) if unused
- [ ] Delete tests for `to_matrix`

#### vector3.py (Domain Value Objects)
- [ ] Verify usage: `rg "normalized" --type py`
- [ ] Remove `normalized` method (line 103) if unused
- [ ] Delete tests for `normalized`

#### feature_match.py (Homography Implementation)
- [ ] Verify usage: `rg "compute_homography" --type py`
- [ ] Remove `compute_homography` method (line 501) if unused
- [ ] Verify usage: `rg "project_point" --type py`
- [ ] Remove `project_point` method (line 741) if unused
- [ ] Delete related tests

#### interface.py (Homography Interface)
- [ ] Verify usage: `rg "LEARNED|MAP_BASED_ORIGIN" --type py`
- [ ] Remove unused constants `LEARNED`, `MAP_BASED_ORIGIN` (lines 38, 45)
- [ ] Verify usage: `rg "identity" --type py`
- [ ] Remove `identity` method (line 89) if unused
- [ ] Verify usage: `rg "project_point" --type py`
- [ ] Remove `project_point` method (line 205) if unused
- [ ] Verify usage: `rg "project_points" --type py`
- [ ] Remove `project_points` method (line 218) if unused
- [ ] Verify usage: `rg "get_confidence" --type py`
- [ ] Remove `get_confidence` method (line 233) if unused
- [ ] Delete tests for all removed interface methods

#### intrinsic_extrinsic.py (Camera Model)
- [ ] Verify usage: `rg "get_distortion_coefficients" --type py`
- [ ] Remove `get_distortion_coefficients` (line 280) if unused
- [ ] Verify usage: `rg "compute_from_config" --type py`
- [ ] Remove `compute_from_config` (line 485) if unused
- [ ] Verify usage: `rg "project_point_static" --type py`
- [ ] Remove `project_point_static` (line 665) if unused
- [ ] Verify usage: `rg "project_image_to_map" --type py`
- [ ] Remove `project_image_to_map` (line 700) if unused
- [ ] Verify usage: `rg "result_to_homography_result" --type py`
- [ ] Remove `result_to_homography_result` (line 750) if unused
- [ ] Delete tests for all removed methods

#### map_points.py (Map Projection)
- [ ] Verify usage: `rg "compute_from_gcps" --type py`
- [ ] Remove `compute_from_gcps` (line 95) if unused
- [ ] Verify usage: `rg "camera_to_map" --type py`
- [ ] Remove `camera_to_map` (line 215) if unused
- [ ] Verify usage: `rg "map_to_camera" --type py`
- [ ] Remove `map_to_camera` (line 241) if unused
- [ ] Verify usage: `rg "camera_to_map_batch" --type py`
- [ ] Remove `camera_to_map_batch` (line 258) if unused
- [ ] Verify usage: `rg "map_to_camera_batch" --type py`
- [ ] Remove `map_to_camera_batch` (line 287) if unused
- [ ] Verify usage: `rg "get_result" --type py`
- [ ] Remove `get_result` (line 308) if unused
- [ ] Verify usage: `rg "get_homography_matrix" --type py`
- [ ] Remove `get_homography_matrix` (line 312) if unused
- [ ] Delete tests for all removed methods

### Exit Criteria for Phase 2
- [ ] All 100% confidence items removed
- [ ] All verified 60% confidence items removed
- [ ] Only <60% confidence items remain (manual review)
- [ ] All tests passing
- [ ] Full validation passing
- [ ] Vulture shows only low-confidence findings

---

## PHASE 3: FINAL VALIDATION (Pending Phase 2 Completion)

- [ ] Run full test suite: `uv run pytest tests/ -v` → 100% pass
- [ ] Run full validation: `uv run poe validate` → clean
- [ ] Run vulture: `uv run vulture poc_homography` → only low confidence
- [ ] Review remaining low-confidence findings for manual cleanup
- [ ] Verify no test coverage drop: `pytest --cov=poc_homography`
- [ ] Run type checking: `uv run pyright poc_homography` → no errors
- [ ] Final commit: "refactor: complete deadcode elimination cycle"

---

## Progress Tracking

### Mode Indicators
- **Current Mode**: TEST_REPAIR (will auto-switch to DEADCODE_ELIMINATION once tests pass)
- **Loop Count**: 0
- **Tests Fixed**: 0
- **Tests Deleted**: 0
- **Code Eliminated**: 0
- **Vulture Findings**: ~30 (initial baseline)

### Session Checkpoints
After every 5 loops, verify:
- [ ] Making forward progress (not stuck on same issue)
- [ ] Git commits are clean and revertible
- [ ] Tests still at 100% pass rate
- [ ] No critical functionality broken

---

## Notes for Human Review

Items that may need manual decision:
- Any <60% confidence vulture findings (defer to manual review)
- Methods with extensive documentation (might be public API)
- Code in `__init__.py` exports (likely part of public interface)
- Recently added code (last 30 days per git log)

These will be flagged with `EXIT_SIGNAL: true` and `STATUS: NEEDS_REVIEW`
