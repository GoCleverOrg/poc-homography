# Production Code W0212 Violations Fix Plan

## Overview
12 W0212 violations in production code must be fixed before test violations can be addressed.

## Priority Order (Fix Sequentially)

### Phase 1: Value Object API (8 violations) - HIGHEST PRIORITY
**Why First**: Most violations, affects domain layer architecture

#### Step 1.1: Add Public Conversion Methods to VOs
**Files**: `domain/vo/matrix3x3.py`, `domain/vo/vector3.py`

**Action**: Add public `to_numpy()` methods as alternatives to `._to_array()`

```python
# In Matrix3x3
def to_numpy(self) -> npt.NDArray[np.float64]:
    """Convert to numpy array for external computation.

    Returns:
        3x3 numpy array
    """
    return self._to_array()

# In Vector3
def to_numpy(self) -> npt.NDArray[np.float64]:
    """Convert to numpy array for external computation.

    Returns:
        3-element numpy array [x, y, z]
    """
    return self._to_array()
```

**Then Update Callers**:
- `camera_geometry.py:348-350`: Replace `._to_array()` → `.to_numpy()`
- `cli/camera.py:135`: Replace `._to_array()` → `.to_numpy()`
- `rotation.py:117`: Replace `._to_array()` → `.to_numpy()`

**Validation After Each Change**:
```bash
uv run pylint <file> | grep W0212
uv run pytest tests/ -v
```

**Expected Result**: 8 violations → 0 violations in this category

---

### Phase 2: VO Internal Communication (2 violations) - MEDIUM PRIORITY
**Why Second**: Affects VO architecture, fewer violations

#### Step 2.1: Fix Orientation → Rotation Access
**File**: `domain/vo/orientation.py:152`

**Current**: `other._rotation` (protected access)

**Options**:
A. Add public getter: `rotation.get_rotation() -> Rotation`
B. Make `._rotation` public: `self.rotation`
C. Add comparison method to Rotation itself

**Recommended**: Option B (simplest, rotation is a core property)

**Action**:
1. Read `orientation.py:152` context to understand usage
2. Decide best approach
3. Implement fix
4. Validate: `uv run pylint domain/vo/orientation.py`

#### Step 2.2: Fix Rotation → Matrix Access
**File**: `domain/vo/rotation.py:108`

**Current**: `other._matrix` (protected access)

**Context**: Composing two rotations (matrix multiplication)

**Recommended**: Add public property or method for matrix multiplication

**Action**:
1. Read `rotation.py:108` context
2. Add public interface for composition
3. Validate: `uv run pylint domain/vo/rotation.py`

**Expected Result**: 2 violations → 0 violations in this category

---

### Phase 3: UI Internal Methods (4 violations) - LOW PRIORITY
**Why Third**: Isolated to UI layer, doesn't affect domain

#### Step 3.1: CalibrationSession._update_display()
**Files**:
- `calibration/interactive.py:465, 521`
- `cli/frame.py:407, 486`

**Current**: Module functions calling `session._update_display()`

**Root Cause**: `_update_display()` is internal but called externally

**Options**:
A. Make `_update_display()` public: `update_display()`
B. Move calling code into CalibrationSession methods
C. Create public `refresh()` wrapper

**Recommended**: Option A (simplest, method is UI-specific)

**Action**:
1. Rename `_update_display()` → `update_display()` in CalibrationSession
2. Update all 4 call sites
3. Validate: `uv run pylint calibration/ cli/`

**Expected Result**: 4 violations → 0 violations in this category

---

### Phase 4: Service Layer (1 violation) - INVESTIGATE FIRST
**File**: `application/services/frame_capture_service.py:94`

**Current**: Accessing `._data_dir` of unknown class

**Action**:
1. Read context around line 94
2. Identify what class owns `_data_dir`
3. Determine if this should be public or if service is wrong layer
4. Fix based on architectural decision

**Expected Result**: 1 violation → 0 violations

---

## Execution Strategy for Ralph

### Approach: Incremental, One Category at a Time

**Loop Pattern**:
1. Fix 1-2 related violations
2. Run validation: `uv run pylint <file>`
3. Run tests: `uv run pytest tests/ -v`
4. Commit if green
5. If tests fail → switch to TEST_REPAIR mode
6. Once tests pass → continue to next violations

### Exit Conditions

**After Each Phase**:
```bash
# Verify violations reduced
uv run pylint poc_homography/ 2>&1 | grep -c W0212
# Should decrease: 12 → 4 → 2 → 0

# Ensure no new violations
uv run poe validate

# Ensure tests still pass (or switch to TEST_REPAIR)
uv run pytest tests/ -v
```

**Final Goal**:
- Production code: 0 W0212 violations
- Tests: May still have violations (will be fixed after production is clean)
- All tests passing

---

## RALPH_STATUS Updates

Ralph should report progress like:

```
RALPH_STATUS:
MODE: PRODUCTION_W0212_REPAIR
STATUS: IN_PROGRESS
EXIT_SIGNAL: false
PHASE: 1 (Value Object API)
VIOLATIONS_FIXED: 3/8 in current phase
TOTAL_REMAINING: 9/12
FILES_MODIFIED: ["domain/vo/matrix3x3.py", "camera_geometry.py"]
NEXT_ACTION: "Add to_numpy() to Vector3, update rotation.py:117"
SUMMARY: "Added Matrix3x3.to_numpy(), fixed camera_geometry.py (3 violations)"
```

---

## Success Criteria

✅ **Phase 1 Complete**: 8 VO violations → 0
✅ **Phase 2 Complete**: 2 VO internal violations → 0
✅ **Phase 3 Complete**: 4 UI violations → 0
✅ **Phase 4 Complete**: 1 service violation → 0
✅ **Final Check**: `uv run pylint poc_homography/ 2>&1 | grep W0212` → no results
✅ **Tests**: All passing or ready for TEST_REPAIR mode

**Then**: Move to fixing test file violations (20 violations in tests/)
