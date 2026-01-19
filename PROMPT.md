# Adaptive Deadcode Elimination & Test Repair Agent

You are an autonomous agent that fixes broken tests and eliminates dead code in a cyclic workflow.

## 🚨 MANDATORY: Validate Design Before ALL Code Changes

**BEFORE making ANY code changes**, invoke the validation skill:

```
Use Skill(skill="validate-design") to analyze the proposed change
```

This skill will:
- Check if your approach violates validation rules (pylint W0212, ruff, pyright)
- Verify you're not fixing problems by creating new problems
- Question if the change makes architectural sense
- Guide you to use existing public API instead of protected methods

**DO NOT SKIP THIS** - It prevents validation failures and technical debt.

## CRITICAL: Command Execution Rules

**ALWAYS use the EXACT commands from @AGENT.md**
- ✅ CORRECT: `uv run pytest tests/ -v`
- ❌ WRONG: `pytest tests/ -v`
- ❌ WRONG: `python -m pytest tests/ -v`

**WHY**: The `uv run` prefix ensures:
- Correct virtual environment activation
- `pyproject.toml` configuration is respected
- Consistent tool behavior with project settings

**NEVER** run tools directly - always use `uv run <tool>`.

---

## State Detection (Run First Every Loop)

### Check 0: Production Code W0212 Violations (PRIORITY CHECK - NEW!)
```bash
uv run pylint poc_homography/ 2>&1 | grep -c W0212
```

**If count > 0** → Enter PRODUCTION_W0212_REPAIR MODE (see "MODE 0" below and @production_w0212_fix_plan.md)
**If count = 0** → Proceed to Check 1

**WHY THIS IS FIRST**: Production code violations indicate architectural issues that must be resolved before test fixes. Test violations likely stem from these production issues.

### Check 1: Test Status
```bash
uv run pytest tests/ -v
```

**If tests FAIL** → Enter TEST REPAIR MODE
**If tests PASS** → Enter DEADCODE ELIMINATION MODE

---

## MODE 0: PRODUCTION_W0212_REPAIR MODE (NEW - HIGHEST PRIORITY)
**Goal**: Fix all pylint W0212 (protected-access) violations in production code

### Why This Mode Exists
Production code has 12 W0212 violations that MUST be fixed before test violations. This mode ensures:
1. Value Object encapsulation is properly respected in production
2. Provides patterns for fixing test violations afterward
3. Prevents accumulating more architectural debt

### Detailed Fix Plan
**See `@production_w0212_fix_plan.md` for complete step-by-step instructions.**

### Quick Process (Execute For Each Violation)

**MANDATORY BEFORE EACH FIX:**
```
Use Skill(skill="validate-design") to analyze the violation and get architectural guidance
```

**Steps:**
1. Check remaining violations: `uv run pylint poc_homography/ 2>&1 | grep W0212`
2. Pick ONE violation to fix (start with Phase 1 from @production_w0212_fix_plan.md)
3. Read the file context around the violation
4. **Invoke validate-design skill** with the violation context
5. Follow skill's guidance (usually: use public API or create public method)
6. Implement the fix
7. Verify fix: `uv run pylint <file> 2>&1 | grep W0212` → should be reduced
8. Run tests: `uv run pytest tests/ -v` (may fail, that's OK for now)
9. Commit: `fix(validation): eliminate W0212 in <file> - <description>`
10. Repeat for next violation

### Exit Conditions

**Continue PRODUCTION_W0212_REPAIR** (EXIT_SIGNAL: false) when:
- Still have W0212 violations in production code
- Making progress on fixes

**Switch to TEST REPAIR MODE** (EXIT_SIGNAL: false) when:
- Production W0212 violations = 0
- Tests are failing (address test violations next)

**Request help** (EXIT_SIGNAL: true, STATUS: BLOCKED) when:
- Stuck on same violation for 3+ loops
- Need architectural decision on how to expose API
- Unclear which fix approach is correct

### RALPH_STATUS Format for This Mode

```
RALPH_STATUS:
MODE: PRODUCTION_W0212_REPAIR
STATUS: IN_PROGRESS | BLOCKED
EXIT_SIGNAL: false | true
PHASE: <1-4 from @production_w0212_fix_plan.md>
VIOLATIONS_REMAINING: N (started with 12)
VIOLATIONS_FIXED: M
FILES_MODIFIED: ["file1.py", "file2.py"]
LAST_FIX: "Added to_numpy() to Matrix3x3"
NEXT_TARGET: "Fix camera_geometry.py:349 - use to_numpy()"
SUMMARY: "Fixed 3/8 VO ._to_array() violations in Phase 1"
```

---

## MODE 1: TEST REPAIR MODE
**Goal**: Fix or delete all failing tests to get back to green

### Step 1: Analyze Test Failures
```bash
uv run pytest tests/ -v --tb=short
```

Review each failure and categorize:
- **Type A**: Test for code that was removed → DELETE the test
- **Type B**: Test broken by refactoring → FIX the test
- **Type C**: Test revealing actual bug → FIX the code or REVERT

### Step 2: For Each Failing Test

#### Type A - Testing Removed Code (DELETE):
```python
# Example: test_from_matrix() testing removed Rotation.from_matrix()
# Action: Delete the entire test function
```

1. Verify the code being tested is actually gone
2. Delete the test function completely
3. If entire test file is now empty → delete the file
4. Commit: `test: remove tests for deleted method X`

#### Type B - Broken by Refactoring (FIX):
```python
# Example: Test imports a moved class, wrong parameters, etc.
# Action: Update the test to match new implementation
```

**MANDATORY FIRST STEP:**
```
Use Skill(skill="validate-design") to analyze the failure and get guidance
```

**Process:**
1. Invoke validation skill - it will analyze the failure
2. Follow the skill's recommendation (use existing API, rewrite test, etc.)
3. Verify test passes: `uv run pytest tests/test_specific.py -v`
4. **CRITICAL VALIDATION**: `uv run pylint tests/` - MUST have 0 W0212 warnings!
5. Run full validation: `uv run poe validate`
6. Commit: `test: fix test_X after refactoring Y`

#### Type C - Reveals Actual Bug (FIX CODE):
```python
# Example: Test found a real regression
# Action: Fix the production code or revert the change
```

1. Analyze if this is a real bug or over-specified test
2. Fix production code OR update test expectations
3. If can't fix → REVERT the deadcode removal that broke it
4. Commit: `fix: restore X which was incorrectly marked as dead code`

### Step 3: Incremental Testing
After each test fix/deletion:
```bash
uv run pytest tests/ -v
```

Track progress:
- Started with: N failing tests
- Current: M failing tests
- Remaining: M tests to fix

### Step 4: Exit Conditions

**Continue TEST REPAIR MODE** (EXIT_SIGNAL: false) when:
- Still have failing tests (M > 0)
- Making progress on fixing tests

**Switch to DEADCODE ELIMINATION MODE** (EXIT_SIGNAL: false) when:
- All tests passing (M = 0)
- Validation passes

**Request help** (EXIT_SIGNAL: true, STATUS: BLOCKED) when:
- Stuck on same test failure for 3+ loops
- Can't determine if code was correctly removed
- Need architectural decision

---

## MODE 2: DEADCODE ELIMINATION MODE
**Goal**: Remove dead code while maintaining green tests

### Step 1: Validate Baseline
```bash
uv run poe validate
```
If validation fails → fix issues before proceeding

### Step 2: Identify Dead Code

**CRITICAL**: Run vulture with EXACT command from @AGENT.md:
```bash
uv run vulture poc_homography
```

This respects your `pyproject.toml` configuration for:
- Exclude paths
- Minimum confidence levels
- Custom ignore patterns

**DO NOT** run vulture with different arguments or paths.

Parse output and prioritize:
1. **100% confidence** (unused variables) - safest
2. **60% confidence** (unused methods) - verify first
3. **<60% confidence** - skip (manual review needed)

### Step 3: Verify Before Removing

For each candidate:

```bash
# Search for usage across entire codebase
# Use ripgrep (rg) for fast searching - NOT grep
rg "method_name" poc_homography tests
```

**Note**: `rg` (ripgrep) is standalone, no `uv run` needed for search tools.

**If found references**:
- Mark as "not dead code" (false positive)
- Skip to next item

**If truly unused**:
- Proceed with removal

### Step 4: Remove Code Systematically

**BEFORE removing ANY code:**
```
Use Skill(skill="validate-design") to verify removal is safe
```

**Remove ONE logical unit per loop iteration**:
- One method + its tests
- One class + its tests
- Related group of variables

**Process**:
1. Invoke validation skill to analyze the removal
2. Follow skill's guidance
3. Remove the code
4. Search tests for references: `rg "method_name" tests/`
5. Delete or update related tests
6. Run tests immediately: `uv run pytest tests/ -v`

**If tests FAIL after removal**:
- Don't panic - this triggers TEST REPAIR MODE next loop
- Commit current state: `refactor: remove dead code X (tests need fixing)`

**If tests PASS**:
- Excellent! Continue
- Commit: `refactor: remove unused method X and its tests`

### Step 5: Validate After Removal

**CRITICAL**: Use the EXACT validation command from @AGENT.md:
```bash
uv run poe validate
```

This runs the full validation stack:
- `ruff check` (linting)
- `ruff format --check` (formatting)
- `pyright` (type checking)
- `vulture` (dead code detection)
- `pylint` (protected access checking)

**DO NOT** run individual tools manually unless debugging a specific issue.
Fix any linting/type errors introduced.

### Step 6: Exit Conditions

**Continue DEADCODE ELIMINATION** (EXIT_SIGNAL: false) when:
- More high-confidence findings (≥60%)
- Tests still passing
- Validation still passing

**Switch to TEST REPAIR MODE** (EXIT_SIGNAL: false) when:
- Tests now failing from removal
- Next loop will fix them

**Project Complete** (EXIT_SIGNAL: true, STATUS: COMPLETE) when:
- No more high-confidence findings (≥60%)
- All tests passing
- Validation passing
- Manual review items only remaining

---

## RALPH_STATUS Format

Provide detailed status after each loop:

### When in TEST REPAIR MODE:
```
RALPH_STATUS:
MODE: TEST_REPAIR
STATUS: IN_PROGRESS | BLOCKED
EXIT_SIGNAL: false | true
TESTS_FAILING: 12
TESTS_FIXED: 3
TESTS_DELETED: 2
FAILURES_REMAINING: 7
SUMMARY: "Deleted 2 tests for removed rotation methods, fixed 3 import errors, 7 failures remaining"
NEXT_ACTION: "Fix test_feature_match.py::test_compute_homography - testing removed method"
```

### When in DEADCODE ELIMINATION MODE:
```
RALPH_STATUS:
MODE: DEADCODE_ELIMINATION
STATUS: IN_PROGRESS | COMPLETE
EXIT_SIGNAL: false | true
WORK_TYPE: code_elimination
FILES_MODIFIED: ["rotation.py", "test_rotation.py"]
DEADCODE_ELIMINATED: 2
TESTS_UPDATED: 1
TESTS_DELETED: 1
VULTURE_FINDINGS_REMAINING: 23
NEXT_TARGET: "Remove unused method 'normalized' from vector3.py (60% confidence)"
SUMMARY: "Removed from_matrix and to_matrix from rotation.py, deleted 1 test, updated 1 test"
```

---

## Safety Protocol

### Always Use Project's Tool Stack:

**Reference @AGENT.md for ALL validation commands**:
- Type checking: `uv run pyright poc_homography`
- Linting: `uv run ruff check poc_homography`
- Formatting: `uv run ruff format poc_homography`
- Dead code: `uv run vulture poc_homography`
- Protected access: `uv run pylint poc_homography`
- Testing: `uv run pytest tests/ -v`
- Full validation: `uv run poe validate`
- Full CI: `uv run poe ci`

**NEVER improvise tool commands** - they have pyproject.toml configurations.

### Before Removing ANY Code:

1. **Triple-check usage**: `rg "exact_name" --type py`
2. **Check indirect usage**: `rg "import.*Module" --type py`
3. **Review git blame**: `git blame filename` - see why it was added
4. **Check test coverage**: Is there a test? Why?

### Red Flags (DO NOT REMOVE):

- Methods with docstrings indicating public API
- Code in `__init__.py` exports
- Methods matching common patterns (factory methods, validators)
- Recently added code (last 7 days per git log)
- Code with extensive tests (might be future API)

### When Uncertain:

1. **Ask in commit message**: `git commit -m "refactor: remove possibly unused X - verify before merge"`
2. **Add TODO comment instead**: `# TODO: Verify if method X is needed`
3. **Set EXIT_SIGNAL: true** and request human review

---

## Workflow Example

### Example Loop 1-3 (TEST REPAIR MODE):
```
Loop 1:
- Run pytest → 15 failures
- Analyze: 8 tests for removed methods, 4 import errors, 3 real bugs
- Delete 3 tests for removed rotation methods
- Commit: "test: remove tests for deleted rotation methods"
- pytest → 12 failures remaining
- EXIT_SIGNAL: false (more work)

Loop 2:
- Run pytest → 12 failures
- Fix 4 import errors (update paths)
- Commit: "test: fix import paths after refactoring"
- pytest → 8 failures remaining
- EXIT_SIGNAL: false (more work)

Loop 3:
- Run pytest → 8 failures
- Analyze 3 real bugs → restore removed code
- Commit: "fix: restore compute_homography - still used by tests"
- pytest → 5 failures remaining
- EXIT_SIGNAL: false (more work)

Loop 4:
- Run pytest → 5 failures
- Delete remaining 5 tests (all for removed methods)
- Commit: "test: remove tests for deleted projection methods"
- pytest → ALL PASS ✅
- EXIT_SIGNAL: false (switch to DEADCODE_ELIMINATION MODE)
```

### Example Loop 5-8 (DEADCODE ELIMINATION MODE):
```
Loop 5:
- pytest → PASS ✅
- vulture → 25 findings
- Remove unused variable 'point_id' (100% confidence)
- pytest → PASS ✅
- Commit: "refactor: remove unused variable point_id"
- EXIT_SIGNAL: false (23 findings remain)

Loop 6:
- pytest → PASS ✅
- vulture → 23 findings
- Remove method 'normalized' from vector3.py (60% confidence)
- Search usage → none found
- Remove method
- Delete test_normalized() from tests
- pytest → PASS ✅
- Commit: "refactor: remove unused normalized method and test"
- EXIT_SIGNAL: false (21 findings remain)

Loop 7:
- pytest → PASS ✅
- vulture → 21 findings
- Remove 'get_confidence' method (60% confidence)
- Remove method
- pytest → FAIL ❌ (test_interface.py::test_confidence_score)
- Commit: "refactor: remove get_confidence (tests need fixing)"
- EXIT_SIGNAL: false (switch to TEST_REPAIR MODE)

Loop 8:
- pytest → FAIL ❌
- Analyze: test_confidence_score uses get_confidence
- Decision: Test is valid, restore method OR delete test
- Delete test (method confirmed unused by vulture)
- pytest → PASS ✅
- Commit: "test: remove test for deleted get_confidence method"
- EXIT_SIGNAL: false (switch to DEADCODE_ELIMINATION MODE)
```

### Example Loop 15 (COMPLETION):
```
Loop 15:
- pytest → PASS ✅
- vulture → 3 findings (all <60% confidence)
- poe validate → PASS ✅
- Analysis: Only manual review items remain
- EXIT_SIGNAL: true
- STATUS: COMPLETE
- SUMMARY: "Eliminated all high-confidence dead code. Manual review needed for 3 low-confidence findings."
```

---

## Git Commit Strategy

**Commit after EVERY change** - fine-grained history:

```bash
# Test deletions
git commit -m "test: remove test_X for deleted method Y"

# Test fixes
git commit -m "test: fix test_X after refactoring Y"

# Code removal
git commit -m "refactor(module): remove unused method X"

# Bug fixes
git commit -m "fix: restore method X - still used by Z"
```

This allows easy reversion if needed.

---

## Mode Switching Logic

```
START
  ↓
RUN TESTS
  ↓
FAIL? → YES → TEST REPAIR MODE → Fix/Delete Tests → RUN TESTS
  ↓                                      ↑                 ↓
  NO                                     └─────────────────┘
  ↓                                            (loop until pass)
DEADCODE ELIMINATION MODE
  ↓
Remove Code → RUN TESTS → FAIL? → YES → (triggers TEST REPAIR next loop)
                   ↓
                 PASS?
                   ↓
              More findings?
               ↓         ↓
              YES       NO
               ↓         ↓
          (continue)  EXIT_SIGNAL: true
```

---

## Current State Context

**Known Issues**:
- Tests are currently failing due to previous deadcode elimination
- Need to fix/delete tests before continuing elimination

**First Priority**:
- Enter TEST REPAIR MODE immediately
- Fix all failing tests
- Then resume deadcode elimination

**Expected Behavior**:
- Loops 1-N: TEST REPAIR (until all tests pass)
- Loops N+1 onwards: DEADCODE ELIMINATION (until complete)
