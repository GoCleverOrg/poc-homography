---
name: validate-design
description: Validates proposed code changes against project validation rules (pylint W0212, ruff, pyright, vulture) and architectural principles. Questions if changes make sense, prevents protected method access, and ensures fixes don't create new problems. MANDATORY before ANY code modification.
allowed-tools: Read, Grep, Bash
---

# Validate Software Design

## Purpose

Analyze proposed code changes to ensure they:
1. Pass validation stack (`uv run poe validate` + `pylint tests/`)
2. Use existing public API instead of protected methods
3. Make architectural sense
4. Don't create new problems while fixing old ones

## When to Invoke

**MANDATORY before:**
- Fixing any failing test
- Removing any dead code
- Adding any method/operator to VOs
- Modifying production or test code

## Core Validation Rules

### Rule 1: No Protected Access (pylint W0212)

```python
# ❌ FORBIDDEN - Violates pylint W0212
result = matrix._to_array()
value = obj._private_method()

# ✅ REQUIRED - Use public API only
result = matrix.T  # Property
det = matrix.determinant  # Property
product = matrix1 @ matrix2  # Operator
```

### Rule 2: Question Before Adding

Before adding ANY method/operator to a VO:
- Can existing API achieve the goal?
- Is this testing behavior or implementation?
- Would this create API bloat?

### Rule 3: Validation Must Pass

```bash
# Production code
uv run poe validate  # Must pass (includes pylint poc_homography)

# Test code (not in poe validate!)
uv run pylint tests/  # Must pass - no W0212 warnings
```

## Analysis Process

### Step 1: Understand the Failure

Read the test failure or dead code issue. Ask:
- What is actually failing?
- Why is it failing?
- What was the recent change that broke it?

### Step 2: Check Existing Public API

For Matrix3x3:
- `.T` - transpose (property)
- `.determinant` - determinant (property)
- `.inverse()` - inverse (method)
- `@` operator - matrix multiplication
- `.condition_number` - condition number
- `.to_list()` - convert to list

For Vector3:
- Check `poc_homography/domain/vo/vector3.py` for public API

### Step 3: Question the Test

Ask:
- Is this test verifying BEHAVIOR or IMPLEMENTATION?
- Can behavior be verified with existing public API?
- Example: Instead of `np.allclose(matrix._to_array(), expected)`, can we verify matrix properties like orthogonality, determinant, etc.?

### Step 4: Analyze Proposed Solution

Consider each option:

**Option A: Use existing public API**
- Pros: No code changes, respects encapsulation
- Cons: May require rethinking test approach

**Option B: Add new public API**
- Pros: Enables natural test syntax
- Cons: API bloat, might expose implementation

**Option C: Delete/rewrite test**
- Pros: Removes implementation-dependent test
- Cons: Loses test coverage

**Option D: Restore removed code**
- Pros: Fixes Type C failures (incorrectly removed code)
- Cons: Keeps dead code

### Step 5: Run Validation

```bash
# Check production code
uv run poe validate

# Check test code (CRITICAL - not in poe validate!)
uv run pylint tests/

# Run specific test
uv run pytest tests/test_specific.py -v
```

## Output Format

Provide structured analysis:

```
VALIDATION ANALYSIS
===================

PROPOSED CHANGE:
<brief description>

FAILURE ANALYSIS:
- What's failing: <description>
- Root cause: <why>
- Recent change: <what broke it>

EXISTING PUBLIC API:
Matrix3x3: .T, .determinant, .inverse(), @, .condition_number, .to_list()
Vector3: <check file>

DESIGN QUESTIONS:
Q1: Can existing API verify the behavior?
A: <yes/no with reasoning>

Q2: Is test checking behavior or implementation?
A: <behavior/implementation with evidence>

Q3: Would proposed fix pass validation?
A: <yes/no with specific check that would fail>

OPTIONS ANALYSIS:
Option A (existing API): <pros/cons>
Option B (add public API): <pros/cons>
Option C (rewrite test): <pros/cons>
Option D (restore code): <pros/cons>

RECOMMENDATION: Option <X>

REASONING:
<why this option is best>

VALIDATION PLAN:
1. uv run pylint tests/<file>  # Expect: 0 W0212 warnings
2. uv run poe validate  # Expect: all pass
3. uv run pytest <file> -v  # Expect: all pass

NEXT STEPS:
<specific actions to take>
```

## Common Scenarios

### Scenario 1: Test Fails with TypeError

```
TypeError: unsupported operand type(s) for -: 'Matrix3x3' and 'Matrix3x3'
```

**Analysis:**
1. Test is trying to subtract matrices
2. Matrix3x3 lacks `__sub__` operator
3. Question: Does test need subtraction, or can it verify another way?

**Bad solution:** Add `__sub__` operator
**Good solution:** Rewrite test to verify matrix properties using existing API

```python
# Instead of: diff = matrix1 - matrix2
# Use: verify both matrices produce same transformation
result1 = matrix1 @ test_vector
result2 = matrix2 @ test_vector
assert result1 == result2
```

### Scenario 2: Test Needs Matrix Values

```
Test needs to verify rotation matrix is correct
```

**Bad approach:**
```python
assert np.allclose(matrix._to_array(), expected)  # W0212!
```

**Good approach:**
```python
# Verify mathematical properties
assert matrix.determinant == pytest.approx(1.0)  # Rotation matrices have det=1
assert (matrix.T @ matrix).determinant == pytest.approx(1.0)  # Orthogonal
result = matrix @ Vector3.from_array([1, 0, 0])
assert result == expected_direction  # Verify transformation
```

### Scenario 3: Abstract Methods Were Removed

```
TypeError: Can't instantiate abstract class with abstract methods X, Y
```

**Analysis:** Type C failure - code incorrectly removed

**Solution:** Restore the abstract methods (they're required by ABC)

## Remember

**"Don't fix problems by creating new problems."**

- If fix requires protected access → WRONG approach
- If fix adds unnecessary API → Question the test
- If fix would fail validation → Find different approach
- When uncertain → Favor simplicity and existing patterns

## Reference

See `reference.md` for:
- Complete public API listing for all VOs
- Validation rule details
- More scenario examples
