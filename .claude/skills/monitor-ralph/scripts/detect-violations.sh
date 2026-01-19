#!/bin/bash
# Detect validation violations (pylint W0212 protected-access)

TARGET="${1:-.}"  # Default: check working directory, or pass "HEAD" for last commit

check_pylint_violations() {
  local target="$1"

  if [[ "$target" == "HEAD" ]] || [[ "$target" =~ ^[a-f0-9]+$ ]]; then
    # Check a specific commit
    echo "Checking commit: $target"

    # Get list of Python files changed
    FILES=$(git diff-tree --no-commit-id --name-only -r "$target" | grep '\.py$')

    if [[ -z "$FILES" ]]; then
      echo "No Python files in commit"
      return 0
    fi

    # Check each file
    VIOLATIONS=0
    for file in $FILES; do
      if [[ -f "$file" ]]; then
        COUNT=$(pylint "$file" 2>/dev/null | grep -c "W0212.*protected" || echo "0")
        if [[ $COUNT -gt 0 ]]; then
          echo "❌ $file: $COUNT W0212 violations"
          VIOLATIONS=$((VIOLATIONS + COUNT))
        fi
      fi
    done

    if [[ $VIOLATIONS -gt 0 ]]; then
      echo "Total violations: $VIOLATIONS"
      return 1
    else
      echo "✅ No W0212 violations found in commit"
      return 0
    fi

  else
    # Check working directory or specific path
    echo "Checking: $target"

    # Run pylint
    if command -v uv &> /dev/null; then
      PYLINT_CMD="uv run pylint"
    else
      PYLINT_CMD="pylint"
    fi

    VIOLATIONS=$($PYLINT_CMD "$target" 2>/dev/null | grep -c "W0212.*protected" || echo "0")

    if [[ $VIOLATIONS -gt 0 ]]; then
      echo "❌ Found $VIOLATIONS W0212 violations in $target"
      $PYLINT_CMD "$target" 2>&1 | grep "W0212.*protected" | head -10
      return 1
    else
      echo "✅ No W0212 violations found"
      return 0
    fi
  fi
}

check_protected_method_usage() {
  local target="$1"

  # Look for ._to_array() and other common protected method calls
  PROTECTED_CALLS=$(grep -r "\._to_array()\|\.\_\w\+(" "$target" --include="*.py" 2>/dev/null | grep -v "^#" | wc -l)

  if [[ $PROTECTED_CALLS -gt 0 ]]; then
    echo "⚠️  Found $PROTECTED_CALLS protected method calls (._method_name)"
    grep -rn "\._to_array()\|\.\_\w\+(" "$target" --include="*.py" 2>/dev/null | grep -v "^#" | head -5
    return 1
  fi

  return 0
}

# Main execution
echo "=== Validation Violation Detection ==="
echo

# Check pylint violations
check_pylint_violations "$TARGET"
PYLINT_STATUS=$?

echo

# Check for protected method usage in code
if [[ "$TARGET" != "HEAD" ]] && [[ ! "$TARGET" =~ ^[a-f0-9]+$ ]]; then
  check_protected_method_usage "$TARGET"
  PATTERN_STATUS=$?
else
  PATTERN_STATUS=0
fi

# Exit with error if either check failed
if [[ $PYLINT_STATUS -ne 0 ]] || [[ $PATTERN_STATUS -ne 0 ]]; then
  echo
  echo "🚨 VALIDATION VIOLATIONS DETECTED!"
  exit 1
else
  echo
  echo "✅ All checks passed"
  exit 0
fi
