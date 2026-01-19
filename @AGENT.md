# Build and Validation Instructions

## Environment Setup

This project uses `uv` for Python dependency management.

```bash
# Ensure uv is available
which uv
```

## Validation Commands

### Full Validation (Use this before committing)
```bash
uv run poe validate
```

This runs:
- `ruff check` - Linting
- `ruff format --check` - Format checking
- `pyright` - Type checking
- `vulture` - Dead code detection
- `pylint` - Protected access checking

### Individual Checks

```bash
# Type checking
uv run pyright poc_homography

# Linting
uv run ruff check poc_homography

# Formatting
uv run ruff format poc_homography

# Dead code detection
uv run vulture poc_homography

# Protected access check
uv run pylint poc_homography
```

## Testing

```bash
# Run all tests with verbose output
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_specific.py -v

# Run with coverage
uv run pytest tests/ --cov=poc_homography --cov-report=term-missing
```

## Full CI Pipeline (What Ralph Should Run)

```bash
# Complete validation + tests
uv run poe ci
```

## Git Workflow

After each successful code elimination:

```bash
git add .
git commit -m "refactor(module): remove unused method/variable X"
```

## Exit Criteria

Ralph should continue until:
1. `uv run vulture poc_homography` returns no high-confidence findings
2. `uv run poe validate` passes cleanly
3. `uv run pytest tests/ -v` passes all tests
