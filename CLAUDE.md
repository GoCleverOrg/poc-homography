# Claude Code Project Guide

## Tools

All tools run via `uv run`:

| Tool | Command |
|------|---------|
| Type check | `uv run pyright poc_homography` |
| Lint | `uv run ruff check poc_homography` |
| Format | `uv run ruff format poc_homography` |
| Test | `uv run pytest tests/ -v` |
| Dead code | `uv run vulture poc_homography` |

## Task Runner

```bash
uv run poe validate  # lint + format-check + typecheck + vulture
uv run poe ci        # validate + tests
```
