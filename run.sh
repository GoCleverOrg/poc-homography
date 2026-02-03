#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Install/sync dependencies if needed
uv sync --quiet

# Run Django development server
cd webapp
exec uv run python manage.py runserver "$@"
