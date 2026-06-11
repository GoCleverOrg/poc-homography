#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ── Help ─────────────────────────────────────────────────────────────
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    cat <<'EOF'
Usage: ./run.sh [--help] [manage.py runserver options...]

Starts the Django development server after validating the environment.

Steps performed:
  1. Check .env file exists (required for camera credentials)
  2. Install/sync Python dependencies (uv sync)
  3. Kill stale Django server on target port (if running)
  4. Apply Django migrations
  5. Start Django dev server (webapp/manage.py runserver)

Map GeoTIFF assets are served from S3/MinIO (see docs/map_asset_storage.md);
no local map materialization is required to run the server.

First-time setup:
  1. Copy and edit the environment file:
       cp .env.example .env
     Required variables: CAMERA_USERNAME, CAMERA_PASSWORD
     Optional per-tenant: {TENANT_ID}_CAMERA_USERNAME, {TENANT_ID}_CAMERA_PASSWORD

  2. Run the server:
       ./run.sh                    # default: 127.0.0.1:8000
       ./run.sh 0.0.0.0:8000      # bind to all interfaces
EOF
    exit 0
fi

# ── 0. Worktree setup (symlink .env, copy test fixtures) ────────────
bash scripts/worktree-setup.sh

# ── 1. Check .env file ──────────────────────────────────────────────
if [ ! -f .env ]; then
    echo "ERROR: .env file not found."
    echo ""
    echo "  cp .env.example .env"
    echo "  # then edit .env with your actual camera credentials"
    echo ""
    echo "Required variables: CAMERA_USERNAME, CAMERA_PASSWORD"
    echo "Optional per-tenant: {TENANT_ID}_CAMERA_USERNAME, {TENANT_ID}_CAMERA_PASSWORD"
    exit 1
fi

# ── 2. Install/sync dependencies ────────────────────────────────────
uv sync --quiet

# Map GeoTIFF assets are served from S3/MinIO (resolved via Map.asset_key);
# the server needs no local .tif materialization. See docs/map_asset_storage.md.

# ── 3. Kill stale Django server on target port ──────────────────────
# Parse port from args (default: 8000)
_port=8000
for _arg in "$@"; do
    case "$_arg" in
        *:*) _port="${_arg##*:}" ;;
        [0-9]*) _port="$_arg" ;;
    esac
done

_pid=$(lsof -ti :"$_port" 2>/dev/null || true)
if [ -n "$_pid" ]; then
    # Only kill if it's a Python process (i.e. another Django runserver)
    _cmd=$(ps -o comm= -p "$_pid" 2>/dev/null || true)
    if [[ "$_cmd" == *Python* || "$_cmd" == *python* ]]; then
        echo "Killing existing server on port $_port (PID $_pid)..."
        kill "$_pid"
        # Wait up to 3s for clean shutdown
        for _ in 1 2 3; do
            kill -0 "$_pid" 2>/dev/null || break
            sleep 1
        done
        # Force kill if still running
        kill -0 "$_pid" 2>/dev/null && kill -9 "$_pid" 2>/dev/null || true
    fi
fi

# ── 4. Apply Django migrations (creates db.sqlite3 if missing) ─────
cd webapp
uv run python manage.py migrate --verbosity 0

# ── 5. Run Django development server ────────────────────────────────
exec uv run python manage.py runserver "$@"
