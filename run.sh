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
  3. Patch pydrive2 for Shared Drive folder-level access
  4. Pull DVC-tracked data from Google Drive (test data + map images)
  5. Validate map images are present
  6. Kill stale Django server on target port (if running)
  7. Apply Django migrations
  8. Start Django dev server (webapp/manage.py runserver)

First-time setup:
  1. Copy and edit the environment file:
       cp .env.example .env
     Required variables: CAMERA_USERNAME, CAMERA_PASSWORD
     Optional per-tenant: {TENANT_ID}_CAMERA_USERNAME, {TENANT_ID}_CAMERA_PASSWORD

  2. Configure DVC Google Drive authentication:
     The default DVC OAuth app is blocked by Google. You need to set up
     credentials in .dvc/config.local (gitignored, never committed):

       dvc remote modify gdrive gdrive_client_id 'YOUR_CLIENT_ID' --local
       dvc remote modify gdrive gdrive_client_secret 'YOUR_SECRET' --local

     To get credentials, either:
       a) Ask a teammate for the shared client_id/secret, or
       b) Create your own at https://console.cloud.google.com/apis
          (enable Drive API, create Desktop OAuth client)

     On first 'dvc pull', a browser opens for one-time Google Drive OAuth.
     Credentials are cached at ~/Library/Caches/pydrive2fs/ (macOS).

  3. Run the server:
       ./run.sh                    # default: 127.0.0.1:8000
       ./run.sh 0.0.0.0:8000      # bind to all interfaces
EOF
    exit 0
fi

# ── 0. Worktree setup (symlink .env, DVC config, map images) ────────
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

# ── 3. Patch pydrive2 for Shared Drive folder-level access ─────────
uv run python scripts/patch_pydrive2_shared_drive.py

# ── 4. Pull DVC-tracked data (skip in worktrees — data comes via setup) ─
_git_common_dir="$(git rev-parse --git-common-dir)"
if [ "$_git_common_dir" = ".git" ]; then
    echo "Pulling DVC-tracked data..."
    if ! uv run dvc pull --force 2>&1; then
        echo "WARNING: dvc pull failed."
        echo "  Possible causes:"
        echo "    - Missing GDrive auth: configure .dvc/config.local with gdrive_client_id/secret"
        echo "    - First run: 'poe dvc-pull' will open a browser for Google Drive OAuth"
    fi
else
    echo "Worktree detected — skipping dvc pull (data provided by worktree-setup)"
fi

DVC_DATA_DIR="tests/homography/test_data"
if [ ! -d "$DVC_DATA_DIR" ] || [ -z "$(ls -A "$DVC_DATA_DIR" 2>/dev/null)" ]; then
    echo "ERROR: DVC test data missing at $DVC_DATA_DIR"
    echo ""
    echo "  poe dvc-pull"
    echo ""
    echo "On first run this will open a browser for Google Drive OAuth."
    exit 1
fi
echo "DVC data OK ($(ls "$DVC_DATA_DIR" | wc -l | tr -d ' ') files)"

# ── 5. Validate map images are present (DVC-tracked GeoTIFFs) ─────
MAPS_DIR="data/maps"
missing_maps=0
for dvc_file in "$MAPS_DIR"/*.tif.dvc; do
    tif_file="${dvc_file%.dvc}"
    if [ ! -f "$tif_file" ]; then
        echo "ERROR: Map image missing: $tif_file"
        missing_maps=1
    fi
done
if [ "$missing_maps" -eq 1 ]; then
    echo ""
    echo "  poe dvc-pull"
    echo ""
    exit 1
fi
echo "Map images OK"

# ── 6. Kill stale Django server on target port ──────────────────────
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

# ── 7. Apply Django migrations (creates db.sqlite3 if missing) ─────
cd webapp
uv run python manage.py migrate --verbosity 0

# ── 8. Run Django development server ────────────────────────────────
exec uv run python manage.py runserver "$@"
