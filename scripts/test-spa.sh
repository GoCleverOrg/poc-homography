#!/usr/bin/env bash
set -euo pipefail

# ── SPA Testing Session Bootstrap ────────────────────────────────────
# Starts the FastAPI backend + Vite dev server and ensures a test user
# exists in the database. Opens the browser when ready.
#
# Usage:
#   ./scripts/test-spa.sh              # default: user=admin pass=admin
#   ./scripts/test-spa.sh myuser mypass
#
# Prerequisites:
#   - DATABASE_URL env var set (or .env file with it)
#   - uv installed
#   - node/npm installed
#   - spa/node_modules installed (script runs npm install if missing)

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPA_DIR="$PROJECT_ROOT/spa"
TEST_USER="${1:-admin}"
TEST_PASS="${2:-admin}"

API_PORT=8000
SPA_PORT=5173

cleanup() {
    echo ""
    echo "Shutting down..."
    [ -n "${API_PID:-}" ] && kill "$API_PID" 2>/dev/null || true
    [ -n "${SPA_PID:-}" ] && kill "$SPA_PID" 2>/dev/null || true
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# ── 1. Check prerequisites ───────────────────────────────────────────

if ! command -v uv &>/dev/null; then
    echo "Error: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if ! command -v node &>/dev/null; then
    echo "Error: node not found. Install Node.js >= 18."
    exit 1
fi

if [ ! -f "$SPA_DIR/package.json" ]; then
    echo "Error: spa/package.json not found. Is the SPA scaffolded?"
    exit 1
fi

# ── 2. Check DATABASE_URL ────────────────────────────────────────────

if [ -z "${DATABASE_URL:-}" ]; then
    if [ -f "$PROJECT_ROOT/.env" ]; then
        # shellcheck disable=SC2046
        export $(grep -v '^\s*#' "$PROJECT_ROOT/.env" | grep 'DATABASE_URL' | xargs)
    fi
fi

if [ -z "${DATABASE_URL:-}" ]; then
    echo "Error: DATABASE_URL is not set."
    echo "Set it in your environment or add it to .env"
    exit 1
fi

echo "DATABASE_URL is set."

# ── 3. Run Alembic migrations ────────────────────────────────────────

echo "Running database migrations..."
cd "$PROJECT_ROOT"
uv run alembic upgrade head

# ── 4. Seed test user (idempotent) ───────────────────────────────────

echo "Ensuring test user '$TEST_USER' exists..."
cd "$PROJECT_ROOT"
uv run python -c "
import uuid, bcrypt, sys
from poc_homography.infrastructure.database import get_sessionmaker
from poc_homography.infrastructure.models.user import UserModel

session = get_sessionmaker()()
try:
    existing = session.query(UserModel).filter(UserModel.username == '$TEST_USER').first()
    if existing:
        print(f'User \"$TEST_USER\" already exists (id={existing.id}).')
    else:
        pw_hash = bcrypt.hashpw(b'$TEST_PASS', bcrypt.gensalt()).decode()
        # Pick the first tenant from the database
        from poc_homography.infrastructure.repositories import RepoPostgresTenant
        tenants = RepoPostgresTenant(session).get_all()
        if not tenants:
            print('Warning: no tenants found in the database.', file=sys.stderr)
            sys.exit(1)
        tenant_id = tenants[0].id
        user = UserModel(
            id=str(uuid.uuid4()),
            username='$TEST_USER',
            hashed_password=pw_hash,
            tenant_id=tenant_id,
            is_active=True,
        )
        session.add(user)
        session.commit()
        print(f'Created user \"$TEST_USER\" (tenant={tenant_id}).')
finally:
    session.close()
"

# ── 5. Install SPA dependencies (if needed) ─────────────────────────

if [ ! -d "$SPA_DIR/node_modules" ]; then
    echo "Installing SPA dependencies..."
    cd "$SPA_DIR"
    npm install --legacy-peer-deps
fi

# ── 6. Kill any existing processes on our ports ──────────────────────

for port in $API_PORT $SPA_PORT; do
    pid=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "Port $port in use (pid $pid), stopping it..."
        kill "$pid" 2>/dev/null || true
        sleep 1
    fi
done

# ── 7. Start FastAPI backend ─────────────────────────────────────────

echo "Starting FastAPI backend on port $API_PORT..."
cd "$PROJECT_ROOT"
uv run uvicorn api.main:app --reload --port "$API_PORT" &
API_PID=$!

# ── 8. Start Vite dev server ─────────────────────────────────────────

echo "Starting Vite dev server on port $SPA_PORT..."
cd "$SPA_DIR"
npm run dev -- --port "$SPA_PORT" &
SPA_PID=$!

# ── 9. Wait for both servers to be ready ─────────────────────────────

echo "Waiting for servers..."

for i in $(seq 1 30); do
    if curl -sf "http://localhost:$API_PORT/health" >/dev/null 2>&1; then
        echo "  FastAPI ready."
        break
    fi
    sleep 1
done

for i in $(seq 1 20); do
    if curl -sf "http://localhost:$SPA_PORT/" >/dev/null 2>&1; then
        echo "  Vite ready."
        break
    fi
    sleep 1
done

# ── 10. Open browser ─────────────────────────────────────────────────

URL="http://localhost:$SPA_PORT"
echo ""
echo "=========================================="
echo "  SPA ready at: $URL"
echo "  Login with:   $TEST_USER / $TEST_PASS"
echo "=========================================="
echo ""

if command -v open &>/dev/null; then
    open "$URL"
elif command -v xdg-open &>/dev/null; then
    xdg-open "$URL"
fi

echo "Press Ctrl+C to stop both servers."
wait
