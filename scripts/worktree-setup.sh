#!/usr/bin/env bash
# Symlink/copy shared assets from the main repo into a git worktree.
# Idempotent — safe to run multiple times. No-op when run from the main repo.
set -euo pipefail

cd "$(dirname "$0")/.."

git_common_dir="$(git rev-parse --git-common-dir)"

# In the main repo --git-common-dir returns ".git"; in a worktree it's absolute.
if [ "$git_common_dir" = ".git" ]; then
    exit 0
fi

main_repo="$(dirname "$git_common_dir")"
echo "Worktree detected — linking shared assets from $main_repo"

# ── Helper: symlink a single file ────────────────────────────────────
link_file() {
    local src="$1" dst="$2"
    if [ -e "$dst" ]; then return; fi
    if [ ! -e "$src" ]; then
        echo "  WARN: $src not found in main repo, skipping"
        return
    fi
    mkdir -p "$(dirname "$dst")"
    ln -s "$src" "$dst"
    echo "  linked $dst"
}

# ── .env ──────────────────────────────────────────────────────────────
link_file "$main_repo/.env" ".env"

# Map GeoTIFF assets are served from S3/MinIO (resolved via Map.asset_key),
# so worktrees need no local .tif materialization. See docs/map_asset_storage.md.

# ── tests/homography/test_data/ (copy for write isolation) ───────────
test_data_dir="tests/homography/test_data"
if [ ! -d "$test_data_dir" ] || [ -z "$(ls -A "$test_data_dir" 2>/dev/null)" ]; then
    src="$main_repo/$test_data_dir"
    if [ -d "$src" ]; then
        mkdir -p "$test_data_dir"
        cp -R "$src"/* "$test_data_dir"/
        echo "  copied $test_data_dir ($(ls "$test_data_dir" | wc -l | tr -d ' ') files)"
    else
        echo "  WARN: $src not found in main repo, skipping"
    fi
fi

echo "Worktree setup complete."
