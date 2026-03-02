#!/usr/bin/env bash
# Sync database passwords from Bitwarden Secrets Manager to GitHub repository secrets.
# Prerequisites: bws CLI authenticated, gh CLI authenticated.
set -euo pipefail

if [ -z "${BWS_PROJECT_ID:-}" ]; then
    echo "Error: BWS_PROJECT_ID environment variable is required." >&2
    exit 1
fi

bws run --project-id "$BWS_PROJECT_ID" -- bash -c '
  echo "$HOM_DB_PASSWORD_APP" | gh secret set DATABASE_PASSWORD_APP
  echo "$HOM_DB_PASSWORD_MIGRATE" | gh secret set DATABASE_PASSWORD_MIGRATE
'

echo "GitHub secrets synced successfully."
