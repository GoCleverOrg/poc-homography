#!/usr/bin/env bash
# Generate local .env file from Bitwarden Secrets Manager for development.
# Prerequisites: bws CLI authenticated.
set -euo pipefail

if [ -z "${BWS_PROJECT_ID:-}" ]; then
    echo "Error: BWS_PROJECT_ID environment variable is required." >&2
    exit 1
fi

bws run --project-id "$BWS_PROJECT_ID" -- bash -c '
  printf "DATABASE_PASSWORD=%s\nDATABASE_USER=hom_app\n" "$HOM_DB_PASSWORD_APP" > .env
'

echo ".env file generated successfully."
