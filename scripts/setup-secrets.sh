#!/usr/bin/env bash
set -euo pipefail

# Setup GitHub Actions secrets from Bitwarden Secrets Manager
# Usage: BWS_ACCESS_TOKEN=<token> ./scripts/setup-secrets.sh

# Bitwarden Secrets Manager project: "GoClever"
BWS_PROJECT_ID="19f59ce7-1fec-4885-b86b-b3b701408989"

: "${BWS_ACCESS_TOKEN:?BWS_ACCESS_TOKEN not set}"

echo "Fetching secrets from Bitwarden Secrets Manager..."

# Set GitHub secrets for database passwords
bws run --project-id "$BWS_PROJECT_ID" -- bash -c '
  echo "Setting DATABASE_PASSWORD_APP secret (for runtime)..."
  echo "$HOM_DB_PASSWORD_APP" | gh secret set DATABASE_PASSWORD_APP

  echo "Setting DATABASE_PASSWORD_MIGRATE secret (for migrations)..."
  echo "$HOM_DB_PASSWORD_MIGRATE" | gh secret set DATABASE_PASSWORD_MIGRATE
'

echo ""
echo "GitHub secrets configured:"
gh secret list
