#!/bin/bash
# Get the latest Claude output from Ralph loop

LOGS_DIR="${1:-logs}"

# Find most recent claude_output file
LATEST=$(ls -t "$LOGS_DIR"/claude_output_*.log 2>/dev/null | head -1)

if [[ -z "$LATEST" ]]; then
  echo "No Claude output logs found in $LOGS_DIR"
  exit 1
fi

echo "=== Latest Claude Output ==="
echo "File: $LATEST"
echo "Size: $(wc -l < "$LATEST") lines"
echo "Modified: $(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LATEST" 2>/dev/null || stat -c "%y" "$LATEST" 2>/dev/null)"
echo

# Try to parse as JSON first
if jq -e . "$LATEST" > /dev/null 2>&1; then
  echo "Format: JSON"
  echo

  # Extract key fields
  jq -r '.result // .error // "No result field"' "$LATEST" 2>/dev/null | head -50
else
  echo "Format: Text"
  echo

  # Show raw content
  cat "$LATEST"
fi
