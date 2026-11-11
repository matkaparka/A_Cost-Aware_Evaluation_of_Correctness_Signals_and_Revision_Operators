#!/usr/bin/env bash
# Scan staged changes for credentials before committing (CLAUDE.md rule 10).
# Usage: scripts/check_secrets.sh        (exit 1 and list matches if anything looks like a secret)
set -euo pipefail

patterns=(
  'JL_API_KEY[[:space:]]*[=:][[:space:]]*"?[A-Za-z0-9_-]{20,}'
  'hf_[A-Za-z0-9]{30,}'
  '-----BEGIN [A-Z ]*PRIVATE KEY-----'
  'AKIA[0-9A-Z]{16}'
  'sk-[A-Za-z0-9_-]{20,}'
  'gh[pousr]_[A-Za-z0-9]{30,}'
)

found=0
for p in "${patterns[@]}"; do
  if git diff --cached -U0 --no-color | grep -E '^\+' | grep -v '^+++' | grep -nE -e "$p" >/dev/null; then
    echo "possible secret matching: $p" >&2
    found=1
  fi
done

if [ "$found" -ne 0 ]; then
  echo "check_secrets: refusing; unstage or remove the matches above." >&2
  exit 1
fi
echo "check_secrets: clean"
