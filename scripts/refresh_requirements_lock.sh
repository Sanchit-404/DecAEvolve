#!/usr/bin/env bash
# Regenerate requirements-lock.txt from the active Python (use after conda activate llmsr).
set -euo pipefail
cd "$(dirname "$0")/.."
pip freeze > requirements-lock.txt
echo "Wrote requirements-lock.txt ($(wc -l < requirements-lock.txt) lines)."
