#!/usr/bin/env bash
# Recreate .venv with Python 3.10+ (fixes Google EOL and urllib3/OpenSSL warnings).
# Run from project root:  bash scripts/setup_venv.sh

set -e
cd "$(dirname "$0")/.."

PYTHON=""
for cmd in python3.12 python3.11 python3.10; do
  if command -v "$cmd" &>/dev/null; then
    PYTHON="$cmd"
    break
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "No Python 3.10+ found. Install one, e.g.:"
  echo "  macOS:  brew install python@3.12"
  echo "  Then run this script again."
  exit 1
fi

echo "Using: $($PYTHON --version)"
rm -rf .venv
"$PYTHON" -m venv .venv
.venv/bin/pip install -r requirements.txt
echo "Done. Activate with:  source .venv/bin/activate"
