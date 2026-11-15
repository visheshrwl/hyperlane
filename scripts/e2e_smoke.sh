#!/usr/bin/env bash
# End-to-end check of the vertical slice. See scripts/e2e_smoke.py for what it checks.
#
#   scripts/e2e_smoke.sh --backend cpu
#   scripts/e2e_smoke.sh --backend metal --model tinyllama
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --extra oracle python scripts/e2e_smoke.py "$@"
