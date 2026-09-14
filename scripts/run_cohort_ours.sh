#!/usr/bin/env bash
# Compatibility entry point for the current paper's 30-patient protocol.
# Usage: scripts/run_cohort_ours.sh OUT [--ckpt FILE] [--root CQ500] [--bridge linear]
set -euo pipefail
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 OUT [reproduce.py cohort options]" >&2
    exit 2
fi
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COHORT_OUT="$1"
shift
PY_EXEC="${PYTHON:-python}"
exec "$PY_EXEC" "$REPO_DIR/scripts/reproduce.py" cohort --out "$COHORT_OUT" \
    --ckpt "$REPO_DIR/logs/fm3d_databridge/ckpt_iter500000.pth" "$@"
