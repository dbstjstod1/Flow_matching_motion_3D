#!/usr/bin/env bash
# Run inside the upstream JRM-ADM clone after copying this adapter there.
# MODEL_PATH selects OUR retrained prior explicitly; no fallback to released weights.
set -euo pipefail
PY_EXEC="${PYTHON:-python}"
: "${MODEL_PATH:?Set MODEL_PATH to the retrained EMA model weights}"
cd "$(dirname "$0")"
CASE_DIR="${CASE_DIR:-data/ours_cohort}"
RESULT_DIR="${RESULT_DIR:-data/recon_ours_v2}"
mkdir -p "$RESULT_DIR"
for i in $(seq 0 29); do
    tag=$(printf 'p%02d' "$i")
    if [ -e "$RESULT_DIR/${tag}_result.pt" ]; then
        echo "Existing result: $RESULT_DIR/${tag}_result.pt; choose a fresh RESULT_DIR." >&2
        exit 1
    fi
    "$PY_EXEC" -u run_on_ours.py --case "$CASE_DIR/$tag.pt" \
        --out "$RESULT_DIR" --vol 224 224 224 --angle_batch 36 \
        --gamma 33000 --prior_zflip --model_path "$MODEL_PATH" \
        > "$RESULT_DIR/$tag.log" 2>&1
    echo "Completed $tag"
done
