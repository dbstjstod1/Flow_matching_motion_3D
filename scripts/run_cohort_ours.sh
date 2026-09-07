#!/bin/bash
# The 30-patient test cohort of the paper: (split=test, run=i, seed=1000+i), i = 0..29, one
# patient at a time, resume-safe (skips patients whose result.pt exists).
#
#   scripts/run_cohort_ours.sh <out_root> [extra run_posterior3d flags...]
#   scripts/run_cohort_ours.sh data/test30
#   scripts/run_cohort_ours.sh data/linbridge_test30 --ckpt logs/fm3d_linbridge/ckpt_iter500000.pth
#
# Every comparison method is run on the same (split, run, seed) triples, which is what makes the
# comparisons paired. About 9.3 min per patient on one RTX A6000.
set -u
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
OUT_ROOT=$1; shift
CKPT=logs/fm3d_databridge/ckpt_iter500000.pth
for a in "$@"; do [ "$a" = "--ckpt" ] && CKPT=""; done   # caller supplies their own
mkdir -p "$OUT_ROOT"
for i in $(seq 0 29); do
    tag=$(printf "p%02d" "$i")
    out="$OUT_ROOT/$tag"
    if [ -f "$out/result.pt" ]; then
        echo "== $tag: exists, skip"
        continue
    fi
    echo "== $tag: start $(date '+%F %T')"
    $PY -u scripts/run_posterior3d.py \
        ${CKPT:+--ckpt $CKPT} --split test --run "$i" --seed $((1000 + i)) \
        --out "$out" "$@" > "$OUT_ROOT/$tag.log" 2>&1
    echo "== $tag: rc=$? $(date '+%F %T')"
done
echo "COHORT DONE $(date '+%F %T')"
