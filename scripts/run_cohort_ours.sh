#!/bin/bash
# Generic test30 cohort driver for OUR posterior loop (2026-08-18). One patient at a time on
# GPU0, the standing (split=test, run=i, seed=1000+i) triples, resume-safe (skips existing
# result.pt). Usage:
#   scripts/run_cohort_ours.sh <out_root> [extra run_posterior3d flags...]
# e.g.
#   scripts/run_cohort_ours.sh data/w3dm_test30 --prior w3dm --w3dm_tmax 500
#   scripts/run_cohort_ours.sh data/linbridge_test30 --ckpt logs/fm3d_linbridge/ckpt_iter500000.pth
# Default --ckpt (databridge 500k) applies unless overridden in the extra flags.
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
cd "$(dirname "$0")/.."
OUT_ROOT=$1; shift
CKPT=logs/fm3d_databridge/ckpt_iter500000.pth
for a in "$@"; do [ "$a" = "--ckpt" ] && CKPT=""; done   # caller supplies their own
for i in $(seq 0 29); do
    tag=$(printf "p%02d" "$i")
    out="$OUT_ROOT/$tag"
    if [ -f "$out/result.pt" ]; then
        echo "== $tag: exists, skip"
        continue
    fi
    echo "== $tag: start $(date '+%F %T')"
    CUDA_VISIBLE_DEVICES=0 $PY -u scripts/run_posterior3d.py \
        ${CKPT:+--ckpt $CKPT} --split test --run "$i" --seed $((1000 + i)) \
        --out "$out" "$@" > "/tmp/cohort_$(basename "$OUT_ROOT")_${tag}.log" 2>&1
    echo "== $tag: rc=$? $(date '+%F %T')"
done
echo "COHORT DONE $(date '+%F %T')"
