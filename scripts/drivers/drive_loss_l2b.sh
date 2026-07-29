#!/usr/bin/env bash
# Continuation of drive_loss_l2.sh with the BENCH CUT DOWN (user: the sweep takes too long).
#
# WHAT WAS CUT AND WHY. The original Phase 2 was 6 configs x 2 references x 20,000 iterations
# (~2 h). Two of those axes were not earning their time:
#   - the `cold` reference is dropped outright. Measured 2026-07-26: on a cold FDK EVERY estimator
#     config plateaus at ~2.0 deg and the ranking collapses (1.99-2.12 across lr, loss, bandwidth
#     and views) because the early loop is IMAGE-limited, not estimator-limited. A loss comparison
#     there cannot resolve anything. `xt` is where theta accuracy is actually decided.
#   - 20,000 -> 5,000 iterations. The curve is logged every 250 iterations, so this truncates the
#     x-axis rather than coarsening it, and l2-vs-l2si is a question about the gradient, which is
#     visible in the first thousand iterations if it is visible at all.
# ~15 min instead of ~2 h. The decision itself is already made and wired (--loss l2 is the
# default); this is confirmation, not evidence the default is waiting on.
#
# Phase 1 (in-loop, val 1 and 2 -- val 0 was already running when this was written) is UNCHANGED:
# it is the decisive end-to-end A/B against the existing l2si arm `c2f_v*`, and 3 patients is the
# minimum that can resolve anything given the loop's ~15-20% run-to-run spread on rot.
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while pgrep -f "run_posterior3d.py --ckpt" > /dev/null; do sleep 60; done
echo "=== l2 vs l2si (trimmed) resumes $(date) ==="

for i in 1 2; do
    case $i in 1) S=7;; 2) S=11;; esac
    OUT=data/runs/akima55/l2_v$i
    mkdir -p $OUT
    if [ -f "$OUT/result.pt" ]; then echo "skip $OUT (done)"; continue; fi
    echo "--- l2_v$i  (val $i seed $S)  $(date) ---"
    $PY scripts/run_posterior3d.py --ckpt $CK --split val --run $i --seed $S \
        --loss l2 --out $OUT > $OUT.log 2>&1
    echo "--- l2_v$i done $(date) ---"
done

$PY scripts/cmp_runs.py data/runs/akima55/c2f_v0 data/runs/akima55/c2f_v1 data/runs/akima55/c2f_v2 \
    data/runs/akima55/l2_v0 data/runs/akima55/l2_v1 data/runs/akima55/l2_v2 \
    2>&1 | tee data/runs/akima55/cmp_loss_l2.txt

$PY scripts/exp_est_sweep.py --suite loss --ref xt --coarse 2 --iters_per_config 5000 \
    --chunk 250 --xt_from data/runs/akima55/c2f_v0/result.pt \
    2>&1 | tee data/runs/akima55/est_sweep/loss_xt.log

echo "=== l2 vs l2si  ALL DONE $(date) ==="
