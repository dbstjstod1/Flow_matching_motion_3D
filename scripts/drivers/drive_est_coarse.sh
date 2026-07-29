#!/usr/bin/env bash
# P1-a: does the estimator need the 256^3 @ 1 mm grid at all?
#
# Thies fits motion on 128^3 @ 2 mm and only RECONSTRUCTS at 256^3 @ 1 mm; we have always done
# both at 256^3. This matters far beyond taste, because the 5/5 diagnosis is that the estimator
# is BUDGET-limited (its theta is still descending at the last ODE step). A coarse grid buys
# budget directly: volume avg-pooled N, panel binned N (N^2 fewer rays), ray sampling decimated N.
#
# The two questions, in order:
#   1. Does coarse cost ACCURACY per iteration? -> same suite, same 2500 iterations, --coarse 2
#      and 4, against the fine run in sweep_akima55_{ref}.json. First evidence says no: the
#      geometry gate's 200 coarse iters reached rot 2.927 deg, matching 250 FINE iters (2.927).
#   2. If not, what does the saved time buy? -> that is the follow-up leg (more iterations at
#      equal wall-clock), deliberately NOT queued here: its iteration count should be chosen from
#      the measured speed-up rather than guessed.
#
# Geometry of the coarse path is gated by scripts/gate_coarse_est.py (PASS 2026-07-26: 1.07% RMS
# vs the binned fine projection, best-fit scale 0.9992, and a deliberate 10% panel mis-scale is
# caught at 27%).
#
# Waits for the fine sweep to finish so the timings stay clean (they are the point of leg 2).
#   GPU=0 setsid nohup bash scripts/drivers/drive_est_coarse.sh > logs/est_coarse.log 2>&1 &
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=data/runs/akima55/est_sweep

while pgrep -f "exp_est_sweep.py --suite akima55 --ref" > /dev/null; do sleep 60; done
echo "=== fine sweep done, starting coarse  $(date) ==="

for c in 2 4; do
    for ref in gt cold xt; do
        echo "=== coarse=$c ref=$ref  $(date) ==="
        $PY scripts/exp_est_sweep.py --suite akima55 --ref "$ref" --coarse "$c" \
            --iters_per_config 2500 --run 0 --seed 3 --out "$OUT"
    done
done
echo "=== EST COARSE DONE  $(date) ==="
