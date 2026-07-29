#!/usr/bin/env bash
# P0 of the akima-5mm/5deg re-tune (2026-07-26): re-establish the MOTION ESTIMATOR at the
# corrected setting, outside the posterior loop.
#
# WHY THIS SUITE, AND WHY THREE REFERENCES. The in-loop theta trace at 5/5 is still descending
# at the final ODE step (val0 3.51 -> 0.46 deg over 50 steps, val2 4.51 -> 1.34, both monotone),
# while at the old 3mm/2deg it was flat from step ~35 at 0.08 deg. The estimator is therefore
# BUDGET-limited, not stuck. Two things follow, and this sweep separates them:
#   --ref gt    the CEILING. 2500 iterations = the loop's own N50 x PER50 budget, on a perfect
#               reference. If theta still lands ~0.5 deg here, no amount of image-path work can
#               help and the estimator's structure/rate is the entire gap.
#   --ref cold  what step 0 actually gets (uncorrected FDK, ~23 dB).
#   --ref xt    what the late loop gets (the carried x_t of the akima55 baseline, ~34 dB).
# A config that only wins on a clean reference is useless at step 0, hence all three.
#
# Axes (see SUITES["akima55"] in exp_est_sweep.py): lr on the DEPLOYED fullband encoder (earlier
# lr sweeps varied it on hashbl only), the net's tanh HEADROOM (default rot_max 8 deg vs the
# measured 5.5 deg peak attenuates the gradient ~2x exactly where the trajectory is largest),
# views/iter, and lncc WITH ITS OWN LR.
#
# Fixed iterations, not iso-cost: the loop grants ITERATIONS, so views48 genuinely costs 2x and
# must be read as such. ~8 min per 24-view config, ~4.5 h in total.
#   GPU=0 setsid nohup bash scripts/drivers/drive_est_akima55.sh > logs/est_akima55.log 2>&1 &
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=data/runs/akima55/est_sweep

for ref in gt cold xt; do
    echo "=== ref=$ref  $(date) ==="
    $PY scripts/exp_est_sweep.py --suite akima55 --ref "$ref" \
        --iters_per_config 2500 --run 0 --seed 3 --out "$OUT"
done
echo "=== EST SWEEP akima55 DONE  $(date) ==="
