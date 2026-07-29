#!/usr/bin/env bash
# The user's two questions (2026-07-27), in the order agreed.
#
# Q2  IS THIES' ESTIMATOR BETTER THAN OUR FULL-BAND MLP?  "Thies' way" changes the MODEL (Akima /
#     B-spline, 30 control points = 180 dof) AND the OPTIMIZER (plain GD, s0 = 100, exponential
#     decay 0.97) at once, so `struct` is a 2x2 that can attribute the result, plus band-limit
#     controls (10 / 30 / 60 control points, and `direct` = none).
#     THE USER'S HYPOTHESIS: the spline regularizes toward smoother motion, so the FDK comes out
#     smoother and better. `rot_rmse` is blind to that, so every row is also scored with
#     `trajectory_roughness`. A 200-iteration smoke already points that way -- bspl30+Adam reached
#     rot 0.840 with roughness 2.6x the true trajectory and 0.8% high-frequency energy, against
#     our MLP's 2.642 deg, 23.1x and 6.3%.
#     Two references: `gt` (the ceiling) and `xt` (what the late loop actually hands it). `cold` is
#     skipped -- every config plateaus at ~2 deg there, so it cannot rank anything.
#
# Q1a WHERE TO SPEND: views_per_iter, ISO-COST. Every views comparison this project has run fixed
#     the ITERATION count, which just rediscovers that more views is better. `--budget` fixes
#     VIEW-EVALUATIONS instead, so views and iterations trade against each other at constant cost,
#     which is the only form of the question that can decide anything. 60,000 view-evals = the
#     deployed 24 x 2500.
#
#   GPU=0 setsid nohup bash scripts/drivers/drive_q_estimator.sh > logs/q_estimator.log 2>&1 &
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=data/runs/akima55/est_sweep

while pgrep -f "run_posterior3d.py" > /dev/null; do sleep 30; done
echo "=== Q2 structure bench  $(date) ==="
for ref in gt xt; do
    echo "--- struct ref=$ref  $(date) ---"
    $PY scripts/exp_est_sweep.py --suite struct --ref "$ref" \
        --iters_per_config 2500 --run 0 --seed 3 --out "$OUT"
done

echo "=== Q1a views iso-cost  $(date) ==="
for ref in gt xt; do
    echo "--- views ref=$ref  $(date) ---"
    $PY scripts/exp_est_sweep.py --suite views --ref "$ref" \
        --budget 60000 --run 0 --seed 3 --out "$OUT"
done
echo "=== Q-ESTIMATOR BENCH DONE  $(date) ==="
