#!/usr/bin/env bash
# Overnight screening, 2026-07-27, run autonomously while the user sleeps.
# GOAL: hand back the BEST INFERENCE LOOP at akima 5 mm / 5 deg.
#
# THE TWO MEASUREMENTS THIS QUEUE IS BUILT ON.
#  (1) theta is worth 6.2-10.7 dB on the MAIN deliverable x_t (--theta_oracle, val0 34.25 -> 40.45,
#      val2 32.27 -> 42.97 dB vs GT) but only ~3 dB / 0.017 SSIM on the FDK(theta_hat) evidence
#      channel, whose ceiling is the FDK operator itself. So estimator accuracy is THE lever.
#  (2) the estimator is only ~15% of a step: 49.0 s blind vs 41.7 s with --theta_oracle, i.e.
#      7.3 s per PER=50, or 0.146 s/iteration. It is the CHEAPEST thing in the loop and the most
#      valuable -- which is an argument for spending far more of it, not for tuning it.
# Plus: on a COLD reference every estimator config plateaus at ~2.0 deg and the config ranking
# collapses entirely (1.99-2.12 across lr/loss/bandwidth/views) while on a GT reference the same
# budget reaches 0.087-0.265. The early loop is IMAGE-limited, the late loop ESTIMATOR-limited.
#
# So the screen is about WHERE AND HOW MUCH estimator work to buy, not which knob to turn:
#   per200      4x the budget on the reconstruction grid           (~+22 s/step)
#   c2per400    8x the budget on Thies' 128^3 @ 2 mm grid, which costs ~1/8 per iteration, so
#               this is ~the SAME price as the deployed per50 (gate: scripts/gate_coarse_est.py)
#   ramp        the SAME total budget, spent linearly in t instead of uniformly -- free
#   lncc        the other loss at its own best lr (0.087 deg on the GT leg vs l2si's 0.093)
# All on val 0 / seed 3 against data/runs/akima55/thies_v0 (x_t 34.25 dB / 0.9488 vs GT), with
# --lr 3e-3 (the GT leg's best l2si lr, 0.093 vs the deployed 1e-3's 0.159) except where stated.
# Winners get confirmed on val 1/2 afterwards.
#
#   GPU=0 setsid nohup bash scripts/drivers/drive_night_est_budget.sh > logs/night_est.log 2>&1 &
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="--ckpt $CK --split val --est_band fullband --n_steps 50 --views_per_iter 24 \
      --theta_avg 1 --dc_op cg --cg_iters 5 --kappa 0.3 --run 0 --seed 3"

while pgrep -f "run_posterior3d.py" > /dev/null; do sleep 60; done
echo "=== night screen starts  $(date) ==="

go () {  # go <tag> <args...>
    local tag=$1; shift
    local out=data/runs/akima55/$tag
    if [ -f "$out/result.pt" ]; then echo "SKIP $tag"; return; fi
    echo "=== $tag  $(date) ==="
    $PY scripts/run_posterior3d.py $BASE --out "$out" "$@"
}

go per200_v0   --loss l2si --lr 3e-3 --per 200
go c2per400_v0 --loss l2si --lr 3e-3 --per 400 --est_coarse 2
go ramp_v0     --loss l2si --lr 3e-3 --per 50  --per_sched ramp
go lncc1e3_v0  --loss lncc --lncc_win 9 --lr 1e-3 --per 50
echo "=== NIGHT SCREEN DONE  $(date) ==="
