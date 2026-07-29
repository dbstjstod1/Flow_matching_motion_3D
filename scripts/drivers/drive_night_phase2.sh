#!/usr/bin/env bash
# Phase 2 of the overnight screen, rewritten 02:20 after phase 1's result made two of its own
# planned runs obsolete.
#
# WHAT PHASE 1 ESTABLISHED (val 0, akima 5/5, against the thies_v0 baseline 34.25 dB / 0.9488):
#   lr 1e-3 -> 3e-3     x_t 35.85 / 0.9695, rot 0.456 -> 0.259, RPE zc 0.478 -> 0.308
#   + PER 50 -> 200     x_t 37.59 / 0.9802, rot 0.176, RPE zc 0.190,  at +42% wall clock
# and per200's EVIDENCE channel (FDK(θ̂) vs sFDK, 0.7930) is at **99.9% of the oracle-theta
# ceiling** (0.7935): that volume is finished, and the only headroom left is x_t, 37.59 dB
# against the oracle's 40.45.
#
# WHY THE PLAN CHANGED. Phase 1 still had `ramp` and `lncc` queued AT PER=50 -- i.e. measuring a
# free schedule change and a loss swap at an operating point we are abandoning. Both were killed.
# `lncc` is dropped outright: in the GT sweep it TIED l2si once each had its own best lr (0.087 vs
# 0.093), so its expected value here is ~0. `ramp` is kept but moved to the NEW operating point,
# where a schedule can actually matter (at PER=200 there are enough iterations for the early,
# image-limited steps to be genuinely wasteful).
#
#   c2per1600  1600 iterations on Thies' 128^3 @ 2 mm grid. A coarse iteration MEASURED at ~1/6.4
#              of a fine one (c2per400 runs at 54.0 s/step vs the baseline's 49.0 and per200's
#              69.8), so this is roughly per200's price for 8x per200's work. The frontier run.
#   ramp200    the same 200-iteration average, spent linearly in t. Free if it helps.
#   per200_v1/v2  the 3-patient confirmation of the current best. val 2 matters most: worst
#              estimator (rot 1.338) and largest oracle gain (+10.7 dB).
#
#   GPU=0 setsid nohup bash scripts/drivers/drive_night_phase2.sh > logs/night_phase2.log 2>&1 &
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="--ckpt $CK --split val --est_band fullband --n_steps 50 --views_per_iter 24 \
      --theta_avg 1 --dc_op cg --cg_iters 5 --kappa 0.3 --loss l2si --lr 3e-3"

# Wait on any live loop, then on the phase-1 driver if it is somehow still alive. Waiting on the
# python alone would let this start inside the gap between a driver's runs.
while pgrep -f "drive_night_est_budget.sh" > /dev/null; do sleep 60; done
while pgrep -f "run_posterior3d.py" > /dev/null; do sleep 60; done
echo "=== phase 2 starts  $(date) ==="

go () {  # go <tag> <run> <seed> <args...>
    local tag=$1 p=$2 s=$3; shift 3
    local out=data/runs/akima55/$tag
    if [ -f "$out/result.pt" ]; then echo "SKIP $tag"; return; fi
    echo "=== $tag  $(date) ==="
    $PY scripts/run_posterior3d.py $BASE --run "$p" --seed "$s" --out "$out" "$@"
}

go c2per1600_v0 0 3  --per 1600 --est_coarse 2
go ramp200_v0   0 3  --per 200  --per_sched ramp
go per200_v1    1 7  --per 200
go per200_v2    2 11 --per 200
echo "=== PHASE 2 DONE  $(date) ==="
