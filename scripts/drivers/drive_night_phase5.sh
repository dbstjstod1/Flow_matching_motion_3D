#!/usr/bin/env bash
# Phase 5, 05:50. Confirm the LEADER on the other two patients.
#
# c2f (coarse 128^3 @ 2 mm to t=0.5, then fine 256^3 @ 1 mm, PER=400, lr 3e-3) leads on val 0:
#   config      s/step | x_t vs GT (MAIN) | OUT vs sFDK | x_t vs sFDK | rot   | RPE zc
#   thies_v0     49.0  | 34.25 / 0.9488   | 33.22/0.7764| 32.50/0.8048| 0.456 | 0.478
#   per200       69.8  | 37.59 / 0.9802   | 35.08/0.7930| 33.92/0.8237| 0.176 | 0.190
#   per400       99.3  | 38.16 / 0.9796   | 35.15/0.7927| 33.98/0.8236| 0.125 | 0.181
#   c2f         ~75.5  | 37.88 / 0.9826   | 35.13/0.7926| 34.03/0.8257| 0.090 | 0.125
#   oracle theta 41.7  | 40.45 / 0.9889   | 36.27/0.7935| 34.08/0.8274| 0     | 0
# It wins or ties every cell except x_t PSNR (per400 is +0.28 dB) at 76% of per400's cost, and its
# x_t-vs-sFDK (34.03/0.8257) is essentially AT the oracle's (34.08/0.8274).
#
# CAVEAT THAT SHAPES THIS PHASE. The loop is NOT run-to-run deterministic: c2f and c2per400 share
# an identical configuration for t<0.5 (same patient, same seed) yet their rot traces differ by
# 0.05-0.14 deg (2.33/0.47/0.36/0.29 vs 2.47/0.55/0.42/0.34 at steps 0/5/10/15) -- ~15-20%
# relative. The estimator's backward goes through the Triton projector's scatter, whose atomic
# adds make the float summation order vary, and 400 iterations x 50 steps amplify it. So c2f vs
# per400 (0.090 vs 0.125) sits at the edge of noise while everything vs the baseline does not,
# and a SECOND PATIENT is worth more right now than a fourth val-0 variant.
#
#   GPU=0 setsid nohup bash scripts/drivers/drive_night_phase5.sh > logs/night_phase5.log 2>&1 &
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="--ckpt $CK --split val --est_band fullband --views_per_iter 24 --n_steps 50 \
      --theta_avg 1 --dc_op cg --cg_iters 5 --kappa 0.3 --loss l2si --lr 3e-3"
C2F="--per 400 --est_coarse 2 --est_coarse_until 0.5"

while pgrep -f "drive_night_phase4.sh" > /dev/null; do sleep 30; done
while pgrep -f "run_posterior3d.py" > /dev/null; do sleep 30; done
echo "=== phase 5 starts  $(date) ==="

go () { local tag=$1 p=$2 s=$3; shift 3
    local out=data/runs/akima55/$tag
    [ -f "$out/result.pt" ] && { echo "SKIP $tag"; return; }
    echo "=== $tag  $(date) ==="
    $PY scripts/run_posterior3d.py $BASE --run "$p" --seed "$s" --out "$out" "$@"; }

go c2f_v1 1 7  $C2F
go c2f_v2 2 11 $C2F
go per200_v1 1 7  --per 200     # the cheaper alternative, same patients, for the cost/quality call
go per200_v2 2 11 --per 200
echo "=== PHASE 5 DONE  $(date) ==="
